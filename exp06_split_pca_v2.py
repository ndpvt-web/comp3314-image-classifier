"""Experiment 6: Split PCA v2 - Focused on best region.
Key findings from exp5:
- HOG-150 better than HOG-175+ (diminishing returns/noise)
- More Other PCA dims = consistently better (up to 150 tested)
- C=5 consistently best
Try: HOG 100-175 with Other 125-200, fine-grained C values.
"""
import os, sys, time, gc, json, csv
import numpy as np
import psutil

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
MODELS = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

def mem():
    return psutil.Process().memory_info().rss / 1e9

HOG_END = 4824
NON_HOG_START = 4824

print(f"[START] RAM: {mem():.2f}GB", flush=True)

# === Load ===
print("Loading features...", flush=True)
all_feat = np.load(os.path.join(CACHE, 'all_features_combined.npy'), mmap_mode='r')
train_feat = np.nan_to_num(np.array(all_feat[:50000], dtype=np.float32))
labels = np.load(os.path.join(CACHE, 'train_labels.npy'))
del all_feat; gc.collect()

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train_feat, labels, test_size=0.2, random_state=42, stratify=labels)
del train_feat, labels; gc.collect()

# === Scale ===
mean = X_train.mean(axis=0, dtype=np.float32)
std = X_train.std(axis=0, dtype=np.float32)
std[std == 0] = 1.0
X_train = np.nan_to_num((X_train - mean) / std)
X_val = np.nan_to_num((X_val - mean) / std)

# Save scaler
np.save(os.path.join(MODELS, 'scaler_mean.npy'), mean)
np.save(os.path.join(MODELS, 'scaler_std.npy'), std)
del mean, std; gc.collect()

# === Power Transform ===
from sklearn.preprocessing import PowerTransformer
import joblib
print("Yeo-Johnson power transform...", flush=True)

np.save(os.path.join(CACHE, 'X_val_scaled_f32.npy'), X_val)
np.save(os.path.join(CACHE, 'y_val.npy'), y_val)
del X_val; gc.collect()

BATCH = 500
n_cols = X_train.shape[1]
X_train_pt = np.empty_like(X_train)
pt_models = []
for start in range(0, n_cols, BATCH):
    end = min(start + BATCH, n_cols)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_pt[:, start:end] = pt.fit_transform(X_train[:, start:end]).astype(np.float32)
    pt_models.append(pt)
X_train_pt = np.nan_to_num(X_train_pt)
del X_train; gc.collect()

X_val_raw = np.load(os.path.join(CACHE, 'X_val_scaled_f32.npy'))
y_val = np.load(os.path.join(CACHE, 'y_val.npy'))
X_val_pt = np.empty_like(X_val_raw)
for idx, start in enumerate(range(0, n_cols, BATCH)):
    end = min(start + BATCH, n_cols)
    X_val_pt[:, start:end] = pt_models[idx].transform(X_val_raw[:, start:end]).astype(np.float32)
X_val_pt = np.nan_to_num(X_val_pt)
del X_val_raw; gc.collect()

# Save PT models for later submission generation
joblib.dump(pt_models, os.path.join(MODELS, 'pt_models.pkl'))
del pt_models; gc.collect()
print(f"  PT done, RAM: {mem():.2f}GB", flush=True)

# === Split features ===
X_tr_hog = X_train_pt[:, :HOG_END]
X_tr_other = X_train_pt[:, NON_HOG_START:]
X_va_hog = X_val_pt[:, :HOG_END]
X_va_other = X_val_pt[:, NON_HOG_START:]
del X_train_pt, X_val_pt; gc.collect()

# === PCA ===
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.save(os.path.join(CACHE, 'X_va_hog.npy'), X_va_hog)
np.save(os.path.join(CACHE, 'X_va_other.npy'), X_va_other)
del X_va_hog, X_va_other; gc.collect()

MAX_HOG_PCA = 200
print(f"PCA on HOG -> max {MAX_HOG_PCA}...", flush=True)
pca_hog = PCA(n_components=MAX_HOG_PCA, random_state=42)
X_tr_hog_pca = pca_hog.fit_transform(X_tr_hog).astype(np.float32)
del X_tr_hog; gc.collect()

MAX_OTHER_PCA = 200
print(f"PCA on Other -> max {MAX_OTHER_PCA}...", flush=True)
pca_other = PCA(n_components=MAX_OTHER_PCA, random_state=42)
X_tr_other_pca = pca_other.fit_transform(X_tr_other).astype(np.float32)
del X_tr_other; gc.collect()

X_va_hog = np.load(os.path.join(CACHE, 'X_va_hog.npy'))
X_va_other = np.load(os.path.join(CACHE, 'X_va_other.npy'))
X_va_hog_pca = pca_hog.transform(X_va_hog).astype(np.float32)
X_va_other_pca = pca_other.transform(X_va_other).astype(np.float32)
del X_va_hog, X_va_other; gc.collect()

# Save PCA models
joblib.dump(pca_hog, os.path.join(MODELS, 'pca_hog.pkl'))
joblib.dump(pca_other, os.path.join(MODELS, 'pca_other.pkl'))
print(f"  PCA done, RAM: {mem():.2f}GB", flush=True)

# === Focused search ===
configs = []

# Sweep: HOG 100-175, Other 125-200
for hog_d in [100, 125, 150, 175]:
    for other_d in [125, 150, 175, 200]:
        for C in [3, 5, 8, 10]:
            configs.append((hog_d, other_d, C, 'scale'))

print(f"\n{'='*60}", flush=True)
print(f"SPLIT PCA v2 SEARCH ({len(configs)} configs)", flush=True)
print(f"{'='*60}", flush=True)

results = []
best_acc = 0.0
best_params = {}

for i, (hog_d, other_d, C, gamma) in enumerate(configs):
    t0 = time.time()
    X_tr = np.hstack([X_tr_hog_pca[:, :hog_d], X_tr_other_pca[:, :other_d]])
    X_va = np.hstack([X_va_hog_pca[:, :hog_d], X_va_other_pca[:, :other_d]])
    total_d = hog_d + other_d

    print(f"[{i+1}/{len(configs)}] HOG-{hog_d}+Other-{other_d}={total_d}, C={C}...", end='', flush=True)

    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42,
              cache_size=1500, decision_function_shape='ovr')
    svm.fit(X_tr, y_train)
    pred = svm.predict(X_va)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0

    tag = " *** NEW BEST ***" if acc > best_acc else ""
    print(f" {acc:.4f} ({elapsed:.0f}s, RAM: {mem():.2f}GB){tag}", flush=True)

    results.append({
        'hog_pca': hog_d, 'other_pca': other_d, 'total_dim': total_d,
        'C': C, 'gamma': str(gamma),
        'val_accuracy': round(acc, 5), 'time_s': round(elapsed, 1)
    })

    if acc > best_acc:
        best_acc = acc
        best_params = {'hog_pca': hog_d, 'other_pca': other_d, 'C': C}
        # Save best model
        if acc > 0.740:
            joblib.dump(svm, os.path.join(MODELS, f'svm_best_{acc:.4f}.pkl'))
            print(f"  Saved model!", flush=True)

    del svm, pred, X_tr, X_va; gc.collect()

# === Results ===
print(f"\n{'='*60}", flush=True)
print(f"BEST: {best_acc:.5f} with {best_params}", flush=True)
print(f"PREV BEST: 0.74350 (HOG-150+Other-150, C=5, exp5)", flush=True)
delta = best_acc - 0.74350
print(f"DELTA: {delta:+.5f}", flush=True)
print(f"{'='*60}", flush=True)

csv_path = os.path.join(RESULTS, 'exp06_split_pca_v2.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['hog_pca', 'other_pca', 'total_dim', 'C', 'gamma', 'val_accuracy', 'time_s'])
    writer.writeheader()
    for r in sorted(results, key=lambda x: -x['val_accuracy']):
        writer.writerow(r)

print("\nTop 20:", flush=True)
for r in sorted(results, key=lambda x: -x['val_accuracy'])[:20]:
    print(f"  HOG-{r['hog_pca']:3d}+Other-{r['other_pca']:3d}={r['total_dim']:3d} C={r['C']:5} -> {r['val_accuracy']:.5f} ({r['time_s']:.0f}s)", flush=True)

print(f"\nFinal RAM: {mem():.2f}GB", flush=True)
