"""Experiment 4: Fine-tune power transform SVM.
Extend PCA range to 350-500 and fine-tune around PCA-250.
Also test manual gamma values.
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
del mean, std; gc.collect()

# === Power Transform (batch) ===
from sklearn.preprocessing import PowerTransformer
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
del X_val_raw, pt_models; gc.collect()
print(f"  PT done, RAM: {mem():.2f}GB", flush=True)

# === PCA up to 500 ===
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.save(os.path.join(CACHE, 'X_val_pt_f32.npy'), X_val_pt)
del X_val_pt; gc.collect()

MAX_PCA = 500
print(f"Fitting PCA-{MAX_PCA}...", flush=True)
pca = PCA(n_components=MAX_PCA, random_state=42)
X_tr_pca = pca.fit_transform(X_train_pt).astype(np.float32)
del X_train_pt; gc.collect()

ev = pca.explained_variance_ratio_.cumsum()
for d in [230, 240, 250, 260, 270, 300, 350, 400, 500]:
    print(f"  PCA-{d}: {ev[d-1]*100:.1f}% variance", flush=True)

X_val_pt = np.load(os.path.join(CACHE, 'X_val_pt_f32.npy'))
X_va_pca = pca.transform(X_val_pt).astype(np.float32)
del X_val_pt; gc.collect()
print(f"  PCA done, RAM: {mem():.2f}GB", flush=True)

# === Search: fine-tune around PCA-250, extend to higher dims ===
configs = []

# Fine-tune around PCA-250 with gamma=scale
for n_pca in [230, 240, 245, 250, 255, 260, 270, 280]:
    for C in [5, 8, 10, 15]:
        configs.append((n_pca, C, 'scale'))

# Test higher PCA dims
for n_pca in [350, 400, 500]:
    for C in [5, 8, 10]:
        configs.append((n_pca, C, 'scale'))

# Test manual gamma at PCA-250
for C in [5, 8, 10, 15]:
    for gamma in [0.001, 0.002, 0.003, 0.005]:
        configs.append((250, C, gamma))

print(f"\n{'='*60}", flush=True)
print(f"FINE-TUNE SEARCH ({len(configs)} configs)", flush=True)
print(f"{'='*60}", flush=True)

results = []
best_acc = 0.0
best_params = {}

for i, (n_pca, C, gamma) in enumerate(configs):
    t0 = time.time()
    X_tr = X_tr_pca[:, :n_pca]
    X_va = X_va_pca[:, :n_pca]

    print(f"[{i+1}/{len(configs)}] PCA={n_pca}, C={C}, gamma={gamma}...", end='', flush=True)

    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42,
              cache_size=1500, decision_function_shape='ovr')
    svm.fit(X_tr, y_train)
    pred = svm.predict(X_va)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0

    tag = " *** NEW BEST ***" if acc > best_acc else ""
    print(f" {acc:.4f} ({elapsed:.0f}s, RAM: {mem():.2f}GB){tag}", flush=True)

    results.append({
        'pca_dim': n_pca, 'C': C, 'gamma': str(gamma),
        'val_accuracy': round(acc, 5), 'time_s': round(elapsed, 1)
    })

    if acc > best_acc:
        best_acc = acc
        best_params = {'pca_dim': n_pca, 'C': C, 'gamma': str(gamma)}

    del svm, pred; gc.collect()

# === Results ===
print(f"\n{'='*60}", flush=True)
print(f"BEST: {best_acc:.5f} with {best_params}", flush=True)
print(f"PREV BEST: 0.72030 (PCA=250, C=8, PT+scale)", flush=True)
delta = best_acc - 0.72030
print(f"DELTA: {delta:+.5f}", flush=True)
print(f"{'='*60}", flush=True)

csv_path = os.path.join(RESULTS, 'exp04_pt_finetune.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pca_dim', 'C', 'gamma', 'val_accuracy', 'time_s'])
    writer.writeheader()
    for r in sorted(results, key=lambda x: -x['val_accuracy']):
        writer.writerow(r)

print("\nTop 20:", flush=True)
for r in sorted(results, key=lambda x: -x['val_accuracy'])[:20]:
    print(f"  PCA={r['pca_dim']:3d} C={r['C']:5} gamma={r['gamma']:8s} -> {r['val_accuracy']:.5f} ({r['time_s']:.0f}s)", flush=True)

print(f"\nFinal RAM: {mem():.2f}GB", flush=True)
