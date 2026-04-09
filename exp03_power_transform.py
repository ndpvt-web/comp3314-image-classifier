"""Experiment 3: Power transform (Yeo-Johnson) + PCA + SVM.
Hypothesis: Yeo-Johnson normalization before PCA makes features more Gaussian,
which should help RBF kernel performance.
Memory-safe: <2.5GB peak.
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
print(f"  Split done, RAM: {mem():.2f}GB", flush=True)

# === StandardScaler first ===
mean = X_train.mean(axis=0, dtype=np.float32)
std = X_train.std(axis=0, dtype=np.float32)
std[std == 0] = 1.0
X_train = np.nan_to_num((X_train - mean) / std)
X_val = np.nan_to_num((X_val - mean) / std)
del mean, std; gc.collect()

# === Power Transform (Yeo-Johnson) ===
# Yeo-Johnson works on any real-valued data (unlike Box-Cox which needs positive)
# Process in batches to control memory - can't do all 5350 features at once
from sklearn.preprocessing import PowerTransformer
print("Applying Yeo-Johnson power transform (batch)...", flush=True)

# Save val to disk during transform fitting
np.save(os.path.join(CACHE, 'X_val_scaled_f32.npy'), X_val)
np.save(os.path.join(CACHE, 'y_val.npy'), y_val)
del X_val; gc.collect()

# Fit power transform in column batches to avoid memory issues
BATCH = 500
n_cols = X_train.shape[1]
X_train_pt = np.empty_like(X_train)

lambdas_all = []
for start in range(0, n_cols, BATCH):
    end = min(start + BATCH, n_cols)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_pt[:, start:end] = pt.fit_transform(X_train[:, start:end]).astype(np.float32)
    lambdas_all.append(pt)
    if start % 2000 == 0:
        print(f"  Cols {start}-{end}, RAM: {mem():.2f}GB", flush=True)

X_train_pt = np.nan_to_num(X_train_pt)
del X_train; gc.collect()
print(f"  Train transform done, RAM: {mem():.2f}GB", flush=True)

# Transform val
X_val_raw = np.load(os.path.join(CACHE, 'X_val_scaled_f32.npy'))
y_val = np.load(os.path.join(CACHE, 'y_val.npy'))
X_val_pt = np.empty_like(X_val_raw)
for idx, start in enumerate(range(0, n_cols, BATCH)):
    end = min(start + BATCH, n_cols)
    X_val_pt[:, start:end] = lambdas_all[idx].transform(X_val_raw[:, start:end]).astype(np.float32)
X_val_pt = np.nan_to_num(X_val_pt)
del X_val_raw, lambdas_all; gc.collect()
print(f"  Val transform done, RAM: {mem():.2f}GB", flush=True)

# === PCA ===
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Save val again before PCA fit
np.save(os.path.join(CACHE, 'X_val_pt_f32.npy'), X_val_pt)
del X_val_pt; gc.collect()

MAX_PCA = 300
print(f"Fitting PCA-{MAX_PCA} on power-transformed features...", flush=True)
pca = PCA(n_components=MAX_PCA, random_state=42)
X_tr_pca = pca.fit_transform(X_train_pt).astype(np.float32)
del X_train_pt; gc.collect()

ev = pca.explained_variance_ratio_.cumsum()
for d in [150, 200, 250, 300]:
    print(f"  PCA-{d}: {ev[d-1]*100:.1f}% variance", flush=True)

X_val_pt = np.load(os.path.join(CACHE, 'X_val_pt_f32.npy'))
X_va_pca = pca.transform(X_val_pt).astype(np.float32)
del X_val_pt; gc.collect()
print(f"  PCA done, RAM: {mem():.2f}GB", flush=True)

# === SVM search ===
configs = []
for n_pca in [150, 175, 200, 225, 250, 300]:
    for C in [5, 8, 10, 15, 20]:
        configs.append((n_pca, C, 'scale'))

print(f"\n{'='*60}", flush=True)
print(f"POWER TRANSFORM + SVM SEARCH ({len(configs)} configs)", flush=True)
print(f"{'='*60}", flush=True)

results = []
best_acc = 0.0
best_params = {}

for i, (n_pca, C, gamma) in enumerate(configs):
    t0 = time.time()
    X_tr = X_tr_pca[:, :n_pca]
    X_va = X_va_pca[:, :n_pca]

    print(f"[{i+1}/{len(configs)}] PCA={n_pca}, C={C}...", end='', flush=True)

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
print(f"BEST (PowerTransform): {best_acc:.5f} with {best_params}", flush=True)
print(f"BASELINE (no PT): 0.71420 (PCA=200, C=8)", flush=True)
delta = best_acc - 0.71420
print(f"DELTA: {delta:+.5f} ({'IMPROVEMENT' if delta > 0 else 'no improvement'})", flush=True)
print(f"{'='*60}", flush=True)

csv_path = os.path.join(RESULTS, 'exp03_power_transform.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pca_dim', 'C', 'gamma', 'val_accuracy', 'time_s'])
    writer.writeheader()
    for r in sorted(results, key=lambda x: -x['val_accuracy']):
        writer.writerow(r)

print("\nAll results:", flush=True)
for r in sorted(results, key=lambda x: -x['val_accuracy']):
    print(f"  PCA={r['pca_dim']:3d} C={r['C']:5} -> {r['val_accuracy']:.5f} ({r['time_s']:.0f}s)", flush=True)

print(f"\nFinal RAM: {mem():.2f}GB", flush=True)
