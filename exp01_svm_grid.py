"""Experiment 1: Wide SVM RBF grid search.
PCA dims: 150-500, C: 5-100, gamma: scale + manual.
Memory-safe: <2.5GB peak. One SVM at a time.
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

# === Load features (mmap then copy train only) ===
print("Loading features...", flush=True)
all_feat = np.load(os.path.join(CACHE, 'all_features_combined.npy'), mmap_mode='r')
train_feat = np.nan_to_num(np.array(all_feat[:50000], dtype=np.float32))
labels = np.load(os.path.join(CACHE, 'train_labels.npy'))
del all_feat
gc.collect()
print(f"  Train: {train_feat.shape}, RAM: {mem():.2f}GB", flush=True)

# === Split ===
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train_feat, labels, test_size=0.2, random_state=42, stratify=labels)
del train_feat, labels
gc.collect()

# === Scale ===
print("Scaling...", flush=True)
mean = X_train.mean(axis=0, dtype=np.float32)
std = X_train.std(axis=0, dtype=np.float32)
std[std == 0] = 1.0
X_train = np.nan_to_num((X_train - mean) / std)
X_val = np.nan_to_num((X_val - mean) / std)
np.save(os.path.join(MODELS, 'scaler_mean.npy'), mean)
np.save(os.path.join(MODELS, 'scaler_std.npy'), std)
del mean, std
gc.collect()
print(f"  RAM: {mem():.2f}GB", flush=True)

# === PCA - save val to disk to reduce peak ===
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Save val to disk temporarily
np.save(os.path.join(CACHE, 'X_val_scaled_f32.npy'), X_val)
np.save(os.path.join(CACHE, 'y_val.npy'), y_val)
del X_val
gc.collect()
print(f"  Val saved to disk, RAM: {mem():.2f}GB", flush=True)

MAX_PCA = 500
print(f"Fitting PCA-{MAX_PCA}...", flush=True)
pca = PCA(n_components=MAX_PCA, random_state=42)
X_tr_pca = pca.fit_transform(X_train)
del X_train
gc.collect()

# Explained variance
ev = pca.explained_variance_ratio_.cumsum()
for d in [150, 200, 250, 300, 400, 500]:
    print(f"  PCA-{d}: {ev[d-1]*100:.1f}% variance explained", flush=True)

# Transform val
X_val = np.load(os.path.join(CACHE, 'X_val_scaled_f32.npy'))
y_val = np.load(os.path.join(CACHE, 'y_val.npy'))
X_va_pca = pca.transform(X_val)
del X_val
gc.collect()
print(f"  PCA done, RAM: {mem():.2f}GB", flush=True)

joblib.dump(pca, os.path.join(MODELS, 'pca500_search.pkl'))

# === Grid search ===
configs = []
pca_dims = [150, 200, 250, 300, 400, 500]
C_values = [5, 8, 10, 15, 20, 50, 100]

for n_pca in pca_dims:
    for C in C_values:
        configs.append((n_pca, C, 'scale'))

# Also test manual gamma at promising PCA dims
for n_pca in [200, 250, 300]:
    for C in [10, 20, 50]:
        for gamma in [0.001, 0.005, 0.01]:
            configs.append((n_pca, C, gamma))

print(f"\n{'='*60}", flush=True)
print(f"SVM GRID SEARCH ({len(configs)} configs)", flush=True)
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

    tag = f" *** NEW BEST ***" if acc > best_acc else ""
    print(f" {acc:.4f} ({elapsed:.0f}s, RAM: {mem():.2f}GB){tag}", flush=True)

    results.append({
        'pca_dim': n_pca, 'C': C, 'gamma': str(gamma),
        'val_accuracy': round(acc, 5), 'time_s': round(elapsed, 1)
    })

    if acc > best_acc:
        best_acc = acc
        best_params = {'pca_dim': n_pca, 'C': C, 'gamma': str(gamma)}

    del svm, pred
    gc.collect()

# === Results ===
print(f"\n{'='*60}", flush=True)
print(f"BEST: {best_acc:.5f} with {best_params}", flush=True)
print(f"{'='*60}", flush=True)

csv_path = os.path.join(RESULTS, 'exp01_svm_grid.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pca_dim', 'C', 'gamma', 'val_accuracy', 'time_s'])
    writer.writeheader()
    for r in sorted(results, key=lambda x: -x['val_accuracy']):
        writer.writerow(r)

with open(os.path.join(RESULTS, 'exp01_best.json'), 'w') as f:
    json.dump({'best_accuracy': best_acc, **best_params}, f, indent=2)

print("\nTop 15:", flush=True)
for r in sorted(results, key=lambda x: -x['val_accuracy'])[:15]:
    print(f"  PCA={r['pca_dim']:3d} C={r['C']:5} gamma={r['gamma']:8s} -> {r['val_accuracy']:.5f} ({r['time_s']:.0f}s)", flush=True)

print(f"\nFinal RAM: {mem():.2f}GB", flush=True)
