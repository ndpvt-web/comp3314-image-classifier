"""Experiment 1b: Fast targeted SVM search.
Key insight from exp01: PCA-150 C=10 = 71.20%. Higher PCA should help.
Focus: PCA 200-350, C 8-30, gamma=scale only. ~20 configs, each ~4-5min.
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
print(f"  Train: {train_feat.shape}, RAM: {mem():.2f}GB", flush=True)

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

# === PCA ===
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Save val to disk to reduce peak mem during PCA fit
np.save(os.path.join(CACHE, 'X_val_scaled_f32.npy'), X_val)
np.save(os.path.join(CACHE, 'y_val.npy'), y_val)
del X_val; gc.collect()

MAX_PCA = 350
print(f"Fitting PCA-{MAX_PCA}...", flush=True)
pca = PCA(n_components=MAX_PCA, random_state=42)
X_tr_pca = pca.fit_transform(X_train)
del X_train; gc.collect()

X_val = np.load(os.path.join(CACHE, 'X_val_scaled_f32.npy'))
y_val = np.load(os.path.join(CACHE, 'y_val.npy'))
X_va_pca = pca.transform(X_val)
del X_val; gc.collect()
print(f"  PCA done, RAM: {mem():.2f}GB", flush=True)

# Convert to float32 to save memory for SVM
X_tr_pca = X_tr_pca.astype(np.float32)
X_va_pca = X_va_pca.astype(np.float32)
gc.collect()

# === Targeted grid ===
configs = []
for n_pca in [200, 225, 250, 275, 300, 350]:
    for C in [8, 10, 12, 15, 20, 30]:
        configs.append((n_pca, C, 'scale'))

print(f"\n{'='*60}", flush=True)
print(f"TARGETED SVM SEARCH ({len(configs)} configs)", flush=True)
print(f"{'='*60}", flush=True)

# Also log from exp01 partial results
known_results = [
    {'pca_dim': 150, 'C': 5, 'gamma': 'scale', 'val_accuracy': 0.71080, 'time_s': 235},
    {'pca_dim': 150, 'C': 8, 'gamma': 'scale', 'val_accuracy': 0.71190, 'time_s': 244},
    {'pca_dim': 150, 'C': 10, 'gamma': 'scale', 'val_accuracy': 0.71200, 'time_s': 240},
]

results = list(known_results)
best_acc = 0.71200
best_params = {'pca_dim': 150, 'C': 10, 'gamma': 'scale'}
print(f"Starting from known best: {best_acc:.5f} (PCA=150, C=10)", flush=True)

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
