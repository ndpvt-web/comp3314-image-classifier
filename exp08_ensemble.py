"""Experiment 8: Ensemble of diverse SVMs.
Train multiple SVMs with different preprocessing/PCA configs,
combine predictions via majority voting.
Memory-safe: train one at a time, save predictions, then vote.
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
X_train_s = np.nan_to_num((X_train - mean) / std)
X_val_s = np.nan_to_num((X_val - mean) / std)
del X_train, X_val, mean, std; gc.collect()

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import stats

# === Power Transform ===
print("Power transform...", flush=True)
np.save(os.path.join(CACHE, 'X_val_scaled_f32.npy'), X_val_s)
gc.collect()

BATCH = 500
n_cols = X_train_s.shape[1]
X_tr_pt = np.empty_like(X_train_s)
pt_models = []
for start in range(0, n_cols, BATCH):
    end = min(start + BATCH, n_cols)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_tr_pt[:, start:end] = pt.fit_transform(X_train_s[:, start:end]).astype(np.float32)
    pt_models.append(pt)
X_tr_pt = np.nan_to_num(X_tr_pt)
del X_train_s; gc.collect()

X_val_s = np.load(os.path.join(CACHE, 'X_val_scaled_f32.npy'))
X_va_pt = np.empty_like(X_val_s)
for idx, start in enumerate(range(0, n_cols, BATCH)):
    end = min(start + BATCH, n_cols)
    X_va_pt[:, start:end] = pt_models[idx].transform(X_val_s[:, start:end]).astype(np.float32)
X_va_pt = np.nan_to_num(X_va_pt)
del X_val_s, pt_models; gc.collect()
print(f"  PT done, RAM: {mem():.2f}GB", flush=True)

# === Split features ===
X_tr_hog = X_tr_pt[:, :HOG_END]
X_tr_other = X_tr_pt[:, NON_HOG_START:]
X_va_hog = X_va_pt[:, :HOG_END]
X_va_other = X_va_pt[:, NON_HOG_START:]
del X_tr_pt, X_va_pt; gc.collect()

# === PCA (max dims for flexibility) ===
np.save(os.path.join(CACHE, 'X_va_hog.npy'), X_va_hog)
np.save(os.path.join(CACHE, 'X_va_other.npy'), X_va_other)
del X_va_hog, X_va_other; gc.collect()

pca_hog = PCA(n_components=250, random_state=42)
X_tr_hog_pca = pca_hog.fit_transform(X_tr_hog).astype(np.float32)
del X_tr_hog; gc.collect()

pca_other = PCA(n_components=200, random_state=42)
X_tr_other_pca = pca_other.fit_transform(X_tr_other).astype(np.float32)
del X_tr_other; gc.collect()

X_va_hog = np.load(os.path.join(CACHE, 'X_va_hog.npy'))
X_va_other = np.load(os.path.join(CACHE, 'X_va_other.npy'))
X_va_hog_pca = pca_hog.transform(X_va_hog).astype(np.float32)
X_va_other_pca = pca_other.transform(X_va_other).astype(np.float32)
del X_va_hog, X_va_other, pca_hog, pca_other; gc.collect()
print(f"  PCA done, RAM: {mem():.2f}GB", flush=True)

# === Define diverse ensemble members ===
# Use different PCA dims and C values for diversity
ensemble_configs = [
    # (hog_d, other_d, C, gamma) - diverse configs
    (150, 150, 5, 'scale'),      # exp5 best
    (150, 175, 5, 'scale'),      # exp7 best
    (150, 125, 5, 'scale'),      # different Other dim
    (125, 150, 5, 'scale'),      # different HOG dim
    (175, 150, 5, 'scale'),      # higher HOG
    (150, 150, 3, 'scale'),      # lower C
    (150, 150, 10, 'scale'),     # higher C
    (200, 150, 5, 'scale'),      # much higher HOG
    (150, 200, 5, 'scale'),      # max Other
]

print(f"\n{'='*60}", flush=True)
print(f"TRAINING {len(ensemble_configs)} ENSEMBLE MEMBERS", flush=True)
print(f"{'='*60}", flush=True)

all_preds = []
individual_accs = []

for i, (hog_d, other_d, C, gamma) in enumerate(ensemble_configs):
    t0 = time.time()
    X_tr = np.hstack([X_tr_hog_pca[:, :hog_d], X_tr_other_pca[:, :other_d]])
    X_va = np.hstack([X_va_hog_pca[:, :hog_d], X_va_other_pca[:, :other_d]])

    print(f"[{i+1}/{len(ensemble_configs)}] HOG-{hog_d}+Other-{other_d} C={C}...", end='', flush=True)

    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42,
              cache_size=1500, decision_function_shape='ovr')
    svm.fit(X_tr, y_train)
    pred = svm.predict(X_va)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0

    print(f" {acc:.4f} ({elapsed:.0f}s)", flush=True)

    all_preds.append(pred)
    individual_accs.append(acc)

    del svm, X_tr, X_va; gc.collect()

# === Majority voting ensembles ===
print(f"\n{'='*60}", flush=True)
print(f"ENSEMBLE RESULTS", flush=True)
print(f"{'='*60}", flush=True)

pred_array = np.array(all_preds)  # shape: (n_models, n_val)

# Try different ensemble sizes (top-k by individual accuracy)
sorted_indices = np.argsort(individual_accs)[::-1]

for k in range(3, len(ensemble_configs)+1, 2):  # odd numbers for clean majority vote
    top_k_idx = sorted_indices[:k]
    top_k_preds = pred_array[top_k_idx]
    # Majority vote
    ensemble_pred = stats.mode(top_k_preds, axis=0)[0].flatten()
    ensemble_acc = accuracy_score(y_val, ensemble_pred)

    members = [f"HOG-{ensemble_configs[j][0]}+O-{ensemble_configs[j][1]}C{ensemble_configs[j][2]}" for j in top_k_idx]
    print(f"  Top-{k} ensemble: {ensemble_acc:.4f} (members: {', '.join(members[:3])}...)", flush=True)

# Also try all models
all_pred = stats.mode(pred_array, axis=0)[0].flatten()
all_acc = accuracy_score(y_val, all_pred)
print(f"  ALL-{len(ensemble_configs)} ensemble: {all_acc:.4f}", flush=True)

# Best individual vs best ensemble
best_individual = max(individual_accs)
print(f"\n  Best individual: {best_individual:.4f}", flush=True)
print(f"  Best ensemble:   {all_acc:.4f}", flush=True)
delta = all_acc - best_individual
print(f"  Delta: {delta:+.4f}", flush=True)

# Save results
results = {
    'individual_accs': [round(a, 5) for a in individual_accs],
    'configs': [{'hog': c[0], 'other': c[1], 'C': c[2]} for c in ensemble_configs],
    'all_ensemble_acc': round(all_acc, 5),
    'best_individual': round(best_individual, 5),
}
with open(os.path.join(RESULTS, 'exp08_ensemble.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save prediction arrays for later use
np.save(os.path.join(CACHE, 'ensemble_val_preds.npy'), pred_array)

print(f"\nFinal RAM: {mem():.2f}GB", flush=True)
