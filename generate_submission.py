"""Generate submission.csv using best model config.
Uses split PCA (HOG + non-HOG) with power transform.
Trains on ALL 50k training samples, predicts on 10k test samples.

Best config from experiments:
- Power Transform (Yeo-Johnson)
- Split PCA: HOG-150 + Other-150 = 300 total dims
- SVM RBF: C=5, gamma=scale
- Val accuracy: 74.46%
"""
import os, sys, time, gc, json
import numpy as np
import psutil

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
MODELS = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')

# === CONFIG - Change these for different submissions ===
HOG_PCA_DIM = 150
OTHER_PCA_DIM = 175
SVM_C = 5
SVM_GAMMA = 'scale'
# =======================================================

HOG_END = 4824
NON_HOG_START = 4824

def mem():
    return psutil.Process().memory_info().rss / 1e9

print(f"[START] RAM: {mem():.2f}GB", flush=True)
print(f"Config: HOG-{HOG_PCA_DIM}+Other-{OTHER_PCA_DIM}, C={SVM_C}, gamma={SVM_GAMMA}", flush=True)

# === Load ALL features (train + test) ===
print("Loading all features...", flush=True)
all_feat = np.load(os.path.join(CACHE, 'all_features_combined.npy'), mmap_mode='r')
train_feat = np.nan_to_num(np.array(all_feat[:50000], dtype=np.float32))
test_feat = np.nan_to_num(np.array(all_feat[50000:], dtype=np.float32))
train_labels = np.load(os.path.join(CACHE, 'train_labels.npy'))
del all_feat; gc.collect()
print(f"  Train: {train_feat.shape}, Test: {test_feat.shape}, RAM: {mem():.2f}GB", flush=True)

# === Scale using train stats ===
print("Scaling...", flush=True)
mean = train_feat.mean(axis=0, dtype=np.float32)
std = train_feat.std(axis=0, dtype=np.float32)
std[std == 0] = 1.0
train_feat = np.nan_to_num((train_feat - mean) / std)
test_feat = np.nan_to_num((test_feat - mean) / std)
del mean, std; gc.collect()

# === Power Transform ===
from sklearn.preprocessing import PowerTransformer
print("Power transform...", flush=True)

# Save test to disk during PT fitting
np.save(os.path.join(CACHE, 'test_scaled_f32.npy'), test_feat)
del test_feat; gc.collect()

BATCH = 500
n_cols = train_feat.shape[1]
train_pt = np.empty_like(train_feat)
pt_models = []
for start in range(0, n_cols, BATCH):
    end = min(start + BATCH, n_cols)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    train_pt[:, start:end] = pt.fit_transform(train_feat[:, start:end]).astype(np.float32)
    pt_models.append(pt)
train_pt = np.nan_to_num(train_pt)
del train_feat; gc.collect()

test_feat = np.load(os.path.join(CACHE, 'test_scaled_f32.npy'))
test_pt = np.empty_like(test_feat)
for idx, start in enumerate(range(0, n_cols, BATCH)):
    end = min(start + BATCH, n_cols)
    test_pt[:, start:end] = pt_models[idx].transform(test_feat[:, start:end]).astype(np.float32)
test_pt = np.nan_to_num(test_pt)
del test_feat, pt_models; gc.collect()
print(f"  PT done, RAM: {mem():.2f}GB", flush=True)

# === Split features ===
tr_hog = train_pt[:, :HOG_END]
tr_other = train_pt[:, NON_HOG_START:]
te_hog = test_pt[:, :HOG_END]
te_other = test_pt[:, NON_HOG_START:]
del train_pt, test_pt; gc.collect()

# === Split PCA ===
from sklearn.decomposition import PCA

# Save test parts to disk during PCA fitting
np.save(os.path.join(CACHE, 'te_hog.npy'), te_hog)
np.save(os.path.join(CACHE, 'te_other.npy'), te_other)
del te_hog, te_other; gc.collect()

print(f"PCA on HOG -> {HOG_PCA_DIM}...", flush=True)
pca_hog = PCA(n_components=HOG_PCA_DIM, random_state=42)
tr_hog_pca = pca_hog.fit_transform(tr_hog).astype(np.float32)
del tr_hog; gc.collect()

print(f"PCA on Other -> {OTHER_PCA_DIM}...", flush=True)
pca_other = PCA(n_components=OTHER_PCA_DIM, random_state=42)
tr_other_pca = pca_other.fit_transform(tr_other).astype(np.float32)
del tr_other; gc.collect()

# Transform test
te_hog = np.load(os.path.join(CACHE, 'te_hog.npy'))
te_other = np.load(os.path.join(CACHE, 'te_other.npy'))
te_hog_pca = pca_hog.transform(te_hog).astype(np.float32)
te_other_pca = pca_other.transform(te_other).astype(np.float32)
del te_hog, te_other, pca_hog, pca_other; gc.collect()

# Concatenate
X_train = np.hstack([tr_hog_pca, tr_other_pca])
X_test = np.hstack([te_hog_pca, te_other_pca])
del tr_hog_pca, tr_other_pca, te_hog_pca, te_other_pca; gc.collect()
print(f"  Final train: {X_train.shape}, test: {X_test.shape}, RAM: {mem():.2f}GB", flush=True)

# === Train SVM on ALL training data ===
from sklearn.svm import SVC
print(f"Training SVM (C={SVM_C}, gamma={SVM_GAMMA}) on all 50k samples...", flush=True)
t0 = time.time()
svm = SVC(C=SVM_C, gamma=SVM_GAMMA, kernel='rbf', random_state=42,
          cache_size=1500, decision_function_shape='ovr')
svm.fit(X_train, train_labels)
print(f"  Training done in {time.time()-t0:.0f}s, RAM: {mem():.2f}GB", flush=True)

# === Predict test set ===
print("Predicting test set...", flush=True)
test_preds = svm.predict(X_test)
print(f"  Predictions: {test_preds.shape}, classes: {np.unique(test_preds)}", flush=True)

# === Generate submission ===
import pandas as pd
submission = pd.DataFrame({
    'id': range(len(test_preds)),
    'label': test_preds.astype(int)
})
sub_path = os.path.join(BASE, 'submission.csv')
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved to {sub_path}", flush=True)
print(f"  Shape: {submission.shape}", flush=True)
print(f"  Label distribution:\n{submission['label'].value_counts().sort_index()}", flush=True)

# Also save config info
config_info = {
    'hog_pca': HOG_PCA_DIM,
    'other_pca': OTHER_PCA_DIM,
    'svm_c': SVM_C,
    'svm_gamma': SVM_GAMMA,
    'val_accuracy': 0.7446,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}
with open(os.path.join(RESULTS, 'submission_config.json'), 'w') as f:
    json.dump(config_info, f, indent=2)

print(f"\nFinal RAM: {mem():.2f}GB", flush=True)
print("DONE!", flush=True)
