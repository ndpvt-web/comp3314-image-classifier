"""Phase 3: Train classifiers. Fast tree methods first, then SVM with reduced data."""
import os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
MODELS = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)

print("Loading features...")
all_features = np.load(os.path.join(CACHE, 'all_features_combined.npy'))
train_labels = np.load(os.path.join(CACHE, 'train_labels.npy'))
n_train, n_test = 50000, 10000

train_features = np.nan_to_num(all_features[:n_train], nan=0.0, posinf=0.0, neginf=0.0)
test_features = np.nan_to_num(all_features[n_train:], nan=0.0, posinf=0.0, neginf=0.0)
print(f"Train: {train_features.shape}, Test: {test_features.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
print(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}")

print("Scaling...")
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_va = scaler.transform(X_val)
joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))

results = {}

# ---- FAST TREE METHODS FIRST ----

print("\n[1/7] XGBoost (500 trees, depth=8)...")
t0 = time.time()
xgb_clf = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    n_jobs=-1, random_state=42, tree_method='hist', eval_metric='mlogloss')
xgb_clf.fit(X_tr, y_train)
xgb_pred = xgb_clf.predict(X_va)
xgb_acc = accuracy_score(y_val, xgb_pred)
print(f"  -> {xgb_acc:.4f} ({time.time()-t0:.0f}s)")
results['xgboost_500_d8'] = xgb_acc
joblib.dump(xgb_clf, os.path.join(MODELS, 'xgboost.pkl'))

print("\n[2/7] LightGBM (1000 trees)...")
t0 = time.time()
lgb_clf = lgb.LGBMClassifier(
    n_estimators=1000, num_leaves=127, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    n_jobs=-1, random_state=42, verbose=-1)
lgb_clf.fit(X_tr, y_train)
lgb_pred = lgb_clf.predict(X_va)
lgb_acc = accuracy_score(y_val, lgb_pred)
print(f"  -> {lgb_acc:.4f} ({time.time()-t0:.0f}s)")
results['lightgbm_1000'] = lgb_acc
joblib.dump(lgb_clf, os.path.join(MODELS, 'lightgbm.pkl'))

print("\n[3/7] Extra Trees (1000 trees)...")
t0 = time.time()
et_clf = ExtraTreesClassifier(
    n_estimators=1000, max_features='sqrt', min_samples_leaf=2,
    n_jobs=-1, random_state=42)
et_clf.fit(X_tr, y_train)
et_pred = et_clf.predict(X_va)
et_acc = accuracy_score(y_val, et_pred)
print(f"  -> {et_acc:.4f} ({time.time()-t0:.0f}s)")
results['extra_trees_1000'] = et_acc
joblib.dump(et_clf, os.path.join(MODELS, 'extra_trees.pkl'))

print("\n[4/7] Random Forest (1000 trees)...")
t0 = time.time()
rf_clf = RandomForestClassifier(
    n_estimators=1000, max_features='sqrt', min_samples_leaf=2,
    n_jobs=-1, random_state=42)
rf_clf.fit(X_tr, y_train)
rf_pred = rf_clf.predict(X_va)
rf_acc = accuracy_score(y_val, rf_pred)
print(f"  -> {rf_acc:.4f} ({time.time()-t0:.0f}s)")
results['random_forest_1000'] = rf_acc
joblib.dump(rf_clf, os.path.join(MODELS, 'random_forest.pkl'))

print("\n[5/7] XGBoost tuned (1000 trees, depth=10)...")
t0 = time.time()
xgb2 = xgb.XGBClassifier(
    n_estimators=1000, max_depth=10, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
    n_jobs=-1, random_state=42, tree_method='hist',
    eval_metric='mlogloss', reg_alpha=0.1, reg_lambda=1.0)
xgb2.fit(X_tr, y_train)
xgb2_pred = xgb2.predict(X_va)
xgb2_acc = accuracy_score(y_val, xgb2_pred)
print(f"  -> {xgb2_acc:.4f} ({time.time()-t0:.0f}s)")
results['xgboost_1000_d10'] = xgb2_acc
if xgb2_acc > xgb_acc:
    joblib.dump(xgb2, os.path.join(MODELS, 'xgboost_best.pkl'))

# ---- SVM with PCA (reduced dims for speed) ----

print("\n[6/7] PCA + SVM RBF (C=10, PCA=300)...")
t0 = time.time()
pca300 = PCA(n_components=300, random_state=42)
X_tr_pca = pca300.fit_transform(X_tr)
X_va_pca = pca300.transform(X_va)
print(f"  PCA variance: {pca300.explained_variance_ratio_.sum():.4f}")
svm = SVC(C=10, gamma='scale', kernel='rbf', random_state=42, cache_size=2000)
svm.fit(X_tr_pca, y_train)
svm_pred = svm.predict(X_va_pca)
svm_acc = accuracy_score(y_val, svm_pred)
print(f"  -> {svm_acc:.4f} ({time.time()-t0:.0f}s)")
results['svm_rbf_pca300_c10'] = svm_acc
joblib.dump(svm, os.path.join(MODELS, 'svm_rbf.pkl'))
joblib.dump(pca300, os.path.join(MODELS, 'pca300.pkl'))

print("\n[7/7] SVM RBF (C=50, PCA=300)...")
t0 = time.time()
svm2 = SVC(C=50, gamma='scale', kernel='rbf', random_state=42, cache_size=2000)
svm2.fit(X_tr_pca, y_train)
svm2_pred = svm2.predict(X_va_pca)
svm2_acc = accuracy_score(y_val, svm2_pred)
print(f"  -> {svm2_acc:.4f} ({time.time()-t0:.0f}s)")
results['svm_rbf_pca300_c50'] = svm2_acc
if svm2_acc > svm_acc:
    joblib.dump(svm2, os.path.join(MODELS, 'svm_rbf.pkl'))

# ---- SUMMARY ----
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for name, a in sorted(results.items(), key=lambda x: -x[1]):
    tag = " *** BEST ***" if a == max(results.values()) else ""
    print(f"  {name:30s}: {a:.4f}{tag}")

best_name = max(results, key=lambda x: results[x])
best_acc = results[best_name]
print(f"\nBest: {best_name} = {best_acc:.4f}")

# Classification report for best
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
preds_map = {'xgboost_500_d8': xgb_pred, 'lightgbm_1000': lgb_pred,
             'extra_trees_1000': et_pred, 'random_forest_1000': rf_pred,
             'xgboost_1000_d10': xgb2_pred, 'svm_rbf_pca300_c10': svm_pred,
             'svm_rbf_pca300_c50': svm2_pred}
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_val, preds_map[best_name], target_names=CLASS_NAMES))

# Save all val predictions for ensemble
np.save(os.path.join(CACHE, 'val_preds_xgb.npy'), xgb_pred)
np.save(os.path.join(CACHE, 'val_preds_lgb.npy'), lgb_pred)
np.save(os.path.join(CACHE, 'val_preds_et.npy'), et_pred)
np.save(os.path.join(CACHE, 'val_preds_rf.npy'), rf_pred)
np.save(os.path.join(CACHE, 'val_preds_xgb2.npy'), xgb2_pred)
np.save(os.path.join(CACHE, 'val_preds_svm.npy'), svm_pred)
np.save(os.path.join(CACHE, 'val_labels.npy'), y_val)

# Save train/val split indices for reproducibility
np.save(os.path.join(CACHE, 'X_train_idx.npy'), np.array([]))  # placeholder
np.save(os.path.join(CACHE, 'y_train.npy'), y_train)

# Write experiment log
with open(os.path.join(BASE, 'experiment_log.md'), 'w') as f:
    f.write("# Experiment Log\n\n")
    f.write("## Features: 5350 dims\n")
    f.write("HOG(4cfg)=4824, ColorHist=288, ColorMoments=24, LBP=54, Gabor=80, Spatial=60, Edge=13, Hu=7\n\n")
    f.write("## Phase 3 Results (80/20 stratified split)\n\n")
    f.write("| Classifier | Val Accuracy |\n|---|---|\n")
    for name, a in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f"| {name} | {a:.4f} |\n")
    f.write(f"\n**Best: {best_name} = {best_acc:.4f}**\n")

print("\n=== Phase 3 Complete ===")
