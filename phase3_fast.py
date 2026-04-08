"""Phase 3: Fast classifier training with PCA-reduced features."""
import os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
MODELS = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)

print("Loading features...", flush=True)
all_features = np.load(os.path.join(CACHE, 'all_features_combined.npy'))
train_labels = np.load(os.path.join(CACHE, 'train_labels.npy'))

train_feat = np.nan_to_num(all_features[:50000], nan=0.0, posinf=0.0, neginf=0.0)
test_feat = np.nan_to_num(all_features[50000:], nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_val, y_train, y_val = train_test_split(
    train_feat, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
print(f"Train={X_train.shape[0]}, Val={X_val.shape[0]}", flush=True)

# Scale + PCA to 800 dims (fast for all classifiers)
print("Scaling + PCA(800)...", flush=True)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_va = scaler.transform(X_val)

pca = PCA(n_components=800, random_state=42)
X_tr_pca = pca.fit_transform(X_tr)
X_va_pca = pca.transform(X_va)
print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.4f}", flush=True)

joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))
joblib.dump(pca, os.path.join(MODELS, 'pca.pkl'))

results = {}
predictions = {}

def run_exp(name, clf, X_fit, y_fit, X_pred):
    t0 = time.time()
    clf.fit(X_fit, y_fit)
    pred = clf.predict(X_pred)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0
    print(f"  {name}: {acc:.4f} ({elapsed:.0f}s)", flush=True)
    results[name] = acc
    predictions[name] = pred
    return clf, acc

# 1. LightGBM (fastest tree method)
print("\n[1/8] LightGBM...", flush=True)
lgb_clf, _ = run_exp('lgbm_1000', lgb.LGBMClassifier(
    n_estimators=1000, num_leaves=127, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    n_jobs=-1, random_state=42, verbose=-1), X_tr_pca, y_train, X_va_pca)
joblib.dump(lgb_clf, os.path.join(MODELS, 'lightgbm.pkl'))

# 2. XGBoost
print("\n[2/8] XGBoost...", flush=True)
xgb_clf, _ = run_exp('xgb_500', xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    n_jobs=-1, random_state=42, tree_method='hist',
    eval_metric='mlogloss'), X_tr_pca, y_train, X_va_pca)
joblib.dump(xgb_clf, os.path.join(MODELS, 'xgboost.pkl'))

# 3. Extra Trees
print("\n[3/8] Extra Trees...", flush=True)
et_clf, _ = run_exp('et_1000', ExtraTreesClassifier(
    n_estimators=1000, max_features='sqrt', min_samples_leaf=2,
    n_jobs=-1, random_state=42), X_tr_pca, y_train, X_va_pca)
joblib.dump(et_clf, os.path.join(MODELS, 'extra_trees.pkl'))

# 4. Random Forest
print("\n[4/8] Random Forest...", flush=True)
rf_clf, _ = run_exp('rf_1000', RandomForestClassifier(
    n_estimators=1000, max_features='sqrt', min_samples_leaf=2,
    n_jobs=-1, random_state=42), X_tr_pca, y_train, X_va_pca)
joblib.dump(rf_clf, os.path.join(MODELS, 'random_forest.pkl'))

# 5. SVM RBF with PCA-300 (fast)
print("\n[5/8] SVM RBF (PCA-300)...", flush=True)
pca300 = PCA(n_components=300, random_state=42)
X_tr_pca300 = pca300.fit_transform(X_tr)
X_va_pca300 = pca300.transform(X_va)
svm_clf, _ = run_exp('svm_c10', SVC(
    C=10, gamma='scale', kernel='rbf', random_state=42,
    cache_size=2000, probability=True), X_tr_pca300, y_train, X_va_pca300)
joblib.dump(svm_clf, os.path.join(MODELS, 'svm_rbf.pkl'))
joblib.dump(pca300, os.path.join(MODELS, 'pca300.pkl'))

# 6. SVM RBF C=50
print("\n[6/8] SVM RBF C=50...", flush=True)
svm2, _ = run_exp('svm_c50', SVC(
    C=50, gamma='scale', kernel='rbf', random_state=42,
    cache_size=2000, probability=True), X_tr_pca300, y_train, X_va_pca300)
if results['svm_c50'] > results['svm_c10']:
    joblib.dump(svm2, os.path.join(MODELS, 'svm_rbf_best.pkl'))

# 7. LightGBM tuned
print("\n[7/8] LightGBM tuned...", flush=True)
lgb2, _ = run_exp('lgbm_2000', lgb.LGBMClassifier(
    n_estimators=2000, num_leaves=255, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=5,
    n_jobs=-1, random_state=42, verbose=-1, reg_alpha=0.1), X_tr_pca, y_train, X_va_pca)
if results['lgbm_2000'] > results['lgbm_1000']:
    joblib.dump(lgb2, os.path.join(MODELS, 'lightgbm_best.pkl'))

# 8. XGBoost tuned
print("\n[8/8] XGBoost tuned...", flush=True)
xgb2, _ = run_exp('xgb_1000', xgb.XGBClassifier(
    n_estimators=1000, max_depth=10, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
    n_jobs=-1, random_state=42, tree_method='hist',
    eval_metric='mlogloss', reg_alpha=0.1), X_tr_pca, y_train, X_va_pca)
if results['xgb_1000'] > results['xgb_500']:
    joblib.dump(xgb2, os.path.join(MODELS, 'xgboost_best.pkl'))

# ---- SUMMARY ----
print("\n" + "="*60, flush=True)
print("RESULTS SUMMARY", flush=True)
print("="*60, flush=True)
for name, a in sorted(results.items(), key=lambda x: -x[1]):
    tag = " *** BEST ***" if a == max(results.values()) else ""
    print(f"  {name:20s}: {a:.4f}{tag}", flush=True)

best_name = max(results, key=lambda x: results[x])
best_acc = results[best_name]

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print(f"\nBest: {best_name} = {best_acc:.4f}", flush=True)
print(classification_report(y_val, predictions[best_name], target_names=CLASS_NAMES), flush=True)

# Save predictions and labels for ensemble phase
for name, pred in predictions.items():
    np.save(os.path.join(CACHE, f'val_pred_{name}.npy'), pred)
np.save(os.path.join(CACHE, 'val_labels.npy'), y_val)
np.save(os.path.join(CACHE, 'y_train.npy'), y_train)

# Write experiment log
with open(os.path.join(BASE, 'experiment_log.md'), 'w') as f:
    f.write("# Experiment Log\n\n")
    f.write("## Features: 5350 dims -> PCA 800 (trees) / PCA 300 (SVM)\n\n")
    f.write("## Phase 3 Results (80/20 stratified split, 40k/10k)\n\n")
    f.write("| Classifier | Val Accuracy |\n|---|---|\n")
    for name, a in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f"| {name} | {a:.4f} |\n")
    f.write(f"\n**Best: {best_name} = {best_acc:.4f}**\n")

print("\n=== Phase 3 Complete ===", flush=True)
