"""Phase 3b: Train SVM (fast, no probability) + Phase 4: Build ensemble.
Uses already-trained tree models from phase3_fast.py partial run.
"""
import os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
MODELS = os.path.join(BASE, 'models')

print("Loading features...", flush=True)
all_features = np.load(os.path.join(CACHE, 'all_features_combined.npy'))
train_labels = np.load(os.path.join(CACHE, 'train_labels.npy'))

train_feat = np.nan_to_num(all_features[:50000], nan=0.0, posinf=0.0, neginf=0.0)
test_feat = np.nan_to_num(all_features[50000:], nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_val, y_train, y_val = train_test_split(
    train_feat, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Scale
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_va = scaler.transform(X_val)
# Scale full train + test for final predictions
X_all_train_s = scaler.transform(train_feat)
X_all_test_s = scaler.transform(test_feat)

# PCA 800 for tree methods
pca800 = PCA(n_components=800, random_state=42)
X_tr_800 = pca800.fit_transform(X_tr)
X_va_800 = pca800.transform(X_va)
X_all_train_800 = pca800.transform(X_all_train_s)
X_all_test_800 = pca800.transform(X_all_test_s)
print(f"PCA-800 variance: {pca800.explained_variance_ratio_.sum():.4f}", flush=True)

# PCA 200 for SVM (fast)
pca200 = PCA(n_components=200, random_state=42)
X_tr_200 = pca200.fit_transform(X_tr)
X_va_200 = pca200.transform(X_va)
X_all_train_200 = pca200.transform(X_all_train_s)
X_all_test_200 = pca200.transform(X_all_test_s)
print(f"PCA-200 variance: {pca200.explained_variance_ratio_.sum():.4f}", flush=True)

joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))
joblib.dump(pca800, os.path.join(MODELS, 'pca800.pkl'))
joblib.dump(pca200, os.path.join(MODELS, 'pca200.pkl'))

results = {}
val_preds = {}
classifiers = {}

# ============== TRAIN ALL CLASSIFIERS ==============

# 1. SVM RBF C=10 on PCA-200 (NO probability - much faster)
print("\n[1/8] SVM RBF C=10 PCA-200...", flush=True)
t0 = time.time()
svm1 = SVC(C=10, gamma='scale', kernel='rbf', random_state=42, cache_size=2000,
           decision_function_shape='ovr')
svm1.fit(X_tr_200, y_train)
p = svm1.predict(X_va_200)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['svm_c10'] = a; val_preds['svm_c10'] = p; classifiers['svm_c10'] = svm1

# 2. SVM RBF C=50 on PCA-200
print("\n[2/8] SVM RBF C=50 PCA-200...", flush=True)
t0 = time.time()
svm2 = SVC(C=50, gamma='scale', kernel='rbf', random_state=42, cache_size=2000)
svm2.fit(X_tr_200, y_train)
p = svm2.predict(X_va_200)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['svm_c50'] = a; val_preds['svm_c50'] = p; classifiers['svm_c50'] = svm2

# 3. SVM RBF C=100 on PCA-200
print("\n[3/8] SVM RBF C=100 PCA-200...", flush=True)
t0 = time.time()
svm3 = SVC(C=100, gamma='scale', kernel='rbf', random_state=42, cache_size=2000)
svm3.fit(X_tr_200, y_train)
p = svm3.predict(X_va_200)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['svm_c100'] = a; val_preds['svm_c100'] = p; classifiers['svm_c100'] = svm3

# 4. LightGBM
print("\n[4/8] LightGBM 1000...", flush=True)
t0 = time.time()
lgb1 = lgb.LGBMClassifier(
    n_estimators=1000, num_leaves=127, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    n_jobs=-1, random_state=42, verbose=-1)
lgb1.fit(X_tr_800, y_train)
p = lgb1.predict(X_va_800)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['lgbm'] = a; val_preds['lgbm'] = p; classifiers['lgbm'] = lgb1

# 5. XGBoost
print("\n[5/8] XGBoost 500...", flush=True)
t0 = time.time()
xgb1 = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    n_jobs=-1, random_state=42, tree_method='hist', eval_metric='mlogloss')
xgb1.fit(X_tr_800, y_train)
p = xgb1.predict(X_va_800)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['xgb'] = a; val_preds['xgb'] = p; classifiers['xgb'] = xgb1

# 6. Extra Trees
print("\n[6/8] Extra Trees 1500...", flush=True)
t0 = time.time()
et1 = ExtraTreesClassifier(n_estimators=1500, max_features='sqrt',
                            min_samples_leaf=1, n_jobs=-1, random_state=42)
et1.fit(X_tr_800, y_train)
p = et1.predict(X_va_800)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['et'] = a; val_preds['et'] = p; classifiers['et'] = et1

# 7. Random Forest
print("\n[7/8] Random Forest 1500...", flush=True)
t0 = time.time()
rf1 = RandomForestClassifier(n_estimators=1500, max_features='sqrt',
                              min_samples_leaf=1, n_jobs=-1, random_state=42)
rf1.fit(X_tr_800, y_train)
p = rf1.predict(X_va_800)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['rf'] = a; val_preds['rf'] = p; classifiers['rf'] = rf1

# 8. LightGBM tuned
print("\n[8/8] LightGBM tuned 2000...", flush=True)
t0 = time.time()
lgb2 = lgb.LGBMClassifier(
    n_estimators=2000, num_leaves=255, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=5,
    n_jobs=-1, random_state=42, verbose=-1, reg_alpha=0.1)
lgb2.fit(X_tr_800, y_train)
p = lgb2.predict(X_va_800)
a = accuracy_score(y_val, p)
print(f"  -> {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
results['lgbm_tuned'] = a; val_preds['lgbm_tuned'] = p; classifiers['lgbm_tuned'] = lgb2

# ============== RESULTS ==============
print("\n" + "="*60, flush=True)
print("INDIVIDUAL CLASSIFIER RESULTS", flush=True)
print("="*60, flush=True)
for name, a in sorted(results.items(), key=lambda x: -x[1]):
    tag = " ***" if a == max(results.values()) else ""
    print(f"  {name:20s}: {a:.4f}{tag}", flush=True)

# ============== PHASE 4: ENSEMBLE ==============
print("\n" + "="*60, flush=True)
print("PHASE 4: ENSEMBLE METHODS", flush=True)
print("="*60, flush=True)

# Majority voting (all classifiers)
print("\nMajority Voting (all)...", flush=True)
all_pred_matrix = np.array([val_preds[k] for k in val_preds]).T  # (10000, n_clf)
from scipy.stats import mode
majority_pred = mode(all_pred_matrix, axis=1)[0].flatten().astype(int)
majority_acc = accuracy_score(y_val, majority_pred)
print(f"  All classifiers voting: {majority_acc:.4f}", flush=True)

# Top-5 voting
top5 = sorted(results, key=lambda x: results[x], reverse=True)[:5]
print(f"\nTop-5 voting ({top5})...", flush=True)
top5_matrix = np.array([val_preds[k] for k in top5]).T
top5_pred = mode(top5_matrix, axis=1)[0].flatten().astype(int)
top5_acc = accuracy_score(y_val, top5_pred)
print(f"  Top-5 voting: {top5_acc:.4f}", flush=True)

# Top-3 voting
top3 = sorted(results, key=lambda x: results[x], reverse=True)[:3]
print(f"\nTop-3 voting ({top3})...", flush=True)
top3_matrix = np.array([val_preds[k] for k in top3]).T
top3_pred = mode(top3_matrix, axis=1)[0].flatten().astype(int)
top3_acc = accuracy_score(y_val, top3_pred)
print(f"  Top-3 voting: {top3_acc:.4f}", flush=True)

# Stacking: use classifier predictions as meta-features
print("\nStacking with Logistic Regression meta-learner...", flush=True)
# Split train into base-train and meta-train
from sklearn.model_selection import cross_val_predict
# Use val predictions as meta-features, fit on a portion
meta_features_val = np.array([val_preds[k] for k in sorted(val_preds.keys())]).T
# For stacking, we need meta-features on train set too
# Use 5-fold cross-val predictions on train set
print("  Generating cross-val predictions for stacking...", flush=True)
meta_train = np.zeros((len(y_train), len(val_preds)))

clf_names_sorted = sorted(val_preds.keys())
for i, name in enumerate(clf_names_sorted):
    print(f"    CV predict: {name}...", flush=True)
    clf = classifiers[name]
    if name.startswith('svm'):
        cv_pred = cross_val_predict(clf, X_tr_200, y_train, cv=3, n_jobs=-1)
    else:
        cv_pred = cross_val_predict(clf, X_tr_800, y_train, cv=3, n_jobs=-1)
    meta_train[:, i] = cv_pred

meta_val = np.array([val_preds[k] for k in clf_names_sorted]).T

# One-hot encode meta features for better stacking
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, categories='auto')
meta_train_ohe = ohe.fit_transform(meta_train)
meta_val_ohe = ohe.transform(meta_val)

meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
meta_clf.fit(meta_train_ohe, y_train)
stack_pred = meta_clf.predict(meta_val_ohe)
stack_acc = accuracy_score(y_val, stack_pred)
print(f"  Stacking accuracy: {stack_acc:.4f}", flush=True)

# ============== BEST ENSEMBLE ==============
ensemble_results = {
    'majority_all': majority_acc,
    'top5_voting': top5_acc,
    'top3_voting': top3_acc,
    'stacking_lr': stack_acc,
}
best_individual = max(results.values())
best_ensemble_name = max(ensemble_results, key=lambda x: ensemble_results[x])
best_ensemble_acc = ensemble_results[best_ensemble_name]
overall_best = max(best_individual, best_ensemble_acc)

print("\n" + "="*60, flush=True)
print("ENSEMBLE RESULTS", flush=True)
print("="*60, flush=True)
for name, a in sorted(ensemble_results.items(), key=lambda x: -x[1]):
    print(f"  {name:20s}: {a:.4f}", flush=True)
print(f"\nBest individual: {max(results, key=lambda x: results[x])} = {best_individual:.4f}", flush=True)
print(f"Best ensemble: {best_ensemble_name} = {best_ensemble_acc:.4f}", flush=True)
print(f"Overall best: {overall_best:.4f}", flush=True)

# Classification report for best
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
if best_ensemble_acc >= best_individual:
    if best_ensemble_name == 'stacking_lr':
        best_pred = stack_pred
    elif best_ensemble_name == 'top5_voting':
        best_pred = top5_pred
    elif best_ensemble_name == 'top3_voting':
        best_pred = top3_pred
    else:
        best_pred = majority_pred
    print(f"\nClassification Report (best ensemble: {best_ensemble_name}):", flush=True)
else:
    bn = max(results, key=lambda x: results[x])
    best_pred = val_preds[bn]
    print(f"\nClassification Report (best individual: {bn}):", flush=True)
print(classification_report(y_val, best_pred, target_names=CLASS_NAMES), flush=True)

# ============== SAVE MODELS ==============
print("\nSaving models...", flush=True)
for name, clf in classifiers.items():
    joblib.dump(clf, os.path.join(MODELS, f'{name}.pkl'))
joblib.dump(meta_clf, os.path.join(MODELS, 'stacking_meta.pkl'))
joblib.dump(ohe, os.path.join(MODELS, 'stacking_ohe.pkl'))

# ============== GENERATE SUBMISSION ==============
print("\n" + "="*60, flush=True)
print("GENERATING SUBMISSION", flush=True)
print("="*60, flush=True)

import pandas as pd
test_df = pd.read_csv(os.path.join(BASE, 'test.csv'))

# Generate predictions from all classifiers on test set
test_preds = {}
for name, clf in classifiers.items():
    if name.startswith('svm'):
        test_preds[name] = clf.predict(X_all_test_200)
    else:
        test_preds[name] = clf.predict(X_all_test_800)
    print(f"  Predicted test set with {name}", flush=True)

# Use best method for submission
if best_ensemble_acc >= best_individual:
    if 'voting' in best_ensemble_name:
        # Use the voting approach
        if best_ensemble_name == 'top5_voting':
            test_matrix = np.array([test_preds[k] for k in top5]).T
        elif best_ensemble_name == 'top3_voting':
            test_matrix = np.array([test_preds[k] for k in top3]).T
        else:
            test_matrix = np.array([test_preds[k] for k in test_preds]).T
        final_pred = mode(test_matrix, axis=1)[0].flatten().astype(int)
    else:
        # Stacking
        meta_test = np.array([test_preds[k] for k in clf_names_sorted]).T
        meta_test_ohe = ohe.transform(meta_test)
        final_pred = meta_clf.predict(meta_test_ohe)
else:
    bn = max(results, key=lambda x: results[x])
    final_pred = test_preds[bn]

submission = pd.DataFrame({'im_name': test_df['im_name'], 'label': final_pred})
submission.to_csv(os.path.join(BASE, 'submission.csv'), index=False)
print(f"\nSubmission saved: {len(submission)} predictions", flush=True)
print(f"Label distribution: {dict(zip(*np.unique(final_pred, return_counts=True)))}", flush=True)

# Write experiment log
with open(os.path.join(BASE, 'experiment_log.md'), 'w') as f:
    f.write("# Experiment Log\n\n")
    f.write("## Features: 5350 dims -> PCA 800 (trees) / PCA 200 (SVM)\n\n")
    f.write("## Individual Classifiers (80/20 split)\n\n")
    f.write("| Classifier | Val Accuracy |\n|---|---|\n")
    for name, a in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f"| {name} | {a:.4f} |\n")
    f.write("\n## Ensemble Methods\n\n")
    f.write("| Method | Val Accuracy |\n|---|---|\n")
    for name, a in sorted(ensemble_results.items(), key=lambda x: -x[1]):
        f.write(f"| {name} | {a:.4f} |\n")
    f.write(f"\n**Overall Best: {overall_best:.4f}**\n")

print("\n=== Phase 3+4 Complete ===", flush=True)
