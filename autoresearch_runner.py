#!/usr/bin/env python3
"""
AutoResearch Runner - Karpathy-style autonomous experimentation.
Iterates through strategies, trains models, evaluates, logs results,
adapts based on feedback. Runs FULLY AUTONOMOUSLY.

Memory-safe: Max 3GB Python. Uses gc.collect() aggressively.
"""
import os, sys, gc, time, json, subprocess, datetime
import numpy as np
import psutil

sys.stdout.reconfigure(line_buffering=True)

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
MODELS = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')
VIS = os.path.join(BASE, 'visualizations')
for d in [MODELS, RESULTS, VIS]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(BASE, 'experiment_log.md')
BEST_FILE = os.path.join(BASE, 'best_result.json')
STRATEGY_FILE = os.path.join(BASE, 'strategy.md')

def mem_gb():
    return psutil.Process().memory_info().rss / 1e9

def mem_check(label=""):
    m = mem_gb()
    print(f"  [MEM {label}] {m:.2f}GB", flush=True)
    if m > 2.5:
        print("  [MEM WARNING] >2.5GB, running gc.collect()", flush=True)
        gc.collect()
    return m

def log_experiment(exp_id, desc, acc, details, duration):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Exp {exp_id} [{ts}]\n")
        f.write(f"- **Accuracy**: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"- **Description**: {desc}\n")
        f.write(f"- **Duration**: {duration:.0f}s\n")
        f.write(f"- **Details**: {details}\n---\n")
    print(f"  [LOG] Exp {exp_id}: {acc:.4f} - {desc}", flush=True)

def load_best():
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE) as f:
            return json.load(f)
    return {"accuracy": 0.0, "exp_id": 0, "desc": "none"}

def save_best(exp_id, acc, desc):
    with open(BEST_FILE, "w") as f:
        json.dump({"accuracy": acc, "exp_id": exp_id, "desc": desc,
                    "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)

def git_push(msg):
    try:
        os.chdir(BASE)
        subprocess.run(["git", "add", "-A"], capture_output=True)
        subprocess.run(["git", "commit", "-m", msg], capture_output=True)
        subprocess.run(["git", "push"], capture_output=True)
        print(f"  [GIT] Pushed: {msg}", flush=True)
    except Exception as e:
        print(f"  [GIT] Failed: {e}", flush=True)

def git_branch(name):
    try:
        os.chdir(BASE)
        subprocess.run(["git", "checkout", "-b", f"strategy/{name}"], capture_output=True)
        print(f"  [GIT] Branch: strategy/{name}", flush=True)
    except:
        pass

def git_main():
    try:
        os.chdir(BASE)
        subprocess.run(["git", "checkout", "main"], capture_output=True)
    except:
        pass

# ============================================================
# STRATEGY DEFINITIONS
# Each strategy is a dict with: name, features, pca_dim, classifier_fn
# ============================================================

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
import xgboost as xgb
import lightgbm as lgb
import joblib

def load_features_and_split():
    """Load cached features, split train/val. Memory-safe."""
    print("Loading features...", flush=True)
    mem_check("before load")
    all_feat = np.load(os.path.join(CACHE, 'all_features_combined.npy'))
    labels = np.load(os.path.join(CACHE, 'train_labels.npy'))
    train_feat = np.nan_to_num(all_feat[:50000], nan=0.0, posinf=0.0, neginf=0.0)
    test_feat = np.nan_to_num(all_feat[50000:], nan=0.0, posinf=0.0, neginf=0.0)
    del all_feat; gc.collect()
    mem_check("after load+cleanup")
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_feat, labels, test_size=0.2, random_state=42, stratify=labels)
    return X_tr, X_val, y_tr, y_val, test_feat, train_feat, labels

def run_svm_experiment(X_tr, X_val, y_tr, y_val, C, gamma, pca_dim):
    """Train SVM with given params. Returns accuracy."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_val)
    pca = PCA(n_components=pca_dim, random_state=42)
    Xtr = pca.fit_transform(Xtr)
    Xva = pca.transform(Xva)
    mem_check("after PCA")
    clf = SVC(C=C, gamma=gamma, kernel='rbf', cache_size=1000)
    t0 = time.time()
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xva)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0
    del Xtr, Xva; gc.collect()
    return acc, elapsed, clf, scaler, pca

def run_lgbm_experiment(X_tr, X_val, y_tr, y_val, pca_dim, n_est, lr, leaves):
    """Train LightGBM. Returns accuracy."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_val)
    pca = PCA(n_components=pca_dim, random_state=42)
    Xtr = pca.fit_transform(Xtr)
    Xva = pca.transform(Xva)
    mem_check("after PCA")
    clf = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=leaves,
        learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, n_jobs=-1, random_state=42, verbose=-1)
    t0 = time.time()
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xva)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0
    del Xtr, Xva; gc.collect()
    return acc, elapsed, clf, scaler, pca

def run_xgb_experiment(X_tr, X_val, y_tr, y_val, pca_dim, n_est, lr, depth):
    """Train XGBoost. Returns accuracy."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_val)
    pca = PCA(n_components=pca_dim, random_state=42)
    Xtr = pca.fit_transform(Xtr)
    Xva = pca.transform(Xva)
    mem_check("after PCA")
    clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth,
        learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, tree_method='hist', eval_metric='mlogloss')
    t0 = time.time()
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xva)
    acc = accuracy_score(y_val, pred)
    elapsed = time.time() - t0
    del Xtr, Xva; gc.collect()
    return acc, elapsed, clf, scaler, pca

def generate_submission(clf, scaler, pca, test_feat, filename="submission.csv"):
    """Generate submission CSV from model."""
    import pandas as pd
    Xt = scaler.transform(test_feat)
    Xt = pca.transform(Xt)
    preds = clf.predict(Xt)
    test_df = pd.read_csv(os.path.join(BASE, 'test.csv'))
    test_df['label'] = preds.astype(int)
    out = os.path.join(BASE, filename)
    test_df.to_csv(out, index=False)
    print(f"  [SUBMISSION] Saved {out} ({len(preds)} predictions)", flush=True)
    del Xt; gc.collect()
    return out

def build_ensemble_predictions(X_tr, X_val, y_tr, y_val, test_feat, configs):
    """Build ensemble by saving predictions from each model separately.
    configs: list of (name, train_fn) where train_fn returns (val_preds, test_preds)
    Memory-safe: only one model loaded at a time."""
    val_preds = []
    test_preds = []
    for name, train_fn in configs:
        print(f"  Ensemble member: {name}", flush=True)
        vp, tp = train_fn(X_tr, X_val, y_tr, y_val, test_feat)
        val_preds.append(vp)
        test_preds.append(tp)
        gc.collect()
        mem_check(f"after {name}")
    # Majority voting
    val_stack = np.array(val_preds)  # (n_models, n_samples)
    test_stack = np.array(test_preds)
    from scipy.stats import mode
    val_ensemble = mode(val_stack, axis=0)[0].flatten()
    test_ensemble = mode(test_stack, axis=0)[0].flatten()
    val_acc = accuracy_score(y_val, val_ensemble)
    return val_acc, val_ensemble, test_ensemble


# ============================================================
# MAIN AUTORESEARCH LOOP
# ============================================================
def main():
    print("=" * 60)
    print("AUTORESEARCH LOOP - COMP3314 Image Classification")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Memory limit: 3GB | Mac RAM: 8GB")
    print("=" * 60, flush=True)

    # Init log
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("# Experiment Log\n\nAutomated experiment tracking.\n\n")

    best = load_best()
    print(f"Current best: {best['accuracy']:.4f}\n", flush=True)

    # Load data once
    X_tr, X_val, y_tr, y_val, test_feat, full_train, full_labels = load_features_and_split()
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {test_feat.shape}\n", flush=True)

    exp_id = best.get('exp_id', 0)
    best_acc = best['accuracy']
    best_clf = None
    best_scaler = None
    best_pca = None

    # ---- PHASE 1: SVM Hyperparameter Search ----
    print("\n" + "=" * 40)
    print("PHASE 1: SVM Hyperparameter Search")
    print("=" * 40, flush=True)

    svm_configs = [
        # (C, gamma, pca_dim)
        (10, 'scale', 200),   # baseline: 71.20%
        (10, 'scale', 250),
        (10, 'scale', 300),
        (10, 'scale', 150),
        (15, 'scale', 200),
        (20, 'scale', 200),
        (5, 'scale', 200),
        (8, 'scale', 200),
        (10, 0.001, 200),
        (10, 0.005, 200),
        (10, 'auto', 200),
        (10, 'scale', 350),
        (10, 'scale', 400),
        (15, 'scale', 250),
        (20, 'scale', 250),
        (15, 'scale', 300),
        (10, 0.002, 250),
        (10, 0.003, 200),
        (12, 'scale', 200),
        (12, 'scale', 250),
    ]

    for C, gamma, pca_dim in svm_configs:
        exp_id += 1
        desc = f"SVM RBF C={C} gamma={gamma} PCA-{pca_dim}"
        print(f"\n[Exp {exp_id}] {desc}", flush=True)
        try:
            acc, elapsed, clf, scaler, pca = run_svm_experiment(
                X_tr, X_val, y_tr, y_val, C, gamma, pca_dim)
            log_experiment(exp_id, desc, acc, f"C={C},gamma={gamma},pca={pca_dim}", elapsed)
            if acc > best_acc:
                best_acc = acc
                best_clf, best_scaler, best_pca = clf, scaler, pca
                save_best(exp_id, acc, desc)
                print(f"  *** NEW BEST: {acc:.4f} ***", flush=True)
                # Save submission for best
                generate_submission(clf, scaler, pca, test_feat, f"submission_exp{exp_id}.csv")
                generate_submission(clf, scaler, pca, test_feat, "submission.csv")
                git_push(f"SVM exp{exp_id}: {acc:.4f} - C={C} gamma={gamma} PCA-{pca_dim}")
            else:
                del clf; gc.collect()
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            gc.collect()

    # ---- PHASE 2: LightGBM Tuning ----
    print("\n" + "=" * 40)
    print("PHASE 2: LightGBM Tuning")
    print("=" * 40, flush=True)

    lgbm_configs = [
        # (pca_dim, n_est, lr, leaves)
        (800, 1000, 0.05, 127),   # baseline: 64.46%
        (800, 2000, 0.03, 255),
        (800, 3000, 0.02, 255),
        (500, 1500, 0.05, 200),
        (800, 2000, 0.05, 127),
        (1000, 1500, 0.03, 200),
        (600, 2000, 0.03, 200),
    ]

    for pca_dim, n_est, lr, leaves in lgbm_configs:
        exp_id += 1
        desc = f"LGBM n={n_est} lr={lr} leaves={leaves} PCA-{pca_dim}"
        print(f"\n[Exp {exp_id}] {desc}", flush=True)
        try:
            acc, elapsed, clf, scaler, pca = run_lgbm_experiment(
                X_tr, X_val, y_tr, y_val, pca_dim, n_est, lr, leaves)
            log_experiment(exp_id, desc, acc, f"n={n_est},lr={lr},leaves={leaves}", elapsed)
            if acc > best_acc:
                best_acc = acc
                best_clf, best_scaler, best_pca = clf, scaler, pca
                save_best(exp_id, acc, desc)
                print(f"  *** NEW BEST: {acc:.4f} ***", flush=True)
                generate_submission(clf, scaler, pca, test_feat, "submission.csv")
                git_push(f"LGBM exp{exp_id}: {acc:.4f}")
            else:
                del clf; gc.collect()
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            gc.collect()

    # ---- PHASE 3: XGBoost Tuning ----
    print("\n" + "=" * 40)
    print("PHASE 3: XGBoost Tuning")
    print("=" * 40, flush=True)

    xgb_configs = [
        # (pca_dim, n_est, lr, depth)
        (800, 500, 0.1, 8),     # baseline: 63.34%
        (800, 1000, 0.05, 10),
        (800, 1500, 0.03, 10),
        (500, 1000, 0.05, 8),
        (800, 2000, 0.02, 12),
        (600, 1000, 0.05, 10),
    ]

    for pca_dim, n_est, lr, depth in xgb_configs:
        exp_id += 1
        desc = f"XGB n={n_est} lr={lr} depth={depth} PCA-{pca_dim}"
        print(f"\n[Exp {exp_id}] {desc}", flush=True)
        try:
            acc, elapsed, clf, scaler, pca = run_xgb_experiment(
                X_tr, X_val, y_tr, y_val, pca_dim, n_est, lr, depth)
            log_experiment(exp_id, desc, acc, f"n={n_est},lr={lr},depth={depth}", elapsed)
            if acc > best_acc:
                best_acc = acc
                best_clf, best_scaler, best_pca = clf, scaler, pca
                save_best(exp_id, acc, desc)
                print(f"  *** NEW BEST: {acc:.4f} ***", flush=True)
                generate_submission(clf, scaler, pca, test_feat, "submission.csv")
                git_push(f"XGB exp{exp_id}: {acc:.4f}")
            else:
                del clf; gc.collect()
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            gc.collect()

    # ---- PHASE 4: Ensemble ----
    print("\n" + "=" * 40)
    print("PHASE 4: Memory-Safe Ensemble")
    print("=" * 40, flush=True)

    # Train each model, save only predictions, delete model
    def svm_member(Xtr, Xva, ytr, yva, Xtest):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xtest)
        pca = PCA(n_components=200, random_state=42)
        Xtr_p = pca.fit_transform(Xtr_s); Xva_p = pca.transform(Xva_s); Xte_p = pca.transform(Xte_s)
        del Xtr_s, Xva_s, Xte_s; gc.collect()
        clf = SVC(C=10, gamma='scale', kernel='rbf', cache_size=1000)
        clf.fit(Xtr_p, ytr)
        vp = clf.predict(Xva_p); tp = clf.predict(Xte_p)
        del clf, Xtr_p, Xva_p, Xte_p; gc.collect()
        return vp, tp

    def lgbm_member(Xtr, Xva, ytr, yva, Xtest):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xtest)
        pca = PCA(n_components=800, random_state=42)
        Xtr_p = pca.fit_transform(Xtr_s); Xva_p = pca.transform(Xva_s); Xte_p = pca.transform(Xte_s)
        del Xtr_s, Xva_s, Xte_s; gc.collect()
        clf = lgb.LGBMClassifier(n_estimators=2000, num_leaves=255, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1)
        clf.fit(Xtr_p, ytr)
        vp = clf.predict(Xva_p); tp = clf.predict(Xte_p)
        del clf, Xtr_p, Xva_p, Xte_p; gc.collect()
        return vp, tp

    def xgb_member(Xtr, Xva, ytr, yva, Xtest):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xtest)
        pca = PCA(n_components=800, random_state=42)
        Xtr_p = pca.fit_transform(Xtr_s); Xva_p = pca.transform(Xva_s); Xte_p = pca.transform(Xte_s)
        del Xtr_s, Xva_s, Xte_s; gc.collect()
        clf = xgb.XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
            tree_method='hist', eval_metric='mlogloss')
        clf.fit(Xtr_p, ytr)
        vp = clf.predict(Xva_p); tp = clf.predict(Xte_p)
        del clf, Xtr_p, Xva_p, Xte_p; gc.collect()
        return vp, tp

    try:
        exp_id += 1
        configs = [("SVM-C10-PCA200", svm_member),
                   ("LGBM-2000", lgbm_member),
                   ("XGB-1000", xgb_member)]
        ens_acc, val_ens, test_ens = build_ensemble_predictions(
            X_tr, X_val, y_tr, y_val, test_feat, configs)
        desc = "Ensemble(SVM+LGBM+XGB) majority vote"
        log_experiment(exp_id, desc, ens_acc, "3-model majority vote", 0)
        print(f"\n  ENSEMBLE ACCURACY: {ens_acc:.4f}", flush=True)
        if ens_acc > best_acc:
            best_acc = ens_acc
            save_best(exp_id, ens_acc, desc)
            print(f"  *** NEW BEST: {ens_acc:.4f} ***", flush=True)
            # Save ensemble submission
            import pandas as pd
            test_df = pd.read_csv(os.path.join(BASE, 'test.csv'))
            test_df['label'] = test_ens.astype(int)
            test_df.to_csv(os.path.join(BASE, 'submission.csv'), index=False)
            git_push(f"Ensemble exp{exp_id}: {ens_acc:.4f}")
    except Exception as e:
        print(f"  [ENSEMBLE ERROR] {e}", flush=True)

    # ---- PHASE 5: Feature Selection Experiments ----
    print("\n" + "=" * 40)
    print("PHASE 5: Feature Selection + Subsets")
    print("=" * 40, flush=True)

    # Try individual feature types
    feature_files = {
        'hog1': 'hog1.npy', 'hog2': 'hog2.npy', 'hog3': 'hog3.npy',
        'hog_color': 'hog_color.npy', 'color_hist': 'color_hist.npy',
        'color_moments': 'color_moments.npy', 'gabor': 'gabor.npy',
        'lbp_r1': 'lbp_r1.npy', 'lbp_r2': 'lbp_r2.npy', 'lbp_r3': 'lbp_r3.npy',
        'edge': 'edge.npy', 'spatial_2x2': 'spatial_2x2.npy',
        'spatial_4x4': 'spatial_4x4.npy', 'hu': 'hu.npy',
    }

    # Try HOG+color combo (likely strongest subset)
    subsets = [
        ("hog_all+color", ['hog1', 'hog2', 'hog3', 'hog_color', 'color_hist', 'color_moments']),
        ("hog+color+lbp", ['hog1', 'hog2', 'hog3', 'color_hist', 'lbp_r1', 'lbp_r2', 'lbp_r3']),
        ("hog+gabor+color", ['hog1', 'hog2', 'hog3', 'gabor', 'color_hist', 'color_moments']),
        ("all_no_spatial", ['hog1','hog2','hog3','hog_color','color_hist','color_moments','gabor','lbp_r1','lbp_r2','lbp_r3','edge','hu']),
    ]

    for subset_name, feat_keys in subsets:
        exp_id += 1
        desc = f"SVM C=10 PCA-200 features={subset_name}"
        print(f"\n[Exp {exp_id}] {desc}", flush=True)
        try:
            arrays = []
            for k in feat_keys:
                arr = np.load(os.path.join(CACHE, feature_files[k]))
                arrays.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
                del arr
            combined = np.hstack(arrays)
            del arrays; gc.collect()
            Xtr_s = combined[:50000]; Xte_s = combined[50000:]
            del combined; gc.collect()
            X_tr2, X_va2, y_tr2, y_va2 = train_test_split(
                Xtr_s, full_labels, test_size=0.2, random_state=42, stratify=full_labels)
            del Xtr_s; gc.collect()
            pca_dim = min(200, X_tr2.shape[1])
            acc, elapsed, clf, scaler, pca = run_svm_experiment(
                X_tr2, X_va2, y_tr2, y_va2, 10, 'scale', pca_dim)
            log_experiment(exp_id, desc, acc, f"features={subset_name}", elapsed)
            if acc > best_acc:
                best_acc = acc
                save_best(exp_id, acc, desc)
                print(f"  *** NEW BEST: {acc:.4f} ***", flush=True)
                generate_submission(clf, scaler, pca, Xte_s, "submission.csv")
                git_push(f"Feature subset {subset_name}: {acc:.4f}")
            del clf, X_tr2, X_va2, Xte_s; gc.collect()
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            gc.collect()

    # ---- FINAL: Generate best submission if not done ----
    print("\n" + "=" * 40)
    print("AUTORESEARCH COMPLETE")
    print(f"Best accuracy: {best_acc:.4f}")
    print("=" * 40, flush=True)

    if best_clf is not None:
        generate_submission(best_clf, best_scaler, best_pca, test_feat, "submission.csv")

    git_push(f"AutoResearch complete - best accuracy: {best_acc:.4f}")

    # Write strategy summary
    with open(STRATEGY_FILE, "w") as f:
        f.write(f"# Strategy Summary\n\n")
        f.write(f"Best accuracy: {best_acc:.4f}\n")
        f.write(f"Best config: {load_best()}\n\n")
        f.write(f"## Next steps to try:\n")
        f.write(f"- SVM with polynomial kernel\n")
        f.write(f"- Feature engineering: PCA whitening\n")
        f.write(f"- Stacking ensemble instead of voting\n")
        f.write(f"- Data augmentation (flip/rotate train images, re-extract features)\n")

if __name__ == "__main__":
    main()
