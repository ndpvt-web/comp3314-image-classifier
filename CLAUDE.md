# COMP3314 Image Classification - Project Instructions

## ENVIRONMENT SETUP - READ FIRST
A Python virtual environment is already set up with ALL required packages installed.
**ALWAYS use the venv Python for ALL python/pip commands:**
```bash
source /Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge/venv/bin/activate
```
Packages installed: scikit-learn, opencv-python, xgboost, lightgbm, scikit-image, scipy, joblib, matplotlib, seaborn, pandas, pillow, numpy

## MEMORY SAFETY - CRITICAL (8GB Mac, WILL CRASH IF VIOLATED)
This Mac has ONLY 8GB RAM. You MUST follow these rules:

1. **MAX 3GB Python usage** - Leave 5GB for OS + Claude Code
2. **NEVER train Extra Trees or Random Forest** - They create 2-4GB model files and use massive RAM
3. **NEVER load multiple large models simultaneously** - Load one, predict, delete, load next
4. **Use gc.collect() after every major step** - Free memory aggressively
5. **Use batch processing for features** - Process images in batches of 5000, not all at once
6. **Preferred classifiers (memory-safe):**
   - SVM RBF (best: 71.2% at C=10, PCA-200) - ~200MB peak
   - LightGBM (~100MB model, fast)
   - XGBoost (~30MB model, fast)
   - Logistic Regression (tiny)
7. **BANNED classifiers:**
   - ExtraTreesClassifier (4GB+ model file)
   - RandomForestClassifier with >200 trees (2GB+ model file)
   - Any bagging ensemble with n_estimators > 200
8. **For ensemble:** Use soft voting with SVM + LightGBM + XGBoost only. Load predictions sequentially, not models.
9. **Monitor memory:** Add `import psutil; print(f"RAM: {psutil.Process().memory_info().rss/1e9:.1f}GB")` in scripts
10. **All output goes to project directory** - NEVER use /tmp for results

## ABSOLUTE RULES - VIOLATION MEANS ZERO POINTS
1. **NO NEURAL NETWORKS** - No CNNs, RNNs, Transformers, or ANY deep learning
2. **NO EXTERNAL DATA** - Only use the provided training images
3. **NO PRE-TRAINED MODELS** - No transfer learning
4. **NEVER MENTION AI ASSISTANTS IN GIT COMMITS** - No Claude, AI, LLM references

## Dataset
- Location: /Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge/
- train.csv: 50,000 images with labels (0-9), CIFAR-10 style
- test.csv: 10,000 images (to predict)
- train_ims/: Training images (32x32 RGB)
- test_ims/: Test images (32x32 RGB)

## CURRENT BEST RESULTS
- Split PCA (HOG-150+Other-150) + PT + SVM C=5: **74.35%** (BEST, submission generated)
- Unified PCA-245 + PT + SVM C=8: 72.12%
- Unified PCA-200 + SVM C=8 (no PT): 71.42%
- SVM RBF C=10 PCA-200 (original): 71.20%
- LightGBM 1000: 64.46%
- XGBoost 500: 63.34%

## KEY TECHNIQUE: Split PCA + Power Transform
The breakthrough was splitting features into HOG (4824d) and non-HOG (526d) groups,
applying separate PCA to each, then concatenating. Non-HOG features (color, LBP, Gabor,
spatial, edge) were being crushed in unified PCA. Yeo-Johnson power transform before PCA
also helps significantly.

Features are already extracted and cached in features_cache/:
- all_features_combined.npy (1.2GB, 5350 features per image)
- Individual feature files (HOG, color, LBP, Gabor, spatial, etc.)

Models saved in models/:
- lightgbm.pkl (100MB)
- xgboost.pkl (28MB)
- scaler.pkl, pca*.pkl

## YOUR MISSION - CONTINUE FROM HERE
The heavy lifting is done. Features are cached. Now:

1. **Load existing features** (features_cache/all_features_combined.npy)
2. **Train SVM with more hyperparams** - Try C=[5,10,20,50], gamma=[scale,auto,0.001,0.01], PCA=[150,200,250,300]
3. **Build memory-safe ensemble** - Save predictions from each model separately, then combine with voting/stacking
4. **Generate submission.csv** - Predict on test set with best model/ensemble
5. **AutoResearch** - Try new feature combos, but ALWAYS check memory first
6. **Git push after every experiment** - Use strategy branches

## File Structure (ALL output here, NOT in /tmp)
- models/ - Saved model files
- features_cache/ - Cached feature .npy files
- visualizations/ - Charts and plots
- results/ - Experiment results and logs
- experiment_log.md - Log ALL experiments
- strategy.md - Current strategy notes
- submission.csv - Final submission file

## Git Strategy
- Create strategy/<name> branches for experiments
- Merge to main if accuracy improves
- Push after EVERY experiment
- No AI/Claude mentions in commits

## Working Style
- DO NOT STOP - work continuously
- DO NOT ASK QUESTIONS - make decisions
- Install packages: source venv/bin/activate && pip install <pkg>
- Target: 70%+ accuracy (already at 71.2%), aim for 75%+
