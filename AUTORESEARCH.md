# AutoResearch Guide

## What is AutoResearch?
Inspired by Andrej Karpathy's approach: an autonomous loop that tries strategies, evaluates, adapts, and iterates until optimality.

## How to Run
```bash
source venv/bin/activate
PYTHONUNBUFFERED=1 python3 autoresearch_runner.py 2>&1 | tee autoresearch_output.log
```

## The Loop
```
1. Pick a strategy (from the queue below)
2. Create git branch: strategy/<name>
3. Implement the strategy in a Python script
4. Train the model
5. Evaluate on validation set (20% split, random_state=42)
6. Log results to experiment_log.md
7. If accuracy > best:
   - Save submission.csv
   - Save model to models/
   - Merge to main
   - Update best_result.json
8. Push to GitHub
9. Pick next strategy -> goto 1
```

## Strategy Queue (Priority Order)

### Tier 1: SVM Hyperparameter Grid (Current Best: ~71%)
- C: [5, 8, 10, 12, 15, 20, 50]
- gamma: [scale, auto, 0.001, 0.002, 0.005, 0.01]
- PCA dims: [150, 200, 250, 300, 350, 400]
- Best combos first: C=10,gamma=scale,PCA=200 (71.2% baseline)

### Tier 2: Feature Subset Selection
- HOG-only (hog1+hog2+hog3+hog_color) -> SVM
- HOG + Color (add color_hist, color_moments) -> SVM
- HOG + Color + LBP -> SVM
- All features minus spatial -> SVM
- SelectKBest with mutual_info_classif -> SVM

### Tier 3: Ensemble Methods
- Majority voting: SVM + LightGBM + XGBoost
- Weighted voting: weight by val accuracy
- Stacking: LogisticRegression meta-learner on base predictions
- Different PCA dims per base model

### Tier 4: Advanced
- SVM with poly kernel (degree 2,3)
- LinearSVC with L1 regularization (built-in feature selection)
- Data augmentation: horizontal flips of training images -> re-extract features
- PCA whitening (whiten=True)
- StandardScaler vs MinMaxScaler vs RobustScaler

## Memory Rules
- Max 3GB Python process
- gc.collect() after every model train/predict
- Delete model after saving predictions
- Never load >1 large model at a time
- Monitor with psutil

## Files
- autoresearch_runner.py: The main autonomous runner (run this!)
- autoresearch.py: Helper functions (log_experiment, save_best, git_push)
- experiment_log.md: All experiment results
- best_result.json: Current best model info
- strategy.md: Strategy notes and what to try next
