# COMP3314 Image Classification - Detailed Execution Plan

## Project Goal
Achieve maximum accuracy (target >= 70%, aim for top 10%) on a CIFAR-10-like 32x32 RGB image classification task using ONLY classical ML methods.

## Phase 0: Environment Setup (15 min)
- [ ] Install dependencies: scikit-learn, opencv-python, xgboost, lightgbm, scikit-image, scipy, joblib, matplotlib, seaborn, pandas
- [ ] Verify all imports work
- [ ] Set up directory structure: features_cache/, results/, models/, visualizations/

## Phase 1: Data Understanding & Analysis (30 min)
- [ ] Load train.csv and test.csv
- [ ] Load sample images, confirm 32x32 RGB
- [ ] Plot class distribution (should be ~5000 per class)
- [ ] Visualize 5 random images per class in a grid
- [ ] Compute per-class mean images
- [ ] Compute overall pixel statistics (mean, std per channel)
- [ ] Identify if this is CIFAR-10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- [ ] Save all visualizations to visualizations/

## Phase 2: Feature Engineering (2-3 hours)
### 2.1 Raw Pixel Features (baseline)
- [ ] Flatten 32x32x3 = 3072 features
- [ ] Normalize to [0,1]

### 2.2 HOG Features (most important)
- [ ] Config 1: pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9 -> ~324 features
- [ ] Config 2: pixels_per_cell=(4,4), cells_per_block=(2,2), orientations=9 -> ~1764 features
- [ ] Config 3: pixels_per_cell=(8,8), cells_per_block=(3,3), orientations=12
- [ ] Extract on grayscale AND per-channel (RGB)
- [ ] Multi-scale HOG: resize to 64x64, extract HOG, then original 32x32

### 2.3 Color Features
- [ ] RGB histogram: 32 bins per channel = 96 features
- [ ] HSV histogram: 32 bins per channel = 96 features
- [ ] Lab histogram: 32 bins per channel = 96 features
- [ ] Color moments: mean, std, skew per channel (RGB+HSV) = 18 features
- [ ] Spatial color: 4x4 grid, mean RGB per cell = 48 features
- [ ] Spatial color: 2x2 grid, full histogram per cell = more features
- [ ] Dominant colors via K-means (k=3-5) on pixel colors

### 2.4 Texture Features
- [ ] LBP: radius=1,2,3 with different n_points, histogram = ~78 features each
- [ ] Gabor filters: 5 frequencies x 8 orientations, mean+var per filter = 80 features
- [ ] GLCM: contrast, dissimilarity, homogeneity, energy, correlation at 4 angles

### 2.5 Shape/Edge Features
- [ ] Canny edge density (overall + per quadrant)
- [ ] Edge orientation histogram
- [ ] Hu moments (7 features)
- [ ] Contour-based features

### 2.6 Spatial/Structure Features
- [ ] Spatial pyramid: features at 1x1, 2x2, 4x4 grids
- [ ] Center vs periphery statistics
- [ ] Horizontal/vertical symmetry measures

### 2.7 Feature Caching
- [ ] Save all extracted features as .npy files in features_cache/
- [ ] Create feature metadata file tracking what each feature set contains

## Phase 3: Baseline Experiments (1-2 hours)
### Experiment 1: Raw Pixels
- [ ] KNN (k=5,10,20) on raw pixels
- [ ] SVM (linear) on raw pixels
- [ ] Expected: ~35-40%

### Experiment 2: HOG Only
- [ ] SVM (RBF, tune C=[0.1,1,10,100], gamma=['scale','auto']) on HOG
- [ ] Expected: ~52-58%

### Experiment 3: HOG + Color
- [ ] SVM (RBF) on concatenated HOG + color histograms
- [ ] Expected: ~58-64%

### Experiment 4: All Features
- [ ] Concatenate all features
- [ ] PCA to reduce dimensionality (500, 1000, 2000 components)
- [ ] SVM (RBF) on PCA-reduced features
- [ ] Expected: ~62-70%

## Phase 4: Classifier Comparison (1-2 hours)
Using the best feature set from Phase 3:
- [ ] SVM (RBF kernel) - tune C, gamma with GridSearchCV
- [ ] Random Forest (n_estimators=500-2000, max_features=['sqrt','log2'])
- [ ] XGBoost (learning_rate, max_depth, n_estimators, subsample)
- [ ] LightGBM (num_leaves, learning_rate, n_estimators)
- [ ] Extra Trees (n_estimators=1000-3000)
- [ ] Logistic Regression (with PCA features)
- [ ] Record all results in experiment_log.md

## Phase 5: Ensemble Building (1-2 hours)
- [ ] Soft Voting: combine top 3-4 classifiers
- [ ] Stacking: use top classifiers as base, logistic regression as meta
- [ ] Weighted voting based on per-class performance
- [ ] Blending with holdout set
- [ ] Test different combinations

## Phase 6: AutoResearch Loop (continuous, remaining time)
Automated iteration cycle:
```
for each strategy in strategy_queue:
    1. Modify feature pipeline or classifier config
    2. Train on training split
    3. Evaluate on validation split
    4. Log result to experiment_log.md
    5. If improvement: update best model, push to git
    6. Generate next strategy based on analysis
```

Strategy ideas to explore:
- [ ] Data augmentation: horizontal flips, small rotations
- [ ] Per-class specialized features
- [ ] Feature selection (mutual information, chi-squared)
- [ ] Calibrated classifiers for better probability estimates
- [ ] Cost-sensitive learning if some classes harder
- [ ] Different PCA dimensions
- [ ] Kernel PCA instead of linear PCA
- [ ] Different HOG configurations
- [ ] Multi-resolution features (resize images to 16x16, 64x64)

## Phase 7: Final Deliverables (1-2 hours)
- [ ] Generate submission.csv with best model
- [ ] Create clean Jupyter notebook with full pipeline
- [ ] Write PDF report covering:
  - Dataset analysis with visualizations
  - Classifier comparison table
  - Final solution description
- [ ] Final git push with all results

## Success Criteria
- Accuracy >= 70% on validation set (proxy for private test)
- Clean, reproducible Jupyter notebook
- Comprehensive experiment log
- All results pushed to GitHub

## Time Budget (4 days)
- Day 1: Phases 0-3 (setup, data, features, baselines)
- Day 2: Phases 4-5 (classifiers, ensemble)
- Day 3: Phase 6 (autoresearch loop, optimization)
- Day 4: Phase 7 (deliverables, cleanup, submission)
