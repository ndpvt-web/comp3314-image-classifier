Read TASK_CONTEXT.md, RULES.md, and ARISTOTLE_ANALYSIS.md in this directory first. Then execute this full pipeline autonomously without stopping:

PHASE 1 - Data Understanding: Load the dataset, visualize sample images from each class, confirm this is CIFAR-10-like, analyze class distribution. Save visualizations.

PHASE 2 - Feature Engineering: Build a comprehensive feature extraction pipeline:
- HOG (Histogram of Oriented Gradients) with multiple configs
- Color histograms (RGB + HSV) with spatial grids  
- LBP (Local Binary Patterns) for texture
- Gabor filter responses
- Color moments (mean, std, skewness per channel)
- Spatial pyramid features
- PCA for dimensionality reduction
Cache extracted features to disk for reuse.

PHASE 3 - Classifier Exploration: Train and compare at least these classifiers:
- SVM with RBF kernel (tune C and gamma)
- Random Forest (tune n_estimators, max_depth)
- XGBoost/LightGBM gradient boosting
- Extra Trees
- Stacking/Voting ensemble of the above
Use 80/20 train/val split or 5-fold cross-validation.

PHASE 4 - AutoResearch Loop (Karpathy-style): Iteratively improve:
- Try a strategy -> train -> evaluate on validation -> analyze -> adapt
- Log every experiment to experiment_log.md
- Keep a strategy.md updated with current best approach and results
- Try different feature combinations, hyperparameters, ensemble weights
- Continue until accuracy plateaus

PHASE 5 - Final Output: Generate submission.csv with predictions for all test images in the exact format of test.csv but with predicted labels.

CRITICAL RULES:
- NO neural networks (CNNs, RNNs, Transformers) - ZERO TOLERANCE
- NO pre-trained models or external data
- Only classical ML: scikit-learn, OpenCV, numpy, scipy, xgboost, lightgbm
- sudo password is 87654321 if needed for pip install or brew install
- Target: maximize accuracy, aim for 70%+ accuracy
- Work FULLY autonomously - do not stop to ask questions, make decisions and keep going
- Install any needed packages with pip or brew as needed