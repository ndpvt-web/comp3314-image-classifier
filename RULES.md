# COMP3314 Assignment 3 - Hard Rules & Constraints

## Absolute Prohibitions (Violation = 0 Points)

### 1. NO Neural Networks
- **NO** Convolutional Neural Networks (CNNs)
- **NO** Recurrent Neural Networks (RNNs)
- **NO** Transformers
- **NO** any deep learning architecture
- **NO** PyTorch, TensorFlow, Keras neural network layers
- This means: NO backpropagation-trained multi-layer networks

### 2. NO External Data or Pre-trained Models
- **NO** additional datasets beyond what is provided
- **NO** pre-trained models (no transfer learning, no pre-trained feature extractors)
- **NO** pre-computed embeddings from external sources
- Must work ONLY with the 50,000 training images provided

### 3. NO Plagiarism
- **NO** copying code from external sources
- **NO** copying prediction results
- TA will verify notebook results match Kaggle submission

## What IS Allowed (Classical ML)
- **Feature extraction**: HOG, SIFT, LBP, color histograms, Gabor filters, edge detection, pixel values, PCA, etc.
- **Classical classifiers**: SVM, Random Forest, KNN, Gradient Boosting (XGBoost/LightGBM), Logistic Regression, Naive Bayes, Decision Trees, AdaBoost, Bagging, Extra Trees
- **Ensemble methods**: Voting, Stacking, Blending (using classical ML models)
- **Data augmentation**: Flips, rotations, crops (on training data only)
- **Feature engineering**: Any hand-crafted or computed features from the images
- **Dimensionality reduction**: PCA, t-SNE, UMAP (for features, not neural)
- **Hyperparameter tuning**: Grid search, random search, Bayesian optimization
- **Cross-validation**: Any CV strategy

## Submission Rules
- CSV must match test.csv image name order exactly
- Jupyter notebook must be self-contained and reproducible
- Notebook generates .csv in same directory when run
- All team members submit notebook + PDF on Moodle
- One member submits on Kaggle

## Scoring
- Accuracy-based scoring (see TASK_CONTEXT.md for thresholds)
- Final score based on PRIVATE test set (8,000 samples), NOT public (2,000)
- Top 10% ranking = +5 bonus points
