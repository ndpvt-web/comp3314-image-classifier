# Aristotelian First-Principles Analysis: COMP3314 Image Classification

## I. The Four Causes (Aristotle's Causal Framework)

### Material Cause - "What is it made of?"
- **Data**: 50,000 tiny images (32x32x3 = 3,072 raw features per image)
- **Nature**: This is almost certainly CIFAR-10. 32x32 RGB, 10 balanced classes, 50k train / 10k test.
- **CIFAR-10 classes** (if confirmed): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Information density**: 3,072 pixels per image. Very low resolution. Fine-grained features are lost.
- **Key insight**: At 32x32, there is NO fine detail. The signal lives in:
  - Color distributions (sky=blue for planes/ships, green for frogs/deer)
  - Coarse spatial structure (shape silhouettes)
  - Texture patterns (fur vs metal vs water)
  - Edge orientations (horizontal for ships, mixed for animals)

### Formal Cause - "What is its form/structure?"
- **Problem type**: Multi-class classification (10 classes)
- **Balanced classes**: ~5,000 each, so no class imbalance to handle
- **Constraint**: Classical ML only. No neural networks.
- **This fundamentally shapes everything**: We cannot learn features. We must DESIGN them.
- **The form of our solution**: Feature Engineering Pipeline -> Classifier -> Ensemble

### Efficient Cause - "What produces it?"
- **Our tools**: scikit-learn, OpenCV, numpy, scipy (feature extraction + ML)
- **Compute**: Mac M1 Pro (powerful for CPU-based ML, good parallelism)
- **Time**: ~4 days remaining
- **Agent**: Claude Code Opus 4.6 as autonomous research engine

### Final Cause - "What is its purpose/goal?"
- **Primary**: Maximize accuracy on the PRIVATE test set (8,000 samples)
- **Target**: >= 70% accuracy for full marks, top 10% for +5 bonus
- **Secondary**: Produce clean notebook + report
- **Meta-goal**: Not just high accuracy, but robust accuracy (private set matters more than public)

---

## II. First Principles Decomposition

### Principle 1: "What makes images distinguishable WITHOUT neural networks?"

At the most fundamental level, two images of different classes differ in:

1. **Color** - A frog is green, a truck is red/blue/multi, sky is blue
2. **Texture** - Fur has a texture, metal is smooth, water has ripples
3. **Shape/Edge** - A ship is flat/horizontal, a horse has legs, a plane has wings
4. **Spatial Layout** - Sky is typically on top, ground on bottom

Therefore our features must capture ALL four dimensions.

### Principle 2: "What is the CEILING for classical ML on CIFAR-10?"

Known benchmarks (no CNNs):
- Raw pixels + KNN: ~35-40%
- HOG + SVM: ~50-55%
- Multiple features + SVM: ~55-60%
- Heavy feature engineering + ensemble: ~65-75%
- State-of-art classical (competition-level): ~75-80%

**The theoretical limit** with perfect classical features on 32x32 is roughly 78-82%.
Our target of 70%+ is achievable but requires serious feature engineering.

### Principle 3: "Feature extraction IS the entire game"

Since we cannot learn features (no CNNs), the QUALITY of hand-crafted features
determines the ceiling. The classifier choice matters less than the features.

**This is the single most important insight.**

---

## III. Feature Engineering Strategy (From First Principles)

### Tier 1: Color Features (capture Cause 1)
- **Color histograms**: Per-channel (R, G, B) histograms, 32 bins each = 96 features
- **HSV histograms**: Hue, Saturation, Value histograms = 96 features
- **Color moments**: Mean, std, skewness for each channel = 9 features
- **Spatial color**: Divide image into 4x4 grid, compute color stats per cell = 144+ features
- **Reasoning**: Color alone separates several classes (frog=green, sky=blue for planes/ships)

### Tier 2: Texture Features (capture Cause 2)
- **LBP (Local Binary Patterns)**: Classic texture descriptor. Multiple radii/points. ~256 features
- **Gabor filters**: Multi-scale, multi-orientation. ~40-80 features
- **GLCM (Gray-Level Co-occurrence Matrix)**: Contrast, correlation, energy, homogeneity
- **Reasoning**: Texture distinguishes animals (fur) from vehicles (metal) from natural scenes

### Tier 3: Shape/Edge Features (capture Cause 3)
- **HOG (Histogram of Oriented Gradients)**: THE gold standard for classical vision
  - Multiple cell sizes (4x4, 8x8), multiple orientations
  - ~324-1764 features depending on config
- **Edge histograms**: Canny edges + orientation histograms
- **Hu Moments**: 7 rotation-invariant shape descriptors
- **Reasoning**: Shape is the primary discriminator between classes at low resolution

### Tier 4: Spatial Features (capture Cause 4)
- **Spatial pyramid**: Extract features at multiple spatial subdivisions (1x1, 2x2, 4x4)
- **Gist-like descriptors**: Global scene structure using oriented filters at multiple scales
- **Reasoning**: Spatial layout tells us "sky on top = outdoor" vs "road on bottom = vehicle"

### Tier 5: Dimensionality Reduction
- **PCA**: Reduce combined feature vector to manageable size (500-2000 components)
- **Reasoning**: Too many features = curse of dimensionality. PCA preserves variance.

---

## IV. Classifier Strategy (From First Principles)

### Why SVM is King Here
- SVMs with RBF kernel are proven best for medium-dimensional feature spaces
- They handle the "many features, not huge dataset" regime well
- CIFAR-10 + HOG + SVM is a well-known strong baseline

### Why Gradient Boosting is Strong
- XGBoost/LightGBM handle heterogeneous features naturally
- No need for feature scaling
- Built-in feature selection via importance
- Can capture non-linear decision boundaries

### Why Ensemble is the Answer
- Stacking/voting of SVM + GBM + Random Forest + Extra Trees
- Different classifiers capture different aspects of the feature space
- Meta-learner (logistic regression) on top combines strengths

### Proposed Classifier Hierarchy
1. **Baseline**: Raw pixels + KNN (establish floor ~35%)
2. **Strong baseline**: HOG + SVM (RBF) (~52-58%)
3. **Feature-rich**: Multi-feature + SVM (~60-68%)
4. **Ensemble**: Multi-feature + Stacking ensemble (~68-75%)
5. **Optimized**: Hyperparameter-tuned ensemble + augmented features (~72-78%)

---

## V. AutoResearch Loop Design (Karpathy-Style)

### The Loop
```
WHILE accuracy < target OR time_budget_remaining:
    1. HYPOTHESIZE: "Adding feature X should improve accuracy because..."
    2. IMPLEMENT: Extract feature X, add to feature pipeline
    3. TRAIN: Train classifier on train split
    4. EVALUATE: Test on validation split
    5. ANALYZE: Compare to previous best
       - If better: KEEP feature/change, update best
       - If worse: REVERT, analyze WHY
    6. LOG: Record result, reasoning, and decision
    7. ADAPT: Based on analysis, form next hypothesis
```

### Key Principles for the Loop
- **Always have a holdout validation set** (e.g., 20% of training data)
- **Never tune on public test set** (overfitting to 2,000 public samples = bad)
- **Log everything** (each experiment: features used, classifier, params, accuracy)
- **Incremental improvement** (don't change everything at once)
- **Feature importance analysis** after each round
- **Time-box experiments** (don't spend 2 hours on a 0.1% improvement)

---

## VI. Execution Architecture

### Mac Pro as the Engine
- Claude Code CLI (Opus 4.6) runs VISIBLY on Mac Terminal
- Acts as autonomous research agent
- Reads TASK_CONTEXT.md and RULES.md at start
- Creates strategy.md with experiment plan
- Executes experiments sequentially
- Logs results to experiment_log.md
- Adapts strategy based on results

### Workflow
```
Phase 1: Data Understanding (30 min)
  - Load and visualize data
  - Confirm it's CIFAR-10
  - Establish class characteristics

Phase 2: Baseline (~1 hour)
  - Raw pixels + simple classifiers
  - Establish accuracy floor

Phase 3: Feature Engineering (~3-4 hours)
  - Implement feature extractors one by one
  - Test each incrementally
  - Build optimal feature vector

Phase 4: Classifier Optimization (~2-3 hours)
  - Try multiple classifiers
  - Hyperparameter tuning
  - Build ensemble

Phase 5: AutoResearch Loop (~remaining time)
  - Automated experimentation
  - Fine-tuning features and classifiers
  - Push for maximum accuracy

Phase 6: Deliverables (~2 hours)
  - Clean Jupyter notebook
  - Generate final predictions CSV
  - PDF report
```

---

## VII. Risk Analysis

### Risk 1: Overfitting to Public Test Set
- **Mitigation**: Use proper cross-validation, never tune on public leaderboard

### Risk 2: Feature Extraction Too Slow
- **Mitigation**: 32x32 images are tiny. 50k images is manageable. Cache features.

### Risk 3: Not Reaching 70%
- **Mitigation**: Stack more features, use ensemble. Known classical approaches hit 75%+.

### Risk 4: Time Constraint (4 days)
- **Mitigation**: Automated pipeline. Claude Code runs experiments autonomously.

---

## VIII. The Aristotelian Synthesis

**The essence of this problem**: Transform raw pixels (material) into discriminative features (form)
using engineered extractors (efficient cause) to achieve maximum classification accuracy (final cause),
all within the constraint of classical machine learning.

**The key syllogism**:
1. To classify images, we need discriminative representations
2. Without neural networks, representations must be hand-engineered
3. The best hand-engineered representations combine color, texture, shape, and spatial information
4. Therefore, our solution must be a rich multi-feature pipeline with strong classical classifiers

**The golden mean**: Not too few features (underfitting) nor too many (curse of dimensionality).
Use PCA to find the optimal dimensionality. Not too simple a classifier (underfitting) nor too
complex (overfitting). Use cross-validation to find the sweet spot.
