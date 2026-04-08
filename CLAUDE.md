# COMP3314 Image Classification - Project Instructions

## ABSOLUTE RULES - VIOLATION MEANS ZERO POINTS
1. **NO NEURAL NETWORKS** - No CNNs, RNNs, Transformers, or ANY deep learning. No PyTorch/TensorFlow neural network layers. No backpropagation-trained multi-layer networks. ZERO TOLERANCE.
2. **NO EXTERNAL DATA** - Only use the 50,000 training images provided in train_ims/
3. **NO PRE-TRAINED MODELS** - No transfer learning, no pre-trained feature extractors, no pre-computed embeddings
4. **NEVER MENTION AI ASSISTANTS IN GIT COMMITS** - No references to Claude, AI, LLM, or any AI tool in commit messages, code comments, or any pushed files. This is an absolute requirement.

## Dataset
- Location: This directory (`/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge/`)
- train.csv: 50,000 images with labels (0-9)
- test.csv: 10,000 images with label=0 (to predict)
- train_ims/: Training images (32x32 RGB JPEGs)
- test_ims/: Test images (32x32 RGB JPEGs)
- 10 balanced classes (~5,000 each) - likely CIFAR-10

## Your Mission
Work FULLY AUTONOMOUSLY through these phases:
1. Data analysis and visualization
2. Feature engineering (HOG, color, LBP, texture, spatial)
3. Train multiple classifiers (SVM, RF, XGBoost, LightGBM, ExtraTrees)
4. Build ensemble (stacking/voting)
5. AutoResearch loop - iterate to maximize accuracy
6. Generate submission.csv

## Key Files
- PLAN.md: Detailed execution plan (follow this)
- RULES.md: All competition rules
- TASK_CONTEXT.md: Full task context
- ARISTOTLE_ANALYSIS.md: Strategy analysis
- autoresearch.py: Import log_experiment, save_best, git_push functions
- experiment_log.md: Log ALL experiments here
- strategy.md: Current best strategy and what to try next

## Working Style
- Install packages with: pip3 install <package> (use sudo with password 87654321 if needed)
- Cache features to features_cache/ directory (create it)
- Save models to models/ directory
- Save visualizations to visualizations/
- Git push after every significant result (the cron also auto-pushes every 5 min)
- Target accuracy: >= 70%, aim for top 10%
- Log EVERY experiment with results
- DO NOT STOP - work continuously until the task is complete

## Allowed Libraries
scikit-learn, opencv-python, xgboost, lightgbm, scikit-image, scipy, numpy, pandas, matplotlib, seaborn, joblib, Pillow

## Git Commit Messages
Write normal human commit messages describing what changed. Examples:
- "Add HOG feature extraction pipeline"
- "Implement SVM classifier with RBF kernel"
- "Improve accuracy to 65% with ensemble method"
