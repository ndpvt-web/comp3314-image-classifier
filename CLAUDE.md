# COMP3314 Image Classification - Project Instructions


## ENVIRONMENT SETUP - READ FIRST
A Python virtual environment is already set up with ALL required packages installed.
**ALWAYS use the venv Python for ALL python/pip commands:**
```bash
source /Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge/venv/bin/activate
```
Run this BEFORE any Python command. Or use the full path:
```bash
/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge/venv/bin/python3 script.py
```
**DO NOT use system pip or system python3. ALWAYS activate venv first or use venv/bin/python3.**
Packages already installed: scikit-learn, opencv-python, xgboost, lightgbm, scikit-image, scipy, joblib, matplotlib, seaborn, pandas, pillow, numpy

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

## Git Strategy - IMPORTANT

### Frequent Pushes
- Push to GitHub after EVERY significant step (not just at the end)
- After data analysis: push visualizations and stats
- After feature extraction: push the pipeline code
- After each classifier training: push results
- After each autoresearch experiment: push logs and updated results
- The cron job auto-pushes every 5 min, but YOU should also push manually after milestones

### Branching for AutoResearch Experiments
Use git branches to test different strategies cleanly:
```
main                  <- best known working solution lives here
strategy/hog-svm      <- experiment: HOG + SVM variations
strategy/color-boost   <- experiment: color features + gradient boosting
strategy/ensemble-v1   <- experiment: first ensemble attempt
strategy/ensemble-v2   <- experiment: improved ensemble
strategy/feature-select <- experiment: feature selection methods
```

Workflow for each experiment:
1. Create branch: `git checkout -b strategy/<name>`
2. Implement the strategy
3. Train, evaluate, log results
4. Push branch: `git push -u origin strategy/<name>`
5. If accuracy IMPROVES over main:
   - Merge to main: `git checkout main && git merge strategy/<name>`
   - Push main: `git push`
   - Update best_result.json
6. If accuracy does NOT improve:
   - Keep the branch for reference (shows what was tried)
   - Switch back: `git checkout main`
7. Start next experiment on new branch

This way teammates can see ALL experiments on GitHub branches.

### Commit Message Style
Write normal human commit messages. Examples:
- "Add HOG feature extraction pipeline"
- "Implement SVM classifier with RBF kernel, val acc 55.2%"
- "Improve ensemble to 68.3% with weighted voting"
- "Try Gabor filters - marginal improvement to 69.1%"
- Include accuracy numbers in commits when relevant!

## Working Style
- Install packages with: pip3 install <package> (use sudo with password 87654321 if needed)
- Cache features to features_cache/ directory (create it)
- Save models to models/ directory
- Save visualizations to visualizations/
- Target accuracy: >= 70%, aim for top 10%
- Log EVERY experiment with results to experiment_log.md
- DO NOT STOP - work continuously until the task is complete
- DO NOT ASK QUESTIONS - make decisions and keep going

## Allowed Libraries
scikit-learn, opencv-python, xgboost, lightgbm, scikit-image, scipy, numpy, pandas, matplotlib, seaborn, joblib, Pillow
