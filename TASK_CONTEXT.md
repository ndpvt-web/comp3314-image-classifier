# COMP3314 Assignment 3 - Image Classification Challenge (HKU Spring 2026)

## Overview
Build an image classifier for a 10-class image dataset. Submit predictions on Kaggle, a Jupyter notebook on Moodle, and a PDF report on Moodle.

## Dataset Characteristics (Empirically Verified)
- **Image size**: 32x32 pixels, RGB (3 channels)
- **Training set**: 50,000 images (50,001 lines including header)
- **Test set**: 10,000 images (10,001 lines including header)
- **Public test**: 2,000 samples (visible on leaderboard)
- **Private test**: 8,000 samples (determines final score)
- **Classes**: 10 categories (labels 0-9)
- **Class distribution (training)**:
  - Label 0: 5,027
  - Label 1: 5,008
  - Label 2: 5,026
  - Label 3: 4,946
  - Label 4: 5,061
  - Label 5: 5,005
  - Label 6: 4,962
  - Label 7: 5,006
  - Label 8: 5,012
  - Label 9: 4,947
- **Balance**: Nearly perfectly balanced (~5,000 per class)
- **Nature**: Almost certainly CIFAR-10 or CIFAR-10-like dataset

## File Structure
```
data/
  train.csv         -> im_name, label
  test.csv          -> im_name, label (all labels = 0, to be predicted)
  train_ims/        -> 50,000 training images (.jpg)
  test_ims/         -> 10,000 test images (.jpg)
```

## Submission Requirements

### 1. Kaggle Prediction CSV (50 points)
- Format: im_name, label (same order as test.csv)
- Scoring by accuracy:
  - >= 0.70: 50 pts (full marks)
  - >= 0.60: 45 pts
  - >= 0.50: 40 pts
  - >= 0.40: 35 pts
  - >= 0.30: 30 pts
  - >= 0.20: 25 pts
  - Top 0-10% ranking: +5 bonus points

### 2. Jupyter Notebook (20 points, Moodle)
- Entire pipeline of final solution
- Must be executable
- Must generate .csv file in same directory
- Pre-executed logs printed
- Do NOT include dataset

### 3. PDF Report (30 points, Moodle)
- Dataset analysis (10 pts): category stats, 1 example per category
- Classifier exploration (10 pts): min 2 classifiers, comparison
- Final solution description (10 pts): core pipeline, key components

## Evaluation
- Public leaderboard: accuracy on 2,000 public test samples
- Final ranking: accuracy on 8,000 private test samples
- Goal: Maximize accuracy, aim for top 10% for bonus

## Dataset Location on Mac Pro
`/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge/`

## Timeline
- Started: ~21 days ago
- Deadline: ~4 days remaining
