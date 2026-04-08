"""Phase 1: Dataset Analysis and Visualization"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
VIS_DIR = os.path.join(BASE, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# CIFAR-10 class names
CLASS_NAMES = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
               5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# Load CSV
train_df = pd.read_csv(os.path.join(BASE, 'train.csv'))
test_df = pd.read_csv(os.path.join(BASE, 'test.csv'))
print(f"Train: {len(train_df)} images, Test: {len(test_df)} images")
print(f"Train columns: {list(train_df.columns)}")
print(f"Label distribution:\n{train_df['label'].value_counts().sort_index()}")

# 1. Class distribution plot
fig, ax = plt.subplots(figsize=(10, 5))
counts = train_df['label'].value_counts().sort_index()
bars = ax.bar(range(10), counts.values, color=plt.cm.tab10(np.arange(10)))
ax.set_xticks(range(10))
ax.set_xticklabels([f"{i}: {CLASS_NAMES[i]}" for i in range(10)], rotation=45, ha='right')
ax.set_ylabel('Count')
ax.set_title('Training Set Class Distribution')
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            str(count), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'class_distribution.png'), dpi=150)
plt.close()
print("Saved class_distribution.png")

# 2. Sample images per class (5 per class)
fig, axes = plt.subplots(10, 5, figsize=(12, 24))
for label in range(10):
    class_imgs = train_df[train_df['label'] == label]['im_name'].values[:5]
    for j, im_name in enumerate(class_imgs):
        img = cv2.imread(os.path.join(BASE, 'train_ims', im_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[label, j].imshow(img)
        axes[label, j].axis('off')
        if j == 0:
            axes[label, j].set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=10, fontweight='bold')
plt.suptitle('Sample Images Per Class (5 per class)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'sample_images_per_class.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved sample_images_per_class.png")

# 3. Compute per-class mean images and overall statistics
print("\nComputing per-class mean images and pixel statistics...")
class_images = {i: [] for i in range(10)}
all_pixels = []

# Load a subset for statistics (all 50k is fine for 32x32)
for idx, row in train_df.iterrows():
    img = cv2.imread(os.path.join(BASE, 'train_ims', row['im_name']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_images[row['label']].append(img)
    if idx < 5000:  # Use 5k for pixel stats
        all_pixels.append(img.reshape(-1, 3))
    if idx % 10000 == 0:
        print(f"  Loaded {idx}/{len(train_df)} images...")

print(f"  Loaded all {len(train_df)} images")

# Pixel statistics
all_pixels = np.vstack(all_pixels)
print(f"\nPixel Statistics (from 5k images):")
print(f"  R - Mean: {all_pixels[:,0].mean():.1f}, Std: {all_pixels[:,0].std():.1f}")
print(f"  G - Mean: {all_pixels[:,1].mean():.1f}, Std: {all_pixels[:,1].std():.1f}")
print(f"  B - Mean: {all_pixels[:,2].mean():.1f}, Std: {all_pixels[:,2].std():.1f}")

# Mean images per class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for label in range(10):
    ax = axes[label // 5, label % 5]
    mean_img = np.mean(class_images[label], axis=0).astype(np.uint8)
    ax.imshow(mean_img)
    ax.set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=10)
    ax.axis('off')
plt.suptitle('Mean Image Per Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'mean_images_per_class.png'), dpi=150)
plt.close()
print("Saved mean_images_per_class.png")

# 4. Image size verification
sample_img = cv2.imread(os.path.join(BASE, 'train_ims', train_df['im_name'].iloc[0]))
print(f"\nImage shape: {sample_img.shape} (height x width x channels)")
print(f"Dtype: {sample_img.dtype}")

# 5. Per-class color distribution
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for label in range(10):
    ax = axes[label // 5, label % 5]
    imgs = np.array(class_images[label][:500])  # Use 500 per class
    for c, color_name, color in zip(range(3), ['R','G','B'], ['red','green','blue']):
        channel_vals = imgs[:, :, :, c].flatten()
        ax.hist(channel_vals, bins=50, alpha=0.5, color=color, label=color_name, density=True)
    ax.set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=9)
    ax.set_xlim(0, 255)
    if label == 0:
        ax.legend(fontsize=7)
plt.suptitle('Per-Class RGB Color Distribution', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'per_class_color_distribution.png'), dpi=150)
plt.close()
print("Saved per_class_color_distribution.png")

print("\n=== Phase 1 Complete ===")
print(f"All visualizations saved to {VIS_DIR}/")
