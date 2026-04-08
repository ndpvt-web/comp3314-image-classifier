"""Phase 2: Comprehensive Feature Extraction for ALL 60k images.
Extracts HOG, Color, LBP, Gabor, Spatial features. Caches to .npy files.
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis
import time
from joblib import Parallel, delayed

BASE = '/Users/nivesh/Downloads/hku-comp3314-2026-spring-challenge'
CACHE = os.path.join(BASE, 'features_cache')
os.makedirs(CACHE, exist_ok=True)

# Load CSV files
train_df = pd.read_csv(os.path.join(BASE, 'train.csv'))
test_df = pd.read_csv(os.path.join(BASE, 'test.csv'))
train_names = train_df['im_name'].values
test_names = test_df['im_name'].values
train_labels = train_df['label'].values
all_names = np.concatenate([train_names, test_names])
all_dirs = np.array(['train_ims'] * len(train_names) + ['test_ims'] * len(test_names))

print(f"Total images: {len(all_names)} (train: {len(train_names)}, test: {len(test_names)})")

def load_image(name, directory):
    path = os.path.join(BASE, directory, name)
    img = cv2.imread(path)
    return img  # BGR format

def extract_hog_features(img_gray, ppc, cpb, orient=9):
    """Extract HOG features."""
    feat = hog(img_gray, orientations=orient, pixels_per_cell=ppc,
               cells_per_block=cpb, block_norm='L2-Hys', feature_vector=True)
    return feat

def extract_color_histograms(img_bgr, bins=32):
    """RGB + HSV + Lab histograms."""
    features = []
    # RGB histograms
    for c in range(3):
        hist = cv2.calcHist([img_bgr], [c], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.append(hist)
    # HSV histograms
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for c in range(3):
        hist = cv2.calcHist([hsv], [c], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.append(hist)
    # Lab histograms
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    for c in range(3):
        hist = cv2.calcHist([lab], [c], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.append(hist)
    return np.concatenate(features)

def extract_color_moments(img_bgr):
    """Mean, std, skewness, kurtosis per channel for RGB + HSV."""
    features = []
    # RGB
    for c in range(3):
        ch = img_bgr[:, :, c].flatten().astype(np.float64)
        features.extend([ch.mean(), ch.std(), skew(ch), kurtosis(ch)])
    # HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    for c in range(3):
        ch = hsv[:, :, c].flatten()
        features.extend([ch.mean(), ch.std(), skew(ch), kurtosis(ch)])
    return np.array(features)

def extract_lbp_features(img_gray, radius, n_points):
    """LBP histogram features."""
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_gabor_features(img_gray):
    """Gabor filter responses - mean and variance for each filter."""
    features = []
    img_f = img_gray.astype(np.float64)
    for theta_idx in range(8):
        theta = theta_idx * np.pi / 8
        for freq in [0.05, 0.1, 0.15, 0.25, 0.4]:
            kernel = cv2.getGaborKernel((9, 9), sigma=3.0, theta=theta,
                                         lambd=1.0/freq, gamma=0.5, psi=0)
            filtered = cv2.filter2D(img_f, cv2.CV_64F, kernel)
            features.extend([filtered.mean(), filtered.var()])
    return np.array(features)

def extract_spatial_color(img_bgr, grid_size=4):
    """Spatial color grid - mean RGB per cell."""
    h, w = img_bgr.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img_bgr[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            features.extend([cell[:,:,c].mean() / 255.0 for c in range(3)])
    return np.array(features)

def extract_edge_features(img_gray):
    """Edge-based features."""
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = edges.mean() / 255.0
    # Quadrant densities
    h, w = edges.shape
    q1 = edges[:h//2, :w//2].mean() / 255.0
    q2 = edges[:h//2, w//2:].mean() / 255.0
    q3 = edges[h//2:, :w//2].mean() / 255.0
    q4 = edges[h//2:, w//2:].mean() / 255.0
    # Sobel orientations
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)
    # Orientation histogram (8 bins)
    hist, _ = np.histogram(angle[mag > mag.mean()], bins=8, range=(-np.pi, np.pi), density=True)
    return np.concatenate([[edge_density, q1, q2, q3, q4], hist])

def extract_hu_moments(img_gray):
    """Hu moments (7 rotation-invariant features)."""
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform for better scale
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu

def extract_all_features(idx):
    """Extract ALL features for one image."""
    name = all_names[idx]
    directory = all_dirs[idx]
    img_bgr = load_image(name, directory)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Resize to 64x64 for multi-scale HOG
    img_64 = cv2.resize(img_bgr, (64, 64))
    gray_64 = cv2.cvtColor(img_64, cv2.COLOR_BGR2GRAY)

    features = {}

    # HOG features (multiple configs)
    # Config 1: 8x8 cells, 2x2 blocks, 9 orientations on 32x32 gray
    features['hog1'] = extract_hog_features(img_gray, (8,8), (2,2), 9)
    # Config 2: 4x4 cells, 2x2 blocks, 9 orientations on 32x32 gray (higher res)
    features['hog2'] = extract_hog_features(img_gray, (4,4), (2,2), 9)
    # Config 3: 8x8 cells on 64x64 (multi-scale)
    features['hog3'] = extract_hog_features(gray_64, (8,8), (2,2), 9)
    # Config 4: Per-channel HOG on 32x32
    hog_channels = []
    for c in range(3):
        hog_channels.append(extract_hog_features(img_bgr[:,:,c], (8,8), (2,2), 9))
    features['hog_color'] = np.concatenate(hog_channels)

    # Color histograms
    features['color_hist'] = extract_color_histograms(img_bgr, bins=32)

    # Color moments
    features['color_moments'] = extract_color_moments(img_bgr)

    # LBP features (multiple radii)
    features['lbp_r1'] = extract_lbp_features(img_gray, radius=1, n_points=8)
    features['lbp_r2'] = extract_lbp_features(img_gray, radius=2, n_points=16)
    features['lbp_r3'] = extract_lbp_features(img_gray, radius=3, n_points=24)

    # Gabor features
    features['gabor'] = extract_gabor_features(img_gray)

    # Spatial color grid
    features['spatial_4x4'] = extract_spatial_color(img_bgr, grid_size=4)
    features['spatial_2x2'] = extract_spatial_color(img_bgr, grid_size=2)

    # Edge features
    features['edge'] = extract_edge_features(img_gray)

    # Hu moments
    features['hu'] = extract_hu_moments(img_gray)

    return features

# Check if features already cached
cache_file = os.path.join(CACHE, 'all_features_combined.npy')
if os.path.exists(cache_file):
    print("Features already cached! Loading...")
    sys.exit(0)

# Extract features in parallel batches
print("Starting feature extraction...")
t0 = time.time()

# Process in chunks to show progress
CHUNK_SIZE = 2000
n_total = len(all_names)
all_results = []

for chunk_start in range(0, n_total, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
    chunk_indices = list(range(chunk_start, chunk_end))

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(extract_all_features)(i) for i in chunk_indices
    )
    all_results.extend(results)

    elapsed = time.time() - t0
    pct = chunk_end / n_total * 100
    eta = elapsed / chunk_end * (n_total - chunk_end) if chunk_end > 0 else 0
    print(f"  Progress: {chunk_end}/{n_total} ({pct:.1f}%) | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

print(f"\nFeature extraction complete in {time.time()-t0:.0f}s")

# Organize features by type
feature_names = list(all_results[0].keys())
print(f"\nFeature types: {feature_names}")
for fname in feature_names:
    print(f"  {fname}: {len(all_results[0][fname])} dims")

# Save individual feature sets AND combined
for fname in feature_names:
    feat_array = np.array([r[fname] for r in all_results], dtype=np.float32)
    np.save(os.path.join(CACHE, f'{fname}.npy'), feat_array)
    print(f"  Saved {fname}.npy: shape {feat_array.shape}")

# Save combined feature vector
combined = np.hstack([np.array([r[fname] for r in all_results], dtype=np.float32)
                       for fname in feature_names])
np.save(os.path.join(CACHE, 'all_features_combined.npy'), combined)
print(f"\nSaved all_features_combined.npy: shape {combined.shape}")

# Save labels
np.save(os.path.join(CACHE, 'train_labels.npy'), train_labels)
np.save(os.path.join(CACHE, 'train_count.npy'), np.array([len(train_names)]))
np.save(os.path.join(CACHE, 'test_count.npy'), np.array([len(test_names)]))

# Save feature metadata
total_dims = sum(len(all_results[0][f]) for f in feature_names)
with open(os.path.join(CACHE, 'feature_metadata.txt'), 'w') as f:
    f.write(f"Total features: {total_dims}\n")
    f.write(f"Total images: {n_total} (train: {len(train_names)}, test: {len(test_names)})\n\n")
    offset = 0
    for fname in feature_names:
        dims = len(all_results[0][fname])
        f.write(f"{fname}: dims={dims}, cols=[{offset}:{offset+dims}]\n")
        offset += dims

print(f"\n=== Phase 2 Complete ===")
print(f"Total feature dimensions: {total_dims}")
print(f"All features cached to {CACHE}/")
