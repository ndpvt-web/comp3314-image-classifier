# Experiment Log

Automated experiment tracking for COMP3314 Image Classification.
Current best: Split PCA HOG-150+Other-175, PT, SVM C=5 = 74.46%

---

## Experiment 1: SVM RBF Grid Search [2026-04-09]
- **Description**: Wide SVM RBF search across PCA dims (150-350) and C values (8-30), gamma=scale
- **Best accuracy**: 71.42% (PCA=200, C=8, gamma=scale)
- **Key findings**:
  - PCA-200 is optimal: 71.41-71.42% across all C=8-30 (very stable)
  - PCA-150: ~71.08-71.20% (too few dims)
  - PCA-225: ~71.20-71.22% (too many dims add noise)
  - C is not a sensitive parameter: 8-30 all give ~71.41%
  - gamma='scale' is correct for these features
- **12 of 36 configs completed** (search terminated early - pattern was clear)
- **Conclusion**: PCA-200 sweet spot confirmed. Need better features or preprocessing to push higher.

---

## Experiment 3: Power Transform (Yeo-Johnson) + SVM [2026-04-09]
- **Description**: Yeo-Johnson power transform before PCA+SVM to Gaussianize features
- **Best accuracy**: 72.03% (PCA=250, C=8, gamma=scale) *** NEW OVERALL BEST ***
- **Baseline**: 71.42% (no power transform, PCA=200, C=8)
- **Improvement**: +0.61%
- **Key findings**:
  - Power transform allows higher PCA dims to work (PCA-250 best vs PCA-200 without)
  - PCA-250: 72.03% (C=8-20 all identical)
  - PCA-300: 71.75-71.80% (still drops at high dims)
  - PCA-225: 71.76-71.80%
  - PCA-200: 71.56-71.69%
  - Power transform helps most at mid-range PCA dims
- **Conclusion**: Yeo-Johnson is a significant improvement. Next: try higher PCA dims (350-500) with power transform, or combine with other preprocessing.

---

## Experiment 4: Fine-tune Power Transform SVM [2026-04-09]
- **Description**: Fine-grained PCA search (230-280) + higher dims (350-500) + manual gamma
- **Best accuracy**: 72.12% (PCA=245, C=8, gamma=scale) *** NEW OVERALL BEST ***
- **Previous best**: 72.03% (PCA=250, C=8, exp3)
- **Key findings** (14/57 configs completed):
  - PCA-230: 71.73-71.77%
  - PCA-240: 72.06-72.10%
  - PCA-245: 72.11-72.12% (PEAK)
  - PCA-250: 72.03-72.04%
  - Sharp peak at PCA 240-250 with power transform
- **Terminated early** - clear pattern, moving to higher-impact experiments

---

## Experiment 5: Split PCA (HOG vs non-HOG) + SVM [2026-04-09]
- **Description**: Separate PCA on HOG (4824d) and non-HOG (526d) features, concatenate, then SVM
- **Best accuracy**: 74.35% (HOG-150+Other-150=300 total, C=5) *** NEW OVERALL BEST ***
- **Previous best**: 72.12% (unified PCA-245, C=8, exp4)
- **Improvement**: +2.23% (MASSIVE)
- **Key findings** (20/75 configs completed):
  - Non-HOG features get crushed in unified PCA - split PCA rescues them
  - More non-HOG PCA dims = monotonically better: 50(72.93%)→75(73.49%)→100(73.75%)→125(73.99%)→150(74.35%)
  - HOG-150 > HOG-175 at same total dims (HOG has diminishing returns)
  - C=5 consistently best (lower C better with split features)
- **Conclusion**: Split PCA is the breakthrough technique. Test more non-HOG dims (175, 200) and fine-tune.

---

## Experiment 6: Split PCA v2 - Fine-tune [2026-04-09]
- **Description**: HOG 100-175, Other 125-200, C 3-10 fine-grained search
- **Best accuracy (this exp)**: 74.12% (HOG-100+Other-175, C=3)
- **Best overall still**: 74.35% (HOG-150+Other-150, C=5, exp5)
- **Key findings** (13/64 configs, HOG-100 block only):
  - HOG-100 peaks at Other-175 C=3: 74.12% (too few HOG dims)
  - HOG-150 from exp5 remains best (74.35%)
  - C=3 is optimal for HOG-100 (lower C for fewer features)
- **Terminated** - exp5 result confirmed as best

---

## Submission Generated [2026-04-09]
- **Config**: Split PCA HOG-150+Other-150, SVM C=5, gamma=scale
- **Val accuracy**: 74.35%
- **Trained on**: All 50k training samples
- **Test predictions**: 10,000 labels, balanced across 10 classes

---

## Experiment 7: QuantileTransformer + Extended Other PCA [2026-04-09]
- **Description**: Compare PowerTransform vs QuantileTransformer, test Other PCA 175-200
- **Key findings**:
  - PowerTransform Other-175 C=5: 74.35% (ties best)
  - PowerTransform Other-200 C=5: 74.34% (plateau)
  - QuantileNormal: 67.09% (MUCH WORSE - destroys distributional structure)
  - Non-HOG PCA saturates around 150-175 dims
- **Conclusion**: PowerTransform confirmed as best. No further gains from more Other PCA dims.

---

## Experiment 8: Ensemble of Diverse SVMs [2026-04-09]
- **Description**: 9 SVMs with different Split PCA configs, majority voting
- **Best individual**: 74.46% (HOG-150+Other-175, C=5) *** NEW OVERALL BEST ***
- **Previous best**: 74.35% (HOG-150+Other-150, C=5, exp5)
- **Improvement**: +0.11%
- **Individual results**:
  - HOG-150+Other-175 C=5: 74.46% (BEST)
  - HOG-150+Other-150 C=5: 74.34%
  - HOG-150+Other-200 C=5: 74.32%
  - HOG-175+Other-150 C=5: 74.19%
  - HOG-150+Other-150 C=3: 74.19%
  - HOG-150+Other-150 C=10: 74.14%
  - HOG-125+Other-150 C=5: 74.05%
  - HOG-150+Other-125 C=5: 74.01%
  - HOG-200+Other-150 C=5: 73.91%
- **Ensemble results**: Top-3 voting: 74.41%, Top-5: 74.40%, ALL-9: 74.30%
- **Conclusion**: Ensemble voting HURT accuracy. Models too similar for diversity benefit. Key finding: Other-175 > Other-150 (+0.12%). New best config: HOG-150+Other-175 C=5.

---

