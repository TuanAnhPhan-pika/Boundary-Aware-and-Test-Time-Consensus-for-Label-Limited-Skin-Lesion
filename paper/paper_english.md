# LENS: A CPU Pilot of Boundary-Aware and Test-Time Consensus Methods for Label-Limited Skin-Lesion Segmentation

## Abstract

This exploratory study evaluates lightweight skin-lesion segmentation under restricted annotation budgets. We compare a compact U-Net trained with binary cross-entropy and soft Dice loss against uniform distance-boundary supervision, uncertainty-gated boundary supervision, four-view mean-probability test-time augmentation (TTA), four-view robust-logit consensus, and a wider single-pass U-Net. Experiments use a locally materialized subset of ISIC 2016 containing 140 training, 40 validation, and 60 test image-mask pairs; the executed CPU pilot further limits evaluation to a fixed 80-image development pool, 20 validation images, and 30 test images at 64 x 64 resolution. Each condition is run for three paired seeds and two epochs at 25% and 50% label budgets. At 50% labels, the compact baseline achieved mean Dice 0.4001, while robust-logit TTA achieved 0.4043 and the wider model achieved 0.3586. At 25% labels, the corresponding values were 0.3883, 0.3922, and 0.3004. Boundary-supervised variants differed from the baseline by less than 0.0003 Dice. No paired comparison provided statistically significant evidence of improvement. The findings therefore do not support the hypothesis that the tested boundary loss, TTA rules, or added width improve segmentation in this short CPU pilot. They instead identify implementation activity, training duration, sample size, and independent replication as priorities for a definitive study.

## 1. Introduction

Skin-lesion segmentation delineates lesion tissue in dermoscopic images. It is an important image-analysis task, but dense masks are costly to create and lesions can exhibit weak contrast, hair occlusion, illumination variation, artifacts, and irregular borders. These characteristics motivate methods that can learn from fewer masks while remaining computationally modest.

U-Net-style encoder-decoder models are a common baseline for medical-image segmentation [siddique2021unet][wang2022medical]. Boundary-aware losses attempt to concentrate supervision near lesion contours, where region losses may provide weak or spatially diffuse gradients [jurdi2021highlevel]. Test-time augmentation instead transforms the input, inverse-aligns multiple predictions, and combines them. This can reduce sensitivity to a particular view, although its benefit depends on the transformations and aggregation operator [ashraf2022melanoma]. Uncertainty-related quantities are also increasingly used in medical-image analysis, but transformation disagreement is not automatically a calibrated estimate of epistemic uncertainty [abdar2021review][mehrtash2020confidence].

This work asks a deliberately narrow question: under a fixed, resource-constrained pilot protocol, do boundary-weighted training or four-view consensus improve segmentation overlap relative to a compact U-Net, and how do they compare with a wider single-pass model? The study reports computational evidence only. It makes no claim about diagnosis, treatment, safety, or clinical deployment.

## 2. Methods

### 2.1 Data and executed protocol

The experiment uses image-mask pairs derived from ISIC 2016. The preparation script materialized 140 training, 40 validation, and 60 test pairs. For the executed pilot, the code selected a fixed development pool of 80 images, 20 validation images, and 30 test images. Images and masks were resized to 64 x 64 pixels. Restricted-label conditions used 25% and 50% of the development pool. The subset is small and is not an official challenge-scale evaluation.

Every method was evaluated with seeds 0, 1, and 2. Models were trained on CPU for at most two epochs with AdamW, batch size 8, cosine learning-rate scheduling, and early stopping. The compact U-Net used channel widths 4, 8, 16, and 32. Wider candidates were selected by the experiment's timing routine. A threshold of 0.5 converted probabilities to binary masks.

### 2.2 Compared conditions

The seven conditions were:

1. **C-UNet:** compact U-Net with binary cross-entropy plus soft Dice loss and single-view inference.
2. **MP-TTA:** C-UNet with four-view mean-probability TTA.
3. **UDB:** compact U-Net with uniform distance-boundary supervision.
4. **UB-TTA:** UDB with mean-probability TTA.
5. **UGDB:** compact U-Net with uncertainty-gated distance-boundary supervision.
6. **RL-TTA:** C-UNet with robust-logit four-view consensus.
7. **W-UNet:** a wider U-Net with single-view inference.

The primary outcome was mean per-image Dice. The experiment also recorded IoU, foreground-balanced Brier score, lesion-centered Brier score, negative log-likelihood, expected calibration error, boundary-band Brier score, boundary F-score, calibration slope and intercept, HD95, and normalized surface Dice. Because this is a small pilot with three seeds, the analysis emphasizes effect direction and variability rather than confirmatory significance.

## 3. Results

### 3.1 Dice results

| Method | 25% labels, Dice mean +/- population SD | 50% labels, Dice mean +/- population SD |
|---|---:|---:|
| C-UNet | 0.3883 +/- 0.0543 | 0.4001 +/- 0.0490 |
| MP-TTA | 0.3854 +/- 0.0737 | 0.3974 +/- 0.0658 |
| UDB | 0.3883 +/- 0.0543 | 0.4002 +/- 0.0490 |
| UB-TTA | 0.3854 +/- 0.0737 | 0.3975 +/- 0.0657 |
| UGDB | 0.3884 +/- 0.0542 | 0.4003 +/- 0.0489 |
| RL-TTA | **0.3922 +/- 0.0639** | **0.4043 +/- 0.0557** |
| W-UNet | 0.3004 +/- 0.0756 | 0.3586 +/- 0.0507 |

Robust-logit consensus produced the highest mean Dice in both label regimes, but its absolute improvement over C-UNet was only 0.0039 at 25% and 0.0042 at 50%. The paired tests were not statistically significant, and the estimated confidence intervals included zero. Mean-probability TTA reduced mean Dice by approximately 0.0029 and 0.0027 at the two budgets.

The boundary-trained models were almost identical to their corresponding references. Relative to C-UNet, UDB changed mean Dice by roughly +0.00002 at 25% and +0.00011 at 50%; UGDB changed it by roughly +0.00004 and +0.00016. These differences are negligible compared with variation across seeds. This result does not establish that boundary losses are generally ineffective. It shows that the boundary intervention in this executed configuration did not materially alter the measured endpoint.

Contrary to the intended capacity hypothesis, W-UNet underperformed C-UNet by 0.0879 Dice at 25% and 0.0415 at 50%. Under two-epoch training, additional width may have increased optimization difficulty or variance without enough updates to realize a capacity benefit.

### 3.2 Interpretation of probability and boundary metrics

The full machine-readable artifact contains the secondary metrics for every seed and condition. These values are useful for auditing, but the small sample and short training schedule preclude strong calibration or contour claims. Brier score mixes calibration and discrimination; ECE is binning-dependent; boundary F-score and HD95 can be unstable for poor or empty predictions. Accordingly, the study does not infer clinical reliability from any single probability or boundary metric [muller2022guideline][maierhein2024metrics].

## 4. Discussion

The main empirical observation is negative: none of the evaluated modifications produced convincing evidence of improvement over the compact single-view baseline. Robust-logit TTA showed a small favorable direction, whereas mean-probability TTA showed a small unfavorable direction. This suggests that the aggregation domain may matter, but three seeds are insufficient to distinguish a stable effect from noise.

The near-identity of C-UNet, UDB, and UGDB is especially important. A nominally more sophisticated objective is not evidence that the objective exerted a meaningful training signal. A definitive follow-up should log the boundary-loss magnitude, its gradient norm relative to the region loss, gate variance, prediction divergence, and parameter divergence. It should also use longer training and direct surface metrics.

The wider model's underperformance should not be generalized to model capacity in medical segmentation. All models received only two epochs on a small, downsampled subset. Wider networks may require different learning rates, regularization, schedules, or more updates. The result is therefore specific to the executed pilot and is best treated as a warning against assuming that extra width automatically helps under a fixed short schedule.

## 5. Limitations

- The study uses a small ISIC 2016 subset rather than a full official benchmark evaluation.
- Images were reduced to 64 x 64, which removes fine boundary detail.
- Training lasted at most two epochs, making underfitting likely.
- Only three seeds and one fixed subset construction were evaluated.
- The 25% and 50% conditions represent fractions of an 80-image development pool, not fractions of the complete ISIC training archive.
- No external dataset, acquisition-shift analysis, patient-level metadata analysis, or clinician study was performed.
- Multiple exploratory outcomes were recorded without a powered, multiplicity-controlled confirmatory analysis.
- CPU timing in this run is not a standardized latency or deployment benchmark.

## 6. Conclusion

In this constrained CPU pilot, a compact U-Net achieved mean Dice 0.3883 and 0.4001 at the 25% and 50% label budgets. Robust-logit TTA increased those means by approximately 0.004, but the evidence did not establish a statistically reliable benefit. Uniform and uncertainty-gated boundary objectives produced negligible changes, while a wider model performed worse under the two-epoch schedule. A larger follow-up should use the full-resolution benchmark, longer training, repeated subset draws, more seeds, logged intervention diagnostics, direct boundary metrics, and external validation. The present findings characterize only segmentation behavior on the selected data and do not establish diagnostic accuracy or clinical utility.

## Reproducibility statement

The code, per-seed results, preparation script, bibliography, charts, and citation-verification report are included with the deliverables. All 31 bibliography entries referenced by the generated research package were verified by DOI lookup in the pipeline's final citation stage. The authoritative numerical source is `code/results.json`; this corrected report supersedes unsupported numerical claims in the automatically generated draft.

