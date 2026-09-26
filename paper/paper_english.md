# Boundary Loss and Test Time Consensus in Label Limited ISIC 2016 Skin Lesion Segmentation

## Abstract

This study tests whether signed-distance boundary supervision and four-view test-time aggregation improve a compact U-Net when pixel-level annotations are limited. We used ISIC 2016 image-mask pairs distributed by the MedOtter mirror. After removing four exact cross-split image duplicates, the experiment contained 716 development, 180 validation, and 379 test images. Nested, lesion-area-stratified training subsets used 179 (25%), 358 (50%), or 716 (100%) images. Seven conditions were evaluated at 256 × 256 with five paired random seeds: compact U-Net; mean-probability test-time augmentation (TTA); uniform boundary loss; boundary loss plus mean TTA; transformation-sensitivity-gated boundary loss; robust-logit consensus TTA; and a wider single-view U-Net. All 105 prespecified evaluations completed. At 25%, 50%, and 100% labels, the compact baseline obtained Dice 0.8444 ± 0.0095, 0.8688 ± 0.0051, and 0.8907 ± 0.0041 (mean ± sample SD). Robust-logit consensus improved these means by 0.0057, 0.0039, and 0.0032. The wider U-Net was best overall at 0.8555, 0.8785, and 0.8971. Both boundary-loss variants were slightly below baseline at every budget. With five paired seeds, two-sided Wilcoxon tests could not attain p < 0.05; positive paired bootstrap intervals are therefore descriptive rather than confirmatory. The results support modest gains from multi-view inference and additional capacity, but not from the tested boundary objectives. They quantify reference-mask agreement and do not establish diagnostic or clinical performance.

**Keywords:** medical image segmentation; skin lesion; U-Net; limited annotations; boundary loss; test-time augmentation

## 1. Introduction

Skin-lesion segmentation identifies lesion pixels in dermoscopic images. Reliable contours are challenging because of weak contrast, hair, illumination changes, artifacts, and irregular borders. Dense masks also require expert time, motivating methods that work with fewer annotated images. U-Net-style encoder-decoder models remain common baselines [1,2].

Two inexpensive strategies are attractive. A boundary term can add spatial information that region losses may underemphasize [3]. Test-time augmentation predicts several geometry-preserving views, aligns the outputs, and aggregates them [4]. View disagreement may indicate transformation sensitivity, although it is not automatically a validated estimate of epistemic uncertainty [5,6].

We ask whether boundary supervision or four-view aggregation improves a compact U-Net and whether either is competitive with a wider single-view network. The experiment uses paired seeds, nested label subsets, held-out validation, and overlap, boundary, distance, and calibration metrics.

## 2. Related work

Reviews of medical-image segmentation consistently identify U-Net as the dominant starting point for supervised encoder-decoder systems [1,2]. Its skip connections combine high-resolution spatial features with deeper semantic features, which is useful for lesions whose position and shape vary substantially. Lightweight variants reduce computation through narrower channels or separable convolutions, but lower capacity can also restrict the model when sufficient labels are available.

Loss design is a second line of work. Region losses such as soft Dice respond directly to mask overlap and handle foreground-background imbalance better than raw accuracy. Boundary-aware objectives instead use contour maps, distances, or shape priors to penalize spatial errors that region overlap can hide [3]. Their behavior depends on sign convention, scale, weight, and interaction with the region loss. A boundary term can therefore be correctly implemented yet contribute too little or duplicate gradients already supplied by soft Dice.

TTA changes inference rather than training. Previous skin-lesion work has combined transformed predictions and reported gains under particular architectures and post-processing choices [4]. Probability averaging is simple, while logit aggregation can avoid compression near zero and one. Neither guarantees improvement when transformations are not true symmetries or when one view produces a systematic error. The present study adds a local rejection rule and includes a wider single-pass model so that multi-view inference is compared with a simple capacity increase.

Predictive uncertainty and calibration are related but separate concerns. Reviews distinguish data uncertainty, model uncertainty, and heuristic disagreement scores [5]. Medical segmentation studies have also shown that confidence calibration can change across cases and acquisition conditions [6]. We therefore call four-view variance transformation sensitivity and test whether it predicts or improves segmentation without presenting it as a validated uncertainty measure.

## 3. Materials and methods

### 3.1 Data and leakage control

The preparation program materialized `MedOtter/ISIC2016` train and test pairs. From 900 source-training pairs, 180 validation cases were chosen deterministically by SHA-256 ranking of image identifiers; the other 720 initially formed the development pool. The source test split supplied 379 pairs. A byte-level image-hash audit retained test over validation over training and removed four training images duplicated across splits. Final counts were 716 development, 180 validation, and 379 test images. Images are not redistributed in this repository.

For each seed, training cases were stratified by lesion-area fraction. The 25%, 50%, and 100% subsets contained 179, 358, and 716 images and were nested within that seed. Every method within a seed and budget used the same cases and augmentation seed. Images and masks were resized to 256 × 256. Patient-level separation could not be verified because suitable identifiers were unavailable; this is a limitation. Careful reporting is important because public skin-image collections can contain overlap and heterogeneous metadata [7,8].

### 3.2 Models and training

The compact four-level U-Net uses widths 16/32/64/128, depthwise-separable blocks, group normalization, SiLU, max pooling, bilinear upsampling, skip connections, and one foreground logit (62,716 trainable parameters). The wider comparator uses widths 24/48/96/192 (134,572 parameters), selected by a pre-run latency rule.

The base objective is binary cross-entropy plus soft Dice. For image pixels indexed by i, the soft Dice term is one minus the ratio between twice the probability-mask intersection and the sum of predicted and reference foreground mass, with a small numerical constant. The uniform boundary model adds the mean product of foreground probability and signed distance to the reference mask. Signed distance is negative inside the lesion, positive outside, and clipped at 20 pixels. The boundary coefficient is 0.01 and increases linearly during the first ten epochs.

The gated model multiplies each boundary contribution by exp(-v/0.01), where v is the variance across four inverse-aligned predictions from a frozen compact reference model. Large disagreement therefore reduces the local boundary penalty. We call v transformation sensitivity, not calibrated uncertainty, because it measures response to four selected transformations rather than a posterior distribution over models.

Models used AdamW (learning rate 3×10⁻⁴, weight decay 10⁻⁴), cosine decay, gradient clipping at 1.0, batch size 24, bfloat16, and at most 60 epochs. Validation occurred every two epochs; early stopping used five validation events. Augmentation comprised horizontal/vertical flips, 90-degree rotations, and mild brightness/contrast jitter. Checkpoint and temperature choices used validation only.

### 3.3 Inference and conditions

Four-view inference used identity, horizontal flip, vertical flip, and 180-degree rotation. Mean TTA averages inverse-aligned probabilities. Robust consensus computes aligned logits, discards at every pixel the view with greatest local deviation from the median in a 5 × 5 window, averages the remaining three logits, clips to ±15, and applies sigmoid.

Seven conditions were prespecified: compact U-Net (C-UNet); mean-probability TTA (MP-TTA); uniform distance-boundary training (UDB); UDB with mean TTA (UDB+TTA); sensitivity-gated boundary training (UGDB); robust-logit consensus (RL-TTA); and wider single-view U-Net (W-UNet).

### 3.4 Outcomes and statistics

The primary endpoint was mean per-image Dice at threshold 0.5. Secondary endpoints included IoU, boundary F-score, normalized surface Dice (NSD), HD95, Brier scores, negative log-likelihood, expected calibration error, calibration slope, and intercept. Multiple metrics are necessary because no score captures every relevant segmentation error [9,10].

We report means and sample SDs over five paired seeds. Each method was compared with C-UNet using a two-sided exact Wilcoxon signed-rank test and a 20,000-resample paired bootstrap interval for the mean Dice difference. With five nonzero pairs, the smallest possible two-sided exact Wilcoxon p-value is 0.0625; this is an estimation-oriented study, not a powered confirmatory test.

### 3.5 Reproducibility and implementation

The experiment ran with Python 3.11.9, PyTorch 2.13.0 with CUDA 13.0, and an NVIDIA RTX 4050 Laptop GPU. The program saved progress after every completed seed and retained failed runs in unconditional summaries at a prespecified worst-case Dice of zero. The final execution contained no failed or truncated run. The repository records package requirements, the deterministic data-preparation program, exact hyperparameters, per-image Dice arrays, all per-seed endpoints, derived CSV tables, and the script used to generate figures. Data images and local environments are excluded.

## 4. Results

All 105 method–budget–seed evaluations completed successfully in 9.73 hours on an NVIDIA RTX 4050 Laptop GPU. No time guard or numerical-divergence flag was triggered.

| Method | 25% (179), Dice | 50% (358), Dice | 100% (716), Dice |
|---|---:|---:|---:|
| C-UNet | 0.8444 ± 0.0095 | 0.8688 ± 0.0051 | 0.8907 ± 0.0041 |
| MP-TTA | 0.8492 ± 0.0102 | 0.8723 ± 0.0056 | 0.8938 ± 0.0048 |
| UDB | 0.8440 ± 0.0096 | 0.8686 ± 0.0053 | 0.8905 ± 0.0040 |
| UDB+TTA | 0.8488 ± 0.0104 | 0.8721 ± 0.0059 | 0.8936 ± 0.0048 |
| UGDB | 0.8437 ± 0.0097 | 0.8686 ± 0.0052 | 0.8905 ± 0.0042 |
| RL-TTA | 0.8500 ± 0.0100 | 0.8726 ± 0.0057 | 0.8939 ± 0.0047 |
| W-UNet | **0.8555 ± 0.0085** | **0.8785 ± 0.0056** | **0.8971 ± 0.0029** |

![Dice across label budgets](../figures/dice_by_label_budget.png)

RL-TTA exceeded C-UNet by 0.0057, 0.0039, and 0.0032 Dice as the budget increased. MP-TTA improved Dice by 0.0048, 0.0036, and 0.0031. W-UNet had the highest mean at every budget, exceeding baseline by 0.0111, 0.0097, and 0.0065. UDB differed from baseline by −0.0004, −0.0002, and −0.0002; UGDB by −0.0007, −0.0002, and −0.0002. UDB+TTA improved over baseline, but almost all of that gain was reproduced by mean TTA without boundary training.

![Paired effects](../figures/paired_dice_effects.png)

No comparison met p < 0.05. The positive bootstrap intervals for TTA describe the observed paired effect but are not independent confirmatory evidence. Complete per-seed results are in `results/per_seed_metrics.csv`, with summaries in `results/summary_metrics.csv`.

Table 2 reports the paired Dice effects for the main practical comparisons. The wider model had the largest mean effect at every budget, but its 25% interval crossed zero because one seed did not improve. Both TTA rules improved all five paired seeds in each budget. The exact Wilcoxon p-value remained 0.0625 because five pairs do not provide a smaller attainable two-sided p-value when all differences share one sign.

| Comparison with C-UNet | 25% difference (95% bootstrap CI) | 50% difference (95% bootstrap CI) | 100% difference (95% bootstrap CI) |
|---|---:|---:|---:|
| MP-TTA | 0.0048 (0.0043, 0.0054) | 0.0036 (0.0028, 0.0042) | 0.0031 (0.0022, 0.0039) |
| RL-TTA | 0.0057 (0.0050, 0.0063) | 0.0039 (0.0030, 0.0045) | 0.0032 (0.0024, 0.0041) |
| W-UNet | 0.0111 (-0.0004, 0.0194) | 0.0097 (0.0035, 0.0143) | 0.0065 (0.0030, 0.0095) |

![Full-label metrics](../figures/full_label_metrics.png)

At the full label budget, W-UNet had the highest Dice, IoU, boundary F-score, and NSD. MP-TTA produced the lowest foreground-balanced Brier score, whereas RL-TTA and W-UNet produced the lowest ECE values. No model dominated every endpoint.

| Method | IoU | Boundary F-score | NSD | HD95 | Balanced Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| C-UNet | 0.8198 | 0.3769 | 0.3769 | 18.50 | 0.0489 | 0.0163 |
| MP-TTA | 0.8245 | 0.3846 | 0.3846 | 16.93 | **0.0469** | 0.0160 |
| UDB | 0.8196 | 0.3765 | 0.3764 | 18.41 | 0.0490 | 0.0165 |
| UDB+TTA | 0.8242 | 0.3838 | 0.3838 | 16.94 | 0.0470 | 0.0162 |
| UGDB | 0.8195 | 0.3759 | 0.3758 | 18.44 | 0.0490 | 0.0165 |
| RL-TTA | 0.8245 | 0.3885 | 0.3884 | 17.00 | 0.0488 | 0.0143 |
| W-UNet | **0.8291** | **0.4057** | **0.4056** | **16.90** | 0.0490 | **0.0143** |

Boundary F-score and NSD were numerically low relative to region Dice, showing that strong overlap does not guarantee precise contours. Calibration metrics were not used for clinical reliability claims: Brier score combines discrimination and calibration, ECE depends on binning, and internal performance does not establish behavior after acquisition or population shift [6,11].

## 5. Discussion

### 5.1 Principal findings

More labeled images produced the largest improvement: baseline Dice rose by 0.0463 from 25% to 100% labels. Among method changes, the wider network gave the best single-view result despite only about twice the parameters. Four-view inference delivered smaller, consistent gains without retraining, at the cost of roughly four forward passes.

Robust-logit consensus was consistently better than mean-probability averaging, but only by 0.0008, 0.0003, and 0.0001 Dice. This is too small for a broad superiority claim. Deployment choices should therefore consider latency and failure behavior, not Dice alone.

The proposed boundary objectives did not help. Possible explanations include redundancy with soft Dice, an undersized coefficient, imperfect scale matching, or boundary simplification after resizing. Future work should measure gradient contributions and tune the coefficient on validation data.

The capacity control changes the engineering conclusion. Without W-UNet, TTA appears best; with it, a modest increase in width produces a larger gain with single-view inference. This does not invalidate TTA, but shows why a simple capacity control is necessary.

### 5.2 Interpretation of the label-budget results

The reduction in TTA benefit from 25% to 100% labels is consistent with a variance-reduction explanation. When the model has seen fewer masks, predictions vary more under flips and rotations, so aggregation removes more view-specific error. This interpretation is plausible but not proven because the study did not manipulate transformation variance independently. The label-budget effect itself was much larger than any method effect: baseline Dice increased by 0.0463 between the smallest and largest subsets, compared with at most 0.0111 for a method change within a budget.

### 5.3 Boundary quality and calibration

Region Dice values near 0.90 coexisted with boundary F-scores near 0.40. This gap matters for applications where contour position or lesion size is more important than broad overlap. The tested boundary losses failed to close it, whereas the wider model improved both overlap and boundary endpoints. MP-TTA improved the balanced Brier score, but its ECE remained above RL-TTA and W-UNet. These differences show why one calibration statistic should not be used as a general reliability claim.

### 5.4 Implications for future work

A follow-up study should treat the negative boundary-loss result as a design clue rather than a final verdict. It should log the magnitude and gradient norm of each objective, tune the boundary weight on validation data, evaluate several distance definitions, and retain full-resolution masks. For inference aggregation, external cohorts should test whether the small TTA gain survives changes in devices, sites, and lesion prevalence. A larger seed count or repeated independent subset draws are needed for a confirmatory statistical analysis.

## 6. Limitations

This study uses one dataset and one fixed validation split. Patient independence could not be checked. Resizing to 256 × 256 removes fine details. Five seeds share nested subsets rather than independent datasets. Boundary and gate hyperparameters were minimally explored. Exact Wilcoxon testing has inadequate resolution at n=5 for two-sided significance below 0.05. There is no external validation, subgroup analysis, clinician review, acquisition-shift test, or prospective assessment. Test results measure reference-mask agreement, not diagnosis, prognosis, treatment benefit, safety, or clinical readiness.

## 7. Conclusion

Across 105 completed ISIC 2016 evaluations, four-view TTA improved a compact U-Net by about 0.003–0.006 Dice, and a modestly wider single-view U-Net achieved the highest mean Dice at every label budget. The tested uniform and sensitivity-gated boundary losses did not improve performance. Additional labels had the largest effect. Independent replication and external datasets are required before generalizing these findings.

## Declarations

**Ethics statement.** This computational study used an existing publicly distributed image dataset and collected no new participant data.

**Clinical scope.** The reported endpoints measure agreement with reference masks. The study did not evaluate diagnosis, prognosis, treatment decisions, clinician performance, patient outcomes, or safety.

**Data availability.** The repository does not redistribute images. It provides a deterministic preparation script for the upstream `MedOtter/ISIC2016` mirror. Users must review the upstream license and terms.

**Code availability.** Executed code, full machine-readable results, derived tables, figures, and both language versions of the manuscript are available in the accompanying GitHub repository.

## References

1. Siddique NA, Paheding S, Elkin CP, Devabhaktuni V. U-Net and its variants for medical image segmentation: a review of theory and applications. IEEE Access. 2021;9. doi:10.1109/ACCESS.2021.3086020.
2. Azad R, Aghdam EK, Rauland A, et al. Medical image segmentation review: the success of U-Net. arXiv. 2022. doi:10.48550/arXiv.2211.14830.
3. El Jurdi R, Petitjean C, Honeine P, Cheplygina V, Abdallah F. High-level prior-based loss functions for medical image segmentation: a survey. Computer Vision and Image Understanding. 2021;210:103248. doi:10.1016/j.cviu.2021.103248.
4. Ashraf H, Waris A, Ghafoor MF, Gilani SO, Niazi IK. Melanoma segmentation using deep learning with test-time augmentations and conditional random fields. Scientific Reports. 2022;12. doi:10.1038/s41598-022-07885-y.
5. Abdar M, Pourpanah F, Hussain S, et al. A review of uncertainty quantification in deep learning: techniques, applications and challenges. Information Fusion. 2021;76:243-297. doi:10.1016/j.inffus.2021.05.008.
6. Mehrtash A, Wells WM, Tempany CM, Abolmaesumi P, Kapur T. Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. IEEE Transactions on Medical Imaging. 2020;39. doi:10.1109/TMI.2020.3006437.
7. Cassidy B, Kendrick C, Brodzicki A, Jaworek-Korjakowska J, Yap MH. Analysis of the ISIC image datasets: usage, benchmarks and recommendations. Medical Image Analysis. 2022;75:102305. doi:10.1016/j.media.2021.102305.
8. Wen D, Khan SM, Xu AJ, et al. Characteristics of publicly available skin cancer image datasets: a systematic review. The Lancet Digital Health. 2022;4:e64-e74. doi:10.1016/S2589-7500(21)00252-1.
9. Muller D, Soto-Rey I, Kramer F. Towards a guideline for evaluation metrics in medical image segmentation. BMC Research Notes. 2022;15:210. doi:10.1186/s13104-022-06096-y.
10. Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nature Methods. 2024;21:195-212. doi:10.1038/s41592-023-02151-z.
11. Karimi D, Gholipour A. Improving calibration and out-of-distribution detection in deep models for medical image segmentation. IEEE Transactions on Artificial Intelligence. 2022. doi:10.1109/TAI.2022.3159510.
