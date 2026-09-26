# Boundary Loss or Test-Time Consensus? A Five-Seed Study of Label-Limited ISIC 2016 Skin-Lesion Segmentation

## Abstract

This study tests whether signed-distance boundary supervision and four-view test-time aggregation improve a compact U-Net when pixel-level annotations are limited. We used ISIC 2016 image-mask pairs distributed by the MedOtter mirror. After removing four exact cross-split image duplicates, the experiment contained 716 development, 180 validation, and 379 test images. Nested, lesion-area-stratified training subsets used 179 (25%), 358 (50%), or 716 (100%) images. Seven conditions were evaluated at 256 × 256 with five paired random seeds: compact U-Net; mean-probability test-time augmentation (TTA); uniform boundary loss; boundary loss plus mean TTA; transformation-sensitivity-gated boundary loss; robust-logit consensus TTA; and a wider single-view U-Net. All 105 prespecified evaluations completed. At 25%, 50%, and 100% labels, the compact baseline obtained Dice 0.8444 ± 0.0095, 0.8688 ± 0.0051, and 0.8907 ± 0.0041 (mean ± sample SD). Robust-logit consensus improved these means by 0.0057, 0.0039, and 0.0032. The wider U-Net was best overall at 0.8555, 0.8785, and 0.8971. Both boundary-loss variants were slightly below baseline at every budget. With five paired seeds, two-sided Wilcoxon tests could not attain p < 0.05; positive paired bootstrap intervals are therefore descriptive rather than confirmatory. The results support modest gains from multi-view inference and additional capacity, but not from the tested boundary objectives. They quantify reference-mask agreement and do not establish diagnostic or clinical performance.

**Keywords:** medical image segmentation; skin lesion; U-Net; limited annotations; boundary loss; test-time augmentation

## 1. Introduction

Skin-lesion segmentation identifies lesion pixels in dermoscopic images. Reliable contours are challenging because of weak contrast, hair, illumination changes, artifacts, and irregular borders. Dense masks also require expert time, motivating methods that work with fewer annotated images. U-Net-style encoder-decoder models remain common baselines [siddique2021unet; azad2022medical].

Two inexpensive strategies are attractive. A boundary term can add spatial information that region losses may underemphasize [jurdi2021highlevel]. Test-time augmentation predicts several geometry-preserving views, aligns the outputs, and aggregates them [ashraf2022melanoma]. View disagreement may indicate transformation sensitivity, although it is not automatically a validated estimate of epistemic uncertainty [abdar2021review; mehrtash2020confidence].

We ask whether boundary supervision or four-view aggregation improves a compact U-Net and whether either is competitive with a wider single-view network. The experiment uses paired seeds, nested label subsets, held-out validation, and overlap, boundary, distance, and calibration metrics.

## 2. Materials and methods

### 2.1 Data and leakage control

The preparation program materialized `MedOtter/ISIC2016` train and test pairs. From 900 source-training pairs, 180 validation cases were chosen deterministically by SHA-256 ranking of image identifiers; the other 720 initially formed the development pool. The source test split supplied 379 pairs. A byte-level image-hash audit retained test over validation over training and removed four training images duplicated across splits. Final counts were 716 development, 180 validation, and 379 test images. Images are not redistributed in this repository.

For each seed, training cases were stratified by lesion-area fraction. The 25%, 50%, and 100% subsets contained 179, 358, and 716 images and were nested within that seed. Every method within a seed and budget used the same cases and augmentation seed. Images and masks were resized to 256 × 256. Patient-level separation could not be verified because suitable identifiers were unavailable; this is a limitation. Careful reporting is important because public skin-image collections can contain overlap and heterogeneous metadata [cassidy2021analysis; wen2021characteristics].

### 2.2 Models and training

The compact four-level U-Net uses widths 16/32/64/128, depthwise-separable blocks, group normalization, SiLU, max pooling, bilinear upsampling, skip connections, and one foreground logit (62,716 trainable parameters). The wider comparator uses widths 24/48/96/192 (134,572 parameters), selected by a pre-run latency rule.

The base objective is binary cross-entropy plus soft Dice. The uniform boundary model adds the mean product of foreground probability and signed distance to the reference mask, clipped at 20 pixels, with weight 0.01 and a ten-epoch warm-up. The gated model multiplies that term by `exp(-variance/0.01)`, where variance comes from four inverse-aligned predictions of a frozen compact reference. We call this transformation sensitivity, not calibrated uncertainty.

Models used AdamW (learning rate 3×10⁻⁴, weight decay 10⁻⁴), cosine decay, gradient clipping at 1.0, batch size 24, bfloat16, and at most 60 epochs. Validation occurred every two epochs; early stopping used five validation events. Augmentation comprised horizontal/vertical flips, 90-degree rotations, and mild brightness/contrast jitter. Checkpoint and temperature choices used validation only.

### 2.3 Inference and conditions

Four-view inference used identity, horizontal flip, vertical flip, and 180-degree rotation. Mean TTA averages inverse-aligned probabilities. Robust consensus computes aligned logits, discards at every pixel the view with greatest local deviation from the median in a 5 × 5 window, averages the remaining three logits, clips to ±15, and applies sigmoid.

Seven conditions were prespecified: compact U-Net (C-UNet); mean-probability TTA (MP-TTA); uniform distance-boundary training (UDB); UDB with mean TTA (UDB+TTA); sensitivity-gated boundary training (UGDB); robust-logit consensus (RL-TTA); and wider single-view U-Net (W-UNet).

### 2.4 Outcomes and statistics

The primary endpoint was mean per-image Dice at threshold 0.5. Secondary endpoints included IoU, boundary F-score, normalized surface Dice (NSD), HD95, Brier scores, negative log-likelihood, expected calibration error, calibration slope, and intercept. Multiple metrics are necessary because no score captures every relevant segmentation error [muller2022guideline; maierhein2024metrics].

We report means and sample SDs over five paired seeds. Each method was compared with C-UNet using a two-sided exact Wilcoxon signed-rank test and a 20,000-resample paired bootstrap interval for the mean Dice difference. With five nonzero pairs, the smallest possible two-sided exact Wilcoxon p-value is 0.0625; this is an estimation-oriented study, not a powered confirmatory test.

## 3. Results

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

![Full-label metrics](../figures/full_label_metrics.png)

Boundary F-score and NSD were numerically low relative to region Dice, showing that strong overlap does not guarantee precise contours. Calibration metrics were not used for clinical reliability claims: Brier score combines discrimination and calibration, ECE depends on binning, and internal performance does not establish behavior after acquisition or population shift [mehrtash2020confidence; karimi2022improving].

## 4. Discussion

More labeled images produced the largest improvement: baseline Dice rose by 0.0463 from 25% to 100% labels. Among method changes, the wider network gave the best single-view result despite only about twice the parameters. Four-view inference delivered smaller, consistent gains without retraining, at the cost of roughly four forward passes.

Robust-logit consensus was consistently better than mean-probability averaging, but only by 0.0008, 0.0003, and 0.0001 Dice. This is too small for a broad superiority claim. Deployment choices should therefore consider latency and failure behavior, not Dice alone.

The proposed boundary objectives did not help. Possible explanations include redundancy with soft Dice, an undersized coefficient, imperfect scale matching, or boundary simplification after resizing. Future work should measure gradient contributions and tune the coefficient on validation data.

The capacity control changes the engineering conclusion. Without W-UNet, TTA appears best; with it, a modest increase in width produces a larger gain with single-view inference. This does not invalidate TTA, but shows why a simple capacity control is necessary.

## 5. Limitations

This study uses one dataset and one fixed validation split. Patient independence could not be checked. Resizing to 256 × 256 removes fine details. Five seeds share nested subsets rather than independent datasets. Boundary and gate hyperparameters were minimally explored. Exact Wilcoxon testing has inadequate resolution at n=5 for two-sided significance below 0.05. There is no external validation, subgroup analysis, clinician review, acquisition-shift test, or prospective assessment. Test results measure reference-mask agreement, not diagnosis, prognosis, treatment benefit, safety, or clinical readiness.

## 6. Conclusion

Across 105 completed ISIC 2016 evaluations, four-view TTA improved a compact U-Net by about 0.003–0.006 Dice, and a modestly wider single-view U-Net achieved the highest mean Dice at every label budget. The tested uniform and sensitivity-gated boundary losses did not improve performance. Additional labels had the largest effect. Independent replication and external datasets are required before generalizing these findings.

## Reproducibility and data statement

The repository contains the exact executed code, preparation script, per-seed results, derived tables, analysis script, figures, and bibliography. It excludes image data, virtual environments, caches, system files, and AutoResearchClaw. Upstream dataset terms must be reviewed before downloading or redistributing data.
