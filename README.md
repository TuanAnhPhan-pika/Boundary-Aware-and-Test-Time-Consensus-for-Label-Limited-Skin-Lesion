# Boundary-Aware and Test-Time Consensus for Label-Limited Skin-Lesion Segmentation

This repository contains the research paper, experiment code, recorded results,
and figures for a small CPU pilot on ISIC 2016 skin-lesion segmentation.

The study compares a compact U-Net baseline with boundary-weighted training,
uncertainty-weighted boundary training, mean-probability test-time augmentation,
robust-logit test-time consensus, and a wider single-pass U-Net.

## Main finding

In the executed two-epoch pilot, none of the tested changes showed a reliable
improvement over the compact U-Net baseline. Robust-logit TTA increased mean
Dice by approximately 0.004, while the boundary-loss variants produced almost
no change. These results are exploratory and do not establish clinical utility.

## Repository structure

- `paper/paper_english.md`: corrected English report.
- `paper/paper_vietnamese.md`: Vietnamese report.
- `paper/references.bib`: bibliography.
- `src/`: experiment implementation.
- `scripts/prepare_isic2016_subset.py`: deterministic data preparation.
- `results/experiment_results.json`: per-condition and per-seed outputs.
- `results/citation_verification.json`: citation verification record.
- `figures/`: generated result plots.

The AutoResearchClaw framework, its internal stages, local virtual environment,
caches, and downloaded image data are intentionally excluded.

## Experiment scope

- Prepared data: 140 train, 40 validation, and 60 test image-mask pairs.
- Executed pilot: 80-image development pool, 20 validation images, and
  30 test images.
- Training subsets: 20 images (25%) and 40 images (50%).
- Input size: 64 x 64 pixels.
- Training: 2 epochs, 3 paired seeds, CPU.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r src/requirements.txt
python scripts/prepare_isic2016_subset.py
python src/main.py
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r src/requirements.txt
python scripts/prepare_isic2016_subset.py
python src/main.py
```

The preparation script downloads a small subset of `MedOtter/ISIC2016` through
the Hugging Face `datasets` package. Review the upstream dataset terms before
redistributing any images. Dataset files are excluded from this repository.

## Important limitations

This is a resource-limited pilot, not a full benchmark evaluation. The images
were resized to 64 x 64, training lasted only two epochs, and each condition
used three seeds. The reported outputs concern mask overlap on the selected
subset and must not be interpreted as evidence of diagnostic accuracy, safety,
or clinical readiness.

