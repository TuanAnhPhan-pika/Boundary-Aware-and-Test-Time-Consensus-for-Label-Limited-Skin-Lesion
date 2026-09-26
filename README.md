# Boundary-Aware and Test-Time Consensus for Label-Limited Skin-Lesion Segmentation

Reproducible research package for a five-seed ISIC 2016 segmentation study. It contains the executed code, full per-seed outputs, derived tables, figures, and English/Vietnamese papers.

## Main result

All 105 prespecified evaluations completed (3 label budgets × 5 seeds × 7 conditions). Four-view TTA improved compact U-Net Dice by about 0.003–0.006. A modestly wider single-view U-Net was best at every budget. The tested uniform and transformation-sensitivity-gated boundary losses did not improve Dice.

| Method | 25% labels | 50% labels | 100% labels |
|---|---:|---:|---:|
| Compact U-Net | 0.8444 | 0.8688 | 0.8907 |
| Robust-logit TTA | 0.8500 | 0.8726 | 0.8939 |
| Wider U-Net | **0.8555** | **0.8785** | **0.8971** |

Values are mean per-image Dice across five paired seeds on 379 test images. They measure agreement with reference masks, not diagnostic performance or clinical utility.

![Dice across label budgets](figures/dice_by_label_budget.png)

## Executed protocol

- Data source: `MedOtter/ISIC2016` mirror.
- Final split after exact-duplicate filtering: 716 development / 180 validation / 379 test images.
- Training budgets: 179 / 358 / 716 images.
- Resolution: 256 × 256.
- Five paired seeds and up to 60 epochs.
- GPU run: NVIDIA RTX 4050 Laptop GPU, PyTorch 2.13.0+cu130.
- Total measured experiment time: 9.73 hours.

## Repository layout

- `paper/paper_english.md` — full English paper.
- `paper/submission_manuscript.docx` — anonymous journal-style manuscript with formatted tables, figures, references, and declarations.
- `paper/paper_vietnamese.md` — Vietnamese paper in accessible technical language.
- `paper/thuyet_minh_nghien_cuu_tieng_viet.md` — Vietnamese research explanation covering motivation, protocol, results, contributions, limitations, and next steps.
- `paper/thuyet_minh_nghien_cuu_tieng_viet.docx` — formatted Word version of the Vietnamese research explanation.
- `paper/references.bib` — bibliography with DOI links.
- `src/` — exact experiment implementation.
- `scripts/prepare_isic2016_full.py` — deterministic data preparation.
- `scripts/generate_analysis.py` — regenerates tables and figures.
- `scripts/build_manuscript_docx.py` — rebuilds the formatted Word manuscript from the English Markdown source.
- `scripts/build_vietnamese_explanation_docx.py` — rebuilds the formatted Vietnamese research explanation.
- `results/experiment_results.json` — authoritative full output.
- `results/per_seed_metrics.csv` — tidy per-seed metrics.
- `results/summary_metrics.csv` — aggregate metrics.
- `results/paired_dice_comparisons.csv` — paired effect analysis.
- `figures/` — publication figures generated from the JSON output.

Image data, virtual environments, caches, system files, model caches, and AutoResearchClaw are intentionally excluded.

## Reproduce

Python 3.11 is recommended. Install dependencies:

```bash
python -m venv .venv
pip install -r src/requirements.txt
```

Prepare data from the upstream mirror:

```bash
python scripts/prepare_isic2016_full.py
```

Run from the `src` directory so the local modules resolve:

```bash
cd src
python main.py
cd ..
python scripts/generate_analysis.py
```

The experiment expects a CUDA-capable PyTorch build for the recorded configuration. Review the upstream dataset terms before downloading or redistributing images.

## Scope and limitations

This is an internal benchmark study on one dataset. Patient-level separation could not be verified from the available metadata, no external cohort was evaluated, and five seeds do not provide a powered confirmatory significance test. See the papers for full methods and limitations.
