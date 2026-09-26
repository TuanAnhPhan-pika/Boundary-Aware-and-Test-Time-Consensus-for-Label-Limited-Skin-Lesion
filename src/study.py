"""English study report generation with separated evidence and implications."""

import json
from pathlib import Path

import numpy as np

STUDY_PROTOCOL = """
Uncertainty-aware lightweight skin-lesion segmentation under limited labels

Research objective
------------------
This study evaluates whether signed-distance boundary supervision and
four-view test-time augmentation change segmentation overlap and probability
calibration relative to a compact U-Net on ISIC 2016. The primary
factorial comparison crosses two training objectives (BCE plus soft Dice,
with or without uniform signed-distance loss) with two inference procedures
(single view or four-view mean-probability TTA). Uncertainty-gated boundary
training, robust logit consensus, and a wider one-pass network are secondary
comparators.

Benchmark and partitions
------------------------
The program requires the locally prepared MedOtter mirror of ISIC 2016 with
development, validation, and test image/mask folders. The preparation script
downloads these data separately. Exact file duplicates, dimensions, binary
masks, empty masks, and patient overlap are checked where metadata permit. Limited-label subsets are
nested, lesion-area-stratified samples of a fixed development pool. Validation
data select checkpoints, uncertainty temperature, and probability temperature.
Test masks are used only for final evaluation.

Methods
-------
The compact model is a four-level U-Net with widths 16, 32, 64, and 128,
group normalization, depthwise-separable convolutional blocks, skip
connections, bilinear decoding, and one foreground logit. Its loss is binary
cross-entropy plus soft Dice.

The uniform-boundary model adds the mean product between foreground
probability and the signed distance to the reference region, with a linear
ten-epoch warm-up. The uncertainty-gated model estimates transformation
sensitivity from four inverse-aligned predictions of a frozen compact model
and weights each boundary term by exp(-variance/tau).

Mean TTA averages inverse-aligned probabilities from identity, horizontal
flip, vertical flip, and 180-degree rotation. Robust consensus instead removes
the view with the greatest local deviation from the median aligned logit and
averages the remaining three logits before applying sigmoid.

Endpoints
---------
The primary accuracy endpoint is mean per-image Dice at threshold 0.5.
Calibration endpoints include foreground-balanced Brier score, lesion-centered
Brier score, boundary-band Brier score, negative log-likelihood, expected
calibration error, calibration slope, and calibration intercept. Secondary
segmentation endpoints include IoU, boundary F-score, normalized surface Dice,
and HD95. Temperature scaling is fitted on validation data only.

Interpretation policy
---------------------
The numerical results measure agreement with the selected ISIC reference masks
under the declared partitions and preprocessing. They do not establish
diagnostic accuracy, clinical benefit, safety, or deployment readiness.
Transformation variance is called transformation sensitivity rather than
epistemic uncertainty unless its residual error-prediction value is established
under the preregistered geometry controls.
""".strip()

def _condition_rows(results):
    rows = []
    for regime, regime_results in results.items():
        for condition, seed_results in regime_results.items():
            values = [
                record["dice_score"]
                for record in seed_results.values()
                if record.get("status") == "success"
            ]
            if values:
                rows.append(
                    (
                        regime,
                        condition,
                        len(values),
                        float(np.mean(values)),
                        float(np.std(values)),
                    )
                )
    return rows

def render_study(results, metadata):
    lines = [
        STUDY_PROTOCOL,
        "",
        "Experimental evidence",
        "---------------------",
    ]
    rows = _condition_rows(results)

    if not rows:
        lines.append(
            "No complete numerical condition was available. No accuracy or "
            "calibration comparison is made."
        )
    else:
        lines.append(
            "The following values are computational outputs from completed "
            "runs and are reported as mean ± population standard deviation "
            "across successful prespecified seeds:"
        )
        lines.append("")
        lines.append(
            "| Label regime | Condition | Successful seeds | Dice mean ± SD |"
        )
        lines.append("|---|---|---:|---:|")
        for regime, condition, count, mean, std in rows:
            lines.append(
                f"| {regime} | {condition} | {count} | "
                f"{mean:.4f} ± {std:.4f} |"
            )

        lines.extend(
            [
                "",
                "These estimates are benchmark-specific. A failed or "
                "time-truncated seed is not silently treated as successful; "
                "unconditional summaries retain failures at the declared "
                "worst-case Dice value of zero.",
            ]
        )

    lines.extend(
        [
            "",
            "Clinical implications",
            "---------------------",
            "No result from this experiment demonstrates diagnostic "
            "performance, treatment benefit, or clinical safety. Before any "
            "clinical interpretation, the method requires external cohorts, "
            "independent contour annotations, acquisition-shift testing, "
            "clinician interaction studies, and prospective outcome evidence.",
            "",
            "Reproducibility record",
            "----------------------",
            f"Device: {metadata.get('device', 'unknown')}",
            f"Manifest: {metadata.get('manifest_path', 'unknown')}",
            f"Stopped by time guard: "
            f"{metadata.get('time_guard_triggered', False)}",
        ]
    )
    return "\n".join(lines)

def save_study(results, metadata, output_path):
    text = render_study(results, metadata)
    Path(output_path).write_text(text, encoding="utf-8")
    return text

def save_machine_readable(payload, output_path):
    Path(output_path).write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
