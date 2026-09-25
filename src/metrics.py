"""Segmentation, surface, calibration, uncertainty, and paired metrics."""

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def dice_per_image(probabilities, targets, threshold, epsilon):
    predictions = probabilities >= threshold
    targets = targets >= 0.5
    intersection = (predictions & targets).sum((1, 2))
    denominator = (
        predictions.sum((1, 2)) + targets.sum((1, 2))
    )
    return (
        2.0 * intersection + epsilon
    ) / (denominator + epsilon)

def iou_per_image(probabilities, targets, threshold, epsilon):
    predictions = probabilities >= threshold
    targets = targets >= 0.5
    intersection = (predictions & targets).sum((1, 2))
    union = (predictions | targets).sum((1, 2))
    return (intersection + epsilon) / (union + epsilon)

def foreground_balanced_brier(probabilities, targets):
    values = []
    for probability, target in zip(probabilities, targets):
        foreground = target >= 0.5
        background = ~foreground
        foreground_error = (
            np.mean((probability[foreground] - 1.0) ** 2)
            if foreground.any()
            else 0.0
        )
        background_error = (
            np.mean(probability[background] ** 2)
            if background.any()
            else 0.0
        )
        values.append(
            0.5 * (foreground_error + background_error)
        )
    return np.asarray(values, dtype=np.float64)

def lesion_centered_brier(probabilities, targets, margin=8):
    values = []
    height, width = targets.shape[-2:]

    for probability, target in zip(probabilities, targets):
        foreground = np.argwhere(target >= 0.5)
        if foreground.size == 0:
            continue
        y0, x0 = foreground.min(axis=0)
        y1, x1 = foreground.max(axis=0)
        y0 = max(0, int(y0) - margin)
        x0 = max(0, int(x0) - margin)
        y1 = min(height, int(y1) + margin + 1)
        x1 = min(width, int(x1) + margin + 1)
        values.append(
            np.mean(
                (
                    probability[y0:y1, x0:x1]
                    - target[y0:y1, x0:x1]
                )
                ** 2
            )
        )

    return float(np.mean(values))

def negative_log_likelihood(probabilities, targets):
    probabilities = np.clip(
        probabilities, 1e-7, 1.0 - 1e-7
    )
    losses = -(
        targets * np.log(probabilities)
        + (1.0 - targets) * np.log(1.0 - probabilities)
    )
    return losses.mean((1, 2))

def expected_calibration_error(probabilities, targets, bins):
    probabilities = probabilities.ravel()
    targets = targets.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0

    for index in range(bins):
        right_closed = index == bins - 1
        selected = probabilities >= edges[index]
        if right_closed:
            selected &= probabilities <= edges[index + 1]
        else:
            selected &= probabilities < edges[index + 1]

        if selected.any():
            error += selected.mean() * abs(
                probabilities[selected].mean()
                - targets[selected].mean()
            )
    return float(error)

def _boundary(mask):
    structure = np.ones((3, 3), dtype=np.bool_)
    return mask ^ binary_erosion(mask, structure=structure)

def boundary_band_brier(probabilities, targets, band_pixels):
    values = []
    for probability, target in zip(probabilities, targets):
        truth = target >= 0.5
        band = binary_dilation(
            _boundary(truth),
            iterations=band_pixels,
        )
        values.append(
            np.mean((probability[band] - target[band]) ** 2)
        )
    return float(np.mean(values))

def boundary_f_score(probabilities, targets, threshold, tolerance):
    scores = []
    for probability, target in zip(probabilities, targets):
        predicted_boundary = _boundary(
            probability >= threshold
        )
        true_boundary = _boundary(target >= 0.5)
        predicted_neighborhood = binary_dilation(
            predicted_boundary, iterations=tolerance
        )
        true_neighborhood = binary_dilation(
            true_boundary, iterations=tolerance
        )
        precision = (
            (predicted_boundary & true_neighborhood).sum()
            / max(1, predicted_boundary.sum())
        )
        recall = (
            (true_boundary & predicted_neighborhood).sum()
            / max(1, true_boundary.sum())
        )
        scores.append(
            2.0 * precision * recall
            / max(1e-12, precision + recall)
        )
    return float(np.mean(scores))

def surface_metrics(probabilities, targets, threshold, tolerance):
    hd95_values = []
    normalized_surface_dice_values = []
    diagonal = float(np.hypot(*targets.shape[-2:]))

    for probability, target in zip(probabilities, targets):
        predicted_boundary = _boundary(
            probability >= threshold
        )
        true_boundary = _boundary(target >= 0.5)

        if not predicted_boundary.any() or not true_boundary.any():
            hd95_values.append(diagonal)
            normalized_surface_dice_values.append(0.0)
            continue

        distance_to_truth = distance_transform_edt(
            ~true_boundary
        )[predicted_boundary]
        distance_to_prediction = distance_transform_edt(
            ~predicted_boundary
        )[true_boundary]
        symmetric_distances = np.concatenate(
            (distance_to_truth, distance_to_prediction)
        )
        hd95_values.append(
            float(np.percentile(symmetric_distances, 95))
        )
        normalized_surface_dice_values.append(
            float(
                np.mean(symmetric_distances <= tolerance)
            )
        )

    return {
        "hd95": float(np.mean(hd95_values)),
        "normalized_surface_dice": float(
            np.mean(normalized_surface_dice_values)
        ),
    }

def calibration_slope_intercept(probabilities, targets):
    clipped = np.clip(
        probabilities.ravel(), 1e-6, 1.0 - 1e-6
    )
    outcomes = targets.ravel().astype(np.int64)
    logits = np.log(clipped / (1.0 - clipped))

    stride = max(1, logits.size // 200_000)
    logits = logits[::stride, None]
    outcomes = outcomes[::stride]

    if np.unique(outcomes).size < 2:
        return float("nan"), float("nan")

    regression = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=200,
        random_state=0,
    )
    regression.fit(logits, outcomes)
    return (
        float(regression.coef_[0, 0]),
        float(regression.intercept_[0]),
    )

def evaluate_arrays(
    probabilities,
    targets,
    threshold,
    dice_epsilon,
    calibration_bins,
    boundary_band_pixels,
    surface_tolerance_pixels,
    disagreement=None,
):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    dice_values = dice_per_image(
        probabilities, targets, threshold, dice_epsilon
    )
    iou_values = iou_per_image(
        probabilities, targets, threshold, dice_epsilon
    )
    balanced_brier = foreground_balanced_brier(
        probabilities, targets
    )
    nll = negative_log_likelihood(probabilities, targets)
    slope, intercept = calibration_slope_intercept(
        probabilities, targets
    )
    surface = surface_metrics(
        probabilities,
        targets,
        threshold,
        surface_tolerance_pixels,
    )

    result = {
        "dice_score": float(dice_values.mean()),
        "dice_score_per_image": dice_values.tolist(),
        "intersection_over_union": float(iou_values.mean()),
        "foreground_balanced_brier_score": float(
            balanced_brier.mean()
        ),
        "lesion_centered_brier_score": lesion_centered_brier(
            probabilities, targets
        ),
        "negative_log_likelihood": float(nll.mean()),
        "expected_calibration_error": expected_calibration_error(
            probabilities, targets, calibration_bins
        ),
        "boundary_band_brier_score": boundary_band_brier(
            probabilities, targets, boundary_band_pixels
        ),
        "boundary_f_score": boundary_f_score(
            probabilities,
            targets,
            threshold,
            surface_tolerance_pixels,
        ),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        **surface,
    }

    if disagreement is not None:
        image_uncertainty = np.asarray(disagreement).mean(
            axis=(1, 2)
        )
        image_error = 1.0 - dice_values
        correlation = stats.spearmanr(
            image_uncertainty, image_error
        ).statistic
        result["image_error_spearman_correlation"] = float(
            correlation
        )

        worst = image_error >= np.quantile(image_error, 0.90)
        if np.unique(worst).size == 2:
            result["worst_dice_decile_identification_auroc"] = (
                float(roc_auc_score(worst, image_uncertainty))
            )

    return result

@torch.no_grad()
def collect_logits_and_targets(model, loader, device):
    logits = []
    targets = []
    model.eval()

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        output = torch.clamp(model(images), -15.0, 15.0)
        logits.append(output.cpu().numpy())
        targets.append(masks.numpy())

    return (
        np.concatenate(logits, axis=0),
        np.concatenate(targets, axis=0),
    )

def fit_temperature(model, validation_loader, device):
    logits, targets = collect_logits_and_targets(
        model, validation_loader, device
    )
    logits = logits.ravel()
    targets = targets.ravel()

    stride = max(1, logits.size // 250_000)
    logits = logits[::stride]
    targets = targets[::stride]

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        scaled = np.clip(
            logits / temperature, -15.0, 15.0
        )
        probabilities = 1.0 / (1.0 + np.exp(-scaled))
        probabilities = np.clip(
            probabilities, 1e-7, 1.0 - 1e-7
        )
        return float(
            -np.mean(
                targets * np.log(probabilities)
                + (1.0 - targets)
                * np.log(1.0 - probabilities)
            )
        )

    solution = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(10.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(np.exp(solution.x))

def paired_seed_analysis(method_values, baseline_values, seed):
    method = np.asarray(method_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    differences = method - baseline

    if differences.size < 2:
        return None

    standard_deviation = differences.std(ddof=1)
    effect = (
        differences.mean() / standard_deviation
        if standard_deviation > 0
        else 0.0
    )

    try:
        wilcoxon = stats.wilcoxon(differences)
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_p_value = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(2000, dtype=np.float64)
    for index in range(bootstrap.size):
        draw = rng.integers(
            0, differences.size, size=differences.size
        )
        bootstrap[index] = differences[draw].mean()

    low, high = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "mean_difference": float(differences.mean()),
        "standard_deviation": float(standard_deviation),
        "paired_standardized_mean_difference": float(effect),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p_value,
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
    }