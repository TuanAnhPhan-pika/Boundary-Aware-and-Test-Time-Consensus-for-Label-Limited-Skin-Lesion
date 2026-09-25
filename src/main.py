"""
Dataset used and loading
------------------------
The experiment loads the locally prepared ISIC 2016 subset from
data/isic2016_experiment. The preparation script creates 140 training,
40 validation, and 60 test image-mask pairs. The executable requires all
three folders and does not silently create replacement splits.

Regimes
-------
A fixed lesion-area-stratified pool of 80 images is selected from the prepared
training partition. The executed regimes use 25% and 50% of this pool (20 and
40 images). Methods within a seed use the same subset, seed, deterministic
augmentation schedule, validation set, and test set.

Architecture
------------
The compact four-level U-Net uses RGB input, widths 4/8/16/32,
depthwise-separable convolutional blocks, group normalization, SiLU
activations, max pooling, bilinear upsampling, skip concatenation, and one
foreground logit. All executable models contain fewer than one million
parameters.

Training protocol
-----------------
Models use AdamW, BCE plus soft Dice, at most 2 epochs, batch size 8, cosine
learning-rate decay, mixed precision on CUDA, and gradient clipping. Boundary
models add uniform or transformation-sensitivity-gated signed-distance loss.
Checkpoint and temperature selection use validation data only.

Evaluation protocol
-------------------
Every completed condition is evaluated on the same fixed 30-image test
subset. Metrics include Dice, IoU, boundary F-score, normalized surface Dice,
HD95, foreground-balanced Brier score, lesion-centered Brier score,
boundary-band Brier score, NLL, ECE, calibration slope, and intercept.

METRIC NAME: dice_score
DIRECTION: maximize
UNITS/SCALE: unitless proportion in [0,1]
FORMULA: (2*|threshold(p,0.5) intersection y|+epsilon) /
         (|threshold(p,0.5)|+|y|+epsilon)
AGGREGATION: arithmetic mean over test images

Scope
-----
This 600-second implementation directly uses authentic skin-lesion images and
masks, but is a resource-limited pilot with three seeds and restricted
development/test sizes. It cannot replace the preregistered ten-seed definitive
protocol. Results are written to results.json; the corrected reports are in
the repository's paper directory.
"""

import copy
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from experiment_harness import ExperimentHarness

from isic_data import (
    ISICSegmentationDataset,
    add_lesion_fractions,
    compute_channel_statistics,
    discover_isic_manifest,
    make_epoch_loader,
    nested_label_subset,
    set_all_seeds,
    stratified_sample,
    validate_manifest,
)
from metrics import (
    evaluate_arrays,
    fit_temperature,
    paired_seed_analysis,
)
from models import (
    CompactUNetBCESoftDice,
    LatencyMatchedWiderUNet,
    MeanProbabilityFourViewTTA,
    NumericalDivergenceError,
    RobustLogitConsensusFourViewTTA,
    UncertaintyGatedDistanceBoundaryUNet,
    UniformDistanceBoundaryUNet,
)
from study import save_machine_readable, save_study

_PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "data" / "isic2016_experiment").is_dir()
)

HYPERPARAMETERS = {
    "dataset_root": str(_PROJECT_ROOT / "data" / "isic2016_experiment"),
    "manifest_filename": "manifest.csv",
    "image_size": 64,
    "development_pool_size": 80,
    "validation_limit": 20,
    "test_limit": 30,
    "label_budget_fractions": [0.25, 0.50],
    "batch_size": 8,
    "evaluation_batch_size": 8,
    "num_epochs": 2,
    "early_stopping_patience": 1,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "dice_epsilon": 1e-6,
    "compact_widths": [4, 8, 16, 32],
    "wider_width_candidates": [
        [6, 12, 24, 48],
        [8, 16, 32, 64],
    ],
    "maximum_parameters": 1_000_000,
    "lambda_boundary": 0.1,
    "boundary_warmup_epochs": 10,
    "distance_clip_pixels": 20,
    "tau_candidates": [0.003, 0.01],
    "variance_floor": 1e-8,
    "threshold": 0.5,
    "num_tta_views": 4,
    "local_discordance_window": 5,
    "retained_views_per_pixel": 3,
    "logit_clip": 15.0,
    "affine_degrees": 8.0,
    "affine_translation": 0.05,
    "color_jitter": 0.08,
    "calibration_bins": 10,
    "boundary_band_pixels": 3,
    "surface_tolerance_pixels": 2,
    "latency_warmup_iterations": 10,
    "latency_timed_iterations": 30,
    "time_budget_seconds": 600,
}

SEEDS = [0, 1, 2]

CONDITIONS = [
    "compact_unet_bce_softdice_single_view",
    "compact_unet_mean_probability_four_view_tta",
    "uniform_distance_boundary_unet_single_view",
    "uniform_boundary_mean_probability_four_view_factorial_arm",
    "uncertainty_gated_distance_boundary_unet_single_view",
    "compact_unet_robust_logit_consensus_four_view_tta",
    "latency_matched_wider_unet_single_view",
]

def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))

def make_scaler(device):
    try:
        return torch.amp.GradScaler(
            device.type, enabled=device.type == "cuda"
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda"
        )

def make_dataset(frame, mean, std, augment, seed):
    return ISICSegmentationDataset(
        frame=frame,
        image_size=HYPERPARAMETERS["image_size"],
        mean=mean,
        std=std,
        distance_clip_pixels=HYPERPARAMETERS[
            "distance_clip_pixels"
        ],
        augment=augment,
        augmentation_seed=seed,
        affine_degrees=HYPERPARAMETERS["affine_degrees"],
        affine_translation=HYPERPARAMETERS[
            "affine_translation"
        ],
        color_jitter=HYPERPARAMETERS["color_jitter"],
    )

def validation_loss(model, dataset, device, seed):
    loader = make_epoch_loader(
        dataset,
        HYPERPARAMETERS["evaluation_batch_size"],
        seed,
        epoch=0,
        shuffle=False,
    )
    losses = []
    model.eval()

    with torch.no_grad():
        for images, masks, _, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = model.segmentation_loss(logits, masks)
            if torch.isfinite(loss).item():
                losses.append(float(loss.cpu()))

    return float(np.mean(losses)) if losses else float("inf")

def train_model(
    model,
    train_dataset,
    validation_dataset,
    seed,
    device,
    harness,
):
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=HYPERPARAMETERS["learning_rate"],
        weight_decay=HYPERPARAMETERS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=HYPERPARAMETERS["num_epochs"],
    )
    scaler = make_scaler(device)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    completed_epochs = 0

    for epoch in range(HYPERPARAMETERS["num_epochs"]):
        if harness.should_stop():
            return model, True, completed_epochs

        loader = make_epoch_loader(
            train_dataset,
            HYPERPARAMETERS["batch_size"],
            seed,
            epoch,
            shuffle=True,
        )
        model.train()
        epoch_losses = []

        for batch in loader:
            if harness.should_stop():
                return model, True, completed_epochs

            loss = model.train_step(
                batch,
                optimizer,
                scaler,
                device,
                HYPERPARAMETERS["gradient_clip_norm"],
                epoch,
            )
            if not np.isfinite(loss) or loss > 100.0:
                print("FAIL: NaN/divergence detected")
                raise NumericalDivergenceError(
                    "invalid loss", loss
                )
            epoch_losses.append(loss)

        completed_epochs += 1
        scheduler.step()
        current_loss = validation_loss(
            model, validation_dataset, device, seed
        )
        print(
            f"TRAIN: seed={seed} epoch={epoch + 1} "
            f"loss={np.mean(epoch_losses):.6f} "
            f"validation_loss={current_loss:.6f}"
        )

        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if (
            stale_epochs
            >= HYPERPARAMETERS["early_stopping_patience"]
        ):
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, False, completed_epochs

def validate_model(model, dataset, device, condition):
    count = parameter_count(model)
    if count >= HYPERPARAMETERS["maximum_parameters"]:
        raise ValueError(
            f"{condition} has {count} parameters, exceeding limit"
        )

    loader = make_epoch_loader(
        dataset, 1, training_seed=0, epoch=0, shuffle=False
    )
    images, _, _, _ = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        output = model(images)

    expected = (
        images.shape[0],
        1,
        HYPERPARAMETERS["image_size"],
        HYPERPARAMETERS["image_size"],
    )
    if tuple(output.shape) != expected:
        raise ValueError(
            f"{condition} output shape {tuple(output.shape)} "
            f"does not match {expected}"
        )

    print(
        f"condition={condition} input_shape={tuple(images.shape)} "
        f"output_shape={tuple(output.shape)} parameters={count}"
    )

@torch.no_grad()
def cache_test_views(model, test_dataset, device):
    loader = make_epoch_loader(
        test_dataset,
        HYPERPARAMETERS["evaluation_batch_size"],
        training_seed=0,
        epoch=0,
        shuffle=False,
    )
    tta = MeanProbabilityFourViewTTA(
        threshold=HYPERPARAMETERS["threshold"],
        num_views=HYPERPARAMETERS["num_tta_views"],
    )

    cached_logits = []
    cached_targets = []
    image_ids = []

    model.eval()
    for images, masks, _, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        aligned = tta.cache_aligned_logits(model, images)
        if not torch.isfinite(aligned).all().item():
            raise NumericalDivergenceError(
                "non-finite cached logits"
            )
        cached_logits.append(
            aligned.cpu().to(torch.float16)
        )
        cached_targets.append(masks.cpu())
        image_ids.extend(batch_ids)

    return {
        "aligned_logits": torch.cat(cached_logits, dim=0),
        "targets": torch.cat(cached_targets, dim=0),
        "image_ids": image_ids,
    }

def metrics_from_cache(cache, aggregation, temperature):
    logits = cache["aligned_logits"].float()
    targets = cache["targets"].squeeze(1).numpy()

    if aggregation == "single":
        probabilities = torch.sigmoid(
            logits[:, 0] / temperature
        )
        disagreement = None
    elif aggregation == "mean_probability":
        probabilities = MeanProbabilityFourViewTTA.aggregate(
            logits, temperature
        )
        disagreement = torch.sigmoid(
            logits / temperature
        ).var(dim=1, unbiased=False)
    elif aggregation == "robust_logit":
        robust = RobustLogitConsensusFourViewTTA(
            local_discordance_window=HYPERPARAMETERS[
                "local_discordance_window"
            ],
            retained_views_per_pixel=HYPERPARAMETERS[
                "retained_views_per_pixel"
            ],
            logit_clip=HYPERPARAMETERS["logit_clip"],
            threshold=HYPERPARAMETERS["threshold"],
        )
        probabilities = robust.predict_proba(
            logits, temperature
        )
        disagreement = robust.predict_disagreement(
            logits, temperature
        )
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")

    probabilities = probabilities.squeeze(1).numpy()
    disagreement_array = (
        disagreement.squeeze(1).numpy()
        if disagreement is not None
        else None
    )

    return evaluate_arrays(
        probabilities=probabilities,
        targets=targets,
        threshold=HYPERPARAMETERS["threshold"],
        dice_epsilon=HYPERPARAMETERS["dice_epsilon"],
        calibration_bins=HYPERPARAMETERS[
            "calibration_bins"
        ],
        boundary_band_pixels=HYPERPARAMETERS[
            "boundary_band_pixels"
        ],
        surface_tolerance_pixels=HYPERPARAMETERS[
            "surface_tolerance_pixels"
        ],
        disagreement=disagreement_array,
    )

def single_view_metrics(model, dataset, device, temperature):
    cache = cache_test_views(model, dataset, device)
    return metrics_from_cache(cache, "single", temperature)

def time_candidate_models(device, example):
    compact = CompactUNetBCESoftDice(
        HYPERPARAMETERS["compact_widths"],
        HYPERPARAMETERS["dice_epsilon"],
    ).to(device)
    compact.eval()
    tta = MeanProbabilityFourViewTTA()

    reference_times = tta.benchmark_latency(
        compact,
        example,
        temperature=1.0,
        warmup_iterations=HYPERPARAMETERS[
            "latency_warmup_iterations"
        ],
        timed_iterations=HYPERPARAMETERS[
            "latency_timed_iterations"
        ],
    )
    reference_median = float(np.median(reference_times))

    candidates = []
    for widths in HYPERPARAMETERS["wider_width_candidates"]:
        candidate = LatencyMatchedWiderUNet(
            widths,
            HYPERPARAMETERS["dice_epsilon"],
        ).to(device)
        if parameter_count(candidate) >= HYPERPARAMETERS[
            "maximum_parameters"
        ]:
            continue

        times = candidate.benchmark_latency(
            example,
            HYPERPARAMETERS["latency_warmup_iterations"],
            HYPERPARAMETERS["latency_timed_iterations"],
        )
        median = float(np.median(times))
        candidates.append(
            (abs(median / reference_median - 1.0), widths, median)
        )

    if not candidates:
        raise RuntimeError("no valid wider-model candidate")

    candidates.sort(key=lambda item: item[0])
    _, selected_widths, selected_median = candidates[0]
    print(
        f"LATENCY_SELECTION: compact_four_view_ms="
        f"{reference_median:.4f} selected_widths={selected_widths} "
        f"selected_single_view_ms={selected_median:.4f} "
        f"ratio={selected_median / reference_median:.4f}"
    )
    return selected_widths

def select_tau(
    frozen_reference,
    train_dataset,
    validation_dataset,
    seed,
    device,
    harness,
):
    candidates = []
    for tau in HYPERPARAMETERS["tau_candidates"]:
        if harness.should_stop():
            return None, True

        set_all_seeds(seed)
        model = UncertaintyGatedDistanceBoundaryUNet(
            frozen_reference=copy.deepcopy(frozen_reference),
            widths=HYPERPARAMETERS["compact_widths"],
            dice_epsilon=HYPERPARAMETERS["dice_epsilon"],
            lambda_boundary=HYPERPARAMETERS[
                "lambda_boundary"
            ],
            boundary_warmup_epochs=HYPERPARAMETERS[
                "boundary_warmup_epochs"
            ],
            tau=tau,
            variance_floor=HYPERPARAMETERS["variance_floor"],
        ).to(device)

        model, truncated, _ = train_model(
            model,
            train_dataset,
            validation_dataset,
            seed,
            device,
            harness,
        )
        if truncated:
            return None, True

        score = validation_loss(
            model, validation_dataset, device, seed
        )
        candidates.append((score, tau, model))

    candidates.sort(key=lambda item: item[0])
    _, selected_tau, selected_model = candidates[0]
    print(f"VALIDATION_SELECTION: tau={selected_tau}")
    return selected_model, False

def pilot_estimate(train_dataset, device, planned_runs):
    set_all_seeds(991)
    model = CompactUNetBCESoftDice(
        HYPERPARAMETERS["compact_widths"],
        HYPERPARAMETERS["dice_epsilon"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMETERS["learning_rate"],
        weight_decay=HYPERPARAMETERS["weight_decay"],
    )
    scaler = make_scaler(device)
    loader = make_epoch_loader(
        train_dataset,
        HYPERPARAMETERS["batch_size"],
        991,
        epoch=0,
        shuffle=True,
    )
    batch = next(iter(loader))

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.train_step(
        batch,
        optimizer,
        scaler,
        device,
        HYPERPARAMETERS["gradient_clip_norm"],
        epoch=0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    step_seconds = max(time.perf_counter() - start, 1e-4)
    steps_per_epoch = math.ceil(
        len(train_dataset) / HYPERPARAMETERS["batch_size"]
    )
    return (
        step_seconds
        * steps_per_epoch
        * HYPERPARAMETERS["num_epochs"]
        * planned_runs
    )

def record_condition(
    results,
    regime,
    condition,
    seed,
    metrics,
    harness,
):
    metric_name = f"{condition}_{regime}_seed_{seed}_dice_score"
    dice = metrics["dice_score"]

    if not harness.check_value(dice, metric_name):
        print("SKIP: NaN/Inf detected")
        results[regime][condition][str(seed)] = {
            "status": "invalid",
            "dice_score": 0.0,
        }
        return

    harness.report_metric(metric_name, dice)
    results[regime][condition][str(seed)] = {
        "status": "success",
        **metrics,
    }
    print(
        f"condition={condition} seed={seed} regime={regime} "
        f"dice_score: {dice:.6f}"
    )
    print(
        f"condition={condition} seed={seed} regime={regime} "
        f"foreground_balanced_brier_score: "
        f"{metrics['foreground_balanced_brier_score']:.6f}"
    )

def finalize(harness, results, metadata):
    harness.finalize()
    harness_payload = None
    try:
        harness_payload = json.loads(
            Path("results.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        pass

    payload = {
        "hyperparameters": HYPERPARAMETERS,
        "metadata": metadata,
        "conditions": results,
        "harness": harness_payload,
    }
    save_machine_readable(payload, "results.json")
    save_study(results, metadata, "study_report.txt")

def main():
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        device = torch.device("cpu")

    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    harness = ExperimentHarness(
        time_budget=HYPERPARAMETERS["time_budget_seconds"]
    )
    print(f"REGISTERED_CONDITIONS: {', '.join(CONDITIONS)}")
    print(
        "METRIC_DEF: dice_score | direction=higher | "
        "desc=mean per-image lesion Dice at threshold 0.5"
    )
    print(
        f"SEED_COUNT: {len(SEEDS)} "
        f"(fixed pilot count, budget="
        f"{HYPERPARAMETERS['time_budget_seconds']}s, "
        f"conditions={len(CONDITIONS)})"
    )
    print(
        "SEED_WARNING: three seeds support a resource-limited pilot, "
        "not the preregistered definitive inference"
    )
    print(
        "MEDSAM_NOTE: the contextual MedSAM reference is excluded from "
        "this lightweight executable because no version-pinned checkpoint "
        "or compatible implementation is assumed; no proxy is substituted"
    )

    root = Path(HYPERPARAMETERS["dataset_root"])
    manifest_path = root / HYPERPARAMETERS["manifest_filename"]
    manifest = discover_isic_manifest(root)
    manifest = add_lesion_fractions(
        manifest, HYPERPARAMETERS["image_size"]
    )

    fixed_pool = stratified_sample(
        manifest[manifest["split"] == "train"],
        HYPERPARAMETERS["development_pool_size"],
        seed=421,
    )
    validation_frame = stratified_sample(
        manifest[manifest["split"] == "validation"],
        HYPERPARAMETERS["validation_limit"],
        seed=422,
    )
    test_frame = stratified_sample(
        manifest[manifest["split"] == "test"],
        HYPERPARAMETERS["test_limit"],
        seed=423,
    )
    selected_manifest = (
        fixed_pool,
        validation_frame,
        test_frame,
    )
    selected_manifest = (
        __import__("pandas").concat(
            selected_manifest, ignore_index=True
        )
    )
    validate_manifest(
        selected_manifest,
        HYPERPARAMETERS["image_size"],
        check_hashes=True,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected_manifest.to_csv(manifest_path, index=False)

    mean, std = compute_channel_statistics(
        fixed_pool, HYPERPARAMETERS["image_size"]
    )
    validation_dataset = make_dataset(
        validation_frame, mean, std, False, 0
    )
    test_dataset = make_dataset(
        test_frame, mean, std, False, 0
    )

    smallest_subset = nested_label_subset(
        fixed_pool,
        HYPERPARAMETERS["label_budget_fractions"][0],
        SEEDS[0],
    )
    pilot_dataset = make_dataset(
        smallest_subset, mean, std, True, SEEDS[0]
    )
    estimated_seconds = pilot_estimate(
        pilot_dataset,
        device,
        planned_runs=(
            len(SEEDS)
            * len(HYPERPARAMETERS["label_budget_fractions"])
            * 5
        ),
    )
    print(f"TIME_ESTIMATE: {estimated_seconds:.1f}s")

    timing_loader = make_epoch_loader(
        validation_dataset, 1, 0, 0, False
    )
    timing_example = next(iter(timing_loader))[0].to(device)
    selected_wider_widths = time_candidate_models(
        device, timing_example
    )

    results = {
        f"label_budget_{int(100 * fraction)}pct": {
            condition: {} for condition in CONDITIONS
        }
        for fraction in HYPERPARAMETERS[
            "label_budget_fractions"
        ]
    }
    metadata = {
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "manifest_path": str(manifest_path),
        "time_guard_triggered": False,
        "fatal_divergence": False,
    }

    fatal_divergence = False

    for fraction in HYPERPARAMETERS["label_budget_fractions"]:
        regime = f"label_budget_{int(100 * fraction)}pct"

        for seed in SEEDS:
            if harness.should_stop():
                metadata["time_guard_triggered"] = True
                break

            subset = nested_label_subset(
                fixed_pool, fraction, seed
            )
            train_dataset = make_dataset(
                subset, mean, std, True, seed
            )

            try:
                set_all_seeds(seed)
                compact = CompactUNetBCESoftDice(
                    HYPERPARAMETERS["compact_widths"],
                    HYPERPARAMETERS["dice_epsilon"],
                ).to(device)
                validate_model(
                    compact,
                    train_dataset,
                    device,
                    CONDITIONS[0],
                )
                compact, truncated, _ = train_model(
                    compact,
                    train_dataset,
                    validation_dataset,
                    seed,
                    device,
                    harness,
                )
                if truncated:
                    metadata["time_guard_triggered"] = True
                    break

                validation_loader = make_epoch_loader(
                    validation_dataset,
                    HYPERPARAMETERS["evaluation_batch_size"],
                    seed,
                    0,
                    False,
                )
                compact_temperature = fit_temperature(
                    compact, validation_loader, device
                )
                compact_cache = cache_test_views(
                    compact, test_dataset, device
                )

                record_condition(
                    results,
                    regime,
                    CONDITIONS[0],
                    seed,
                    metrics_from_cache(
                        compact_cache,
                        "single",
                        compact_temperature,
                    ),
                    harness,
                )
                record_condition(
                    results,
                    regime,
                    CONDITIONS[1],
                    seed,
                    metrics_from_cache(
                        compact_cache,
                        "mean_probability",
                        compact_temperature,
                    ),
                    harness,
                )

                set_all_seeds(seed)
                uniform = UniformDistanceBoundaryUNet(
                    widths=HYPERPARAMETERS["compact_widths"],
                    dice_epsilon=HYPERPARAMETERS[
                        "dice_epsilon"
                    ],
                    lambda_boundary=HYPERPARAMETERS[
                        "lambda_boundary"
                    ],
                    boundary_warmup_epochs=HYPERPARAMETERS[
                        "boundary_warmup_epochs"
                    ],
                ).to(device)
                uniform, truncated, _ = train_model(
                    uniform,
                    train_dataset,
                    validation_dataset,
                    seed,
                    device,
                    harness,
                )
                if truncated:
                    metadata["time_guard_triggered"] = True
                    break

                uniform_temperature = fit_temperature(
                    uniform, validation_loader, device
                )
                uniform_cache = cache_test_views(
                    uniform, test_dataset, device
                )
                record_condition(
                    results,
                    regime,
                    CONDITIONS[2],
                    seed,
                    metrics_from_cache(
                        uniform_cache,
                        "single",
                        uniform_temperature,
                    ),
                    harness,
                )
                record_condition(
                    results,
                    regime,
                    CONDITIONS[3],
                    seed,
                    metrics_from_cache(
                        uniform_cache,
                        "mean_probability",
                        uniform_temperature,
                    ),
                    harness,
                )

                gated, truncated = select_tau(
                    compact,
                    train_dataset,
                    validation_dataset,
                    seed,
                    device,
                    harness,
                )
                if truncated or gated is None:
                    metadata["time_guard_triggered"] = True
                    break

                gated_temperature = fit_temperature(
                    gated, validation_loader, device
                )
                gated_metrics = single_view_metrics(
                    gated,
                    test_dataset,
                    device,
                    gated_temperature,
                )
                record_condition(
                    results,
                    regime,
                    CONDITIONS[4],
                    seed,
                    gated_metrics,
                    harness,
                )

                record_condition(
                    results,
                    regime,
                    CONDITIONS[5],
                    seed,
                    metrics_from_cache(
                        compact_cache,
                        "robust_logit",
                        compact_temperature,
                    ),
                    harness,
                )

                set_all_seeds(seed)
                wider = LatencyMatchedWiderUNet(
                    selected_wider_widths,
                    HYPERPARAMETERS["dice_epsilon"],
                ).to(device)
                validate_model(
                    wider,
                    train_dataset,
                    device,
                    CONDITIONS[6],
                )
                wider, truncated, _ = train_model(
                    wider,
                    train_dataset,
                    validation_dataset,
                    seed,
                    device,
                    harness,
                )
                if truncated:
                    metadata["time_guard_triggered"] = True
                    break

                wider_temperature = fit_temperature(
                    wider, validation_loader, device
                )
                record_condition(
                    results,
                    regime,
                    CONDITIONS[6],
                    seed,
                    single_view_metrics(
                        wider,
                        test_dataset,
                        device,
                        wider_temperature,
                    ),
                    harness,
                )

            except NumericalDivergenceError:
                print("FAIL: NaN/divergence detected")
                metadata["fatal_divergence"] = True
                fatal_divergence = True
                break
            except Exception as error:
                print(
                    f"CONDITION_FAILED: regime={regime} "
                    f"seed={seed} error={str(error).replace(chr(10), ' ')}"
                )

        if (
            metadata["time_guard_triggered"]
            or fatal_divergence
        ):
            break

    baseline = CONDITIONS[0]
    for regime, regime_results in results.items():
        summary = []
        baseline_records = regime_results[baseline]

        for condition in CONDITIONS:
            records = regime_results[condition]
            successful = [
                record["dice_score"]
                for record in records.values()
                if record.get("status") == "success"
            ]
            unconditional = [
                records.get(str(seed), {}).get(
                    "dice_score", 0.0
                )
                for seed in SEEDS
            ]

            if successful:
                mean_value = float(np.mean(successful))
                std_value = float(np.std(successful))
                print(
                    f"condition={condition} regime={regime} "
                    f"dice_score_mean: {mean_value:.6f} "
                    f"dice_score_std: {std_value:.6f}"
                )
                summary.append(f"{condition}={mean_value:.6f}")
            else:
                summary.append(f"{condition}=NA")

            print(
                f"condition={condition} regime={regime} "
                f"success_rate: {len(successful)}/{len(SEEDS)}"
            )
            print(
                f"condition={condition} regime={regime} "
                f"unconditional_dice_score_mean: "
                f"{np.mean(unconditional):.6f}"
            )

            common_seeds = [
                str(seed)
                for seed in SEEDS
                if records.get(str(seed), {}).get("status")
                == "success"
                and baseline_records.get(str(seed), {}).get(
                    "status"
                )
                == "success"
            ]
            if condition != baseline and len(common_seeds) >= 2:
                paired = paired_seed_analysis(
                    [
                        records[seed]["dice_score"]
                        for seed in common_seeds
                    ],
                    [
                        baseline_records[seed]["dice_score"]
                        for seed in common_seeds
                    ],
                    seed=2026,
                )
                print(
                    f"PAIRED: {condition} vs {baseline} "
                    f"regime={regime} "
                    f"mean_diff={paired['mean_difference']:.6f} "
                    f"wilcoxon_p={paired['wilcoxon_p_value']:.6f} "
                    f"ci95=[{paired['bootstrap_ci95_low']:.6f},"
                    f"{paired['bootstrap_ci95_high']:.6f}]"
                )

        print(f"SUMMARY: regime={regime}, " + ", ".join(summary))

    finalize(harness, results, metadata)
    print(
        "EVIDENCE_SCOPE: results quantify reference-mask agreement "
        "on the declared ISIC subset only; no diagnostic or clinical "
        "claim is supported"
    )

if __name__ == "__main__":
    main()
