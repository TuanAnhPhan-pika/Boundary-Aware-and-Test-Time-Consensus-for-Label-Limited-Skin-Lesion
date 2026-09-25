"""ISIC 2016 subset discovery, integrity checks, and deterministic loading."""

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fft import dctn
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except ImportError:
    Image = None

SPLIT_DIRECTORIES = {
    "train": (
        "train/images",
        "train/masks",
    ),
    "validation": (
        "validation/images",
        "validation/masks",
    ),
    "test": (
        "test/images",
        "test/masks",
    ),
}

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _candidate_files(directory):
    extensions = {".jpg", ".jpeg", ".png", ".npy"}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

def discover_isic_manifest(root):
    """
    Discover the prepared ISIC 2016 subset without downloading data.

    Test masks are required; the code never creates a random replacement for
    a missing prepared partition.
    """
    root = Path(root)
    records = []

    for split, (image_name, mask_name) in SPLIT_DIRECTORIES.items():
        image_directory = root / image_name
        mask_directory = root / mask_name
        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"missing prepared {split} image directory: "
                f"{image_directory}"
            )
        if not mask_directory.is_dir():
            raise FileNotFoundError(
                f"missing prepared {split} mask directory: "
                f"{mask_directory}"
            )

        masks_by_stem = {}
        for mask_path in _candidate_files(mask_directory):
            mask_stem = mask_path.stem
            if mask_stem.endswith("_segmentation"):
                mask_stem = mask_stem[: -len("_segmentation")]
            masks_by_stem[mask_stem] = mask_path

        for image_path in _candidate_files(image_directory):
            image_id = image_path.stem
            mask_path = masks_by_stem.get(image_id)
            if mask_path is None:
                raise FileNotFoundError(
                    f"no Task 1 mask for {image_id}"
                )
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "split": split,
                    "patient_id": "",
                }
            )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise RuntimeError("ISIC manifest is empty")
    if manifest["image_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["image_id"].duplicated(), "image_id"
        ].tolist()
        raise ValueError(f"duplicate image identifiers: {duplicates[:5]}")
    return manifest

def _load_array(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)

    if Image is None:
        raise ImportError(
            "Pillow is required to decode the JPEG/PNG files. "
            "Alternatively provide losslessly decoded .npy arrays."
        )
    with Image.open(path) as image:
        return np.asarray(image)

def _resize_array(array, size, is_mask):
    if array.ndim == 2:
        tensor = torch.from_numpy(
            np.ascontiguousarray(array)
        ).float()[None, None]
    elif array.ndim == 3:
        tensor = torch.from_numpy(
            np.ascontiguousarray(array)
        ).permute(2, 0, 1).float()[None]
    else:
        raise ValueError(f"unsupported array shape: {array.shape}")

    mode = "nearest" if is_mask else "bilinear"
    kwargs = {
        "size": (size, size),
        "mode": mode,
    }
    if not is_mask:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True

    resized = F.interpolate(tensor, **kwargs).squeeze(0)
    if is_mask:
        return resized.squeeze(0).numpy()
    return resized.permute(1, 2, 0).numpy()

def load_resized_pair(image_path, mask_path, size):
    image = _load_array(image_path)
    mask = _load_array(mask_path)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"invalid RGB image shape: {image.shape}")
    if image.shape[2] > 3:
        image = image[:, :, :3]

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"invalid mask shape: {mask.shape}")

    image = _resize_array(image, size, is_mask=False)
    mask = _resize_array(mask, size, is_mask=True)

    if image.max() > 1.0:
        image = image / 255.0
    if mask.max() > 1.0:
        mask = mask / 255.0

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    mask = (mask >= 0.5).astype(np.float32)
    return image, mask

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def perceptual_hash(image):
    if image.ndim == 3:
        gray = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        )
    else:
        gray = image

    small = _resize_array(gray, 32, is_mask=False)
    if small.ndim == 3:
        small = small[:, :, 0]
    coefficients = dctn(small, type=2, norm="ortho")[:8, :8]
    median = np.median(coefficients[1:])
    bits = coefficients >= median
    return "".join("1" if value else "0" for value in bits.ravel())

def add_lesion_fractions(manifest, size):
    fractions = []
    dimensions = []
    for row in manifest.itertuples(index=False):
        image, mask = load_resized_pair(
            row.image_path, row.mask_path, size
        )
        fractions.append(float(mask.mean()))
        dimensions.append(
            f"{image.shape[0]}x{image.shape[1]}"
        )

    result = manifest.copy()
    result["lesion_fraction"] = fractions
    result["evaluation_dimensions"] = dimensions
    return result

def validate_manifest(manifest, size, check_hashes=True):
    required = {
        "image_id",
        "image_path",
        "mask_path",
        "split",
        "patient_id",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest columns missing: {sorted(missing)}")

    allowed_splits = {"train", "validation", "test"}
    observed_splits = set(manifest["split"].unique())
    if observed_splits != allowed_splits:
        raise ValueError(
            f"expected prepared splits {allowed_splits}, "
            f"observed {observed_splits}"
        )

    for row in manifest.itertuples(index=False):
        if not Path(row.image_path).is_file():
            raise FileNotFoundError(row.image_path)
        if not Path(row.mask_path).is_file():
            raise FileNotFoundError(row.mask_path)

        image, mask = load_resized_pair(
            row.image_path, row.mask_path, size
        )
        if image.shape != (size, size, 3):
            raise ValueError(
                f"invalid image dimensions for {row.image_id}"
            )
        if mask.shape != (size, size):
            raise ValueError(
                f"invalid mask dimensions for {row.image_id}"
            )
        unique = set(np.unique(mask).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"non-binary mask: {row.image_id}")
        if mask.sum() == 0:
            raise ValueError(f"empty mask: {row.image_id}")

    if check_hashes:
        hashes = {}
        for row in manifest.itertuples(index=False):
            digest = file_sha256(row.image_path)
            if digest in hashes:
                previous = hashes[digest]
                raise ValueError(
                    f"exact duplicate across manifest: "
                    f"{previous} and {row.image_id}"
                )
            hashes[digest] = row.image_id

    patient_rows = manifest[
        manifest["patient_id"].astype(str).str.len() > 0
    ]
    if not patient_rows.empty:
        patient_split_counts = (
            patient_rows.groupby("patient_id")["split"].nunique()
        )
        if (patient_split_counts > 1).any():
            raise ValueError("patient overlap across prepared splits")

def stratified_sample(frame, count, seed):
    if count >= len(frame):
        return frame.copy().reset_index(drop=True)

    working = frame.copy()
    quantiles = min(3, working["lesion_fraction"].nunique())
    if quantiles < 2:
        working["area_stratum"] = 0
    else:
        working["area_stratum"] = pd.qcut(
            working["lesion_fraction"],
            q=quantiles,
            labels=False,
            duplicates="drop",
        )

    rng = np.random.default_rng(seed)
    selected = []
    groups = list(working.groupby("area_stratum"))
    remaining = count

    for group_index, (_, group) in enumerate(groups):
        groups_left = len(groups) - group_index
        target = min(
            len(group),
            max(1, int(round(remaining / groups_left))),
        )
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        selected.extend(indices[:target].tolist())
        remaining = count - len(selected)

    if len(selected) < count:
        unused = np.setdiff1d(
            working.index.to_numpy(),
            np.asarray(selected),
        )
        rng.shuffle(unused)
        selected.extend(unused[: count - len(selected)].tolist())

    return working.loc[selected[:count]].reset_index(drop=True)

def nested_label_subset(pool, fraction, seed):
    working = pool.copy()
    quantiles = min(3, working["lesion_fraction"].nunique())
    if quantiles < 2:
        working["area_stratum"] = 0
    else:
        working["area_stratum"] = pd.qcut(
            working["lesion_fraction"],
            q=quantiles,
            labels=False,
            duplicates="drop",
        )

    selected = []
    rng = np.random.default_rng(seed)
    for _, group in working.groupby("area_stratum"):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        count = max(1, int(math.ceil(len(indices) * fraction)))
        selected.extend(indices[:count].tolist())

    return working.loc[selected].reset_index(drop=True)

def compute_channel_statistics(frame, size):
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for row in frame.itertuples(index=False):
        image, _ = load_resized_pair(
            row.image_path, row.mask_path, size
        )
        pixels = image.reshape(-1, 3).astype(np.float64)
        channel_sum += pixels.sum(axis=0)
        channel_square_sum += np.square(pixels).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = channel_sum / pixel_count
    variance = (
        channel_square_sum / pixel_count - np.square(mean)
    )
    std = np.sqrt(np.maximum(variance, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)

class ISICSegmentationDataset(Dataset):
    """In-memory resized ISIC image/mask dataset with deterministic augmentation."""

    def __init__(
        self,
        frame,
        image_size,
        mean,
        std,
        distance_clip_pixels,
        augment,
        augmentation_seed,
        affine_degrees,
        affine_translation,
        color_jitter,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.distance_clip_pixels = distance_clip_pixels
        self.augment = augment
        self.augmentation_seed = augmentation_seed
        self.affine_degrees = affine_degrees
        self.affine_translation = affine_translation
        self.color_jitter = color_jitter
        self.epoch = 0

        self.images = []
        self.masks = []
        self.image_ids = []

        for row in self.frame.itertuples(index=False):
            image, mask = load_resized_pair(
                row.image_path,
                row.mask_path,
                image_size,
            )
            self.images.append(
                torch.from_numpy(image).permute(2, 0, 1)
            )
            self.masks.append(
                torch.from_numpy(mask).unsqueeze(0)
            )
            self.image_ids.append(row.image_id)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.images)

    def _augment(self, image, mask, index):
        seed = (
            self.augmentation_seed * 1_000_003
            + self.epoch * 10_007
            + index
        )
        rng = np.random.default_rng(seed)

        if rng.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if rng.random() < 0.5:
            image = torch.flip(image, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))

        angle = np.deg2rad(
            rng.uniform(-self.affine_degrees, self.affine_degrees)
        )
        shift_x = rng.uniform(
            -self.affine_translation, self.affine_translation
        )
        shift_y = rng.uniform(
            -self.affine_translation, self.affine_translation
        )
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        theta = torch.tensor(
            [
                [cosine, -sine, shift_x],
                [sine, cosine, shift_y],
            ],
            dtype=image.dtype,
        ).unsqueeze(0)

        grid = F.affine_grid(
            theta,
            size=(1, 3, self.image_size, self.image_size),
            align_corners=False,
        )
        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        brightness = 1.0 + rng.uniform(
            -self.color_jitter, self.color_jitter
        )
        contrast = 1.0 + rng.uniform(
            -self.color_jitter, self.color_jitter
        )
        channel_mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (
            (image - channel_mean) * contrast + channel_mean
        ) * brightness
        return image.clamp(0.0, 1.0), (mask >= 0.5).float()

    def __getitem__(self, index):
        image = self.images[index].clone()
        mask = self.masks[index].clone()

        if self.augment:
            image, mask = self._augment(image, mask, index)

        binary = mask.squeeze(0).numpy() >= 0.5
        outside = distance_transform_edt(~binary)
        inside = distance_transform_edt(binary)
        signed_distance = np.clip(
            outside - inside,
            -self.distance_clip_pixels,
            self.distance_clip_pixels,
        ).astype(np.float32)
        distance = torch.from_numpy(
            signed_distance
        ).unsqueeze(0)

        image = (image - self.mean) / self.std
        return image, mask, distance, self.image_ids[index]

def make_epoch_loader(
    dataset,
    batch_size,
    training_seed,
    epoch,
    shuffle,
):
    dataset.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(training_seed * 1009 + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
