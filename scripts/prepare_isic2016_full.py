"""Prepare a deterministic full ISIC 2016 segmentation benchmark locally."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from datasets import load_dataset
from PIL import Image


OUTPUT = Path("data/isic2016_full")
VALIDATION_COUNT = 180


def stable_rank(image_id: str) -> str:
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def save_pair(row: dict, split: str) -> str:
    image_id = str(row["image_id"])
    image_dir = OUTPUT / split / "images"
    mask_dir = OUTPUT / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = row["image"].convert("RGB")
    mask = row["mask"].convert("L")
    mask = mask.point(lambda value: 255 if value > 0 else 0)
    image.save(image_dir / f"{image_id}.jpg", quality=95)
    mask.save(mask_dir / f"{image_id}.png")
    return image_id


def main() -> None:
    dataset = load_dataset("MedOtter/ISIC2016", streaming=True)

    source_ids: list[str] = []
    for index, row in enumerate(dataset["train"], start=1):
        source_ids.append(save_pair(row, "source_train"))
        if index % 100 == 0:
            print(f"downloaded train: {index}", flush=True)

    validation_ids = set(sorted(source_ids, key=stable_rank)[:VALIDATION_COUNT])
    for image_id in source_ids:
        destination = "validation" if image_id in validation_ids else "train"
        for folder, suffix in (("images", ".jpg"), ("masks", ".png")):
            source = OUTPUT / "source_train" / folder / f"{image_id}{suffix}"
            target_dir = OUTPUT / destination / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target_dir / source.name))
    shutil.rmtree(OUTPUT / "source_train")

    test_ids: list[str] = []
    for index, row in enumerate(dataset["test"], start=1):
        test_ids.append(save_pair(row, "test"))
        if index % 100 == 0:
            print(f"downloaded test: {index}", flush=True)

    print(
        f"prepared train={len(source_ids) - len(validation_ids)}, "
        f"validation={len(validation_ids)}, test={len(test_ids)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
