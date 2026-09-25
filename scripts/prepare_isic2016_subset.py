"""Materialize a small, deterministic ISIC 2016 subset for local experiments."""

from pathlib import Path
import shutil

from datasets import load_dataset
from PIL import Image


OUTPUT = Path("data/isic2016_subset")
COUNTS = {"train": 180, "test": 60}
SIZE = (128, 128)


def main() -> None:
    dataset = load_dataset("MedOtter/ISIC2016", streaming=True)
    for split, count in COUNTS.items():
        image_dir = OUTPUT / split / "images"
        mask_dir = OUTPUT / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for row in dataset[split]:
            image_id = str(row["image_id"])
            image = row["image"].convert("RGB").resize(SIZE, Image.Resampling.BILINEAR)
            mask = row["mask"].convert("L").resize(SIZE, Image.Resampling.NEAREST)
            mask = mask.point(lambda value: 255 if value > 0 else 0)
            image.save(image_dir / f"{image_id}.jpg", quality=90)
            mask.save(mask_dir / f"{image_id}.png")
            written += 1
            if written >= count:
                break
        print(f"{split}: {written} image-mask pairs")

    # Materialize deterministic train/validation/test folders expected by the
    # generated experiment. The original streamed subset remains untouched.
    experiment_root = Path("data/isic2016_experiment")
    partitions = {
        "train": list((OUTPUT / "train" / "images").glob("*.jpg"))[:140],
        "validation": list((OUTPUT / "train" / "images").glob("*.jpg"))[140:180],
        "test": list((OUTPUT / "test" / "images").glob("*.jpg"))[:60],
    }
    for split, image_paths in partitions.items():
        image_dir = experiment_root / split / "images"
        mask_dir = experiment_root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            mask_path = image_path.parents[1] / "masks" / f"{image_path.stem}.png"
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(mask_path, mask_dir / mask_path.name)
        print(f"experiment/{split}: {len(image_paths)} pairs")


if __name__ == "__main__":
    main()
