import json
import os
from pathlib import Path

from PIL import Image


DATASET_ROOT = Path("FH-V1_WITHOUT_AUG_DATASET")
TRAIN_ROOT = DATASET_ROOT / "train"
TEST_ROOT = DATASET_ROOT / "test"
OUTPUT_DIR = Path("styled_data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (256, 256)


def list_person_ids(root: Path) -> set[str]:
    if not root.exists():
        return set()

    return {
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
    }


def process_split(root: Path, person: str, label_id: int, labels: list[list], count: int) -> int:
    person_path = root / person
    if not person_path.exists():
        return count

    for image_path in sorted(person_path.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)

            filename = f"{count:05d}.png"
            img.save(OUTPUT_DIR / filename)

        labels.append([filename, label_id])
        count += 1

    return count


def main() -> None:
    if not TRAIN_ROOT.exists() and not TEST_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folders not found. Expected '{TRAIN_ROOT}' and/or '{TEST_ROOT}'."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    labels: list[list] = []
    count = 0

    persons = sorted(list_person_ids(TRAIN_ROOT) | list_person_ids(TEST_ROOT), key=int)

    for label_id, person in enumerate(persons):
        count = process_split(TRAIN_ROOT, person, label_id, labels, count)
        count = process_split(TEST_ROOT, person, label_id, labels, count)

    print("Total images:", count)

    with open(OUTPUT_DIR / "dataset.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f)


if __name__ == "__main__":
    main()
