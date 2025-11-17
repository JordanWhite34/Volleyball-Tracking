from __future__ import annotations

import math
import shutil
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple


def _collect_pairs(dataset_dir: Path) -> List[Tuple[Path, Path]]:
    """Return (image_path, label_path) tuples from every existing split."""
    pairs: List[Tuple[Path, Path]] = []
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue
        if not label_dir.exists():
            raise ValueError(f"Missing labels directory for split '{split}' in {dataset_dir}")
        for img_path in sorted(img_dir.glob("*")):
            if not img_path.is_file():
                continue
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {img_path}")
            pairs.append((img_path, label_path))
    if not pairs:
        raise ValueError(f"No images found under {dataset_dir}")
    return pairs


def _compute_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        raise ValueError("Total images must be positive")
    exacts: List[Tuple[str, float, int]] = []
    base_total = 0
    for split, ratio in ratios.items():
        exact = total * ratio
        base = math.floor(exact)
        exacts.append((split, exact - base, base))
        base_total += base
    remainder = total - base_total
    # Distribute leftover images to splits with largest fractional remainders
    exacts.sort(key=lambda item: item[1], reverse=True)
    idx = 0
    while remainder > 0:
        split, _, base = exacts[idx % len(exacts)]
        base += 1
        exacts[idx % len(exacts)] = (split, _, base)
        remainder -= 1
        idx += 1
    return {split: base for split, _, base in exacts}


def rebalance_dataset(dataset_dir: Path, ratios: Dict[str, float], seed: int = 42) -> Dict[str, int]:
    pairs = _collect_pairs(dataset_dir)
    rng = Random(seed)
    rng.shuffle(pairs)

    counts = _compute_counts(len(pairs), ratios)
    tmp_root = dataset_dir / "_rebalance_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    for split in ratios:
        (tmp_root / split / "images").mkdir(parents=True, exist_ok=True)
        (tmp_root / split / "labels").mkdir(parents=True, exist_ok=True)

    start = 0
    for split in ratios:
        end = start + counts[split]
        for img_path, label_path in pairs[start:end]:
            shutil.move(str(img_path), str(tmp_root / split / "images" / img_path.name))
            shutil.move(str(label_path), str(tmp_root / split / "labels" / label_path.name))
        start = end

    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        target_dir = tmp_root / split
        if target_dir.exists():
            shutil.move(str(target_dir), str(split_dir))
    shutil.rmtree(tmp_root, ignore_errors=True)
    return counts


def main() -> None:
    repo_root = Path(__file__).parent
    ratios = {"train": 0.70, "valid": 0.15, "test": 0.15}
    for dataset in ("data/ball", "data/people"):
        dataset_dir = repo_root / dataset
        counts = rebalance_dataset(dataset_dir, ratios)
        total = sum(counts.values())
        print(f"{dataset}: {total} images -> "
              f"train {counts['train']}, valid {counts['valid']}, test {counts['test']}")


if __name__ == "__main__":
    main()
