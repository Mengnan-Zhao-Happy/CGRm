"""
Build CIFAR100-LT splits following LDAM/RobustLT.

The long-tailed class counts follow
    n_i = n_max * imbalance_ratio ** (-i / (num_classes - 1))
for class i = 0, ..., num_classes - 1.

By default this script creates a real ImageFolder-style dataset:
    output_dir/train/000/*.png  ... output_dir/train/099/*.png
    output_dir/test/000/*.png   ... output_dir/test/099/*.png

The zero-padded folder names are intentional: torchvision ImageFolder sorts
class folders lexicographically, so "000"..."099" preserves the numeric
CIFAR-100 class order.

It does not modify the original CIFAR-100 files.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR100


def get_args():
    parser = argparse.ArgumentParser(description="Create CIFAR100-LT ImageFolder dataset.")
    parser.add_argument("--data_dir", default="./data/cifar-data100", type=str,
                        help="Root directory containing/downloading CIFAR-100.")
    parser.add_argument("--output_dir", default="./data/CIFAR100-LT-IR50", type=str,
                        help="Directory to store the CIFAR100-LT ImageFolder dataset.")
    parser.add_argument("--imbalance_ratio", "--ir", default=50.0, type=float,
                        help="Head-class to tail-class sample ratio.")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed for selecting examples per class.")
    parser.add_argument("--max_samples_per_class", default=None, type=int,
                        help="Head-class sample count. Defaults to CIFAR-100 maximum per class.")
    parser.add_argument("--download", action="store_true", default=False,
                        help="Download CIFAR-100 if it is missing.")
    parser.add_argument("--save_npz", action="store_true", default=False,
                        help="Also save selected arrays and indices as an .npz file.")
    return parser.parse_args()


def make_long_tailed_counts(max_count, num_classes, imbalance_ratio):
    counts = []
    for class_idx in range(num_classes):
        exponent = -class_idx / (num_classes - 1.0)
        counts.append(int(max_count * (imbalance_ratio ** exponent)))
    return np.asarray(counts, dtype=np.int64)


def save_png(image_array, path):
    Image.fromarray(image_array).save(path)


def class_folder_name(class_idx):
    return "{:03d}".format(class_idx)


def build_cifar100_lt(data_dir, output_dir, imbalance_ratio, seed,
                      max_samples_per_class=None, download=False, save_npz=False):
    train_set = CIFAR100(root=data_dir, train=True, download=download)
    test_set = CIFAR100(root=data_dir, train=False, download=download)
    targets = np.asarray(train_set.targets, dtype=np.int64)
    num_classes = 100

    available_counts = np.bincount(targets, minlength=num_classes)
    if max_samples_per_class is None:
        max_samples_per_class = int(available_counts.max())
    if max_samples_per_class > int(available_counts.min()):
        raise ValueError(
            "max_samples_per_class={} exceeds available per-class count {}."
            .format(max_samples_per_class, int(available_counts.min())))

    class_counts = make_long_tailed_counts(
        max_samples_per_class, num_classes, imbalance_ratio)
    rng = np.random.default_rng(seed)

    selected_indices = []
    per_class_indices = {}
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    for split_dir in (train_dir, test_dir):
        for class_idx in range(num_classes):
            os.makedirs(os.path.join(split_dir, class_folder_name(class_idx)), exist_ok=True)

    for class_idx, count in enumerate(class_counts):
        indices = np.flatnonzero(targets == class_idx)
        rng.shuffle(indices)
        chosen = np.sort(indices[:count])
        per_class_indices[str(class_idx)] = chosen.astype(int).tolist()
        selected_indices.append(chosen)
        folder = class_folder_name(class_idx)
        for source_idx in chosen:
            save_png(
                train_set.data[source_idx],
                os.path.join(train_dir, folder, "{:05d}.png".format(source_idx)),
            )

    for source_idx, class_idx in enumerate(test_set.targets):
        folder = class_folder_name(class_idx)
        save_png(
            test_set.data[source_idx],
            os.path.join(test_dir, folder, "{:05d}.png".format(source_idx)),
        )

    selected_indices = np.concatenate(selected_indices).astype(np.int64)
    rng.shuffle(selected_indices)
    selected_targets = targets[selected_indices]

    os.makedirs(output_dir, exist_ok=True)
    stem = "cifar100_lt_ir{}_seed{}".format(
        str(imbalance_ratio).replace(".", "p"), seed)
    npz_path = os.path.join(output_dir, stem + ".npz") if save_npz else None
    json_path = os.path.join(output_dir, "metadata.json")

    if save_npz:
        np.savez(
            npz_path,
            indices=selected_indices,
            data=train_set.data[selected_indices],
            targets=selected_targets,
            class_counts=class_counts,
            available_counts=available_counts,
            imbalance_ratio=np.asarray(imbalance_ratio),
            seed=np.asarray(seed),
        )

    test_targets = np.asarray(test_set.targets, dtype=np.int64)
    metadata = {
        "dataset": "CIFAR100-LT",
        "format": "ImageFolder",
        "source_dataset": "CIFAR100 train",
        "root": os.path.abspath(output_dir),
        "train_dir": os.path.abspath(train_dir),
        "test_dir": os.path.abspath(test_dir),
        "formula": "n_i = n_max * imbalance_ratio ** (-i / (num_classes - 1))",
        "num_classes": num_classes,
        "class_names": list(train_set.classes),
        "class_folder_names": [class_folder_name(i) for i in range(num_classes)],
        "imbalance_ratio": imbalance_ratio,
        "seed": seed,
        "max_samples_per_class": max_samples_per_class,
        "total_train": int(len(selected_indices)),
        "total_test": int(len(test_set)),
        "available_counts": available_counts.astype(int).tolist(),
        "class_counts_train": class_counts.astype(int).tolist(),
        "class_counts_test": np.bincount(test_targets, minlength=num_classes).astype(int).tolist(),
        "npz_path": os.path.abspath(npz_path) if npz_path is not None else None,
        "per_class_source_indices": per_class_indices,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved ImageFolder dataset:", os.path.abspath(output_dir))
    print("Train dir:", os.path.abspath(train_dir))
    print("Test dir:", os.path.abspath(test_dir))
    if npz_path is not None:
        print("Saved:", os.path.abspath(npz_path))
    print("Saved:", os.path.abspath(json_path))
    print("Class counts:", class_counts.tolist())
    print("Total train:", int(len(selected_indices)))
    print("Total test:", int(len(test_set)))

    return npz_path, json_path


def main():
    args = get_args()
    build_cifar100_lt(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        imbalance_ratio=args.imbalance_ratio,
        seed=args.seed,
        max_samples_per_class=args.max_samples_per_class,
        download=args.download,
        save_npz=args.save_npz,
    )


if __name__ == "__main__":
    main()
