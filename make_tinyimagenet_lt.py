"""
Build TinyImageNet-LT splits following LDAM/RobustLT.

The long-tailed class counts follow
    n_i = n_max * imbalance_ratio ** (-i / (num_classes - 1))
for class i = 0, ..., num_classes - 1.

By default this script creates a real ImageFolder-style dataset:
    output_dir/train/000_n02124075/*.JPEG
    output_dir/test/000_n02124075/*.JPEG

TinyImageNet's official test split has no public labels, so this script uses
the official validation split as output_dir/test. It does not modify the
original tiny-imagenet-200 directory.
"""

import argparse
import json
import os
import shutil

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="Create TinyImageNet-LT ImageFolder dataset.")
    parser.add_argument("--data_dir", default="./data/tiny-imagenet-200", type=str,
                        help="Root directory of the extracted tiny-imagenet-200 dataset.")
    parser.add_argument("--output_dir", default="./data/TinyImageNet-LT-IR50", type=str,
                        help="Directory to store the TinyImageNet-LT ImageFolder dataset.")
    parser.add_argument("--imbalance_ratio", "--ir", default=50.0, type=float,
                        help="Head-class to tail-class sample ratio.")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed for selecting examples per class.")
    parser.add_argument("--max_samples_per_class", default=None, type=int,
                        help="Head-class sample count. Defaults to the minimum available train count.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Remove output_dir before building the dataset.")
    parser.add_argument("--save_npz", action="store_true", default=False,
                        help="Also save selected relative paths and labels as an .npz file.")
    return parser.parse_args()


def make_long_tailed_counts(max_count, num_classes, imbalance_ratio):
    counts = []
    for class_idx in range(num_classes):
        exponent = -class_idx / (num_classes - 1.0)
        counts.append(int(max_count * (imbalance_ratio ** exponent)))
    return np.asarray(counts, dtype=np.int64)


def read_wnids(data_dir):
    wnids_path = os.path.join(data_dir, "wnids.txt")
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError("Missing wnids.txt under {}".format(data_dir))
    with open(wnids_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_words(data_dir):
    words_path = os.path.join(data_dir, "words.txt")
    words = {}
    if not os.path.isfile(words_path):
        return words
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                words[parts[0]] = parts[1]
    return words


def class_folder_name(class_idx, wnid):
    return "{:03d}_{}".format(class_idx, wnid)


def list_train_images(data_dir, wnid):
    image_dir = os.path.join(data_dir, "train", wnid, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError("Missing train image directory: {}".format(image_dir))
    images = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".jpeg", ".jpg", ".png"))
    ]
    return sorted(images)


def read_val_annotations(data_dir):
    annotations_path = os.path.join(data_dir, "val", "val_annotations.txt")
    val_image_dir = os.path.join(data_dir, "val", "images")
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError("Missing val_annotations.txt under {}".format(data_dir))
    if not os.path.isdir(val_image_dir):
        raise FileNotFoundError("Missing val/images under {}".format(data_dir))

    val_by_wnid = {}
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, wnid = parts[0], parts[1]
            image_path = os.path.join(val_image_dir, image_name)
            if os.path.isfile(image_path):
                val_by_wnid.setdefault(wnid, []).append(image_path)
    for wnid in val_by_wnid:
        val_by_wnid[wnid].sort()
    return val_by_wnid


def prepare_output_dir(output_dir, overwrite):
    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        elif os.listdir(output_dir):
            raise FileExistsError(
                "{} is not empty. Use --overwrite or choose a new --output_dir."
                .format(output_dir))
    os.makedirs(output_dir, exist_ok=True)


def copy_images(image_paths, dst_dir, prefix=None):
    os.makedirs(dst_dir, exist_ok=True)
    copied_relpaths = []
    for idx, src_path in enumerate(image_paths):
        name = os.path.basename(src_path)
        if prefix is not None:
            name = "{}_{:05d}_{}".format(prefix, idx, name)
        dst_path = os.path.join(dst_dir, name)
        shutil.copy2(src_path, dst_path)
        copied_relpaths.append(dst_path)
    return copied_relpaths


def build_tinyimagenet_lt(data_dir, output_dir, imbalance_ratio, seed,
                          max_samples_per_class=None, overwrite=False,
                          save_npz=False):
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1.")

    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    train_root = os.path.join(data_dir, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError("Missing TinyImageNet train directory: {}".format(train_root))

    wnids = read_wnids(data_dir)
    words = read_words(data_dir)
    num_classes = len(wnids)
    if num_classes == 0:
        raise ValueError("No classes found in wnids.txt.")

    train_images_by_class = [list_train_images(data_dir, wnid) for wnid in wnids]
    available_counts = np.asarray([len(images) for images in train_images_by_class],
                                  dtype=np.int64)
    if max_samples_per_class is None:
        max_samples_per_class = int(available_counts.min())
    if max_samples_per_class > int(available_counts.min()):
        raise ValueError(
            "max_samples_per_class={} exceeds minimum available train count {}."
            .format(max_samples_per_class, int(available_counts.min())))

    class_counts = make_long_tailed_counts(
        max_samples_per_class, num_classes, imbalance_ratio)
    if np.any(class_counts <= 0):
        raise ValueError(
            "Some classes would receive zero images. Reduce --imbalance_ratio "
            "or increase --max_samples_per_class.")

    val_by_wnid = read_val_annotations(data_dir)
    missing_val = [wnid for wnid in wnids if wnid not in val_by_wnid]
    if missing_val:
        raise ValueError("Validation annotations missing classes: {}".format(missing_val))

    prepare_output_dir(output_dir, overwrite)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    rng = np.random.default_rng(seed)
    selected_train_relpaths = []
    selected_train_labels = []
    per_class_source_files = {}

    for class_idx, (wnid, count) in enumerate(zip(wnids, class_counts)):
        folder = class_folder_name(class_idx, wnid)
        os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        indices = np.arange(len(train_images_by_class[class_idx]))
        rng.shuffle(indices)
        chosen_indices = np.sort(indices[:count])
        chosen_paths = [train_images_by_class[class_idx][int(i)] for i in chosen_indices]
        per_class_source_files[wnid] = [os.path.relpath(path, data_dir) for path in chosen_paths]

        copied_paths = copy_images(chosen_paths, os.path.join(train_dir, folder))
        selected_train_relpaths.extend(os.path.relpath(path, output_dir) for path in copied_paths)
        selected_train_labels.extend([class_idx] * len(copied_paths))

        copy_images(val_by_wnid[wnid], os.path.join(test_dir, folder))

    selected_train_relpaths = np.asarray(selected_train_relpaths)
    selected_train_labels = np.asarray(selected_train_labels, dtype=np.int64)
    test_counts = np.asarray([len(val_by_wnid[wnid]) for wnid in wnids], dtype=np.int64)

    stem = "tinyimagenet_lt_ir{}_seed{}".format(
        str(imbalance_ratio).replace(".", "p"), seed)
    npz_path = os.path.join(output_dir, stem + ".npz") if save_npz else None
    json_path = os.path.join(output_dir, "metadata.json")

    if save_npz:
        np.savez(
            npz_path,
            paths=selected_train_relpaths,
            targets=selected_train_labels,
            class_counts=class_counts,
            available_counts=available_counts,
            imbalance_ratio=np.asarray(imbalance_ratio),
            seed=np.asarray(seed),
        )

    metadata = {
        "dataset": "TinyImageNet-LT",
        "format": "ImageFolder",
        "source_dataset": "tiny-imagenet-200 train",
        "test_source": "tiny-imagenet-200 val",
        "root": output_dir,
        "train_dir": os.path.abspath(train_dir),
        "test_dir": os.path.abspath(test_dir),
        "formula": "n_i = n_max * imbalance_ratio ** (-i / (num_classes - 1))",
        "num_classes": num_classes,
        "class_wnids": wnids,
        "class_names": [words.get(wnid, "") for wnid in wnids],
        "class_folder_names": [
            class_folder_name(class_idx, wnid) for class_idx, wnid in enumerate(wnids)
        ],
        "imbalance_ratio": imbalance_ratio,
        "seed": seed,
        "max_samples_per_class": max_samples_per_class,
        "total_train": int(class_counts.sum()),
        "total_test": int(test_counts.sum()),
        "available_counts": available_counts.astype(int).tolist(),
        "class_counts_train": class_counts.astype(int).tolist(),
        "class_counts_test": test_counts.astype(int).tolist(),
        "npz_path": os.path.abspath(npz_path) if npz_path is not None else None,
        "per_class_source_files": per_class_source_files,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved ImageFolder dataset:", output_dir)
    print("Train dir:", os.path.abspath(train_dir))
    print("Test dir:", os.path.abspath(test_dir))
    if npz_path is not None:
        print("Saved:", os.path.abspath(npz_path))
    print("Saved:", os.path.abspath(json_path))
    print("Class counts:", class_counts.astype(int).tolist())
    print("Total train:", int(class_counts.sum()))
    print("Total test:", int(test_counts.sum()))

    return npz_path, json_path


def main():
    args = get_args()
    build_tinyimagenet_lt(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        imbalance_ratio=args.imbalance_ratio,
        seed=args.seed,
        max_samples_per_class=args.max_samples_per_class,
        overwrite=args.overwrite,
        save_npz=args.save_npz,
    )


if __name__ == "__main__":
    main()
