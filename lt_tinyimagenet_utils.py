"""Shared ImageFolder helpers for CIFAR-LT and TinyImageNet-LT scripts."""

import os

import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms


DATASET_CHOICES = ("auto", "cifar10", "cifar100", "tinyimagenet")


def infer_lt_dataset(args, n_class):
    args.n_class = n_class
    if args.dataset == "auto":
        if n_class == 10:
            args.dataset = "cifar10"
        elif n_class == 100:
            args.dataset = "cifar100"
        elif n_class == 200:
            args.dataset = "tinyimagenet"
        else:
            raise ValueError("Cannot infer dataset name from {} classes.".format(n_class))
    elif args.dataset == "cifar10" and n_class != 10:
        raise ValueError("Dataset argument is cifar10 but ImageFolder has {} classes.".format(n_class))
    elif args.dataset == "cifar100" and n_class != 100:
        raise ValueError("Dataset argument is cifar100 but ImageFolder has {} classes.".format(n_class))
    elif args.dataset == "tinyimagenet" and n_class != 200:
        raise ValueError("Dataset argument is tinyimagenet but ImageFolder has {} classes.".format(n_class))

    if not hasattr(args, "image_size") or args.image_size <= 0:
        args.image_size = 64 if args.dataset == "tinyimagenet" else 32


def imagefolder_transforms(args, train=True):
    image_size = args.image_size
    padding = 8 if image_size >= 64 else 4
    if train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def load_lt_imagefolder_dataset(args):
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            "Expected ImageFolder data at train/ and test/ under: {}".format(args.data_root))

    class_probe = datasets.ImageFolder(train_dir)
    infer_lt_dataset(args, len(class_probe.classes))
    trainset = datasets.ImageFolder(train_dir, transform=imagefolder_transforms(args, train=True))
    testset = datasets.ImageFolder(test_dir, transform=imagefolder_transforms(args, train=False))
    if trainset.classes != testset.classes:
        raise ValueError("Train/test class folders differ.")
    class_counts = np.bincount([target for _, target in trainset.samples],
                               minlength=len(trainset.classes)).astype(np.int64)
    return trainset, testset, class_counts


def get_classifier_module(model):
    net = model.module if hasattr(model, "module") else model
    if hasattr(net, "linear"):
        return net.linear
    if hasattr(net, "fc"):
        return net.fc
    return None


def adapt_model_for_image_size(model, args):
    if args.image_size <= 32:
        return model

    classifier = get_classifier_module(model)
    if classifier is None:
        raise ValueError("Cannot adapt model {}: classifier not found.".format(args.model))
    if classifier.out_features != args.n_class:
        raise ValueError("Model output classes {} do not match dataset classes {}.".format(
            classifier.out_features, args.n_class))

    scale = (args.image_size // 32) ** 2
    if scale <= 1:
        return model

    new_classifier = nn.Linear(classifier.in_features * scale, args.n_class)
    if hasattr(model, "linear"):
        model.linear = new_classifier
    elif hasattr(model, "fc"):
        model.fc = new_classifier
    return model
