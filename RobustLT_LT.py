"""
Standalone RobustLT training on CIFAR-LT ImageFolder datasets.

This file keeps RobustLT.py unchanged. It reuses the RobustLT CPB + AIW loss
implementation, but infers CIFAR10/CIFAR100 from ImageFolder class count.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from RobustLT import (
    adjust_learning_rate,
    aiw_intensity,
    compute_cpb_eps,
    compute_tail_classes,
    evaluate,
    train_epoch,
)
from utils import get_model


def get_args():
    parser = argparse.ArgumentParser(description="RobustLT training for CIFAR-LT ImageFolder data")

    parser.add_argument("--data_root", default="./data/CIFAR100-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str, choices=("auto", "cifar10", "cifar100"))
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/robustlt_lt")
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_freq", default=10, type=int)

    parser.add_argument("--weight_decay", "--wd", default=5e-4, type=float)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_schedule", type=str, default="bag_of_tricks",
                        choices=("trades", "bag_of_tricks", "madry"))
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)

    parser.add_argument("--base_algorithm", "--loss", dest="base_algorithm",
                        default="trades", type=str, choices=("pgd", "trades"))
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--beta", default=4.0, type=float)

    parser.add_argument("--robustlt_alpha", "--alpha", default=0.3, type=float)
    parser.add_argument("--robustlt_beta", "--aiw_beta", default=0.8, type=float)
    parser.add_argument("--disable_aiw", action="store_true", default=False)
    parser.add_argument("--tail_fraction", default=0.8, type=float)

    return parser.parse_args()


def cifar_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def load_imagefolder_dataset(args):
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            "Expected ImageFolder data at train/ and test/ under: {}".format(args.data_root))

    trainset = datasets.ImageFolder(train_dir, transform=cifar_transforms(train=True))
    testset = datasets.ImageFolder(test_dir, transform=cifar_transforms(train=False))
    class_counts = np.bincount([target for _, target in trainset.samples],
                               minlength=len(trainset.classes)).astype(np.int64)
    return trainset, testset, class_counts


def infer_dataset(args, n_class):
    args.n_class = n_class
    if args.dataset == "auto":
        if n_class == 10:
            args.dataset = "cifar10"
        elif n_class == 100:
            args.dataset = "cifar100"
        else:
            raise ValueError("Cannot infer dataset name from {} classes.".format(n_class))
    elif args.dataset == "cifar10" and n_class != 10:
        raise ValueError("Dataset argument is cifar10 but ImageFolder has {} classes.".format(n_class))
    elif args.dataset == "cifar100" and n_class != 100:
        raise ValueError("Dataset argument is cifar100 but ImageFolder has {} classes.".format(n_class))


def main():
    args = get_args()

    trainset, testset, class_counts = load_imagefolder_dataset(args)
    infer_dataset(args, len(trainset.classes))

    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, "training.log")),
            logging.StreamHandler()
        ])
    logging.info("Args: %s", args)
    logging.info("Detected dataset: %s, n_class: %d", args.dataset, args.n_class)
    logging.info("Train class counts: %s", class_counts.tolist())

    final_checkpoint_path = os.path.join(args.model_dir, "final.pt")
    if not args.overwrite and os.path.exists(final_checkpoint_path):
        logging.info("Final checkpoint found - quitting. Use --overwrite to train again.")
        sys.exit(0)

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)
    cpb_eps = compute_cpb_eps(class_counts, args.epsilon, args.robustlt_alpha, device)
    logging.info("Tail classes: %s", tail_classes.tolist())
    logging.info("CPB final class eps: %s", (torch.round(cpb_eps.cpu() * 1000000) / 1000000).tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = get_model(args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        intensity = aiw_intensity(epoch, args.epochs, args.robustlt_beta, args.disable_aiw)
        class_eps = cpb_eps * intensity

        logging.info("Setting learning rate to %g", lr)
        logging.info("AIW intensity %.6f, class eps: %s",
                     intensity, (torch.round(class_eps.cpu() * 1000000) / 1000000).tolist())

        train_epoch(args, model, device, optimizer, train_loader, epoch, class_eps, len(trainset))

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            eval_data = evaluate(args, model, device, test_loader, tail_classes)
            robust_acc = eval_data["robust_acc"]
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                torch.save({"state_dict": model.state_dict()}, os.path.join(args.model_dir, "best.pt"))
                logging.info("Saved best checkpoint: epoch %d, PGD robust accuracy %.2f%%",
                             epoch, 100.0 * best_robust_acc)
            logging.info(120 * "=")

        if epoch == args.epochs:
            torch.save({"state_dict": model.state_dict()}, os.path.join(args.model_dir, "final.pt"))
            logging.info("Saved final checkpoint: epoch %d", epoch)

        elapsed_time = time.time() - init_time
        print("elapsed time : %d h %d m %d s" % (
            elapsed_time / 3600, (elapsed_time % 3600) / 60, elapsed_time % 60))


if __name__ == "__main__":
    main()
