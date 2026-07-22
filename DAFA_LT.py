"""
Standalone DAFA training on CIFAR-LT ImageFolder datasets.

This file keeps DAFA.py unchanged. It reuses losses.py and utils.py, but uses
ImageFolder data so DAFA can be compared on CIFAR10-LT/CIFAR100-LT.
"""

import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from lt_evaluation import compute_tail_classes, evaluate_lt
from lt_tinyimagenet_utils import (
    DATASET_CHOICES,
    adapt_model_for_image_size,
    infer_lt_dataset,
    load_lt_imagefolder_dataset,
)
from losses import madry_loss, trades_loss
from AWP_LT import add_into_weights, diff_in_weights
from robal_base import (
    add_robal_args,
    evaluate_robal_lt,
    is_robal,
    make_robal_state,
    maybe_replace_classifier,
    robal_margin_loss,
)
from lt_advanced_baselines import (
    ADVANCED_BASE_CHOICES,
    add_advanced_base_args,
    advanced_base_loss,
    begin_advanced_epoch,
    close_advanced_base_state,
    ensure_advanced_feature_hook,
    finish_advanced_epoch,
    is_advanced_base,
    make_advanced_base_state,
    normalize_base_algorithm,
)
from utils import calculate_class_weights, get_model


def get_args():
    parser = argparse.ArgumentParser(description="DAFA training for CIFAR-LT ImageFolder data")

    parser.add_argument("--data_root", default="./data/CIFAR100-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/dafa_lt")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--load_epoch", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_freq", default=10, type=int)
    parser.add_argument("--tail_fraction", default=0.8, type=float)

    parser.add_argument("--weight_decay", "--wd", default=5e-4, type=float)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_schedule", type=str, default="bag_of_tricks",
                        choices=("trades", "bag_of_tricks", "madry"))
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)

    parser.add_argument("--base_algorithm", "--loss", dest="base_algorithm",
                        default="trades", type=str,
                        choices=("at", "pgd", "trades", "awp", "robal",
                                 "AT", "PGD", "Trades", "TRADES", "AWP", "RoBal", "ROBAL")
                                + ADVANCED_BASE_CHOICES,
                        help="Base adversarial training algorithm. Use at for PGD-AT.")
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--beta", default=4.0, type=float)

    parser.add_argument("--rob_fairness_algorithm", default="dafa", type=str,
                        choices=("dafa", "none"))
    parser.add_argument("--dafa_warmup", default=70, type=int)
    parser.add_argument("--dafa_lambda", default=1.5, type=float)
    parser.add_argument("--dafa_min_weight", default=0.25, type=float,
                        help="Smallest legal DAFA multiplier for LT training.")
    parser.add_argument("--awp_gamma", default=0.01, type=float)
    parser.add_argument("--awp_warmup", default=10, type=int)
    parser.add_argument("--awp_lr", default=0.01, type=float)
    add_robal_args(parser)
    add_advanced_base_args(parser)

    args = parser.parse_args()
    args.base_algorithm = normalize_base_algorithm(args.base_algorithm)
    args.loss = "pgd" if args.base_algorithm in ("at", "awp", "robal", "at-bsl", "reat", "taet") else "trades"
    return args


def cifar_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], index


def load_imagefolder_dataset(args):
    return load_lt_imagefolder_dataset(args)


def infer_dataset(args, n_class):
    infer_lt_dataset(args, n_class)


def calculate_lt_dafa_class_weights(memory_dict, requested_lambda, min_weight):
    """Apply DAFA's linear class-weight rule with a legal LT lambda.

    The original DAFA code uses w_c = 1 + lambda * d_c directly as both a
    loss multiplier and a PGD budget multiplier. On long-tailed data, the same
    lambda can make head-class multipliers negative, which invalidates the
    perturbation bounds and can collapse head accuracy. We keep DAFA's direction
    d_c unchanged, and only reduce the effective lambda when needed so every
    multiplier remains positive.
    """
    if min_weight <= 0.0 or min_weight >= 1.0:
        raise ValueError("--dafa_min_weight must be in (0, 1).")

    base_weights = calculate_class_weights(memory_dict, 0.0).astype(np.float64)
    unit_weights = calculate_class_weights(memory_dict, 1.0).astype(np.float64)
    direction = unit_weights - base_weights

    if not np.isfinite(direction).all():
        raise ValueError("DAFA class-weight direction contains NaN/Inf values.")

    effective_lambda = float(requested_lambda)
    min_direction = float(direction.min())
    if min_direction < 0.0:
        max_lambda = (1.0 - float(min_weight)) / (-min_direction)
        effective_lambda = min(effective_lambda, max_lambda)

    weights = base_weights + effective_lambda * direction
    weights = np.maximum(weights, float(min_weight))
    return weights.astype(np.float32), direction.astype(np.float32), effective_lambda


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.lr_schedule == "trades":
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    elif args.lr_schedule == "madry":
        if epoch >= 0.5 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    elif args.lr_schedule == "bag_of_tricks":
        if epoch >= args.epochs - 10:
            lr = args.lr * 0.1
        if epoch >= args.epochs - 5:
            lr = args.lr * 0.01
    else:
        raise ValueError("Unknown LR schedule {}".format(args.lr_schedule))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def calc_awp(model, proxy, proxy_optimizer, loss_fn):
    proxy.load_state_dict(model.state_dict())
    proxy.train()
    loss = -loss_fn(proxy)
    proxy_optimizer.zero_grad()
    loss.backward()
    proxy_optimizer.step()
    return diff_in_weights(model, proxy)


def train_epoch(args, model, device, optimizer, train_loader, epoch, class_weights,
                trainset_size, proxy=None, proxy_optimizer=None, robal_state=None,
                advanced_state=None):
    model.train()

    if args.rob_fairness_algorithm == "dafa" and epoch == args.dafa_warmup:
        memory_dict = {
            "probs": np.zeros((trainset_size, args.n_class)),
            "labels": np.zeros(trainset_size),
        }
    else:
        memory_dict = None

    for batch_idx, dataset in enumerate(train_loader):
        (data, target), batch_indices = dataset
        data, target = data.to(device), target.to(device)

        model.train()
        optimizer.zero_grad()

        def loss_fn(net, write_memory=False):
            cur_memory = memory_dict if write_memory else None
            cur_indices = batch_indices if write_memory else None
            if is_robal(args):
                class_eps = class_weights.to(data.device) * args.epsilon
                return robal_margin_loss(
                    net, data, target, optimizer, args, robal_state,
                    class_eps=class_eps, class_weights=class_weights,
                    batch_indices=cur_indices, memory_dict=cur_memory)
            if is_advanced_base(args):
                class_eps = class_weights.to(data.device) * args.epsilon
                return advanced_base_loss(
                    net, data, target, optimizer, args, advanced_state,
                    class_eps=class_eps, class_weights=class_weights)
            if args.loss == "trades":
                return trades_loss(
                    model=net, x_natural=data, y=target, optimizer=optimizer, args=args,
                    class_weights=class_weights, batch_indices=cur_indices, memory_dict=cur_memory)
            if args.loss == "pgd":
                return madry_loss(
                    model=net, x_natural=data, y=target, optimizer=optimizer, args=args,
                    class_weights=class_weights, batch_indices=cur_indices, memory_dict=cur_memory)
            raise ValueError("Unknown loss {}".format(args.loss))

        awp_diff = None
        if args.base_algorithm == "awp" and args.awp_gamma > 0.0 and epoch >= args.awp_warmup:
            awp_diff = calc_awp(model, proxy, proxy_optimizer, lambda net: loss_fn(net)[0])
            add_into_weights(model, awp_diff, args.awp_gamma)

        if is_robal(args):
            class_eps = class_weights.to(data.device) * args.epsilon
            loss, loss_dict = robal_margin_loss(
                model, data, target, optimizer, args, robal_state,
                class_eps=class_eps, class_weights=class_weights,
                batch_indices=batch_indices, memory_dict=memory_dict)
        elif is_advanced_base(args):
            class_eps = class_weights.to(data.device) * args.epsilon
            loss, loss_dict = advanced_base_loss(
                model, data, target, optimizer, args, advanced_state,
                class_eps=class_eps, class_weights=class_weights)
            if memory_dict is not None:
                with torch.no_grad():
                    memory_dict["probs"][batch_indices] = torch.softmax(model(data), dim=1).detach().cpu().numpy()
                    memory_dict["labels"][batch_indices] = target.detach().cpu().numpy()
        elif args.loss == "trades":
            loss, loss_dict = trades_loss(
                model=model, x_natural=data, y=target, optimizer=optimizer, args=args,
                class_weights=class_weights, batch_indices=batch_indices, memory_dict=memory_dict)
        elif args.loss == "pgd":
            loss, loss_dict = madry_loss(
                model=model, x_natural=data, y=target, optimizer=optimizer, args=args,
                class_weights=class_weights, batch_indices=batch_indices, memory_dict=memory_dict)
        else:
            raise ValueError("Unknown loss {}".format(args.loss))

        model.train()
        loss_dict["awp"] = float(awp_diff is not None)
        loss.backward()
        optimizer.step()
        if awp_diff is not None:
            add_into_weights(model, awp_diff, -args.awp_gamma)

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, (batch_idx + 1) * args.batch_size, trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    loss_log += "{} : {:.6f}\t".format(key, value)
                else:
                    loss_log += "{} : {}\t".format(key, value)
            logging.info(default_log + loss_log)

    return memory_dict


def main():
    args = get_args()

    base_trainset, testset, class_counts = load_imagefolder_dataset(args)
    infer_dataset(args, len(base_trainset.classes))
    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)
    trainset = IndexedDataset(base_trainset)

    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, "training.log")),
            logging.StreamHandler()
        ])
    logging.info("Args: %s", args)
    logging.info("Detected dataset: %s, n_class: %d, image_size: %d",
                 args.dataset, args.n_class, args.image_size)
    logging.info("Class counts: %s", class_counts.tolist())
    logging.info("Tail classes: %s", tail_classes.tolist())

    final_checkpoint_path = os.path.join(args.model_dir, "final.pt")
    if not args.overwrite and os.path.exists(final_checkpoint_path):
        logging.info("Final checkpoint found - quitting. Use --overwrite to train again.")
        sys.exit(0)

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = maybe_replace_classifier(adapt_model_for_image_size(get_model(args), args), args)
    logging.info(args.model)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    proxy = copy.deepcopy(model) if args.base_algorithm == "awp" else None
    proxy_optimizer = optim.SGD(proxy.parameters(), lr=args.awp_lr) if proxy is not None else None
    robal_state = make_robal_state(class_counts, args, device)
    advanced_state = make_advanced_base_state(args, class_counts, device)
    advanced_state = ensure_advanced_feature_hook(args, model, advanced_state)

    class_weights = torch.ones(args.n_class, device=device)
    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(args.load_epoch + 1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)
        args.epoch = epoch - 1
        begin_advanced_epoch(args, advanced_state, epoch)

        memory_dict = train_epoch(args, model, device, optimizer, train_loader,
                                  epoch, class_weights, len(trainset),
                                  proxy=proxy, proxy_optimizer=proxy_optimizer,
                                  robal_state=robal_state,
                                  advanced_state=advanced_state)
        finish_advanced_epoch(args, advanced_state)

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch - args.load_epoch == 1 or epoch == args.epochs:
            if is_robal(args):
                eval_data = evaluate_robal_lt(args, model, device, test_loader, tail_classes, robal_state)
            else:
                eval_data = evaluate_lt(args, model, device, test_loader, tail_classes)
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

        if args.rob_fairness_algorithm == "dafa" and epoch == args.dafa_warmup:
            class_weight_values, dafa_direction, effective_lambda = calculate_lt_dafa_class_weights(
                memory_dict, args.dafa_lambda, args.dafa_min_weight)
            if effective_lambda < args.dafa_lambda:
                logging.warning(
                    "Requested DAFA lambda %.6f would create nonpositive LT multipliers. "
                    "Using effective lambda %.6f to keep all weights >= %.4f.",
                    args.dafa_lambda, effective_lambda, args.dafa_min_weight)
            logging.info("DAFA class-weight direction => %s", dafa_direction.tolist())
            class_weights = torch.as_tensor(
                class_weight_values,
                dtype=torch.float32,
                device=device)
            logging.info("Assigned class weights => %s", class_weights.detach().cpu().tolist())

        elapsed_time = time.time() - init_time
        print("elapsed time : %d h %d m %d s" % (
            elapsed_time / 3600, (elapsed_time % 3600) / 60, elapsed_time % 60))

    close_advanced_base_state(advanced_state)


if __name__ == "__main__":
    main()
