"""
Adversarial Weight Perturbation (AWP) for long-tailed ImageFolder datasets.

This script keeps the dataset, evaluation, and checkpoint conventions used by
AT_LT.py and TRADES_LT.py, while adding the AWP weight-space perturbation. It
supports both AT-AWP and TRADES-AWP through --base_algorithm.
"""

import argparse
import copy
import logging
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lt_tinyimagenet_utils import (
    DATASET_CHOICES,
    adapt_model_for_image_size,
    infer_lt_dataset,
    load_lt_imagefolder_dataset,
)
from utils import get_model


EPS = 1e-20


def get_args():
    parser = argparse.ArgumentParser(
        description="AWP training for long-tailed ImageFolder data")

    parser.add_argument("--data_root", default="./data/CIFAR100-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/awp_lt")
    parser.add_argument("--overwrite", action="store_true", default=False)

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
                        default="trades", choices=("at", "pgd", "trades"))
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--no_random_start", action="store_true", default=False,
                        help="Disable random initialization for AT adversarial examples.")
    parser.add_argument("--beta", default=4.0, type=float,
                        help="TRADES robust loss coefficient.")

    parser.add_argument("--awp_gamma", default=0.01, type=float,
                        help="Magnitude of adversarial weight perturbation.")
    parser.add_argument("--awp_warmup", default=10, type=int,
                        help="Start applying AWP from this epoch.")
    parser.add_argument("--awp_lr", default=0.01, type=float,
                        help="Learning rate for the proxy model used to find AWP.")

    return parser.parse_args()


def load_imagefolder_dataset(args):
    return load_lt_imagefolder_dataset(args)


def infer_dataset(args, n_class):
    infer_lt_dataset(args, n_class)


def compute_tail_classes(class_counts, tail_fraction):
    num_tail = max(1, int(np.ceil(len(class_counts) * tail_fraction)))
    return np.argsort(class_counts)[:num_tail].astype(np.int64)


def pgd_adversary(model, x_natural, y, epsilon, step_size, perturb_steps, random_start=True):
    model.eval()

    if random_start:
        delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon)
        delta = torch.min(torch.max(delta, -x_natural), 1.0 - x_natural)
        x_adv = torch.clamp(x_natural + delta, 0.0, 1.0).detach()
    else:
        x_adv = x_natural.detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def trades_adversary(model, x_natural, step_size, epsilon, perturb_steps):
    criterion_kl = nn.KLDivLoss(reduction="sum")
    model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.no_grad():
        natural_prob = F.softmax(model(x_natural), dim=1).detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), natural_prob)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def base_loss(model, x_natural, x_adv, y, args):
    if args.base_algorithm in ("at", "pgd"):
        logits_adv = model(x_adv)
        adv_loss = F.cross_entropy(logits_adv, y)
        with torch.no_grad():
            natural_loss = F.cross_entropy(model(x_natural), y)
        return adv_loss, {
            "natural": natural_loss.item(),
            "adv": adv_loss.item(),
            "total": adv_loss.item(),
        }

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    natural_loss = F.cross_entropy(logits_nat, y)
    natural_prob = F.softmax(logits_nat, dim=1).detach()
    robust_loss = nn.KLDivLoss(reduction="sum")(
        F.log_softmax(logits_adv, dim=1), natural_prob) / len(x_natural)
    loss = natural_loss + args.beta * robust_loss
    return loss, {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "total": loss.item(),
    }


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if old_k != new_k:
            raise RuntimeError("Model and proxy state_dict keys do not match: {} vs {}".format(old_k, new_k))
        if len(old_w.size()) <= 1:
            continue
        if "weight" in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb:
    def __init__(self, model, proxy, proxy_optimizer, gamma):
        self.model = model
        self.proxy = proxy
        self.proxy_optimizer = proxy_optimizer
        self.gamma = gamma

    def calc_awp(self, x_natural, x_adv, y, args):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss, _ = base_loss(self.proxy, x_natural, x_adv, y, args)
        loss = -loss

        self.proxy_optimizer.zero_grad()
        loss.backward()
        self.proxy_optimizer.step()

        return diff_in_weights(self.model, self.proxy)

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-self.gamma)


def awp_loss(model, awp_adversary, x_natural, y, optimizer, args, epoch):
    if args.base_algorithm in ("at", "pgd"):
        x_adv = pgd_adversary(
            model,
            x_natural,
            y,
            args.epsilon,
            args.pgd_step_size,
            args.pgd_num_steps,
            random_start=not args.no_random_start,
        )
    else:
        x_adv = trades_adversary(
            model, x_natural, args.pgd_step_size, args.epsilon, args.pgd_num_steps)

    model.train()
    awp_diff = None
    awp_active = args.awp_gamma > 0 and epoch >= args.awp_warmup
    if awp_active:
        awp_diff = awp_adversary.calc_awp(x_natural, x_adv, y, args)
        awp_adversary.perturb(awp_diff)

    optimizer.zero_grad()
    loss, loss_dict = base_loss(model, x_natural, x_adv, y, args)
    loss_dict["awp"] = float(awp_active)
    return loss, loss_dict, awp_diff


def train_epoch(args, model, awp_adversary, device, optimizer, train_loader, epoch, trainset_size):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss, loss_dict, awp_diff = awp_loss(
            model, awp_adversary, data, target, optimizer, args, epoch)
        loss.backward()
        optimizer.step()
        if awp_diff is not None:
            awp_adversary.restore(awp_diff)

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, min((batch_idx + 1) * args.batch_size, trainset_size), trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)


def pgd_eval(model, x, y, epsilon, num_steps, step_size):
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    delta = torch.min(torch.max(delta, -x), 1.0 - x)

    for _ in range(num_steps):
        delta.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(torch.clamp(x + delta, 0.0, 1.0)), y)
        grad = torch.autograd.grad(loss, [delta])[0]
        delta = delta.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.min(torch.max(delta, -x), 1.0 - x)

    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def evaluate(args, model, device, loader, tail_classes):
    model.eval()
    tail_classes = set(int(c) for c in tail_classes)
    total = clean_correct = robust_correct = 0
    tail_total = tail_clean_correct = tail_robust_correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            clean_pred = model(data).max(1)[1]

        x_adv = pgd_eval(model, data, target, args.test_epsilon,
                         args.test_pgd_num_steps, args.test_pgd_step_size)
        with torch.no_grad():
            robust_pred = model(x_adv).max(1)[1]

        clean = clean_pred.eq(target)
        robust = robust_pred.eq(target)
        total += len(target)
        clean_correct += clean.sum().item()
        robust_correct += robust.sum().item()

        tail_mask = torch.zeros_like(target, dtype=torch.bool)
        for cls in tail_classes:
            tail_mask |= target.eq(cls)
        if tail_mask.any():
            tail_total += tail_mask.sum().item()
            tail_clean_correct += clean[tail_mask].sum().item()
            tail_robust_correct += robust[tail_mask].sum().item()

    clean_acc = clean_correct / total
    robust_acc = robust_correct / total
    tail_clean_acc = tail_clean_correct / max(tail_total, 1)
    tail_robust_acc = tail_robust_correct / max(tail_total, 1)

    logging.info(
        "TEST: Clean(all) %.2f%%, Robust(all) %.2f%%, Clean(tail) %.2f%%, Robust(tail) %.2f%%",
        100.0 * clean_acc,
        100.0 * robust_acc,
        100.0 * tail_clean_acc,
        100.0 * tail_robust_acc,
    )
    return {
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "tail_clean_acc": tail_clean_acc,
        "tail_robust_acc": tail_robust_acc,
    }


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


def main():
    args = get_args()
    if args.base_algorithm == "pgd":
        args.base_algorithm = "at"

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
    logging.info("Method: AWP-LT (%s)", args.base_algorithm.upper())
    logging.info("Args: %s", args)
    logging.info("Detected dataset: %s, n_class: %d, image_size: %d",
                 args.dataset, args.n_class, args.image_size)
    logging.info("Train class counts: %s", class_counts.tolist())

    final_checkpoint_path = os.path.join(args.model_dir, "final.pt")
    if not args.overwrite and os.path.exists(final_checkpoint_path):
        logging.info("Final checkpoint found - quitting. Use --overwrite to train again.")
        sys.exit(0)

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)
    logging.info("Tail classes: %s", tail_classes.tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = adapt_model_for_image_size(get_model(args), args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    proxy = copy.deepcopy(model)
    proxy_optimizer = optim.SGD(proxy.parameters(), lr=args.awp_lr)
    awp_adversary = AdvWeightPerturb(model, proxy, proxy_optimizer, args.awp_gamma)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)

        train_epoch(args, model, awp_adversary, device, optimizer,
                    train_loader, epoch, len(trainset))

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
