"""
Standalone RobustLT training script.

This script reproduces the core RobustLT idea for CIFAR-LT ImageFolder data:
Class-wise Perturbation Balancing (CPB) + Adversarial Iteration Weighting (AIW).
It does not modify DAFA.py, CFA.py, UDR.py, losses.py, or datasets.py.
"""

import argparse
import copy
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lt_tinyimagenet_utils import (
    DATASET_CHOICES,
    adapt_model_for_image_size,
    infer_lt_dataset,
    load_lt_imagefolder_dataset,
)
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
from utils import get_model


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch RobustLT Adversarial Training")

    parser.add_argument("--data_root", default="./data/CIFAR10-LT-IR50", type=str,
                        help="CIFAR-LT ImageFolder root containing train/ and test/.")
    parser.add_argument("--dataset", default="auto", type=str,
                        choices=DATASET_CHOICES,
                        help="Dataset name. Use auto to infer it from ImageFolder class count.")
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/robustlt")
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
                        default="trades", type=str,
                        choices=("at", "pgd", "trades", "awp", "robal",
                                 "AT", "PGD", "Trades", "TRADES", "AWP", "RoBal", "ROBAL")
                                + ADVANCED_BASE_CHOICES,
                        help="Base adversarial training algorithm enhanced by RobustLT.")
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--beta", default=4.0, type=float,
                        help="TRADES robust loss coefficient.")

    parser.add_argument("--robustlt_alpha", "--alpha", default=0.3, type=float,
                        help="CPB alpha. Larger values assign stronger perturbations to tail classes.")
    parser.add_argument("--robustlt_beta", "--aiw_beta", default=0.8, type=float,
                        help="AIW beta. Perturbations warm up over beta * epochs.")
    parser.add_argument("--disable_aiw", action="store_true", default=False,
                        help="Use the final CPB perturbation from epoch 1.")
    parser.add_argument("--tail_fraction", default=0.8, type=float,
                        help="Fraction of classes with the fewest samples used as tail classes.")
    parser.add_argument("--awp_gamma", default=0.01, type=float)
    parser.add_argument("--awp_warmup", default=10, type=int)
    parser.add_argument("--awp_lr", default=0.01, type=float)
    add_robal_args(parser)
    add_advanced_base_args(parser)

    args = parser.parse_args()
    args.base_algorithm = normalize_base_algorithm(args.base_algorithm)
    return args


def cifar10_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def load_imagefolder_dataset(args):
    return load_lt_imagefolder_dataset(args)


def infer_dataset(args, n_class):
    infer_lt_dataset(args, n_class)


def compute_tail_classes(class_counts, tail_fraction):
    num_classes = len(class_counts)
    num_tail = max(1, int(math.ceil(num_classes * tail_fraction)))
    return np.argsort(class_counts)[:num_tail].astype(np.int64)


def compute_cpb_eps(class_counts, epsilon, alpha, device):
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    n_max = counts.max()
    total = counts.sum()
    log_ratio_sqrt = torch.sqrt(torch.clamp(torch.log(n_max / counts), min=0.0))
    denom = ((counts / total) * log_ratio_sqrt).sum()

    if alpha == 0 or denom.item() == 0:
        return torch.ones_like(counts) * epsilon

    tau = alpha / denom
    class_eps = ((1.0 - alpha) + tau * log_ratio_sqrt) * epsilon
    return class_eps


def aiw_intensity(epoch, epochs, beta, disable_aiw=False):
    if disable_aiw or beta <= 0:
        return 1.0
    return min((epoch - 1.0) / (epochs * beta), 1.0)


def expand_per_sample(values, y, x):
    values = torch.as_tensor(values, dtype=x.dtype, device=x.device)
    return values[y].view(-1, 1, 1, 1)


def pgd_adversary(model, x_natural, y, class_eps, base_epsilon, base_step_size, perturb_steps):
    eps_mask = expand_per_sample(class_eps, y, x_natural)
    step_mask = eps_mask / max(base_epsilon, 1e-12) * base_step_size

    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-1.0, 1.0) * eps_mask
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_mask * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def trades_adversary(model, x_natural, y, class_eps, base_epsilon, base_step_size, perturb_steps):
    criterion_kl = nn.KLDivLoss(reduction="sum")
    eps_mask = expand_per_sample(class_eps, y, x_natural)
    step_mask = eps_mask / max(base_epsilon, 1e-12) * base_step_size

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.no_grad():
        natural_prob = F.softmax(model(x_natural), dim=1).detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_logits = model(x_adv)
            loss_kl = criterion_kl(F.log_softmax(adv_logits, dim=1), natural_prob)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_mask * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def robustlt_loss(model, x_natural, y, optimizer, args, class_eps,
                  robal_state=None, advanced_state=None):
    model.eval()
    if is_robal(args):
        loss, loss_dict = robal_margin_loss(
            model, x_natural, y, optimizer, args, robal_state, class_eps=class_eps)
        return loss, loss_dict
    if is_advanced_base(args):
        return advanced_base_loss(
            model, x_natural, y, optimizer, args, advanced_state, class_eps=class_eps)
    if args.base_algorithm in ("at", "awp"):
        x_adv = pgd_adversary(
            model, x_natural, y, class_eps, args.epsilon,
            args.pgd_step_size, args.pgd_num_steps)
    elif args.base_algorithm == "trades":
        x_adv = trades_adversary(
            model, x_natural, y, class_eps, args.epsilon,
            args.pgd_step_size, args.pgd_num_steps)
    else:
        raise ValueError("Unknown base algorithm {}".format(args.base_algorithm))

    model.train()
    optimizer.zero_grad()

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    natural_loss = F.cross_entropy(logits_nat, y)

    if args.base_algorithm in ("at", "awp"):
        robust_loss = F.cross_entropy(logits_adv, y)
        loss = robust_loss
    else:
        natural_prob = F.softmax(logits_nat, dim=1).detach()
        robust_loss = nn.KLDivLoss(reduction="sum")(
            F.log_softmax(logits_adv, dim=1), natural_prob) / len(x_natural)
        loss = natural_loss + args.beta * robust_loss

    eps_batch = torch.as_tensor(class_eps, device=x_natural.device)[y]
    loss_dict = {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "eps_min": eps_batch.min().item(),
        "eps_max": eps_batch.max().item(),
    }
    return loss, loss_dict


def calc_awp(model, proxy, proxy_optimizer, loss_fn):
    proxy.load_state_dict(model.state_dict())
    proxy.train()
    loss = -loss_fn(proxy)
    proxy_optimizer.zero_grad()
    loss.backward()
    proxy_optimizer.step()
    return diff_in_weights(model, proxy)


def train_epoch(args, model, device, optimizer, train_loader, epoch, class_eps,
                trainset_size, proxy=None, proxy_optimizer=None, robal_state=None,
                advanced_state=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        awp_diff = None
        if args.base_algorithm == "awp" and args.awp_gamma > 0.0 and epoch >= args.awp_warmup:
            awp_diff = calc_awp(
                model, proxy, proxy_optimizer,
                lambda net: robustlt_loss(
                    net, data, target, optimizer, args, class_eps,
                    robal_state, advanced_state)[0])
            add_into_weights(model, awp_diff, args.awp_gamma)

        loss, loss_dict = robustlt_loss(
            model, data, target, optimizer, args, class_eps, robal_state, advanced_state)
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

    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info("Args: %s", args)

    final_checkpoint_path = os.path.join(args.model_dir, "final.pt")
    if not args.overwrite and os.path.exists(final_checkpoint_path):
        logging.info("Final checkpoint found - quitting. Use --overwrite to train again.")
        sys.exit(0)

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    trainset, testset, class_counts = load_imagefolder_dataset(args)
    infer_dataset(args, len(trainset.classes))
    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)
    cpb_eps = compute_cpb_eps(class_counts, args.epsilon, args.robustlt_alpha, device)
    logging.info("Detected dataset: %s, n_class: %d, image_size: %d",
                 args.dataset, args.n_class, args.image_size)
    logging.info("Train class counts: %s", class_counts.tolist())
    logging.info("Tail classes: %s", tail_classes.tolist())
    logging.info("CPB final class eps: %s", (torch.round(cpb_eps.cpu() * 1000000) / 1000000).tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = maybe_replace_classifier(adapt_model_for_image_size(get_model(args), args), args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    proxy = copy.deepcopy(model) if args.base_algorithm == "awp" else None
    proxy_optimizer = optim.SGD(proxy.parameters(), lr=args.awp_lr) if proxy is not None else None
    robal_state = make_robal_state(class_counts, args, device)
    advanced_state = make_advanced_base_state(args, class_counts, device)
    advanced_state = ensure_advanced_feature_hook(args, model, advanced_state)

    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        intensity = aiw_intensity(epoch, args.epochs, args.robustlt_beta, args.disable_aiw)
        class_eps = cpb_eps * intensity

        logging.info("Setting learning rate to %g", lr)
        logging.info("AIW intensity %.6f, class eps: %s",
                     intensity, (torch.round(class_eps.cpu() * 1000000) / 1000000).tolist())
        begin_advanced_epoch(args, advanced_state, epoch)

        train_epoch(args, model, device, optimizer, train_loader, epoch, class_eps,
                    len(trainset), proxy=proxy, proxy_optimizer=proxy_optimizer,
                    robal_state=robal_state, advanced_state=advanced_state)
        finish_advanced_epoch(args, advanced_state)

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1:
            if is_robal(args):
                eval_data = evaluate_robal_lt(args, model, device, test_loader, tail_classes, robal_state)
            else:
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

    close_advanced_base_state(advanced_state)


if __name__ == "__main__":
    main()
