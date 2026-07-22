"""
CGR-LT: Confusion-Geometry Rebalanced Robust Long-Tail Training.

This standalone script is intended as a research prototype. It keeps the
existing DAFA/RobustLT/CFA/UDR files untouched and combines several feasible
long-tail robust training ideas:

1) RobustLT-style class-wise perturbation balancing (CPB + AIW).
2) Feedback loss reweighting from per-class robust accuracy and confusion.
3) Class-wise TRADES beta for a per-class clean-robust tradeoff.
4) Balanced-softmax style log-prior adjusted CE.
5) Confusion-geometry graph and pairwise margin for tail-to-head confusion.
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

from AWP_LT import add_into_weights, diff_in_weights
from robal_base import (
    add_robal_args,
    apply_eval_adjustment,
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
    parser = argparse.ArgumentParser(description="CGR-LT training for long-tailed ImageFolder data")

    parser.add_argument("--data_root", default="./data/CIFAR10-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str,
                        choices=("auto", "cifar10", "cifar100", "tinyimagenet"),
                        help="Dataset name passed to the model factory. 'auto' infers it from ImageFolder class count.")
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/cgr_lt")
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
                                + ADVANCED_BASE_CHOICES)
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

    parser.add_argument("--use_feedback_weight", action="store_true", default=True)
    parser.add_argument("--no_feedback_weight", dest="use_feedback_weight", action="store_false")
    parser.add_argument("--feedback_start", default=10, type=int)
    parser.add_argument("--feedback_momentum", default=0.8, type=float)
    parser.add_argument("--weight_lambda", default=1.0, type=float)
    parser.add_argument("--conf_lambda", default=0.5, type=float)
    parser.add_argument("--weight_max", default=3.0, type=float)
    parser.add_argument("--weight_natural", action="store_true", default=False)

    parser.add_argument("--use_feedback_eps", action="store_true", default=False)
    parser.add_argument("--eps_lambda", default=0.25, type=float)
    parser.add_argument("--eps_min", default=0.0, type=float)
    parser.add_argument("--eps_max", default=0.0, type=float)

    parser.add_argument("--use_balanced_ce", action="store_true", default=True)
    parser.add_argument("--no_balanced_ce", dest="use_balanced_ce", action="store_false")
    parser.add_argument("--prior_tau", default=1.0, type=float)

    parser.add_argument("--use_classwise_beta", action="store_true", default=True)
    parser.add_argument("--no_classwise_beta", dest="use_classwise_beta", action="store_false")
    parser.add_argument("--beta_lambda", default=0.5, type=float)
    parser.add_argument("--beta_min_scale", default=0.5, type=float)
    parser.add_argument("--beta_max_scale", default=2.0, type=float)

    parser.add_argument("--use_cgr_margin", action="store_true", default=True)
    parser.add_argument("--no_cgr_margin", dest="use_cgr_margin", action="store_false")
    parser.add_argument("--margin_lambda", default=0.1, type=float)
    parser.add_argument("--margin_m", default=0.5, type=float)
    parser.add_argument("--graph_momentum", default=0.8, type=float)
    parser.add_argument("--graph_topk", default=3, type=int)
    parser.add_argument("--head_gamma", default=0.5, type=float)
    parser.add_argument("--geometry_gamma", default=1.0, type=float)
    parser.add_argument("--graph_eps", default=1e-6, type=float)
    parser.add_argument("--awp_gamma", default=0.01, type=float)
    parser.add_argument("--awp_warmup", default=10, type=int)
    parser.add_argument("--awp_lr", default=0.01, type=float)
    add_robal_args(parser)
    add_advanced_base_args(parser)

    args = parser.parse_args()
    args.base_algorithm = normalize_base_algorithm(args.base_algorithm)
    return args


def imagefolder_transforms(args, train=True):
    image_size = args.image_size
    if image_size <= 0:
        image_size = 64 if args.dataset == "tinyimagenet" else 32
    padding = 8 if image_size >= 64 else 4
    if train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def infer_dataset(args, n_class):
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
    if args.dataset == "cifar10" and n_class != 10:
        raise ValueError("Dataset argument is cifar10 but ImageFolder has {} classes.".format(n_class))
    if args.dataset == "cifar100" and n_class != 100:
        raise ValueError("Dataset argument is cifar100 but ImageFolder has {} classes.".format(n_class))
    if args.dataset == "tinyimagenet" and n_class != 200:
        raise ValueError("Dataset argument is tinyimagenet but ImageFolder has {} classes.".format(n_class))
    if args.image_size <= 0:
        args.image_size = 64 if args.dataset == "tinyimagenet" else 32


def load_imagefolder_dataset(args):
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            "Expected ImageFolder data at train/ and test/ under: {}".format(args.data_root))

    class_probe = datasets.ImageFolder(train_dir)
    infer_dataset(args, len(class_probe.classes))
    trainset = datasets.ImageFolder(train_dir, transform=imagefolder_transforms(args, train=True))
    testset = datasets.ImageFolder(test_dir, transform=imagefolder_transforms(args, train=False))
    if len(trainset.classes) != len(testset.classes):
        raise ValueError("Train/test class counts differ: {} vs {}.".format(
            len(trainset.classes), len(testset.classes)))
    class_counts = np.bincount([target for _, target in trainset.samples],
                               minlength=len(trainset.classes)).astype(np.int64)
    return trainset, testset, class_counts


def compute_tail_classes(class_counts, tail_fraction):
    num_tail = max(1, int(math.ceil(len(class_counts) * tail_fraction)))
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
    return ((1.0 - alpha) + tau * log_ratio_sqrt) * epsilon


def aiw_intensity(epoch, epochs, beta, disable_aiw=False):
    if disable_aiw or beta <= 0:
        return 1.0
    return min((epoch - 1.0) / (epochs * beta), 1.0)


def build_log_prior(class_counts, device):
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    prior = counts / counts.sum()
    return torch.log(prior.clamp_min(1e-12))


def adjusted_ce_each(logits, y, log_prior, args):
    if args.use_balanced_ce:
        logits = logits + args.prior_tau * log_prior.view(1, -1).to(logits.device)
    return F.cross_entropy(logits, y, reduction="none")


def weighted_mean(per_sample_loss, y, class_weights):
    weights = class_weights[y].to(per_sample_loss.device)
    return (per_sample_loss * weights).mean()


def expand_per_sample(values, y, x):
    values = torch.as_tensor(values, dtype=x.dtype, device=x.device)
    return values[y].view(-1, 1, 1, 1)


def get_classifier_module(model):
    net = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(net, "linear"):
        return net.linear
    if hasattr(net, "fc"):
        return net.fc
    return None


def adapt_model_for_dataset(model, args):
    if args.dataset != "tinyimagenet" or args.image_size <= 32:
        return model

    classifier = get_classifier_module(model)
    if classifier is None:
        raise ValueError("Cannot adapt model {} for TinyImageNet: classifier not found.".format(args.model))

    in_features = classifier.in_features
    out_features = classifier.out_features
    if out_features != args.n_class:
        raise ValueError("Model output classes {} do not match dataset classes {}.".format(
            out_features, args.n_class))

    if hasattr(model, "linear"):
        model.linear = nn.Linear(in_features * 4, args.n_class)
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(in_features * 4, args.n_class)
    return model


class FeatureHook:
    def __init__(self, model):
        self.features = None
        self.handle = None
        classifier = get_classifier_module(model)
        if classifier is not None:
            self.handle = classifier.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.features = inputs[0].detach()

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def pgd_adversary(model, x_natural, y, class_eps, base_epsilon, base_step_size,
                  perturb_steps, log_prior, args):
    eps_mask = expand_per_sample(class_eps, y, x_natural)
    step_mask = eps_mask / max(base_epsilon, 1e-12) * base_step_size

    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-1.0, 1.0) * eps_mask
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = adjusted_ce_each(model(x_adv), y, log_prior, args).mean()
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
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), natural_prob)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_mask * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def pairwise_margin_loss(logits, y, class_graph, class_weights, args):
    if (not args.use_cgr_margin) or args.margin_lambda <= 0:
        return logits.new_tensor(0.0)
    graph_y = class_graph[y].to(logits.device)
    if graph_y.sum().item() == 0:
        return logits.new_tensor(0.0)

    true_logits = logits.gather(1, y.view(-1, 1))
    margins = F.relu(args.margin_m - true_logits + logits)
    margins = margins * graph_y
    per_sample = margins.sum(dim=1)
    return (per_sample * class_weights[y].to(logits.device)).mean()


def build_class_betas(args, feedback_score):
    if not args.use_classwise_beta:
        return torch.ones_like(feedback_score) * args.beta
    score = feedback_score / feedback_score.mean().clamp_min(1e-12)
    scale = 1.0 + args.beta_lambda * (score - 1.0)
    scale = torch.clamp(scale, min=args.beta_min_scale, max=args.beta_max_scale)
    scale = scale / scale.mean().clamp_min(1e-12)
    return args.beta * scale


def build_epoch_eps(args, cpb_eps, feedback_score, epoch):
    intensity = aiw_intensity(epoch, args.epochs, args.robustlt_beta, args.disable_aiw)
    class_eps = cpb_eps * intensity
    if args.use_feedback_eps:
        score = feedback_score / feedback_score.mean().clamp_min(1e-12)
        eps_scale = torch.clamp(1.0 + args.eps_lambda * (score - 1.0), min=0.1)
        class_eps = class_eps * eps_scale
        if args.eps_min > 0:
            class_eps = torch.clamp(class_eps, min=args.eps_min * intensity)
        if args.eps_max > 0:
            class_eps = torch.clamp(class_eps, max=args.eps_max * intensity)
    return class_eps


def cgr_lt_loss(model, x_natural, y, optimizer, args, class_eps, class_weights,
                class_betas, class_graph, log_prior, robal_state=None,
                advanced_state=None):
    model.eval()
    if is_robal(args):
        loss, loss_dict, outputs = robal_margin_loss(
            model, x_natural, y, optimizer, args, robal_state,
            class_eps=class_eps, class_weights=class_weights, return_outputs=True)
        margin_loss = pairwise_margin_loss(
            outputs["logits_adv"], y, class_graph, class_weights, args)
        loss = loss + args.margin_lambda * margin_loss
        loss_dict["margin"] = margin_loss.item()
        return loss, loss_dict
    if is_advanced_base(args):
        loss, loss_dict, outputs = advanced_base_loss(
            model, x_natural, y, optimizer, args, advanced_state,
            class_eps=class_eps, class_weights=class_weights,
            return_outputs=True)
        logits_adv = model(outputs["x_adv"])
        margin_loss = pairwise_margin_loss(logits_adv, y, class_graph, class_weights, args)
        loss = loss + args.margin_lambda * margin_loss
        loss_dict["margin"] = margin_loss.item()
        return loss, loss_dict
    if args.base_algorithm in ("at", "awp"):
        x_adv = pgd_adversary(
            model, x_natural, y, class_eps, args.epsilon,
            args.pgd_step_size, args.pgd_num_steps, log_prior, args)
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
    natural_each = adjusted_ce_each(logits_nat, y, log_prior, args)

    if args.weight_natural:
        natural_loss = weighted_mean(natural_each, y, class_weights)
    else:
        natural_loss = natural_each.mean()

    if args.base_algorithm in ("at", "awp"):
        robust_each = adjusted_ce_each(logits_adv, y, log_prior, args)
        robust_loss = weighted_mean(robust_each, y, class_weights)
        loss = robust_loss
        beta_batch = torch.zeros_like(y, dtype=x_natural.dtype)
    else:
        natural_prob = F.softmax(logits_nat, dim=1).detach()
        robust_each = F.kl_div(
            F.log_softmax(logits_adv, dim=1), natural_prob, reduction="none"
        ).sum(dim=1)
        weights = class_weights[y].to(x_natural.device)
        beta_batch = class_betas[y].to(x_natural.device)
        robust_loss = (robust_each * weights * beta_batch).mean()
        loss = natural_loss + robust_loss

    margin_loss = pairwise_margin_loss(logits_adv, y, class_graph, class_weights, args)
    loss = loss + args.margin_lambda * margin_loss

    eps_batch = torch.as_tensor(class_eps, device=x_natural.device)[y]
    weight_batch = class_weights[y].detach()
    return loss, {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "margin": margin_loss.item(),
        "eps_min": eps_batch.min().item(),
        "eps_max": eps_batch.max().item(),
        "w_min": weight_batch.min().item(),
        "w_max": weight_batch.max().item(),
        "beta_min": beta_batch.min().item(),
        "beta_max": beta_batch.max().item(),
    }


def calc_awp(model, proxy, proxy_optimizer, loss_fn):
    proxy.load_state_dict(model.state_dict())
    proxy.train()
    loss = -loss_fn(proxy)
    proxy_optimizer.zero_grad()
    loss.backward()
    proxy_optimizer.step()
    return diff_in_weights(model, proxy)


def train_epoch(args, model, device, optimizer, train_loader, epoch, class_eps,
                class_weights, class_betas, class_graph, log_prior, trainset_size,
                proxy=None, proxy_optimizer=None, robal_state=None,
                advanced_state=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        awp_diff = None
        if args.base_algorithm == "awp" and args.awp_gamma > 0.0 and epoch >= args.awp_warmup:
            awp_diff = calc_awp(
                model, proxy, proxy_optimizer,
                lambda net: cgr_lt_loss(
                    net, data, target, optimizer, args, class_eps, class_weights,
                    class_betas, class_graph, log_prior, robal_state, advanced_state)[0])
            add_into_weights(model, awp_diff, args.awp_gamma)

        loss, loss_dict = cgr_lt_loss(
            model, data, target, optimizer, args, class_eps, class_weights,
            class_betas, class_graph, log_prior, robal_state, advanced_state)
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


def pgd_eval(model, x, y, epsilon, num_steps, step_size, args=None, robal_state=None):
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    delta = torch.min(torch.max(delta, -x), 1.0 - x)

    for _ in range(num_steps):
        delta.requires_grad_()
        with torch.enable_grad():
            logits = model(torch.clamp(x + delta, 0.0, 1.0))
            if robal_state is not None:
                logits = apply_eval_adjustment(logits, robal_state["log_freq"], args)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [delta])[0]
        delta = delta.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.min(torch.max(delta, -x), 1.0 - x)

    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def evaluate(args, model, device, loader, tail_classes, n_class, feature_hook, robal_state=None):
    model.eval()
    tail_classes = set(int(c) for c in tail_classes)
    confusion = torch.zeros(n_class, n_class, dtype=torch.long)
    class_total = torch.zeros(n_class, dtype=torch.long)
    class_clean_correct = torch.zeros(n_class, dtype=torch.long)
    class_robust_correct = torch.zeros(n_class, dtype=torch.long)
    feature_sum = None
    feature_count = torch.zeros(n_class, dtype=torch.long)

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            clean_logits = model(data)
            if robal_state is not None:
                clean_logits = apply_eval_adjustment(clean_logits, robal_state["log_freq"], args)
            clean_pred = clean_logits.max(1)[1]

        x_adv = pgd_eval(model, data, target, args.test_epsilon,
                         args.test_pgd_num_steps, args.test_pgd_step_size,
                         args=args, robal_state=robal_state)
        with torch.no_grad():
            robust_logits = model(x_adv)
            if robal_state is not None:
                robust_logits = apply_eval_adjustment(robust_logits, robal_state["log_freq"], args)
            robust_pred = robust_logits.max(1)[1]

        features = feature_hook.features
        if features is None:
            features = robust_logits.detach()
        features = features.detach().view(features.size(0), -1).cpu()
        if feature_sum is None:
            feature_sum = torch.zeros(n_class, features.size(1), dtype=torch.float32)

        for idx, (y_true, y_clean, y_robust) in enumerate(
                zip(target.cpu(), clean_pred.cpu(), robust_pred.cpu())):
            class_total[y_true] += 1
            class_clean_correct[y_true] += int(y_clean == y_true)
            class_robust_correct[y_true] += int(y_robust == y_true)
            confusion[y_true, y_robust] += 1
            feature_sum[y_true] += features[idx]
            feature_count[y_true] += 1

    clean_acc_per_class = class_clean_correct.float() / class_total.clamp_min(1).float()
    robust_acc_per_class = class_robust_correct.float() / class_total.clamp_min(1).float()
    clean_acc = class_clean_correct.sum().item() / class_total.sum().item()
    robust_acc = class_robust_correct.sum().item() / class_total.sum().item()

    tail_indices = torch.as_tensor(sorted(tail_classes), dtype=torch.long)
    tail_clean_acc = clean_acc_per_class[tail_indices].mean().item()
    tail_robust_acc = robust_acc_per_class[tail_indices].mean().item()

    if feature_sum is None:
        feature_centers = None
    else:
        feature_centers = feature_sum / feature_count.clamp_min(1).float().view(-1, 1)

    logging.info(
        "TEST: Clean(all) %.2f%%, Robust(all) %.2f%%, Clean(tail) %.2f%%, Robust(tail) %.2f%%",
        100.0 * clean_acc, 100.0 * robust_acc,
        100.0 * tail_clean_acc, 100.0 * tail_robust_acc)
    logging.info("Per-class robust acc: %s",
                 (torch.round(robust_acc_per_class * 10000) / 100).tolist())

    return {
        "clean_acc": clean_acc,
        "robust_acc": robust_acc,
        "tail_clean_acc": tail_clean_acc,
        "tail_robust_acc": tail_robust_acc,
        "clean_acc_per_class": clean_acc_per_class,
        "robust_acc_per_class": robust_acc_per_class,
        "confusion": confusion,
        "feature_centers": feature_centers,
    }


def update_feedback(args, eval_data, feedback_score, device):
    if not args.use_feedback_weight:
        return feedback_score.detach(), torch.ones_like(feedback_score)

    robust_acc = eval_data["robust_acc_per_class"].to(device)
    confusion = eval_data["confusion"].float().to(device)
    row_total = confusion.sum(dim=1).clamp_min(1.0)
    confusion_error = 1.0 - torch.diag(confusion) / row_total

    mean_robust = robust_acc.mean()
    robust_gap = torch.clamp(mean_robust - robust_acc, min=0.0)
    raw_score = robust_gap + args.conf_lambda * confusion_error
    raw_score = raw_score / raw_score.mean().clamp_min(1e-12)

    updated_score = (
        args.feedback_momentum * feedback_score
        + (1.0 - args.feedback_momentum) * raw_score
    )
    class_weights = 1.0 + args.weight_lambda * updated_score
    class_weights = class_weights / class_weights.mean().clamp_min(1e-12)
    class_weights = torch.clamp(class_weights, min=0.25, max=args.weight_max)
    class_weights = class_weights / class_weights.mean().clamp_min(1e-12)
    return updated_score.detach(), class_weights.detach()


def update_confusion_geometry_graph(args, eval_data, class_counts, old_graph, device):
    if not args.use_cgr_margin:
        return old_graph

    confusion = eval_data["confusion"].float().to(device)
    row_prob = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
    row_prob.fill_diagonal_(0.0)

    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    head_score = torch.pow(counts / counts.max().clamp_min(1.0), args.head_gamma)

    centers = eval_data["feature_centers"]
    if centers is None:
        geometry = torch.ones_like(row_prob)
    else:
        centers = F.normalize(centers.to(device), dim=1)
        dist = torch.cdist(centers, centers, p=2).clamp_min(args.graph_eps)
        geometry = torch.pow(1.0 / dist, args.geometry_gamma)
    geometry.fill_diagonal_(0.0)

    graph = row_prob * head_score.view(1, -1) * geometry
    graph.fill_diagonal_(0.0)

    if args.graph_topk > 0 and args.graph_topk < graph.size(1):
        keep = torch.zeros_like(graph)
        _, idx = torch.topk(graph, k=args.graph_topk, dim=1)
        keep.scatter_(1, idx, 1.0)
        graph = graph * keep

    nonzero = graph[graph > 0]
    if nonzero.numel() > 0:
        graph = graph / nonzero.mean().clamp_min(1e-12)
    graph = torch.clamp(graph, max=5.0)

    return (args.graph_momentum * old_graph + (1.0 - args.graph_momentum) * graph).detach()


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
    logging.info("Method: CGR-LT")
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

    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)
    cpb_eps = compute_cpb_eps(class_counts, args.epsilon, args.robustlt_alpha, device)
    log_prior = build_log_prior(class_counts, device)

    logging.info("Detected dataset: %s, n_class: %d, image_size: %d",
                 args.dataset, args.n_class, args.image_size)
    logging.info("Train class counts: %s", class_counts.tolist())
    logging.info("Tail classes: %s", tail_classes.tolist())
    logging.info("CPB final class eps: %s",
                 (torch.round(cpb_eps.cpu() * 1000000) / 1000000).tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = maybe_replace_classifier(adapt_model_for_dataset(get_model(args), args), args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    feature_hook = FeatureHook(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    proxy = copy.deepcopy(model) if args.base_algorithm == "awp" else None
    proxy_optimizer = optim.SGD(proxy.parameters(), lr=args.awp_lr) if proxy is not None else None
    robal_state = make_robal_state(class_counts, args, device)
    advanced_state = make_advanced_base_state(args, class_counts, device)
    advanced_state = ensure_advanced_feature_hook(args, model, advanced_state)

    feedback_score = torch.ones(args.n_class, device=device)
    class_weights = torch.ones(args.n_class, device=device)
    class_graph = torch.zeros(args.n_class, args.n_class, device=device)
    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        class_eps = build_epoch_eps(args, cpb_eps, feedback_score, epoch)
        class_betas = build_class_betas(args, feedback_score)

        logging.info("Setting learning rate to %g", lr)
        logging.info("Class weights: %s",
                     (torch.round(class_weights.cpu() * 10000) / 10000).tolist())
        logging.info("Class betas: %s",
                     (torch.round(class_betas.cpu() * 10000) / 10000).tolist())
        logging.info("Graph mean %.6f, max %.6f", class_graph.mean().item(), class_graph.max().item())
        logging.info("Class eps: %s",
                     (torch.round(class_eps.cpu() * 1000000) / 1000000).tolist())
        begin_advanced_epoch(args, advanced_state, epoch)

        train_epoch(args, model, device, optimizer, train_loader, epoch,
                    class_eps, class_weights, class_betas, class_graph,
                    log_prior, len(trainset), proxy=proxy, proxy_optimizer=proxy_optimizer,
                    robal_state=robal_state, advanced_state=advanced_state)
        finish_advanced_epoch(args, advanced_state)

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            eval_data = evaluate(args, model, device, test_loader, tail_classes,
                                 args.n_class, feature_hook, robal_state=robal_state)
            robust_acc = eval_data["robust_acc"]

            if epoch >= args.feedback_start:
                feedback_score, class_weights = update_feedback(
                    args, eval_data, feedback_score, device)
                class_graph = update_confusion_geometry_graph(
                    args, eval_data, class_counts, class_graph, device)
                logging.info("Updated feedback score: %s",
                             (torch.round(feedback_score.cpu() * 10000) / 10000).tolist())
                logging.info("Updated class weights: %s",
                             (torch.round(class_weights.cpu() * 10000) / 10000).tolist())
                logging.info("Updated graph mean %.6f, max %.6f",
                             class_graph.mean().item(), class_graph.max().item())

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

    feature_hook.close()
    close_advanced_base_state(advanced_state)


if __name__ == "__main__":
    main()
