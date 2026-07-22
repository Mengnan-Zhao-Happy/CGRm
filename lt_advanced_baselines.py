"""Shared runner for REAT-LT, AT-BSL-LT, and TAET-LT.

The implementations follow the LT ImageFolder, checkpoint, and evaluation
conventions used by AT_LT.py, TRADES_LT.py, and RoBal_LT.py.
"""

import argparse
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

from lt_tinyimagenet_utils import (
    DATASET_CHOICES,
    adapt_model_for_image_size,
    infer_lt_dataset,
    load_lt_imagefolder_dataset,
)
from utils import get_model


ADVANCED_BASE_CHOICES = (
    "at-bsl", "at_bsl", "atbsl", "AT-BSL", "AT_BSL", "ATBSL",
    "reat", "real", "REAT", "REAL", "REAT_LT", "REAL_LT",
    "taet", "TAET", "TAET_LT",
)


def normalize_base_algorithm(name):
    name = name.lower()
    if name == "pgd":
        return "at"
    if name in ("at_bsl", "atbsl"):
        return "at-bsl"
    if name == "real":
        return "reat"
    if name.endswith("_lt"):
        name = name[:-3]
    if name == "real":
        return "reat"
    if name in ("at-bsl", "reat", "taet"):
        return name
    return name


def is_advanced_base(args):
    return getattr(args, "base_algorithm", "") in ("at-bsl", "reat", "taet")


def add_advanced_base_args(parser):
    parser.add_argument("--reat_beta", default=None, type=float,
                        help="Effective-number beta for REAT-style RBL. Defaults to (N-1)/N.")
    parser.add_argument("--reat_tail_lambda", default=0.1, type=float)
    parser.add_argument("--reat_tail_margin", default=0.2, type=float)
    parser.add_argument("--reat_tail_tau", default=0.2, type=float)
    parser.add_argument("--taet_stage1_epochs", default=40, type=int)
    parser.add_argument("--taet_alpha", default=1.0, type=float)
    parser.add_argument("--taet_beta", default=0.1, type=float)
    parser.add_argument("--taet_gamma", default=0.1, type=float)
    parser.add_argument("--taet_group_weight", default=1.0, type=float)


def get_args(method):
    parser = argparse.ArgumentParser(description="{} training for long-tailed ImageFolder data".format(method))
    parser.add_argument("--data_root", default="./data/CIFAR100-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--image_size", default=0, type=int)
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/{}".format(method.lower().replace("-", "_")))
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
                        default="at", type=str,
                        choices=("at", "pgd", "trades", "AT", "PGD", "Trades", "TRADES"))
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--beta", default=4.0, type=float)
    parser.add_argument("--no_random_start", action="store_true", default=False)

    add_advanced_base_args(parser)

    args = parser.parse_args()
    args.method = method
    args.base_algorithm = normalize_base_algorithm(args.base_algorithm)
    return args


def compute_tail_classes(class_counts, tail_fraction):
    num_tail = max(1, int(np.ceil(len(class_counts) * tail_fraction)))
    return np.argsort(class_counts)[:num_tail].astype(np.int64)


def split_groups(class_counts):
    order = np.argsort(-class_counts)
    n = len(order)
    head = order[:max(1, n // 3)]
    tail = order[int(math.ceil(2 * n / 3.0)):]
    body = np.setdiff1d(order, np.concatenate([head, tail]), assume_unique=False)
    groups = np.zeros(n, dtype=np.int64) + 1
    groups[head] = 0
    groups[tail] = 2
    return groups


def balanced_softmax_loss(logits, y, class_counts, reduction="mean"):
    counts = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device).clamp_min(1.0)
    return F.cross_entropy(logits + counts.log().view(1, -1), y, reduction=reduction)


def effective_number_weights(counts, beta=None):
    counts = torch.as_tensor(counts, dtype=torch.float32).clamp_min(1.0)
    if beta is None:
        beta = float((counts.sum().item() - 1.0) / max(counts.sum().item(), 1.0))
    beta = min(max(float(beta), 0.0), 0.999999)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights


def _expand_eps(class_eps, y, x, base_epsilon):
    if class_eps is None:
        return torch.full_like(x[:, :1, :1, :1], base_epsilon)
    class_eps = torch.as_tensor(class_eps, dtype=x.dtype, device=x.device)
    return class_eps[y].view(-1, 1, 1, 1)


def pgd_adversary(model, x_natural, y, args, attack_loss_fn, random_start=True, class_eps=None):
    model.eval()
    eps_mask = _expand_eps(class_eps, y, x_natural, args.epsilon)
    step_mask = eps_mask / max(args.epsilon, 1e-12) * args.pgd_step_size
    if random_start:
        delta = torch.empty_like(x_natural).uniform_(-1.0, 1.0) * eps_mask
        delta = torch.min(torch.max(delta, -x_natural), 1.0 - x_natural)
        x_adv = torch.clamp(x_natural + delta, 0.0, 1.0).detach()
    else:
        x_adv = x_natural.detach()

    for _ in range(args.pgd_num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = attack_loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_mask * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def trades_adversary(model, x_natural, args, attack_loss_fn=None, y=None, class_eps=None):
    criterion_kl = nn.KLDivLoss(reduction="sum")
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    if y is not None and class_eps is not None:
        eps_mask = _expand_eps(class_eps, y, x_natural, args.epsilon)
        step_mask = eps_mask / max(args.epsilon, 1e-12) * args.pgd_step_size
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
    else:
        eps_mask = None
        step_mask = args.pgd_step_size
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.no_grad():
        natural_prob = F.softmax(model(x_natural), dim=1).detach()

    for _ in range(args.pgd_num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_adv = model(x_adv)
            if attack_loss_fn is None:
                loss = criterion_kl(F.log_softmax(logits_adv, dim=1), natural_prob)
            else:
                loss = attack_loss_fn(logits_adv, natural_prob)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_mask * torch.sign(grad.detach())
        if eps_mask is None:
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon), x_natural + args.epsilon)
        else:
            x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def get_classifier_module(model):
    net = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(net, "linear"):
        return net.linear
    if hasattr(net, "fc"):
        return net.fc
    return None


class FeatureHook:
    def __init__(self, model):
        self.features = None
        classifier = get_classifier_module(model)
        self.handle = classifier.register_forward_pre_hook(self._hook) if classifier is not None else None

    def _hook(self, module, inputs):
        self.features = inputs[0]

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def tail_feature_loss(features, y, tail_classes, margin=0.2, tau=0.2):
    if features is None:
        return None
    tail_classes = torch.as_tensor(tail_classes, device=y.device)
    tail_mask = (y.view(-1, 1) == tail_classes.view(1, -1)).any(dim=1)
    if tail_mask.sum().item() == 0 or len(torch.unique(y)) < 2:
        return features.new_tensor(0.0)

    feats = F.normalize(features.view(features.size(0), -1), dim=1)
    classes = torch.unique(y)
    centers = []
    valid_classes = []
    for cls in classes:
        cls_mask = y.eq(cls)
        if cls_mask.any():
            centers.append(F.normalize(feats[cls_mask].mean(dim=0, keepdim=True), dim=1))
            valid_classes.append(cls)
    centers = torch.cat(centers, dim=0)
    valid_classes = torch.stack(valid_classes)
    logits = feats[tail_mask] @ centers.t() / max(tau, 1e-6)
    target = y[tail_mask]
    target_index = (target.view(-1, 1) == valid_classes.view(1, -1)).float().argmax(dim=1)
    ce = F.cross_entropy(logits, target_index)
    true_sim = logits.gather(1, target_index.view(-1, 1))
    neg_logits = logits.masked_fill(
        torch.arange(logits.size(1), device=logits.device).view(1, -1).eq(target_index.view(-1, 1)),
        -1e4)
    sep = F.relu(margin - true_sim + neg_logits.max(dim=1, keepdim=True)[0]).mean()
    return ce + sep


def _weighted_mean(loss_each, y, class_weights=None):
    if class_weights is None:
        return loss_each.mean()
    weights = torch.as_tensor(class_weights, dtype=loss_each.dtype, device=loss_each.device)[y]
    return (loss_each * weights).mean()


def _balanced_softmax_each(logits, y, class_counts):
    counts = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device).clamp_min(1.0)
    return F.cross_entropy(logits + counts.log().view(1, -1), y, reduction="none")


def at_bsl_loss(model, x_natural, y, optimizer, args, class_counts,
                class_eps=None, class_weights=None, class_beta=None):
    if args.base_algorithm == "trades":
        x_adv = trades_adversary(model, x_natural, args, y=y, class_eps=class_eps)
    else:
        attack_loss = lambda logits, target: F.cross_entropy(logits, target)
        x_adv = pgd_adversary(
            model, x_natural, y, args, attack_loss,
            random_start=not getattr(args, "no_random_start", False), class_eps=class_eps)

    model.train()
    optimizer.zero_grad()
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    natural_each = _balanced_softmax_each(logits_nat, y, class_counts)
    robust_each = _balanced_softmax_each(logits_adv, y, class_counts)
    natural_loss = _weighted_mean(natural_each, y, class_weights)
    if class_beta is not None:
        beta = torch.as_tensor(class_beta, dtype=x_natural.dtype, device=x_natural.device)[y]
        loss = _weighted_mean((1.0 - beta) * natural_each + beta * robust_each, y, class_weights)
        robust_loss = _weighted_mean(robust_each, y, class_weights)
    elif args.base_algorithm == "trades":
        natural_prob = F.softmax(logits_nat, dim=1).detach()
        robust_each = F.kl_div(
            F.log_softmax(logits_adv, dim=1), natural_prob, reduction="none").sum(dim=1)
        robust_loss = _weighted_mean(robust_each, y, class_weights)
        loss = natural_loss + args.beta * robust_loss
    else:
        robust_loss = _weighted_mean(robust_each, y, class_weights)
        loss = robust_loss

    loss_dict = {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "total": loss.item(),
    }
    if class_eps is not None:
        eps_batch = torch.as_tensor(class_eps, dtype=x_natural.dtype, device=x_natural.device)[y]
        loss_dict["eps_min"] = eps_batch.min().item()
        loss_dict["eps_max"] = eps_batch.max().item()
    return loss, loss_dict, x_adv


def reat_loss(model, x_natural, y, optimizer, args, class_counts, ae_counts,
              feature_hook, tail_classes, class_eps=None, class_weights=None,
              class_beta=None):
    weights = effective_number_weights(ae_counts, args.reat_beta).to(x_natural.device)

    def rbl_attack(logits, target):
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            sample_weights = weights[pred]
        ce = F.cross_entropy(logits, target, reduction="none")
        return (sample_weights * ce).mean()

    if args.base_algorithm == "trades":
        x_adv = trades_adversary(model, x_natural, args, y=y, class_eps=class_eps)
    else:
        x_adv = pgd_adversary(
            model, x_natural, y, args, rbl_attack,
            random_start=not getattr(args, "no_random_start", False), class_eps=class_eps)

    model.train()
    optimizer.zero_grad()
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    natural_each = _balanced_softmax_each(logits_nat, y, class_counts)
    robust_each = _balanced_softmax_each(logits_adv, y, class_counts)
    natural_loss = _weighted_mean(natural_each, y, class_weights)
    if class_beta is not None:
        beta = torch.as_tensor(class_beta, dtype=x_natural.dtype, device=x_natural.device)[y]
        robust_loss = _weighted_mean(robust_each, y, class_weights)
        base_loss = _weighted_mean((1.0 - beta) * natural_each + beta * robust_each, y, class_weights)
    elif args.base_algorithm == "trades":
        natural_prob = F.softmax(logits_nat, dim=1).detach()
        robust_each = F.kl_div(
            F.log_softmax(logits_adv, dim=1), natural_prob, reduction="none").sum(dim=1)
        robust_loss = _weighted_mean(robust_each, y, class_weights)
        base_loss = natural_loss + args.beta * robust_loss
    else:
        robust_loss = _weighted_mean(robust_each, y, class_weights)
        base_loss = robust_loss

    feat_loss = tail_feature_loss(
        feature_hook.features, y, tail_classes,
        margin=args.reat_tail_margin, tau=args.reat_tail_tau)
    if feat_loss is None:
        feat_loss = logits_adv.new_tensor(0.0)
    loss = base_loss + args.reat_tail_lambda * feat_loss

    with torch.no_grad():
        adv_pred = logits_adv.argmax(dim=1).detach().cpu()
    loss_dict = {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "tail_feat": feat_loss.item(),
        "total": loss.item(),
    }
    if class_eps is not None:
        eps_batch = torch.as_tensor(class_eps, dtype=x_natural.dtype, device=x_natural.device)[y]
        loss_dict["eps_min"] = eps_batch.min().item()
        loss_dict["eps_max"] = eps_batch.max().item()
    return loss, loss_dict, x_adv, adv_pred


def group_weights_for_batch(y, group_ids, args):
    groups = torch.as_tensor(group_ids, device=y.device, dtype=torch.long)
    y_group = groups[y]
    counts = torch.bincount(y_group, minlength=3).float().to(y.device).clamp_min(1.0)
    weights = counts.sum() / (3.0 * counts)
    weights = weights / weights.mean().clamp_min(1e-12)
    return 1.0 + args.taet_group_weight * (weights[y_group] - 1.0)


def equalization_loss(logits, y, group_ids, args):
    per_sample = F.cross_entropy(logits, y, reduction="none")
    sample_weights = group_weights_for_batch(y, group_ids, args)
    weighted_ce = (per_sample * sample_weights).mean()

    class_losses = []
    for cls in torch.unique(y):
        cls_mask = y.eq(cls)
        class_losses.append(per_sample[cls_mask].mean())
    if class_losses:
        class_losses = torch.stack(class_losses)
        var_loss = ((class_losses - class_losses.mean()) ** 2).mean()
        norm_loss = ((class_losses / class_losses.sum().clamp_min(1e-12)) ** 2).sum()
    else:
        var_loss = logits.new_tensor(0.0)
        norm_loss = logits.new_tensor(0.0)
    return args.taet_alpha * weighted_ce + args.taet_beta * var_loss + args.taet_gamma * norm_loss, {
        "weighted_ce": weighted_ce.item(),
        "eq_var": var_loss.item(),
        "rare": norm_loss.item(),
    }


def taet_loss(model, x_natural, y, optimizer, args, class_counts, group_ids,
              epoch, class_eps=None, class_weights=None, class_beta=None):
    if args.base_algorithm == "trades":
        x_adv = trades_adversary(model, x_natural, args, y=y, class_eps=class_eps)
    else:
        attack_loss = lambda logits, target: F.cross_entropy(logits, target)
        x_adv = pgd_adversary(
            model, x_natural, y, args, attack_loss,
            random_start=not getattr(args, "no_random_start", False), class_eps=class_eps)

    model.train()
    optimizer.zero_grad()
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    if epoch <= args.taet_stage1_epochs:
        natural_each = _balanced_softmax_each(logits_nat, y, class_counts)
        robust_each = _balanced_softmax_each(logits_adv, y, class_counts)
        natural_loss = _weighted_mean(natural_each, y, class_weights)
        if class_beta is not None:
            beta = torch.as_tensor(class_beta, dtype=x_natural.dtype, device=x_natural.device)[y]
            robust_loss = _weighted_mean(robust_each, y, class_weights)
            loss = _weighted_mean((1.0 - beta) * natural_each + beta * robust_each, y, class_weights)
        elif args.base_algorithm == "trades":
            natural_prob = F.softmax(logits_nat, dim=1).detach()
            robust_each = F.kl_div(
                F.log_softmax(logits_adv, dim=1), natural_prob, reduction="none").sum(dim=1)
            robust_loss = _weighted_mean(robust_each, y, class_weights)
            loss = natural_loss + args.beta * robust_loss
        else:
            robust_loss = _weighted_mean(robust_each, y, class_weights)
            loss = robust_loss
        loss_dict = {"stage": 1.0, "natural": natural_loss.item(), "robust": robust_loss.item()}
    else:
        eq_loss, eq_dict = equalization_loss(logits_adv, y, group_ids, args)
        natural_loss = _weighted_mean(_balanced_softmax_each(logits_nat, y, class_counts), y, class_weights)
        if args.base_algorithm == "trades":
            natural_prob = F.softmax(logits_nat, dim=1).detach()
            robust_loss = nn.KLDivLoss(reduction="sum")(
                F.log_softmax(logits_adv, dim=1), natural_prob) / len(x_natural)
            eq_loss = natural_loss + args.beta * robust_loss + eq_loss
        loss = eq_loss
        loss_dict = {"stage": 2.0, "natural": natural_loss.item(), **eq_dict}
    loss_dict["total"] = loss.item()
    if class_eps is not None:
        eps_batch = torch.as_tensor(class_eps, dtype=x_natural.dtype, device=x_natural.device)[y]
        loss_dict["eps_min"] = eps_batch.min().item()
        loss_dict["eps_max"] = eps_batch.max().item()
    return loss, loss_dict, x_adv


def method_loss(model, x_natural, y, optimizer, args, state):
    if args.method == "AT-BSL-LT":
        return at_bsl_loss(model, x_natural, y, optimizer, args, state["class_counts"])[:2]
    if args.method == "REAT-LT":
        loss, loss_dict, _, adv_pred = reat_loss(
            model, x_natural, y, optimizer, args, state["class_counts"],
            state["ae_counts"], state["feature_hook"], state["tail_classes"])
        state["epoch_adv_preds"].append(adv_pred)
        return loss, loss_dict
    if args.method == "TAET-LT":
        return taet_loss(
            model, x_natural, y, optimizer, args, state["class_counts"],
            state["group_ids"], state["epoch"])[:2]
    raise ValueError("Unknown method {}".format(args.method))


def make_advanced_base_state(args, class_counts, device, feature_hook=None):
    if not is_advanced_base(args):
        return None
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device).clamp_min(1.0)
    tail_classes = compute_tail_classes(np.asarray(class_counts), getattr(args, "tail_fraction", 0.8))
    group_ids = split_groups(np.asarray(class_counts))
    state = {
        "class_counts": counts,
        "tail_classes": torch.as_tensor(tail_classes, dtype=torch.long, device=device),
        "group_ids": torch.as_tensor(group_ids, dtype=torch.long, device=device),
        "ae_counts": counts.detach().cpu(),
        "feature_hook": feature_hook,
        "owns_feature_hook": False,
        "epoch": 0,
        "epoch_adv_preds": [],
    }
    logging.info("Advanced base %s tail classes: %s", args.base_algorithm, tail_classes.tolist())
    logging.info("Advanced base group ids (0=head,1=body,2=tail): %s", group_ids.tolist())
    return state


def ensure_advanced_feature_hook(args, model, state):
    if state is None or args.base_algorithm != "reat":
        return state
    if state["feature_hook"] is None:
        state["feature_hook"] = FeatureHook(model)
        state["owns_feature_hook"] = True
    return state


def close_advanced_base_state(state):
    if state is not None and state.get("owns_feature_hook") and state.get("feature_hook") is not None:
        state["feature_hook"].close()


def begin_advanced_epoch(args, state, epoch):
    if state is None:
        return
    state["epoch"] = epoch
    if args.base_algorithm == "reat":
        state["epoch_adv_preds"] = []


def finish_advanced_epoch(args, state):
    if state is None or args.base_algorithm != "reat":
        return
    preds = state.get("epoch_adv_preds", [])
    if not preds:
        return
    preds = torch.cat(preds, dim=0)
    counts = torch.bincount(preds, minlength=getattr(args, "n_class")).float().clamp_min(1.0)
    state["ae_counts"] = counts
    logging.info("REAT-base adversarial prediction counts: %s", counts.long().tolist())


def advanced_base_loss(model, x_natural, y, optimizer, args, state,
                       class_eps=None, class_weights=None, class_beta=None,
                       return_outputs=False):
    if state is None:
        raise ValueError("Advanced base state is required for {}".format(args.base_algorithm))
    if args.base_algorithm == "at-bsl":
        loss, loss_dict, x_adv = at_bsl_loss(
            model, x_natural, y, optimizer, args, state["class_counts"],
            class_eps=class_eps, class_weights=class_weights, class_beta=class_beta)
        if return_outputs:
            return loss, loss_dict, {"x_adv": x_adv}
        return loss, loss_dict
    if args.base_algorithm == "reat":
        loss, loss_dict, x_adv, adv_pred = reat_loss(
            model, x_natural, y, optimizer, args, state["class_counts"],
            state["ae_counts"], state["feature_hook"], state["tail_classes"],
            class_eps=class_eps, class_weights=class_weights, class_beta=class_beta)
        state["epoch_adv_preds"].append(adv_pred)
        if return_outputs:
            return loss, loss_dict, {"x_adv": x_adv}
        return loss, loss_dict
    if args.base_algorithm == "taet":
        loss, loss_dict, x_adv = taet_loss(
            model, x_natural, y, optimizer, args, state["class_counts"],
            state["group_ids"], state["epoch"], class_eps=class_eps,
            class_weights=class_weights, class_beta=class_beta)
        if return_outputs:
            return loss, loss_dict, {"x_adv": x_adv}
        return loss, loss_dict
    raise ValueError("Unknown advanced base {}".format(args.base_algorithm))


def train_epoch(args, model, device, optimizer, train_loader, epoch, trainset_size, state):
    model.train()
    state["epoch"] = epoch
    if args.method == "REAT-LT":
        state["epoch_adv_preds"] = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss, loss_dict = method_loss(model, data, target, optimizer, args, state)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, min((batch_idx + 1) * args.batch_size, trainset_size), trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)

    if args.method == "REAT-LT" and state["epoch_adv_preds"]:
        preds = torch.cat(state["epoch_adv_preds"], dim=0)
        counts = torch.bincount(preds, minlength=args.n_class).float().clamp_min(1.0)
        state["ae_counts"] = counts
        logging.info("REAT adversarial prediction counts: %s", counts.long().tolist())


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
        100.0 * clean_acc, 100.0 * robust_acc,
        100.0 * tail_clean_acc, 100.0 * tail_robust_acc)
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


def run(method):
    args = get_args(method)
    trainset, testset, class_counts = load_lt_imagefolder_dataset(args)
    infer_lt_dataset(args, len(trainset.classes))

    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, "training.log")),
            logging.StreamHandler(),
        ])
    logging.info("Method: %s", method)
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
    group_ids = split_groups(class_counts)
    logging.info("Tail classes: %s", tail_classes.tolist())
    logging.info("Group ids (0=head,1=body,2=tail): %s", group_ids.tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = adapt_model_for_image_size(get_model(args), args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    class_counts_tensor = torch.as_tensor(class_counts, dtype=torch.float32, device=device).clamp_min(1.0)
    state = {
        "class_counts": class_counts_tensor,
        "tail_classes": torch.as_tensor(tail_classes, dtype=torch.long, device=device),
        "group_ids": torch.as_tensor(group_ids, dtype=torch.long, device=device),
        "ae_counts": class_counts_tensor.detach().cpu(),
        "feature_hook": FeatureHook(model) if method == "REAT-LT" else None,
        "epoch": 0,
    }

    best_robust_acc = -1.0
    init_time = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)
        train_epoch(args, model, device, optimizer, train_loader, epoch, len(trainset), state)

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

    if state["feature_hook"] is not None:
        state["feature_hook"].close()
