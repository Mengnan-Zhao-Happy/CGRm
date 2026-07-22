"""Shared RoBal base utilities for long-tailed adversarial training scripts."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoBal_LT import (
    apply_eval_adjustment,
    apply_training_margins,
    build_robal_priors,
    replace_classifier_with_cosine,
)


def add_robal_args(parser):
    parser.add_argument("--robal_scale", default=30.0, type=float,
                        help="Scale of the RoBal cosine classifier logits.")
    parser.add_argument("--robal_class_margin", default=0.5, type=float,
                        help="Maximum RoBal class-aware margin.")
    parser.add_argument("--robal_pair_margin", default=0.2, type=float,
                        help="Strength of RoBal pair-aware source-target margin.")
    parser.add_argument("--robal_uniform_margin", default=0.0, type=float,
                        help="Uniform margin subtracted from the true class logit.")
    parser.add_argument("--robal_eval_tau", default=1.0, type=float,
                        help="RoBal boundary adjustment strength during evaluation.")
    parser.add_argument("--robal_natural_weight", default=1.0, type=float,
                        help="Clean margin CE weight for RoBal-base AT.")


def is_robal(args):
    return getattr(args, "base_algorithm", "").lower() == "robal"


def maybe_replace_classifier(model, args):
    if is_robal(args):
        model = replace_classifier_with_cosine(model, args.robal_scale)
    return model


def make_robal_state(class_counts, args, device):
    if not is_robal(args):
        return None
    log_freq, class_margin, pair_margin = build_robal_priors(class_counts, args, device)
    logging.info("RoBal class margins: %s",
                 (torch.round(class_margin.cpu() * 10000) / 10000).tolist())
    logging.info("RoBal pair margin mean %.6f, max %.6f",
                 pair_margin.mean().item(), pair_margin.max().item())
    return {
        "log_freq": log_freq,
        "class_margin": class_margin,
        "pair_margin": pair_margin,
    }


def _expand_per_sample(values, y, x):
    values = torch.as_tensor(values, dtype=x.dtype, device=x.device)
    return values[y].view(-1, 1, 1, 1)


def robal_pgd_adversary(model, x_natural, y, args, class_eps=None):
    if class_eps is None:
        eps_mask = torch.full_like(x_natural[:, :1, :1, :1], args.epsilon)
    else:
        eps_mask = _expand_per_sample(class_eps, y, x_natural)
    step_mask = eps_mask / max(args.epsilon, 1e-12) * args.pgd_step_size

    random_start = not getattr(args, "no_random_start", False)
    if random_start:
        x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-1.0, 1.0) * eps_mask
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_natural.detach()

    model.eval()
    for _ in range(args.pgd_num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_mask * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def robal_margin_loss(model, x_natural, y, optimizer, args, robal_state,
                      class_eps=None, class_weights=None, class_beta=None,
                      batch_indices=None, memory_dict=None, return_outputs=False):
    if robal_state is None:
        raise ValueError("RoBal state is required when --base_algorithm RoBal is used.")

    x_adv = robal_pgd_adversary(model, x_natural, y, args, class_eps=class_eps)

    model.train()
    optimizer.zero_grad()

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    logits_nat_m = apply_training_margins(
        logits_nat, y, robal_state["class_margin"], robal_state["pair_margin"], args)
    logits_adv_m = apply_training_margins(
        logits_adv, y, robal_state["class_margin"], robal_state["pair_margin"], args)

    natural_each = F.cross_entropy(logits_nat_m, y, reduction="none")
    robust_each = F.cross_entropy(logits_adv_m, y, reduction="none")

    if class_weights is None:
        weights = torch.ones_like(natural_each)
    else:
        weights = torch.as_tensor(class_weights, dtype=x_natural.dtype, device=x_natural.device)[y]

    if class_beta is None:
        loss_each = args.robal_natural_weight * natural_each + robust_each
    else:
        beta = torch.as_tensor(class_beta, dtype=x_natural.dtype, device=x_natural.device)[y]
        loss_each = (1.0 - beta) * natural_each + beta * robust_each

    natural_loss = (natural_each * weights).mean()
    robust_loss = (robust_each * weights).mean()
    loss = (loss_each * weights).mean()

    if memory_dict is not None and batch_indices is not None:
        memory_dict["probs"][batch_indices] = F.softmax(logits_adv, dim=1).detach().cpu().numpy()
        memory_dict["labels"][batch_indices] = y.detach().cpu().numpy()

    loss_dict = {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "total": loss.item(),
    }
    if class_eps is not None:
        eps_batch = torch.as_tensor(class_eps, dtype=x_natural.dtype, device=x_natural.device)[y]
        loss_dict["eps_min"] = eps_batch.min().item()
        loss_dict["eps_max"] = eps_batch.max().item()
    if class_weights is not None:
        loss_dict["w_min"] = weights.min().item()
        loss_dict["w_max"] = weights.max().item()

    outputs = {"x_adv": x_adv, "logits_nat": logits_nat, "logits_adv": logits_adv}
    if return_outputs:
        return loss, loss_dict, outputs
    return loss, loss_dict


def robal_pgd_eval(model, x, y, log_freq, args):
    delta = torch.empty_like(x).uniform_(-args.test_epsilon, args.test_epsilon)
    delta = torch.min(torch.max(delta, -x), 1.0 - x)

    for _ in range(args.test_pgd_num_steps):
        delta.requires_grad_()
        with torch.enable_grad():
            logits = apply_eval_adjustment(
                model(torch.clamp(x + delta, 0.0, 1.0)), log_freq, args)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [delta])[0]
        delta = delta.detach() + args.test_pgd_step_size * torch.sign(grad.detach())
        delta = torch.clamp(delta, -args.test_epsilon, args.test_epsilon)
        delta = torch.min(torch.max(delta, -x), 1.0 - x)
    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def evaluate_robal_lt(args, model, device, loader, tail_classes, robal_state):
    if robal_state is None:
        raise ValueError("RoBal state is required for RoBal evaluation.")

    model.eval()
    log_freq = robal_state["log_freq"]
    tail_classes = set(int(c) for c in tail_classes)
    total = clean_correct = robust_correct = 0
    tail_total = tail_clean_correct = tail_robust_correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            clean_logits = apply_eval_adjustment(model(data), log_freq, args)
            clean_pred = clean_logits.max(1)[1]

        x_adv = robal_pgd_eval(model, data, target, log_freq, args)
        with torch.no_grad():
            robust_logits = apply_eval_adjustment(model(x_adv), log_freq, args)
            robust_pred = robust_logits.max(1)[1]

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
