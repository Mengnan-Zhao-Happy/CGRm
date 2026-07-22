import logging

import numpy as np
import torch
import torch.nn.functional as F


def compute_tail_classes(class_counts, tail_fraction):
    num_tail = max(1, int(np.ceil(len(class_counts) * tail_fraction)))
    return np.argsort(class_counts)[:num_tail].astype(np.int64)


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


def evaluate_lt(args, model, device, loader, tail_classes):
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
