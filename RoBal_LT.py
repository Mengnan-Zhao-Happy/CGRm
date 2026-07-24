"""
Standalone RoBal-style training on long-tailed ImageFolder datasets.
"""

import argparse
import logging
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


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


def get_args():
    parser = argparse.ArgumentParser(description="RoBal-style training for long-tailed ImageFolder data")

    parser.add_argument("--data_root", default="./data/CIFAR100-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/robal_lt")
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
                        choices=("at", "pgd", "trades", "AT", "PGD", "Trades", "TRADES"),
                        help="Base adversarial training objective used by RoBal.")
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--beta", default=4.0, type=float,
                        help="TRADES KL coefficient.")
    parser.add_argument("--no_random_start", action="store_true", default=False)

    parser.add_argument("--robal_scale", default=30.0, type=float,
                        help="Scale of the cosine classifier logits.")
    parser.add_argument("--robal_class_margin", default=0.5, type=float,
                        help="Maximum LDAM-style class-aware margin.")
    parser.add_argument("--robal_pair_margin", default=0.2, type=float,
                        help="Strength of pair-aware source-target margin.")
    parser.add_argument("--robal_uniform_margin", default=0.0, type=float,
                        help="Uniform margin subtracted from the true class logit.")
    parser.add_argument("--robal_eval_tau", default=1.0, type=float,
                        help="Boundary adjustment strength at evaluation.")
    parser.add_argument("--robal_natural_weight", default=1.0, type=float,
                        help="Clean margin CE weight for AT-style RoBal.")

    args = parser.parse_args()
    args.base_algorithm = args.base_algorithm.lower()
    if args.base_algorithm == "pgd":
        args.base_algorithm = "at"
    return args


def load_imagefolder_dataset(args):
    return load_lt_imagefolder_dataset(args)


def infer_dataset(args, n_class):
    infer_lt_dataset(args, n_class)


def compute_tail_classes(class_counts, tail_fraction):
    num_tail = max(1, int(np.ceil(len(class_counts) * tail_fraction)))
    return np.argsort(class_counts)[:num_tail].astype(np.int64)


def replace_classifier_with_cosine(model, scale):
    if hasattr(model, "linear"):
        old = model.linear
        model.linear = CosineLinear(old.in_features, old.out_features, scale=scale)
        return model
    if hasattr(model, "fc"):
        old = model.fc
        model.fc = CosineLinear(old.in_features, old.out_features, scale=scale)
        return model
    raise ValueError("Cannot find classifier layer to replace with CosineLinear.")


def build_robal_priors(class_counts, args, device):
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device).clamp_min(1.0)
    log_freq = torch.log(counts / counts.max()).detach()

    class_margin = counts.pow(-0.25)
    class_margin = class_margin / class_margin.max().clamp_min(1e-12) * args.robal_class_margin

    pair_margin = torch.zeros((len(counts), len(counts)), dtype=torch.float32, device=device)
    if args.robal_pair_margin > 0:
        # Positive for source classes that are rarer than the target class.
        raw = torch.relu(log_freq.view(1, -1) - log_freq.view(-1, 1))
        if raw.max().item() > 0:
            raw = raw / raw.max().clamp_min(1e-12)
        pair_margin = args.robal_pair_margin * raw
        pair_margin.fill_diagonal_(0.0)

    return log_freq, class_margin.detach(), pair_margin.detach()


def apply_training_margins(logits, y, class_margin, pair_margin, args):
    logits = logits.clone()
    idx = torch.arange(len(y), device=logits.device)
    logits[idx, y] -= class_margin[y] + args.robal_uniform_margin
    if args.robal_pair_margin > 0:
        logits = logits + pair_margin[y]
    return logits


def apply_eval_adjustment(logits, log_freq, args):
    if args.robal_eval_tau <= 0:
        return logits
    return logits - args.robal_eval_tau * log_freq.view(1, -1).to(logits.device)


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


def robal_loss(model, x_natural, y, optimizer, args, class_margin, pair_margin):
    if args.base_algorithm == "trades":
        x_adv = trades_adversary(
            model, x_natural, args.pgd_step_size, args.epsilon, args.pgd_num_steps)
    else:
        x_adv = pgd_adversary(
            model, x_natural, y, args.epsilon, args.pgd_step_size,
            args.pgd_num_steps, random_start=not args.no_random_start)

    model.train()
    optimizer.zero_grad()

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    logits_nat_m = apply_training_margins(logits_nat, y, class_margin, pair_margin, args)
    logits_adv_m = apply_training_margins(logits_adv, y, class_margin, pair_margin, args)

    natural_loss = F.cross_entropy(logits_nat_m, y)

    if args.base_algorithm == "trades":
        natural_prob = F.softmax(logits_nat, dim=1).detach()
        robust_loss = nn.KLDivLoss(reduction="sum")(
            F.log_softmax(logits_adv, dim=1), natural_prob) / len(x_natural)
        loss = natural_loss + args.beta * robust_loss
    else:
        robust_loss = F.cross_entropy(logits_adv_m, y)
        loss = args.robal_natural_weight * natural_loss + robust_loss

    return loss, {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "total": loss.item(),
    }


def train_epoch(args, model, device, optimizer, train_loader, epoch,
                class_margin, pair_margin, trainset_size):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss, loss_dict = robal_loss(
            model, data, target, optimizer, args, class_margin, pair_margin)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, min((batch_idx + 1) * args.batch_size, trainset_size),
                trainset_size, 100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)


def pgd_eval(model, x, y, log_freq, args):
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


def evaluate(args, model, device, loader, tail_classes, log_freq):
    model.eval()
    tail_classes = set(int(c) for c in tail_classes)
    total = clean_correct = robust_correct = 0
    tail_total = tail_clean_correct = tail_robust_correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            clean_logits = apply_eval_adjustment(model(data), log_freq, args)
            clean_pred = clean_logits.max(1)[1]

        x_adv = pgd_eval(model, data, target, log_freq, args)
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
    logging.info("Method: RoBal-LT (%s)", args.base_algorithm.upper())
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
    model = replace_classifier_with_cosine(model, args.robal_scale)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    log_freq, class_margin, pair_margin = build_robal_priors(class_counts, args, device)
    logging.info("Class margins: %s", (torch.round(class_margin.cpu() * 10000) / 10000).tolist())
    logging.info("Pair margin mean %.6f, max %.6f", pair_margin.mean().item(), pair_margin.max().item())

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)

        train_epoch(args, model, device, optimizer, train_loader, epoch,
                    class_margin, pair_margin, len(trainset))

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            eval_data = evaluate(args, model, device, test_loader, tail_classes, log_freq)
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
