"""Evaluate a saved LT checkpoint under CW-margin PGD and AutoAttack.

The script uses the same ImageFolder LT data path convention as the training
scripts: data_root/train and data_root/test. It reports all-class and tail-class
robust accuracy.
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lt_evaluation import compute_tail_classes
from lt_tinyimagenet_utils import (
    DATASET_CHOICES,
    adapt_model_for_image_size,
    load_lt_imagefolder_dataset,
)
from robal_base import apply_eval_adjustment, build_robal_priors, maybe_replace_classifier
from utils import get_model


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LT checkpoint with CW-margin attack and AutoAttack")

    parser.add_argument("--data_root", required=True, type=str,
                        help="ImageFolder root containing train/ and test/.")
    parser.add_argument("--checkpoint", "--resume_path", "--model_path",
                        dest="checkpoint", required=True, type=str,
                        help="Path to best.pt or final.pt.")
    parser.add_argument("--dataset", default="auto", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--base_algorithm", "--loss", dest="base_algorithm",
                        default="at", type=str,
                        help="Use robal only when loading a RoBal cosine-classifier checkpoint.")
    parser.add_argument("--robal_scale", default=30.0, type=float)
    parser.add_argument("--robal_class_margin", default=0.5, type=float)
    parser.add_argument("--robal_pair_margin", default=0.2, type=float)
    parser.add_argument("--robal_uniform_margin", default=0.0, type=float)
    parser.add_argument("--robal_eval_tau", default=1.0, type=float)

    parser.add_argument("--attack", default="both", choices=("cw", "aa", "both"))
    parser.add_argument("--test_batch_size", default=200, type=int)
    parser.add_argument("--epsilon", "--test_epsilon", dest="epsilon",
                        default=0.031, type=float)
    parser.add_argument("--tail_fraction", default=0.8, type=float)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--cw_steps", default=20, type=int)
    parser.add_argument("--cw_step_size", default=0.003, type=float)
    parser.add_argument("--cw_restarts", default=1, type=int)
    parser.add_argument("--cw_no_random_start", action="store_true", default=False)

    parser.add_argument("--aa_batch_size", default=200, type=int)
    parser.add_argument("--aa_version", default="standard", type=str,
                        choices=("standard", "plus", "rand"))
    parser.add_argument("--aa_verbose", action="store_true", default=False)
    return parser.parse_args()


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def normalize_state_dict_keys(state_dict, model):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        return state_dict

    model_has_module = model_keys[0].startswith("module.")
    ckpt_has_module = ckpt_keys[0].startswith("module.")
    if model_has_module == ckpt_has_module:
        return state_dict
    if ckpt_has_module and not model_has_module:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if model_has_module and not ckpt_has_module:
        return {"module." + k: v for k, v in state_dict.items()}
    return state_dict


def build_tail_mask(target, tail_classes):
    tail_mask = torch.zeros_like(target, dtype=torch.bool)
    for cls in tail_classes:
        tail_mask |= target.eq(int(cls))
    return tail_mask


def cw_margin(logits, y):
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits[torch.arange(len(y), device=y.device), y] = -1e9
    max_other = other_logits.max(dim=1)[0]
    return max_other - true_logits


def cw_linf_attack(model, x, y, epsilon, step_size, steps, restarts, random_start=True):
    best_margin = torch.full((len(x),), -1e9, device=x.device)
    best_adv = x.detach().clone()
    restarts = max(1, restarts)

    for _ in range(restarts):
        if random_start:
            delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
            delta = torch.min(torch.max(delta, -x), 1.0 - x)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
        else:
            x_adv = x.detach().clone()

        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                margin = cw_margin(model(x_adv), y)
                loss = margin.mean()
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            with torch.no_grad():
                cur_margin = cw_margin(model(x_adv), y)
                improve = cur_margin > best_margin
                best_margin[improve] = cur_margin[improve]
                best_adv[improve] = x_adv[improve]

    return best_adv.detach()


def evaluate_cw(args, model, device, loader, tail_classes):
    model.eval()
    total = robust_correct = 0
    tail_total = tail_robust_correct = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        x_adv = cw_linf_attack(
            model, data, target, args.epsilon, args.cw_step_size,
            args.cw_steps, args.cw_restarts,
            random_start=not args.cw_no_random_start)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        robust = pred.eq(target)
        tail_mask = build_tail_mask(target, tail_classes)

        total += len(target)
        robust_correct += robust.sum().item()
        if tail_mask.any():
            tail_total += tail_mask.sum().item()
            tail_robust_correct += robust[tail_mask].sum().item()

        logging.info("CW batch %d/%d done", batch_idx + 1, len(loader))

    return {
        "robust_acc": robust_correct / max(total, 1),
        "tail_robust_acc": tail_robust_correct / max(tail_total, 1),
    }


def collect_test_tensors(loader, device):
    xs, ys = [], []
    for data, target in loader:
        xs.append(data)
        ys.append(target)
    return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)


def evaluate_predictions(pred, target, tail_classes):
    robust = pred.eq(target)
    tail_mask = build_tail_mask(target, tail_classes)
    return {
        "robust_acc": robust.float().mean().item(),
        "tail_robust_acc": robust[tail_mask].float().mean().item() if tail_mask.any() else 0.0,
    }


def evaluate_autoattack(args, model, device, loader, tail_classes):
    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise ImportError(
            "AutoAttack is not installed. Install it first, for example: "
            "pip install git+https://github.com/fra31/auto-attack"
        ) from exc

    x_test, y_test = collect_test_tensors(loader, device)
    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=args.epsilon,
        seed=args.seed,
        version=args.aa_version,
        device=device,
        verbose=args.aa_verbose,
    )
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.aa_batch_size)
    with torch.no_grad():
        preds = []
        for start in range(0, len(x_adv), args.aa_batch_size):
            preds.append(model(x_adv[start:start + args.aa_batch_size]).argmax(dim=1))
        pred = torch.cat(preds, dim=0)
    return evaluate_predictions(pred, y_test, tail_classes)


def load_model(args, device, use_cuda):
    model = adapt_model_for_image_size(get_model(args), args)
    model = maybe_replace_classifier(model, args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint), model)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class RoBalEvalWrapper(nn.Module):
    def __init__(self, model, log_freq, args):
        super().__init__()
        self.model = model
        self.register_buffer("log_freq", log_freq.detach().clone())
        self.args = args

    def forward(self, x):
        logits = self.model(x)
        return apply_eval_adjustment(logits, self.log_freq, self.args)


def maybe_wrap_eval_model(args, model, class_counts, device):
    if args.base_algorithm != "robal":
        return model
    log_freq, _, _ = build_robal_priors(class_counts, args, device)
    wrapped = RoBalEvalWrapper(model, log_freq, args).to(device)
    wrapped.eval()
    return wrapped


def main():
    args = get_args()
    args.base_algorithm = args.base_algorithm.lower()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import time
    current_time=time.time()
    print(current_time)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Args: %s", args)

    _, testset, class_counts = load_lt_imagefolder_dataset(args)
    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)
    logging.info("Detected dataset: %s, n_class: %d, image_size: %d",
                 args.dataset, args.n_class, args.image_size)
    logging.info("Tail classes: %s", tail_classes.tolist())

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {
        "num_workers": args.num_workers}
    test_loader = DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    model = load_model(args, device, use_cuda)
    eval_model = maybe_wrap_eval_model(args, model, class_counts, device)
    logging.info("Loaded checkpoint: %s", args.checkpoint)

    if args.attack in ("cw", "both"):
        cw = evaluate_cw(args, eval_model, device, test_loader, tail_classes)
        logging.info("CW: Robust(all) %.2f%%, Robust(tail) %.2f%%",
                     100.0 * cw["robust_acc"], 100.0 * cw["tail_robust_acc"])

    if args.attack in ("aa", "both"):
        aa = evaluate_autoattack(args, eval_model, device, test_loader, tail_classes)
        logging.info("AA: Robust(all) %.2f%%, Robust(tail) %.2f%%",
                     100.0 * aa["robust_acc"], 100.0 * aa["tail_robust_acc"])

    print(time.time()-current_time)
if __name__ == "__main__":
    main()
