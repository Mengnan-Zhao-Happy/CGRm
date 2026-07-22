"""
Standalone UDR training on CIFAR-LT ImageFolder datasets.

This file keeps UDR.py unchanged and only replaces the dataset entry with
ImageFolder so UDR can be compared on CIFAR10-LT/CIFAR100-LT.
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from UDR import adjust_learning_rate, mean_l1_distance, train_epoch, udr_adversary, udr_loss, update_lamda
from AWP_LT import add_into_weights, diff_in_weights
from lt_evaluation import compute_tail_classes, evaluate_lt
from lt_tinyimagenet_utils import (
    DATASET_CHOICES,
    adapt_model_for_image_size,
    infer_lt_dataset,
    load_lt_imagefolder_dataset,
)
from robal_base import (
    add_robal_args,
    apply_training_margins,
    evaluate_robal_lt,
    is_robal,
    make_robal_state,
    maybe_replace_classifier,
)
from lt_advanced_baselines import (
    ADVANCED_BASE_CHOICES,
    add_advanced_base_args,
    balanced_softmax_loss,
    begin_advanced_epoch,
    close_advanced_base_state,
    ensure_advanced_feature_hook,
    equalization_loss,
    finish_advanced_epoch,
    is_advanced_base,
    make_advanced_base_state,
    normalize_base_algorithm,
    tail_feature_loss,
)
from utils import get_model


def get_args():
    parser = argparse.ArgumentParser(description="UDR training for CIFAR-LT ImageFolder data")

    parser.add_argument("--data_root", default="./data/CIFAR100-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--image_size", default=0, type=int,
                        help="Input image size. Use 0 to infer 32 for CIFAR and 64 for TinyImageNet.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/udr_lt")
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

    parser.add_argument("--lamda_init", default=1.0, type=float)
    parser.add_argument("--lamda_lr", default=2e-2, type=float)
    parser.add_argument("--lamda_period", default=10, type=int)
    parser.add_argument("--lamda_min", default=0.0, type=float)
    parser.add_argument("--udr_tau", default=None, type=float)
    parser.add_argument("--pgd_natural_weight", default=1.0, type=float)
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


def load_imagefolder_dataset(args):
    return load_lt_imagefolder_dataset(args)


def infer_dataset(args, n_class):
    infer_lt_dataset(args, n_class)


def calc_awp(model, proxy, proxy_optimizer, loss_fn):
    proxy.load_state_dict(model.state_dict())
    proxy.train()
    loss = -loss_fn(proxy)
    proxy_optimizer.zero_grad()
    loss.backward()
    proxy_optimizer.step()
    return diff_in_weights(model, proxy)


def train_epoch_awp(args, model, proxy, proxy_optimizer, device, optimizer,
                    train_loader, epoch, lamda, trainset_size):
    model.train()
    distance_buffer = []

    for batch_idx, dataset in enumerate(train_loader):
        if isinstance(dataset[0], tuple):
            data, target = dataset[0]
        else:
            data, target = dataset
        data, target = data.to(device), target.to(device)

        awp_diff = None
        if args.awp_gamma > 0.0 and epoch >= args.awp_warmup:
            awp_diff = calc_awp(
                model, proxy, proxy_optimizer,
                lambda net: udr_loss(net, data, target, optimizer, args, lamda)[0])
            add_into_weights(model, awp_diff, args.awp_gamma)

        loss, loss_dict, delta = udr_loss(model, data, target, optimizer, args, lamda)
        loss_dict["awp"] = float(awp_diff is not None)
        loss.backward()
        optimizer.step()
        if awp_diff is not None:
            add_into_weights(model, awp_diff, -args.awp_gamma)

        distance_buffer.append(mean_l1_distance(delta).detach().cpu())
        if (batch_idx + 1) % args.lamda_period == 0:
            lamda = update_lamda(args, lamda, distance_buffer)
            distance_buffer = []

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, (batch_idx + 1) * args.batch_size, trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)

    return update_lamda(args, lamda, distance_buffer)


def udr_robal_loss(model, x_natural, y, optimizer, args, lamda, robal_state):
    delta = udr_adversary(model, x_natural, y, args, lamda)
    x_adv = torch.clamp(x_natural + delta, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    logits_nat_m = apply_training_margins(
        logits_nat, y, robal_state["class_margin"], robal_state["pair_margin"], args)
    logits_adv_m = apply_training_margins(
        logits_adv, y, robal_state["class_margin"], robal_state["pair_margin"], args)

    natural_loss = F.cross_entropy(logits_nat_m, y)
    robust_loss = F.cross_entropy(logits_adv_m, y)
    loss = args.robal_natural_weight * natural_loss + robust_loss

    loss_dict = {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "delta_l1": mean_l1_distance(delta).item(),
        "delta_linf": delta.detach().abs().amax().item(),
        "lamda": lamda.item(),
    }
    return loss, loss_dict, delta


def train_epoch_robal(args, model, device, optimizer, train_loader, epoch,
                      lamda, robal_state, trainset_size):
    model.train()
    distance_buffer = []

    for batch_idx, dataset in enumerate(train_loader):
        if isinstance(dataset[0], tuple):
            data, target = dataset[0]
        else:
            data, target = dataset
        data, target = data.to(device), target.to(device)

        loss, loss_dict, delta = udr_robal_loss(
            model, data, target, optimizer, args, lamda, robal_state)
        loss.backward()
        optimizer.step()

        distance_buffer.append(mean_l1_distance(delta).detach().cpu())
        if (batch_idx + 1) % args.lamda_period == 0:
            lamda = update_lamda(args, lamda, distance_buffer)
            distance_buffer = []

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, (batch_idx + 1) * args.batch_size, trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)

    return update_lamda(args, lamda, distance_buffer)


def udr_advanced_loss(model, x_natural, y, optimizer, args, lamda, advanced_state):
    delta = udr_adversary(model, x_natural, y, args, lamda)
    x_adv = torch.clamp(x_natural + delta, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    natural_loss = balanced_softmax_loss(logits_nat, y, advanced_state["class_counts"])
    if args.base_algorithm == "taet" and advanced_state["epoch"] > args.taet_stage1_epochs:
        loss, eq_dict = equalization_loss(logits_adv, y, advanced_state["group_ids"], args)
        robust_loss = loss
        loss_dict = {"natural": natural_loss.item(), **eq_dict}
    else:
        robust_loss = balanced_softmax_loss(logits_adv, y, advanced_state["class_counts"])
        loss = robust_loss
        loss_dict = {"natural": natural_loss.item(), "robust": robust_loss.item()}

    if args.base_algorithm == "reat":
        feat_loss = tail_feature_loss(
            advanced_state["feature_hook"].features, y, advanced_state["tail_classes"],
            margin=args.reat_tail_margin, tau=args.reat_tail_tau)
        if feat_loss is None:
            feat_loss = logits_adv.new_tensor(0.0)
        loss = loss + args.reat_tail_lambda * feat_loss
        loss_dict["tail_feat"] = feat_loss.item()
        with torch.no_grad():
            advanced_state["epoch_adv_preds"].append(logits_adv.argmax(dim=1).detach().cpu())

    loss_dict.update({
        "total": loss.item(),
        "delta_l1": mean_l1_distance(delta).item(),
        "delta_linf": delta.detach().abs().amax().item(),
        "lamda": lamda.item(),
    })
    return loss, loss_dict, delta


def train_epoch_advanced(args, model, device, optimizer, train_loader, epoch,
                         lamda, advanced_state, trainset_size):
    model.train()
    distance_buffer = []
    for batch_idx, dataset in enumerate(train_loader):
        if isinstance(dataset[0], tuple):
            data, target = dataset[0]
        else:
            data, target = dataset
        data, target = data.to(device), target.to(device)

        loss, loss_dict, delta = udr_advanced_loss(
            model, data, target, optimizer, args, lamda, advanced_state)
        loss.backward()
        optimizer.step()

        distance_buffer.append(mean_l1_distance(delta).detach().cpu())
        if (batch_idx + 1) % args.lamda_period == 0:
            lamda = update_lamda(args, lamda, distance_buffer)
            distance_buffer = []

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, (batch_idx + 1) * args.batch_size, trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)
    return update_lamda(args, lamda, distance_buffer)


def main():
    args = get_args()

    trainset, testset, class_counts = load_imagefolder_dataset(args)
    infer_dataset(args, len(trainset.classes))
    tail_classes = compute_tail_classes(class_counts, args.tail_fraction)

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
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)
    proxy = copy.deepcopy(model) if args.base_algorithm == "awp" else None
    proxy_optimizer = optim.SGD(proxy.parameters(), lr=args.awp_lr) if proxy is not None else None
    robal_state = make_robal_state(class_counts, args, device)
    advanced_state = make_advanced_base_state(args, class_counts, device)
    advanced_state = ensure_advanced_feature_hook(args, model, advanced_state)

    lamda = torch.tensor(args.lamda_init, dtype=torch.float32, device=device)
    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)
        begin_advanced_epoch(args, advanced_state, epoch)

        if is_advanced_base(args):
            lamda = train_epoch_advanced(args, model, device, optimizer, train_loader,
                                         epoch, lamda, advanced_state, len(trainset))
        elif is_robal(args):
            lamda = train_epoch_robal(args, model, device, optimizer, train_loader,
                                      epoch, lamda, robal_state, len(trainset))
        elif args.base_algorithm == "awp":
            lamda = train_epoch_awp(args, model, proxy, proxy_optimizer, device,
                                    optimizer, train_loader, epoch, lamda, len(trainset))
        else:
            lamda = train_epoch(args, model, device, optimizer, train_loader,
                                epoch, lamda, len(trainset))
        finish_advanced_epoch(args, advanced_state)
        logging.info("UDR lambda after epoch %d: %.6f", epoch, lamda.item())

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
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

        elapsed_time = time.time() - init_time
        print("elapsed time : %d h %d m %d s" % (
            elapsed_time / 3600, (elapsed_time % 3600) / 60, elapsed_time % 60))

    close_advanced_base_state(advanced_state)


if __name__ == "__main__":
    main()
