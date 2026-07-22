"""
Standalone UDR training script.

Implements a Unified Wasserstein Distributional Robustness style adversary
without changing DAFA.py, CFA.py, or the original loss functions.
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import CustomDataset
from evaluation import eval
from utils import get_model


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch UDR Adversarial Training")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=("cifar10", "cifar100", "stl10"))
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model", "-m", default="resnet", type=str, choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/udr")
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

    parser.add_argument("--loss", default="trades", type=str, choices=("trades", "pgd"))
    parser.add_argument("--epsilon", default=0.031, type=float)
    parser.add_argument("--test_epsilon", default=0.031, type=float)
    parser.add_argument("--pgd_num_steps", default=10, type=int)
    parser.add_argument("--pgd_step_size", default=0.007, type=float)
    parser.add_argument("--test_pgd_num_steps", default=20, type=int)
    parser.add_argument("--test_pgd_step_size", default=0.003, type=float)
    parser.add_argument("--beta", default=4.0, type=float)

    parser.add_argument("--lamda_init", default=1.0, type=float,
                        help="Initial UDR lambda value.")
    parser.add_argument("--lamda_lr", default=2e-2, type=float,
                        help="Learning rate for UDR lambda updates.")
    parser.add_argument("--lamda_period", default=10, type=int,
                        help="Update lambda every N training batches.")
    parser.add_argument("--lamda_min", default=0.0, type=float,
                        help="Minimum lambda value. Theory requires lambda >= 0.")
    parser.add_argument("--udr_tau", default=None, type=float,
                        help="Soft-ball temperature. Defaults to pgd_step_size.")
    parser.add_argument("--pgd_natural_weight", default=1.0, type=float,
                        help="Natural CE weight for UDR-PGD.")

    return parser.parse_args()


def clip_delta_to_valid_image(delta, x):
    return torch.min(torch.max(delta, -x), 1.0 - x)


def mean_l1_distance(delta):
    return delta.detach().abs().view(delta.size(0), -1).mean(1).mean()


def udr_adversary(model, x_natural, y, args, lamda):
    epsilon = args.epsilon
    alpha = args.pgd_step_size
    tau = args.udr_tau if args.udr_tau is not None else alpha
    tau = max(tau, 1e-12)

    delta = torch.empty_like(x_natural).uniform_(-epsilon, epsilon)
    delta = clip_delta_to_valid_image(delta, x_natural)

    model.eval()
    if args.loss == "trades":
        with torch.no_grad():
            natural_prob = F.softmax(model(x_natural), dim=1).detach()

    for _ in range(args.pgd_num_steps):
        delta.requires_grad_()
        x_adv = torch.clamp(x_natural + delta, 0.0, 1.0)

        with torch.enable_grad():
            adv_logits = model(x_adv)
            if args.loss == "pgd":
                attack_loss = F.cross_entropy(adv_logits, y)
            elif args.loss == "trades":
                attack_loss = nn.KLDivLoss(reduction="sum")(
                    F.log_softmax(adv_logits, dim=1), natural_prob)
            else:
                raise ValueError("Unknown loss {}".format(args.loss))

        grad = torch.autograd.grad(attack_loss, [delta])[0]
        delta = delta.detach() + alpha * torch.sign(grad.detach())

        # UDR soft-ball correction. Values inside the epsilon box are untouched;
        # values outside are softly pulled back according to lambda.
        abs_delta = delta.detach().abs()
        correction = (delta - torch.sign(delta) * epsilon) * (abs_delta > epsilon)
        delta = delta - lamda.detach() * alpha / tau * correction
        delta = clip_delta_to_valid_image(delta, x_natural)

    return delta.detach()


def udr_loss(model, x_natural, y, optimizer, args, lamda):
    delta = udr_adversary(model, x_natural, y, args, lamda)
    x_adv = torch.clamp(x_natural + delta, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    natural_loss = F.cross_entropy(logits_nat, y)
    if args.loss == "pgd":
        robust_loss = F.cross_entropy(logits_adv, y)
        loss = args.pgd_natural_weight * natural_loss + args.beta * robust_loss
    elif args.loss == "trades":
        natural_prob = F.softmax(logits_nat, dim=1).detach()
        robust_loss = nn.KLDivLoss(reduction="sum")(
            F.log_softmax(logits_adv, dim=1), natural_prob) / len(x_natural)
        loss = natural_loss + args.beta * robust_loss
    else:
        raise ValueError("Unknown loss {}".format(args.loss))

    loss_dict = {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "delta_l1": mean_l1_distance(delta).item(),
        "delta_linf": delta.detach().abs().amax().item(),
        "lamda": lamda.item(),
    }
    return loss, loss_dict, delta


def update_lamda(args, lamda, distances):
    if not distances:
        return lamda
    avg_distance = torch.stack(distances).mean().to(lamda.device)
    lamda = lamda - args.lamda_lr * (args.epsilon - avg_distance)
    if args.lamda_min is not None:
        lamda = torch.clamp(lamda, min=args.lamda_min)
    return lamda.detach()


def train_epoch(args, model, device, optimizer, train_loader, epoch, lamda, trainset_size):
    model.train()
    distance_buffer = []

    for batch_idx, dataset in enumerate(train_loader):
        if isinstance(dataset[0], tuple):
            data, target = dataset[0]
        else:
            data, target = dataset
        data, target = data.to(device), target.to(device)

        loss, loss_dict, delta = udr_loss(model, data, target, optimizer, args, lamda)
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

    lamda = update_lamda(args, lamda, distance_buffer)
    return lamda


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
    args.n_class = 100 if args.dataset == "cifar100" else 10

    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, "training.log")),
            logging.StreamHandler()
        ])
    logging.info("Args: %s", args)

    final_checkpoint_path = os.path.join(args.model_dir, "final.pt")
    if not args.overwrite and os.path.exists(final_checkpoint_path):
        logging.info("Final checkpoint found - quitting. Use --overwrite to train again.")
        sys.exit(0)

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    trainset = CustomDataset(dataset=args.dataset, root=args.data_dir, train=True,
                             download=True, get_indices=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = CustomDataset(dataset=args.dataset, root=args.data_dir, train=False,
                            download=True)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = get_model(args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    lamda = torch.tensor(args.lamda_init, dtype=torch.float32, device=device)
    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)

        lamda = train_epoch(args, model, device, optimizer, train_loader,
                            epoch, lamda, len(trainset))
        logging.info("UDR lambda after epoch %d: %.6f", epoch, lamda.item())

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            eval_data = eval(args, model, device, "test", test_loader)
            robust_acc = eval_data["test_robust_accuracy"]
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
