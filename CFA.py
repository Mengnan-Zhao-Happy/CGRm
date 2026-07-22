"""
Standalone CFA training script.

Implements Class-wise Calibrated Fair Adversarial Training without changing the
original DAFA training path.
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
    parser = argparse.ArgumentParser(description="PyTorch CFA Adversarial Training")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=("cifar10", "cifar100", "stl10"))
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model", "-m", default="resnet", type=str, choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/cfa")
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

    parser.add_argument("--cfa_begin", default=10, type=int,
                        help="Epoch to start class-wise calibration.")
    parser.add_argument("--cfa_lambda1", default=0.5, type=float,
                        help="Base perturbation budget lambda_1 for CCM.")
    parser.add_argument("--cfa_lambda2", default=0.5, type=float,
                        help="Base regularization budget lambda_2 for CCR.")
    parser.add_argument("--no_ccm", action="store_true", default=False,
                        help="Disable class-wise calibrated margin.")
    parser.add_argument("--no_ccr", action="store_true", default=False,
                        help="Disable class-wise calibrated TRADES regularization.")

    return parser.parse_args()


def expand_per_sample(values, y, x):
    values = torch.as_tensor(values, dtype=x.dtype, device=x.device)
    return values[y].view(-1, 1, 1, 1)


def pgd_adversary(model, x_natural, y, class_eps, step_size, perturb_steps):
    eps_mask = expand_per_sample(class_eps, y, x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def trades_adversary(model, x_natural, y, class_eps, step_size, perturb_steps):
    criterion_kl = nn.KLDivLoss(reduction="sum")
    eps_mask = expand_per_sample(class_eps, y, x_natural)
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
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps_mask), x_natural + eps_mask)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def classwise_correct(logits, y, n_class):
    pred = logits.detach().max(1)[1]
    correct = pred.eq(y)
    counts = torch.bincount(y.detach(), minlength=n_class).float().cpu()
    correct_counts = torch.bincount(y.detach()[correct], minlength=n_class).float().cpu()
    return correct_counts, counts


def cfa_loss(model, x_natural, y, optimizer, args, class_eps, class_beta):
    model.eval()
    if args.loss == "pgd":
        x_adv = pgd_adversary(model, x_natural, y, class_eps, args.pgd_step_size, args.pgd_num_steps)
    elif args.loss == "trades":
        x_adv = trades_adversary(model, x_natural, y, class_eps, args.pgd_step_size, args.pgd_num_steps)
    else:
        raise ValueError("Unknown loss {}".format(args.loss))

    model.train()
    optimizer.zero_grad()

    adv_logits = model(x_adv)
    natural_logits = model(x_natural)

    if args.loss == "pgd":
        loss = F.cross_entropy(adv_logits, y)
        loss_dict = {"robust": loss.item()}
    else:
        beta = torch.as_tensor(class_beta, dtype=x_natural.dtype, device=x_natural.device)[y]
        natural_loss = F.cross_entropy(natural_logits, y, reduction="none")
        natural_prob = F.softmax(natural_logits, dim=1).detach()
        robust_loss = nn.KLDivLoss(reduction="none")(
            F.log_softmax(adv_logits, dim=1), natural_prob).sum(1)
        loss = ((1.0 - beta) * natural_loss + beta * robust_loss).mean()
        loss_dict = {
            "natural": natural_loss.mean().item(),
            "robust": robust_loss.mean().item(),
            "beta_min": beta.min().item(),
            "beta_max": beta.max().item(),
        }

    correct_counts, counts = classwise_correct(adv_logits, y, args.n_class)
    loss_dict["eps_min"] = torch.as_tensor(class_eps, device=x_natural.device)[y].min().item()
    loss_dict["eps_max"] = torch.as_tensor(class_eps, device=x_natural.device)[y].max().item()

    return loss, loss_dict, correct_counts, counts


def train_epoch(args, model, device, optimizer, train_loader, epoch, class_eps, class_beta, trainset_size):
    model.train()
    correct_by_class = torch.zeros(args.n_class)
    count_by_class = torch.zeros(args.n_class)

    for batch_idx, dataset in enumerate(train_loader):
        if isinstance(dataset[0], tuple):
            data, target = dataset[0]
        else:
            data, target = dataset
        data, target = data.to(device), target.to(device)

        loss, loss_dict, correct_counts, counts = cfa_loss(
            model, data, target, optimizer, args, class_eps, class_beta)

        loss.backward()
        optimizer.step()

        correct_by_class += correct_counts
        count_by_class += counts

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, (batch_idx + 1) * args.batch_size, trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)

    train_robust_acc = correct_by_class / torch.clamp(count_by_class, min=1.0)
    return train_robust_acc


def update_cfa_parameters(args, device, train_robust_acc, use_calibration):
    if use_calibration and not args.no_ccm:
        class_eps = (args.cfa_lambda1 + train_robust_acc.to(device)) * args.epsilon
    else:
        class_eps = torch.ones(args.n_class, device=device) * args.epsilon

    if args.loss == "trades":
        if use_calibration and not args.no_ccr:
            beta_raw = (args.cfa_lambda2 + train_robust_acc.to(device)) * args.beta
        else:
            beta_raw = torch.ones(args.n_class, device=device) * args.beta
        class_beta = beta_raw / (1.0 + beta_raw)
    else:
        class_beta = torch.zeros(args.n_class, device=device)

    return class_eps, class_beta


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

    best_robust_acc = -1.0
    train_robust_acc = torch.ones(args.n_class)
    class_eps, class_beta = update_cfa_parameters(args, device, train_robust_acc, False)
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logging.info("Setting learning rate to %g", lr)

        train_robust_acc = train_epoch(
            args, model, device, optimizer, train_loader, epoch,
            class_eps, class_beta, len(trainset))

        use_calibration = epoch >= args.cfa_begin
        class_eps, class_beta = update_cfa_parameters(args, device, train_robust_acc, use_calibration)
        # logging.info("CFA train robust acc: %s",
        #              torch.round(train_robust_acc * 10000) / 10000)
        # logging.info("CFA class eps: %s", torch.round(class_eps.detach().cpu() * 1000000) / 1000000)
        # if args.loss == "trades":
        #     logging.info("CFA class beta: %s", torch.round(class_beta.detach().cpu() * 1000000) / 1000000)

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
