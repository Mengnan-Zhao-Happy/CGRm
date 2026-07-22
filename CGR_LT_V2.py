"""
CGR-LT V2: Gated Confusion-Geometry Robust Long-Tail Training.

This is a standalone V2 research prototype built on CGR_LT.py. Compared with
CGR-LT V1, V2 adds three more conservative and paper-friendly ideas:

1) Reliability-gated confusion geometry graph.
   Edges are emphasized only when robust confusion is large, the source class
   has below-average robust accuracy, and the feature geometry is close.

2) Prototype contrastive regularization.
   Class prototypes are updated from evaluation features. Adversarial features
   are pulled to their own prototype and repelled from graph-selected hard
   negative prototypes.

3) Staged activation.
   Feedback weights, graph margin, prototype loss, and class-wise beta can start
   at different epochs and ramp up smoothly, reducing early-training noise.
"""

import argparse
import logging
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import CGR_LT as cgr
from utils import get_model


def get_args():
    parser = argparse.ArgumentParser(description="CGR-LT V2 training for CIFAR-LT")

    parser.add_argument("--data_root", default="./data/CIFAR10-LT-IR50", type=str)
    parser.add_argument("--dataset", default="auto", type=str,
                        choices=("auto", "cifar10", "cifar100"),
                        help="Dataset name passed to the model factory. 'auto' infers it from ImageFolder class count.")
    parser.add_argument("--model", "-m", default="resnet", type=str,
                        choices=("resnet", "pre-resnet", "wrn-28-10"))
    parser.add_argument("--model_dir", default="./model_output/cgr_lt_v2")
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
                        default="trades", type=str, choices=("pgd", "trades"))
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

    parser.add_argument("--feedback_start", default=10, type=int)
    parser.add_argument("--graph_start", default=20, type=int)
    parser.add_argument("--proto_start", default=20, type=int)
    parser.add_argument("--ramp_epochs", default=20, type=int)

    parser.add_argument("--use_feedback_weight", action="store_true", default=True)
    parser.add_argument("--no_feedback_weight", dest="use_feedback_weight", action="store_false")
    parser.add_argument("--feedback_momentum", default=0.8, type=float)
    parser.add_argument("--weight_lambda", default=0.8, type=float)
    parser.add_argument("--conf_lambda", default=0.5, type=float)
    parser.add_argument("--weight_max", default=2.5, type=float)
    parser.add_argument("--weight_natural", action="store_true", default=False)

    parser.add_argument("--use_feedback_eps", action="store_true", default=False)
    parser.add_argument("--eps_lambda", default=0.2, type=float)
    parser.add_argument("--eps_min", default=0.0, type=float)
    parser.add_argument("--eps_max", default=0.0, type=float)

    parser.add_argument("--use_balanced_ce", action="store_true", default=True)
    parser.add_argument("--no_balanced_ce", dest="use_balanced_ce", action="store_false")
    parser.add_argument("--prior_tau", default=1.0, type=float)

    parser.add_argument("--use_classwise_beta", action="store_true", default=True)
    parser.add_argument("--no_classwise_beta", dest="use_classwise_beta", action="store_false")
    parser.add_argument("--beta_lambda", default=0.4, type=float)
    parser.add_argument("--beta_min_scale", default=0.5, type=float)
    parser.add_argument("--beta_max_scale", default=1.8, type=float)

    parser.add_argument("--use_cgr_margin", action="store_true", default=True)
    parser.add_argument("--no_cgr_margin", dest="use_cgr_margin", action="store_false")
    parser.add_argument("--margin_lambda", default=0.05, type=float)
    parser.add_argument("--margin_m", default=0.5, type=float)
    parser.add_argument("--graph_momentum", default=0.85, type=float)
    parser.add_argument("--graph_topk", default=2, type=int)
    parser.add_argument("--head_gamma", default=0.5, type=float)
    parser.add_argument("--geometry_gamma", default=0.5, type=float)
    parser.add_argument("--graph_eps", default=1e-6, type=float)
    parser.add_argument("--min_edge_conf", default=0.02, type=float)

    parser.add_argument("--use_proto", action="store_true", default=True)
    parser.add_argument("--no_proto", dest="use_proto", action="store_false")
    parser.add_argument("--proto_lambda", default=0.05, type=float)
    parser.add_argument("--proto_temp", default=0.2, type=float)
    parser.add_argument("--proto_margin", default=0.2, type=float)
    parser.add_argument("--proto_momentum", default=0.9, type=float)

    return parser.parse_args()


def ramp_value(epoch, start_epoch, ramp_epochs):
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min((epoch - start_epoch + 1.0) / ramp_epochs, 1.0)


def build_class_betas_v2(args, feedback_score, clean_acc_score, beta_ramp):
    if not args.use_classwise_beta:
        return torch.ones_like(feedback_score) * args.beta
    feedback = feedback_score / feedback_score.mean().clamp_min(1e-12)
    clean_gate = clean_acc_score / clean_acc_score.mean().clamp_min(1e-12)
    score = 1.0 + args.beta_lambda * beta_ramp * (feedback - 1.0) * clean_gate
    score = torch.clamp(score, min=args.beta_min_scale, max=args.beta_max_scale)
    score = score / score.mean().clamp_min(1e-12)
    return args.beta * score


def prototype_losses(features, y, prototypes, class_graph, class_weights, args, proto_ramp):
    if (not args.use_proto) or proto_ramp <= 0 or prototypes is None:
        zero = features.new_tensor(0.0)
        return zero, zero

    valid = prototypes.abs().sum(dim=1) > 0
    if valid.sum().item() < 2:
        zero = features.new_tensor(0.0)
        return zero, zero

    feat = F.normalize(features, dim=1)
    proto = F.normalize(prototypes.to(features.device), dim=1)
    proto_logits = feat.matmul(proto.t()) / max(args.proto_temp, 1e-6)
    proto_logits[:, ~valid.to(features.device)] = -1e4
    proto_ce_each = F.cross_entropy(proto_logits, y, reduction="none")
    proto_ce = cgr.weighted_mean(proto_ce_each, y, class_weights)

    graph_y = class_graph[y].to(features.device)
    true_sim = proto_logits.gather(1, y.view(-1, 1))
    repulse = F.relu(args.proto_margin - true_sim + proto_logits) * graph_y
    proto_repulse = (repulse.sum(dim=1) * class_weights[y].to(features.device)).mean()
    return proto_ramp * proto_ce, proto_ramp * proto_repulse


def cgr_lt_v2_loss(model, feature_hook, x_natural, y, optimizer, args,
                   class_eps, class_weights, class_betas, class_graph,
                   log_prior, prototypes, graph_ramp, proto_ramp):
    model.eval()
    if args.base_algorithm == "pgd":
        x_adv = cgr.pgd_adversary(
            model, x_natural, y, class_eps, args.epsilon,
            args.pgd_step_size, args.pgd_num_steps, log_prior, args)
    elif args.base_algorithm == "trades":
        x_adv = cgr.trades_adversary(
            model, x_natural, y, class_eps, args.epsilon,
            args.pgd_step_size, args.pgd_num_steps)
    else:
        raise ValueError("Unknown base algorithm {}".format(args.base_algorithm))

    model.train()
    optimizer.zero_grad()

    logits_nat = model(x_natural)
    natural_each = cgr.adjusted_ce_each(logits_nat, y, log_prior, args)
    if args.weight_natural:
        natural_loss = cgr.weighted_mean(natural_each, y, class_weights)
    else:
        natural_loss = natural_each.mean()

    logits_adv = model(x_adv)
    adv_features = feature_hook.features
    if adv_features is None:
        adv_features = logits_adv
    adv_features = adv_features.view(adv_features.size(0), -1)

    if args.base_algorithm == "pgd":
        robust_each = cgr.adjusted_ce_each(logits_adv, y, log_prior, args)
        robust_loss = cgr.weighted_mean(robust_each, y, class_weights)
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

    margin_loss = graph_ramp * cgr.pairwise_margin_loss(logits_adv, y, class_graph, class_weights, args)
    proto_ce, proto_repulse = prototype_losses(
        adv_features, y, prototypes, class_graph, class_weights, args, proto_ramp)

    loss = loss + args.margin_lambda * margin_loss
    loss = loss + args.proto_lambda * (proto_ce + proto_repulse)

    eps_batch = torch.as_tensor(class_eps, device=x_natural.device)[y]
    weight_batch = class_weights[y].detach()
    return loss, {
        "natural": natural_loss.item(),
        "robust": robust_loss.item(),
        "margin": margin_loss.item(),
        "proto_ce": proto_ce.item(),
        "proto_rep": proto_repulse.item(),
        "eps_min": eps_batch.min().item(),
        "eps_max": eps_batch.max().item(),
        "w_min": weight_batch.min().item(),
        "w_max": weight_batch.max().item(),
        "beta_min": beta_batch.min().item(),
        "beta_max": beta_batch.max().item(),
    }


def train_epoch(args, model, feature_hook, device, optimizer, train_loader, epoch,
                class_eps, class_weights, class_betas, class_graph, log_prior,
                prototypes, graph_ramp, proto_ramp, trainset_size):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss, loss_dict = cgr_lt_v2_loss(
            model, feature_hook, data, target, optimizer, args, class_eps,
            class_weights, class_betas, class_graph, log_prior, prototypes,
            graph_ramp, proto_ramp)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            default_log = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(
                epoch, (batch_idx + 1) * args.batch_size, trainset_size,
                100.0 * (batch_idx + 1) / len(train_loader))
            loss_log = "[Loss] "
            for key, value in loss_dict.items():
                loss_log += "{} : {:.6f}\t".format(key, value)
            logging.info(default_log + loss_log)


def update_feedback_v2(args, eval_data, feedback_score, clean_acc_score, device):
    if not args.use_feedback_weight:
        return feedback_score.detach(), clean_acc_score.detach(), torch.ones_like(feedback_score)

    robust_acc = eval_data["robust_acc_per_class"].to(device)
    clean_acc = eval_data["clean_acc_per_class"].to(device)
    confusion = eval_data["confusion"].float().to(device)
    row_total = confusion.sum(dim=1).clamp_min(1.0)
    confusion_error = 1.0 - torch.diag(confusion) / row_total

    robust_gap = torch.clamp(robust_acc.mean() - robust_acc, min=0.0)
    raw_score = robust_gap + args.conf_lambda * confusion_error
    raw_score = raw_score / raw_score.mean().clamp_min(1e-12)

    updated_score = (
        args.feedback_momentum * feedback_score
        + (1.0 - args.feedback_momentum) * raw_score
    )
    updated_clean = (
        args.feedback_momentum * clean_acc_score
        + (1.0 - args.feedback_momentum) * clean_acc.clamp_min(0.05)
    )

    class_weights = 1.0 + args.weight_lambda * updated_score
    class_weights = class_weights / class_weights.mean().clamp_min(1e-12)
    class_weights = torch.clamp(class_weights, min=0.25, max=args.weight_max)
    class_weights = class_weights / class_weights.mean().clamp_min(1e-12)
    class_weights = torch.clamp(class_weights, min=0.25, max=args.weight_max)
    return updated_score.detach(), updated_clean.detach(), class_weights.detach()


def update_graph_v2(args, eval_data, class_counts, old_graph, device, graph_ramp):
    if (not args.use_cgr_margin) or graph_ramp <= 0:
        return old_graph

    confusion = eval_data["confusion"].float().to(device)
    row_prob = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
    row_prob.fill_diagonal_(0.0)

    robust_acc = eval_data["robust_acc_per_class"].to(device)
    source_gate = torch.clamp(robust_acc.mean() - robust_acc, min=0.0)
    source_gate = source_gate / source_gate.max().clamp_min(1e-12)

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

    edge_mask = (row_prob >= args.min_edge_conf).float()
    graph = row_prob * source_gate.view(-1, 1) * head_score.view(1, -1) * geometry * edge_mask
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
    graph = graph * graph_ramp
    return (args.graph_momentum * old_graph + (1.0 - args.graph_momentum) * graph).detach()


def update_prototypes(args, eval_data, old_prototypes, device, proto_ramp):
    centers = eval_data["feature_centers"]
    if (not args.use_proto) or centers is None or proto_ramp <= 0:
        return old_prototypes
    centers = centers.to(device)
    if old_prototypes is None:
        return centers.detach()
    valid = centers.abs().sum(dim=1, keepdim=True) > 0
    updated = args.proto_momentum * old_prototypes + (1.0 - args.proto_momentum) * centers
    return torch.where(valid, updated, old_prototypes).detach()


def main():
    args = get_args()

    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info("Method: CGR-LT V2")
    logging.info("Args: %s", args)

    final_checkpoint_path = os.path.join(args.model_dir, "final.pt")
    if not args.overwrite and os.path.exists(final_checkpoint_path):
        logging.info("Final checkpoint found - quitting. Use --overwrite to train again.")
        sys.exit(0)

    cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    trainset, testset, class_counts = cgr.load_imagefolder_dataset(args)
    args.n_class = len(trainset.classes)
    if args.dataset == "auto":
        if args.n_class == 10:
            args.dataset = "cifar10"
        elif args.n_class == 100:
            args.dataset = "cifar100"
        else:
            raise ValueError("Cannot infer dataset name from {} classes.".format(args.n_class))
    if args.dataset == "cifar10" and args.n_class != 10:
        raise ValueError("Dataset argument is cifar10 but ImageFolder has {} classes.".format(args.n_class))
    if args.dataset == "cifar100" and args.n_class != 100:
        raise ValueError("Dataset argument is cifar100 but ImageFolder has {} classes.".format(args.n_class))

    tail_classes = cgr.compute_tail_classes(class_counts, args.tail_fraction)
    cpb_eps = cgr.compute_cpb_eps(class_counts, args.epsilon, args.robustlt_alpha, device)
    log_prior = cgr.build_log_prior(class_counts, device)

    logging.info("Detected dataset: %s, n_class: %d", args.dataset, args.n_class)
    logging.info("Train class counts: %s", class_counts.tolist())
    logging.info("Tail classes: %s", tail_classes.tolist())
    logging.info("CPB final class eps: %s",
                 (torch.round(cpb_eps.cpu() * 1000000) / 1000000).tolist())

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = get_model(args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    feature_hook = cgr.FeatureHook(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    feedback_score = torch.ones(args.n_class, device=device)
    clean_acc_score = torch.ones(args.n_class, device=device)
    class_weights = torch.ones(args.n_class, device=device)
    class_graph = torch.zeros(args.n_class, args.n_class, device=device)
    prototypes = None
    best_robust_acc = -1.0
    init_time = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = cgr.adjust_learning_rate(args, optimizer, epoch)
        graph_ramp = ramp_value(epoch, args.graph_start, args.ramp_epochs)
        proto_ramp = ramp_value(epoch, args.proto_start, args.ramp_epochs)
        beta_ramp = ramp_value(epoch, args.feedback_start, args.ramp_epochs)

        class_eps = cgr.build_epoch_eps(args, cpb_eps, feedback_score, epoch)
        class_betas = build_class_betas_v2(args, feedback_score, clean_acc_score, beta_ramp)

        logging.info("Setting learning rate to %g", lr)
        logging.info("Ramps: graph %.4f, proto %.4f, beta %.4f", graph_ramp, proto_ramp, beta_ramp)
        logging.info("Class weights: %s",
                     (torch.round(class_weights.cpu() * 10000) / 10000).tolist())
        logging.info("Class betas: %s",
                     (torch.round(class_betas.cpu() * 10000) / 10000).tolist())
        logging.info("Graph mean %.6f, max %.6f", class_graph.mean().item(), class_graph.max().item())

        train_epoch(args, model, feature_hook, device, optimizer, train_loader, epoch,
                    class_eps, class_weights, class_betas, class_graph,
                    log_prior, prototypes, graph_ramp, proto_ramp, len(trainset))

        logging.info(120 * "=")
        if epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs:
            eval_data = cgr.evaluate(args, model, device, test_loader, tail_classes,
                                     args.n_class, feature_hook)
            robust_acc = eval_data["robust_acc"]

            if epoch >= args.feedback_start:
                feedback_score, clean_acc_score, class_weights = update_feedback_v2(
                    args, eval_data, feedback_score, clean_acc_score, device)
                class_graph = update_graph_v2(
                    args, eval_data, class_counts, class_graph, device, graph_ramp)
                prototypes = update_prototypes(args, eval_data, prototypes, device, proto_ramp)
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


if __name__ == "__main__":
    main()
