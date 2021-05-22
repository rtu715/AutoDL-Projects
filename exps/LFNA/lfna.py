#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.04 #
#####################################################
# python exps/LFNA/lfna.py --env_version v1 --workers 0
# python exps/LFNA/lfna.py --env_version v1 --device cuda --lr 0.001
# python exps/LFNA/lfna.py --env_version v1 --device cuda --lr 0.002
#####################################################
import sys, time, copy, torch, random, argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "..").resolve()
print("LIB-DIR: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.log_utils import time_string
from xautodl.log_utils import AverageMeter, convert_secs2time

from xautodl.utils import split_str2indexes

from xautodl.procedures.advanced_main import basic_train_fn, basic_eval_fn
from xautodl.procedures.metric_utils import SaveMetric, MSEMetric, ComposeMetric
from xautodl.datasets.synthetic_core import get_synthetic_env, EnvSampler
from xautodl.models.xcore import get_model
from xautodl.xlayers import super_core, trunc_normal_

from lfna_utils import lfna_setup, train_model, TimeData
from lfna_meta_model import LFNA_Meta


def epoch_train(loader, meta_model, base_model, optimizer, criterion, device, logger):
    base_model.train()
    meta_model.train()
    loss_meter = AverageMeter()
    for ibatch, batch_data in enumerate(loader):
        timestamps, (batch_seq_inputs, batch_seq_targets) = batch_data
        timestamps = timestamps.squeeze(dim=-1).to(device)
        batch_seq_inputs = batch_seq_inputs.to(device)
        batch_seq_targets = batch_seq_targets.to(device)

        optimizer.zero_grad()

        batch_seq_containers = meta_model(timestamps)
        losses = []
        for seq_containers, seq_inputs, seq_targets in zip(
            batch_seq_containers, batch_seq_inputs, batch_seq_targets
        ):
            for container, inputs, targets in zip(
                seq_containers, seq_inputs, seq_targets
            ):
                predictions = base_model.forward_with_container(inputs, container)
                loss = criterion(predictions, targets)
                losses.append(loss)
        final_loss = torch.stack(losses).mean()
        final_loss.backward()
        optimizer.step()
        loss_meter.update(final_loss.item())
    return loss_meter


def epoch_evaluate(loader, meta_model, base_model, criterion, device, logger):
    with torch.no_grad():
        base_model.eval()
        meta_model.eval()
        loss_meter = AverageMeter()
        for ibatch, batch_data in enumerate(loader):
            timestamps, (batch_seq_inputs, batch_seq_targets) = batch_data
            timestamps = timestamps.squeeze(dim=-1).to(device)
            batch_seq_inputs = batch_seq_inputs.to(device)
            batch_seq_targets = batch_seq_targets.to(device)

            batch_seq_containers = meta_model(timestamps)
            losses = []
            for seq_containers, seq_inputs, seq_targets in zip(
                batch_seq_containers, batch_seq_inputs, batch_seq_targets
            ):
                for container, inputs, targets in zip(
                    seq_containers, seq_inputs, seq_targets
                ):
                    predictions = base_model.forward_with_container(inputs, container)
                    loss = criterion(predictions, targets)
                    losses.append(loss)
            final_loss = torch.stack(losses).mean()
            loss_meter.update(final_loss.item())
    return loss_meter


def pretrain(base_model, meta_model, criterion, xenv, args, logger):
    optimizer = torch.optim.Adam(
        meta_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    meta_model.set_best_dir(logger.path(None) / "checkpoint-pretrain")
    for iepoch in range(args.epochs):
        total_meta_losses, total_match_losses = [], []
        for ibatch in range(args.meta_batch):
            rand_index = random.randint(0, meta_model.meta_length - xenv.seq_length - 1)
            timestamps = meta_model.meta_timestamps[
                rand_index : rand_index + xenv.seq_length
            ]

            seq_timestamps, (seq_inputs, seq_targets) = xenv.seq_call(timestamps)
            [seq_containers], time_embeds = meta_model(
                torch.unsqueeze(timestamps, dim=0)
            )
            # performance loss
            losses = []
            seq_inputs, seq_targets = seq_inputs.to(args.device), seq_targets.to(
                args.device
            )
            for container, inputs, targets in zip(
                seq_containers, seq_inputs, seq_targets
            ):
                predictions = base_model.forward_with_container(inputs, container)
                loss = criterion(predictions, targets)
                losses.append(loss)
            meta_loss = torch.stack(losses).mean()
            match_loss = criterion(
                torch.squeeze(time_embeds, dim=0),
                meta_model.super_meta_embed[rand_index : rand_index + xenv.seq_length],
            )
            # batch_loss = meta_loss + match_loss * 0.1
            # total_losses.append(batch_loss)
            total_meta_losses.append(meta_loss)
            total_match_losses.append(match_loss)
        final_meta_loss = torch.stack(total_meta_losses).mean()
        final_match_loss = torch.stack(total_match_losses).mean()
        total_loss = final_meta_loss + final_match_loss
        total_loss.backward()
        optimizer.step()
        # success
        success, best_score = meta_model.save_best(-total_loss.item())
        logger.log(
            "{:} [{:04d}/{:}] loss : {:.5f} = {:.5f} + {:.5f} (match)".format(
                time_string(),
                iepoch,
                args.epochs,
                total_loss.item(),
                final_meta_loss.item(),
                final_match_loss.item(),
            )
            + ", batch={:}".format(len(total_meta_losses))
        )


def main(args):
    logger, env_info, model_kwargs = lfna_setup(args)
    train_env = get_synthetic_env(mode="train", version=args.env_version)
    valid_env = get_synthetic_env(mode="valid", version=args.env_version)
    logger.log("training enviornment: {:}".format(train_env))
    logger.log("validation enviornment: {:}".format(valid_env))

    base_model = get_model(**model_kwargs)
    base_model = base_model.to(args.device)
    criterion = torch.nn.MSELoss()

    shape_container = base_model.get_w_container().to_shape_container()

    # pre-train the hypernetwork
    timestamps = train_env.get_timestamp(None)
    meta_model = LFNA_Meta(shape_container, args.layer_dim, args.time_dim, timestamps)
    meta_model = meta_model.to(args.device)

    logger.log("The base-model has {:} weights.".format(base_model.numel()))
    logger.log("The meta-model has {:} weights.".format(meta_model.numel()))

    batch_sampler = EnvSampler(train_env, args.meta_batch, args.sampler_enlarge)
    train_env.reset_max_seq_length(args.seq_length)
    valid_env.reset_max_seq_length(args.seq_length)
    valid_env_loader = torch.utils.data.DataLoader(
        valid_env,
        batch_size=args.meta_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_env_loader = torch.utils.data.DataLoader(
        train_env,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        meta_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1, 2, 3, 4, 5],
        gamma=0.2,
    )
    logger.log("The base-model is\n{:}".format(base_model))
    logger.log("The meta-model is\n{:}".format(meta_model))
    logger.log("The optimizer is\n{:}".format(optimizer))
    logger.log("The scheduler is\n{:}".format(lr_scheduler))
    logger.log("Per epoch iterations = {:}".format(len(train_env_loader)))

    pretrain(base_model, meta_model, criterion, train_env, args, logger)

    if logger.path("model").exists():
        ckp_data = torch.load(logger.path("model"))
        base_model.load_state_dict(ckp_data["base_model"])
        meta_model.load_state_dict(ckp_data["meta_model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        lr_scheduler.load_state_dict(ckp_data["lr_scheduler"])
        last_success_epoch = ckp_data["last_success_epoch"]
        start_epoch = ckp_data["iepoch"] + 1
        check_strs = [
            "epochs",
            "env_version",
            "hidden_dim",
            "lr",
            "layer_dim",
            "time_dim",
            "seq_length",
        ]
        for xstr in check_strs:
            cx = getattr(args, xstr)
            px = getattr(ckp_data["args"], xstr)
            assert cx == px, "[{:}] {:} vs {:}".format(xstr, cx, ps)
        success, _ = meta_model.save_best(ckp_data["cur_score"])
        logger.log("Load ckp from {:}".format(logger.path("model")))
        if success:
            logger.log(
                "Re-save the best model with score={:}".format(ckp_data["cur_score"])
            )
    else:
        start_epoch, last_success_epoch = 0, 0

    # LFNA meta-train
    meta_model.set_best_dir(logger.path(None) / "checkpoint")
    per_epoch_time, start_time = AverageMeter(), time.time()
    for iepoch in range(start_epoch, args.epochs):

        head_str = "[{:}] [{:04d}/{:04d}] ".format(
            time_string(), iepoch, args.epochs
        ) + "Time Left: {:}".format(
            convert_secs2time(per_epoch_time.avg * (args.epochs - iepoch), True)
        )

        loss_meter = epoch_train(
            train_env_loader,
            meta_model,
            base_model,
            optimizer,
            criterion,
            args.device,
            logger,
        )

        valid_loss_meter = epoch_evaluate(
            valid_env_loader, meta_model, base_model, criterion, args.device, logger
        )
        logger.log(
            head_str
            + " meta-train-loss: {meter.avg:.4f} ({meter.count:.0f})".format(
                meter=loss_meter
            )
            + " meta-valid-loss: {meter.val:.4f}".format(meter=valid_loss_meter)
            + " :: lr={:.5f}".format(min(lr_scheduler.get_last_lr()))
            + "  :: last-success={:}".format(last_success_epoch)
        )
        success, best_score = meta_model.save_best(-loss_meter.avg)
        if success:
            logger.log("Achieve the best with best-score = {:.5f}".format(best_score))
            last_success_epoch = iepoch
            save_checkpoint(
                {
                    "meta_model": meta_model.state_dict(),
                    "base_model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "last_success_epoch": last_success_epoch,
                    "cur_score": -loss_meter.avg,
                    "iepoch": iepoch,
                    "args": args,
                },
                logger.path("model"),
                logger,
            )
        if iepoch - last_success_epoch >= args.early_stop_thresh:
            if lr_scheduler.last_epoch > 4:
                logger.log("Early stop at {:}".format(iepoch))
                break
            else:
                last_success_epoch = iepoch
                lr_scheduler.step()
                logger.log("Decay the lr [{:}]".format(lr_scheduler.last_epoch))

        per_epoch_time.update(time.time() - start_time)
        start_time = time.time()

    # meta-test
    meta_model.load_best()
    eval_env = env_info["dynamic_env"]
    w_container_per_epoch = dict()
    for idx in range(args.seq_length, len(eval_env)):
        # build-timestamp
        future_time = env_info["{:}-timestamp".format(idx)].item()
        time_seqs = []
        for iseq in range(args.seq_length):
            time_seqs.append(future_time - iseq * eval_env.timestamp_interval)
        time_seqs.reverse()
        with torch.no_grad():
            meta_model.eval()
            base_model.eval()
            time_seqs = torch.Tensor(time_seqs).view(1, -1).to(args.device)
            [seq_containers] = meta_model(time_seqs)
            future_container = seq_containers[-1]
            w_container_per_epoch[idx] = future_container.no_grad_clone()
            # evaluation
            future_x = env_info["{:}-x".format(idx)].to(args.device)
            future_y = env_info["{:}-y".format(idx)].to(args.device)
            future_y_hat = base_model.forward_with_container(
                future_x, w_container_per_epoch[idx]
            )
            future_loss = criterion(future_y_hat, future_y)
            logger.log(
                "meta-test: [{:03d}] -> loss={:.4f}".format(idx, future_loss.item())
            )

        # creating the new meta-time-embedding
        distance = meta_model.get_closest_meta_distance(future_time)
        if distance < eval_env.timestamp_interval:
            continue
        #
        new_param = meta_model.create_meta_embed()
        optimizer = torch.optim.Adam(
            [new_param], lr=args.refine_lr, weight_decay=1e-5, amsgrad=True
        )
        meta_model.replace_append_learnt(
            torch.Tensor([future_time]).to(args.device), new_param
        )
        meta_model.eval()
        base_model.train()
        for iepoch in range(args.refine_epochs):
            optimizer.zero_grad()
            [seq_containers] = meta_model(time_seqs)
            future_container = seq_containers[-1]
            future_y_hat = base_model.forward_with_container(future_x, future_container)
            future_loss = criterion(future_y_hat, future_y)
            future_loss.backward()
            optimizer.step()
        logger.log(
            "post-meta-test: [{:03d}] -> loss={:.4f}".format(idx, future_loss.item())
        )
        with torch.no_grad():
            meta_model.replace_append_learnt(None, None)
            meta_model.append_fixed(torch.Tensor([future_time]), new_param)

    save_checkpoint(
        {"w_container_per_epoch": w_container_per_epoch},
        logger.path(None) / "final-ckp.pth",
        logger,
    )

    logger.log("-" * 200 + "\n")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(".")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/lfna-synthetic/lfna-battle",
        help="The checkpoint directory.",
    )
    parser.add_argument(
        "--env_version",
        type=str,
        required=True,
        help="The synthetic enviornment version.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=16,
        help="The hidden dimension.",
    )
    parser.add_argument(
        "--layer_dim",
        type=int,
        default=16,
        help="The layer chunk dimension.",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=16,
        help="The timestamp dimension.",
    )
    #####
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="The initial learning rate for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.00001,
        help="The weight decay for the optimizer (default is Adam)",
    )
    parser.add_argument(
        "--meta_batch",
        type=int,
        default=64,
        help="The batch size for the meta-model",
    )
    parser.add_argument(
        "--sampler_enlarge",
        type=int,
        default=5,
        help="Enlarge the #iterations for an epoch",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="The total #epochs.")
    parser.add_argument(
        "--refine_lr",
        type=float,
        default=0.005,
        help="The learning rate for the optimizer, during refine",
    )
    parser.add_argument(
        "--refine_epochs", type=int, default=1000, help="The final refine #epochs."
    )
    parser.add_argument(
        "--early_stop_thresh",
        type=int,
        default=20,
        help="The #epochs for early stop.",
    )
    parser.add_argument(
        "--seq_length", type=int, default=10, help="The sequence length."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="The number of workers in parallel."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "The save dir argument can not be None"
    args.save_dir = "{:}-d{:}_{:}_{:}-s{:}-lr{:}-wd{:}-e{:}-env{:}".format(
        args.save_dir,
        args.hidden_dim,
        args.layer_dim,
        args.time_dim,
        args.seq_length,
        args.lr,
        args.weight_decay,
        args.epochs,
        args.env_version,
    )
    main(args)
