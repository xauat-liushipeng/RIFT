import argparse
import datetime
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import util.misc as utils
from datasets import create_dataset
from engine import train_one_epoch
from eval import eval as evaluate_results
from rift import build_model
from util.logger import get_logger


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning-rate decay without mmengine dependency."""

    def __init__(self, optimizer, max_iters: int, eta_min: float = 1e-6, power: float = 0.9, last_epoch: int = -1):
        self.max_iters = max(1, int(max_iters))
        self.eta_min = float(eta_min)
        self.power = float(power)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = max(0.0, 1.0 - self.last_epoch / self.max_iters) ** self.power
        return [(base_lr - self.eta_min) * factor + self.eta_min for base_lr in self.base_lrs]


def get_args_parser():
    parser = argparse.ArgumentParser("RIFT for crack segmentation", add_help=False)

    # Loss
    parser.add_argument("--BCELoss_ratio", default=0.87, type=float)
    parser.add_argument("--DiceLoss_ratio", default=0.13, type=float)

    # Dataset: folder layout is kept consistent with MixerCSeg.
    parser.add_argument("--dataset_path", default="../datasets/CrackMap", type=str)
    parser.add_argument("--dataset_mode", default="crack", type=str)
    parser.add_argument("--phase", default="train", type=str)
    parser.add_argument("--load_width", default=512, type=int)
    parser.add_argument("--load_height", default=512, type=int)
    parser.add_argument("--batch_size_train", default=1, type=int)
    parser.add_argument("--batch_size_test", default=1, type=int)
    parser.add_argument("--serial_batches", action="store_true")
    parser.add_argument("--num_threads", default=1, type=int)

    # Optimization
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--lr_scheduler", default="PolyLR", choices=["PolyLR", "StepLR", "CosLR"])
    parser.add_argument("--lr_drop", default=30, type=int)
    parser.add_argument("--sgd", action="store_true")

    # Runtime
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)

    # RIFT architecture. Defaults correspond to the recommended base model.
    parser.add_argument("--dims", default="32,64,128,192", type=str)
    parser.add_argument("--depths", default="2,2,3,2", type=str)
    parser.add_argument("--kernel_size", default=13, type=int)
    parser.add_argument("--expand_ratio", default=2.0, type=float)
    parser.add_argument("--drop_path", default=0.05, type=float)
    parser.add_argument("--decoder_dim", default=64, type=int)
    parser.add_argument("--gn_groups", default=8, type=int)
    return parser


def build_scheduler(optimizer, args):
    if args.lr_scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if args.lr_scheduler == "CosLR":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    if args.lr_scheduler == "PolyLR":
        return PolyLR(optimizer, max_iters=args.epochs, eta_min=args.min_lr)
    raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")


def safe_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    max_value = float(np.max(array))
    if max_value <= 1e-12:
        return np.zeros_like(array, dtype=np.uint8)
    return np.clip(255.0 * (array / max_value), 0, 255).astype(np.uint8)


def build_dataloader(args, phase, batch_size):
    args.phase = phase
    args.batch_size = batch_size
    return create_dataset(args)


def run_validation_epoch(model, criterion, args, device, save_root, log_val):
    val_loader = build_dataloader(args, phase="val", batch_size=args.batch_size_test)
    os.makedirs(save_root, exist_ok=True)

    pbar = tqdm(total=len(val_loader), desc="Validation")
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            x = data["image"].to(device)
            target = data["label"].to(device)
            logits = model(x)
            loss = criterion(logits, target.float())

            target_np = target[0, 0].detach().cpu().numpy()
            pred_np = logits[0, 0].detach().cpu().numpy()
            root_name = os.path.basename(data["A_paths"][0]).rsplit(".", 1)[0]

            target_img = safe_to_uint8(target_np)
            pred_img = safe_to_uint8(pred_np)

            lab_path = os.path.join(save_root, f"{root_name}_lab.png")
            pre_path = os.path.join(save_root, f"{root_name}_pre.png")
            cv2.imwrite(lab_path, target_img)
            cv2.imwrite(pre_path, pred_img)

            log_val.info(f"loss -> {loss.item():.6f}")
            log_val.info(lab_path)
            log_val.info(pre_path)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
    pbar.close()


def main(args):
    cur_time = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime(time.time()))
    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    process_folder_path = os.path.join(args.log_dir, f"{cur_time}_{dataset_name}")
    os.makedirs(process_folder_path, exist_ok=True)

    log_train = get_logger(process_folder_path, "train")
    log_val = get_logger(process_folder_path, "val")
    log_eval = get_logger(process_folder_path, "eval")

    log_train.info("RIFT")
    log_train.info("args -> " + str(args))
    print("RIFT")
    print("args ->", args)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters -> {param_count:,}")
    log_train.info(f"Parameters -> {param_count:,}")

    train_loader = build_dataloader(args, phase="train", batch_size=args.batch_size_train)
    print(f"The number of training images = {len(train_loader)}")
    log_train.info(f"The number of training images = {len(train_loader)}")

    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    if args.sgd:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        print("use SGD")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        print("use AdamW")
    lr_scheduler = build_scheduler(optimizer, args)

    output_dir = Path(args.output_dir) / f"{cur_time}_Dataset-{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    max_miou = 0.0
    max_metrics = {"epoch": 0, "mIoU": 0.0}

    for epoch in range(args.start_epoch, args.epochs):
        print("-" * 87)
        print("training epoch start ->", epoch)
        train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, log_train)
        lr_scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        utils.save_on_master(checkpoint, output_dir / "checkpoint.pth")
        utils.save_on_master(checkpoint, output_dir / f"checkpoint{epoch}.pth")
        print("training epoch finish ->", epoch)

        print("validation epoch start ->", epoch)
        results_name = f"{cur_time}_Dataset-{dataset_name}"
        save_root = str(Path("./results") / results_name / f"val_results_{epoch}")
        run_validation_epoch(model, criterion, args, device, save_root, log_val)
        print("validation epoch finish ->", epoch)

        print("evaluating epoch start ->", epoch)
        metrics = evaluate_results(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(f"{key} -> {value}")
        if max_miou < metrics["mIoU"]:
            max_metrics = metrics
            max_miou = metrics["mIoU"]
            utils.save_on_master(checkpoint, output_dir / "checkpoint_best.pth")
            log_train.info("update and save best model -> " + str(epoch))
            print("update and save best model ->", epoch)

        print("evaluating epoch finish ->", epoch)
        print(f"\nmax_mIoU -> {max_metrics['mIoU']}\nmax Epoch -> {max_metrics['epoch']}")
        log_eval.info(f"max_mIoU -> {max_metrics['mIoU']} | max Epoch -> {max_metrics['epoch']}")

    for key, value in max_metrics.items():
        log_eval.info(f"{key} -> {value}")
    log_eval.info(f"\nmax_mIoU -> {max_metrics['mIoU']}\nmax Epoch -> {max_metrics['epoch']}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Process time {total_time_str}")
    log_train.info(f"Process time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RIFT", parents=[get_args_parser()])
    main(parser.parse_args())
