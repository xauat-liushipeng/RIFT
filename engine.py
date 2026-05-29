from typing import Iterable
import time

import torch
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args=None,
    logger=None,
):
    model.train()
    criterion.train()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    pbar = tqdm(total=len(data_loader.dataloader), desc="Loss: pending")
    for data in data_loader:
        samples = data["image"].to(device)
        targets = data["label"].to(device).float()

        logits = model(samples)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        cur_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(time.time()))
        if logger is not None:
            logger.info(
                f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | "
                f"loss -> {loss.item():.4f} | lr -> {lr}"
            )
        pbar.set_description(f"Loss: {loss.item():.4f}")
        pbar.update(1)
    pbar.close()
