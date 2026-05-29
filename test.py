import argparse
import os
import shutil
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from datasets import create_dataset
from rift import build_model
from train import get_args_parser, safe_to_uint8


def main(args):
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for inference.")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.phase = "test"
    args.batch_size = args.batch_size_test
    test_loader = create_dataset(args)

    model, _ = build_model(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    save_root = args.save_dir
    os.makedirs(save_root, exist_ok=True)
    print("Load model successful!")

    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            x = data["image"].to(device)
            target = data["label"].to(device)
            logits = model(x)

            root_name = os.path.basename(data["A_paths"][0]).rsplit(".", 1)[0]
            target_img = safe_to_uint8(target[0, 0].detach().cpu().numpy())
            pred_img = safe_to_uint8(logits[0, 0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), target_img)
            cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), pred_img)

    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RIFT inference", parents=[get_args_parser()])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--save_dir", type=str, default="./results_test", help="Directory to save predictions.")
    main(parser.parse_args())
