"""
train.py — Main Training Entry Point
======================================
Single-GPU:
    python train.py --config config.yaml

Multi-GPU (recommended — uses DDP):
    torchrun --nproc_per_node=4 train.py --config config.yaml

Resume:
    torchrun --nproc_per_node=4 train.py --config config.yaml \\
        --resume checkpoints/bridge_best.pt

Override hyperparameters on the fly:
    torchrun --nproc_per_node=4 train.py --config config.yaml \\
        --overrides training.batch_size=16 training.learning_rate=5e-5
"""

import argparse
import logging
import os
import sys
import yaml

from trainer import Trainer


def override_cfg(cfg: dict, overrides: list):
    for kv in overrides:
        key, _, val = kv.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        node[keys[-1]] = val
        print(f"  Override: {key} = {val}")


def main():
    parser = argparse.ArgumentParser(description="Train Mimi-to-HuBERT Bridge")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    # Only rank-0 writes to the log file to avoid duplicate writes
    global_rank = int(os.environ.get("RANK", 0))
    handlers = [logging.StreamHandler(sys.stdout)]
    if global_rank == 0:
        handlers.append(logging.FileHandler("train.log"))

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  [rank%(process)d]  %(message)s",
        handlers= handlers,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        override_cfg(cfg, args.overrides)

    trainer = Trainer(cfg)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()