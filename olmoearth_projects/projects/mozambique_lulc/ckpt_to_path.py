"""Save a lightning .ckpt as something which can be loaded by rslearn's OlmoEarth wrapper."""

import argparse
from typing import Any

import torch
from rslearn.train.lightning_module import RestoreConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the ckpt",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Place to save the weights, as `weights.pth`",
    )

    args = parser.parse_args()

    config = RestoreConfig(args.ckpt_path)
    state_dict = config.get_state_dict()["state_dict"]

    # remove the ".model"
    new_sd: dict[str, Any] = {}
    for key, val in state_dict.items():
        new_sd[key.replace("model.", "").replace("encoder.0", "encoder")] = val

    torch.save(new_sd, f"{args.save_path}/weights.pth")
