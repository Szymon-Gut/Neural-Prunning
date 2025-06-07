import re
import glob
import os
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils import prune


def finalize_pruning(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(
            module, "weight_mask"
        ):
            prune.remove(module, "weight")
    return model


def build_model_from_name(model_name: str, num_classes: int) -> nn.Module:
    model_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    if model_name not in model_map:
        raise ValueError(f"Not compatible Architecture: {model_name}")

    model = model_map[model_name](weights=None) 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def clean_pruned_checkpoints(
    pattern: str = "prunned*.pth",
    num_classes: int = 10
) -> None:
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        print("Can't find model for prunning.")
        return

    for ckpt_path in checkpoints:
        print(f"\n=== Checkpoint: {ckpt_path}")
        filename = os.path.basename(ckpt_path)
        m = re.search(r"(resnet\d+)", filename)
        if not m:
            print("Unable to load architecture – Skipping.")
            continue
        model_name = m.group(1)

        raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        clean_state = {}
        for k, v in list(raw_state.items()):
            if k.endswith("_orig"):
                base = k[:-5]
                mask = raw_state[base + "_mask"]
                clean_state[base] = v * mask
            elif k.endswith("_mask"):
                continue
            else:
                clean_state[k] = v

        if "fc.weight" in clean_state:
            num_classes = clean_state["fc.weight"].shape[0]
        model = build_model_from_name(model_name, num_classes)
        model.load_state_dict(clean_state, strict=True)
        finalize_pruning(model)
        torch.save(model.state_dict(), filename)
        print(f"Model after prunning saved to - {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="prunned*.pth")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    clean_pruned_checkpoints(
        args.pattern, args.num_classes
    )
