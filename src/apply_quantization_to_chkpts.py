import re
import glob
import os
import torch
import torch.nn as nn
from torchvision.models.quantization import resnet18, resnet50
from dataloader import get_cifar_dataloader


def build_model_from_name(model_name: str, num_classes: int) -> nn.Module:
    model_map = {
        "resnet18": resnet18,
        "resnet50": resnet50,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Not compatible architecture: {model_name}")

    model = model_map[model_name](weights=None, quantize=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def apply_quantization(
    pattern: str = "prunned*.pth",
    num_classes: int = 10,
    data_path: str = "dataset/filtered_dataset.pkl"
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
        model = build_model_from_name(model_name, num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        model.fuse_model()  
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)

        _, calib_loader = get_cifar_dataloader(data_path, batch_size=128, num_workers=4, augment=False)
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                if i > 10:
                    break
                model(images)

        torch.quantization.convert(model, inplace=True)
        print("Model successfully quantized.")

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in calib_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
        print(f"Model accuracy after quantization: {accuracy:.2f}%")

        with open("quant_eval_results.txt", "a") as log_file:
            log_file.write(f"{filename}: {accuracy:.2f}%\n")
            print(f"Quantization results of architecture {filename} saved to quant_eval_results.txt")
            
        quantized_ckpt_path = ckpt_path.replace(".pth", "_quantized.pth")
        torch.save(model, quantized_ckpt_path)
        print(f"Model after quantization saved to - {quantized_ckpt_path}")
        print(f"Model size before quantization: {os.path.getsize(ckpt_path) / 1024:.2f} KB")
        print(f"Model size after quantization: {os.path.getsize(quantized_ckpt_path) / 1024:.2f} KB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply quantization to checkpoints.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="prunned*.pth",
        help="Pattern to match checkpoint files."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of classes in the model."
    )
    args = parser.parse_args()
    apply_quantization(args.pattern, args.num_classes)
