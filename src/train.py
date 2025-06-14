import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from dataloader import get_cifar_dataloader
import argparse
import os
import wandb
import copy
from datetime import datetime
from torch.nn.utils import prune
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on Cifar dataset")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model to train")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Cifar dataset directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer to use for training")
    parser.add_argument("--quantization", action="store_true", help="Apply quantization to the model after training")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation to training set")
    parser.add_argument("--wandb_name", type=str, default="resnet_cifar", help="Name for the Weights & Biases run")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--model_save_path", type=str, default="resnet_model.pth", help="Path to save the trained model")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with no improvement before stopping")
    parser.add_argument("--prune", action="store_true", help="Apply pruning to the model before training")
    parser.add_argument("--prune_amount", type=float, default=0.3, help="Fraction of parameters to prune (default: 0.3)")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "relu6", "tanh", "sigmoid", "leaky_relu"], help="Activation function to use in the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate to apply before activation layers (default: 0.0 - no dropout)")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to a pretrained model checkpoint (.pth)")
    return parser.parse_args()

def replace_activation(model, new_activation):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, child_name, new_activation)

def append_dropout(model, dropout_prob):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, dropout_prob)
        if isinstance(module, nn.ReLU):
                new = nn.Sequential(module, nn.Dropout2d(p=dropout_prob, inplace=False))
                setattr(model, name, new)

def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
    print(f"Pruning applied: {amount * 100:.1f}% of weights in Conv2d and Linear layers.")
    return model

def count_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            weight = module.weight
            total_zeros += torch.sum(weight == 0).item()
            total_elements += weight.nelement()
    sparsity = 100. * total_zeros / total_elements
    return sparsity

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_model(args):
    if args.use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.wandb_name}-{timestamp}"
        wandb.init(project="neural_prunning", 
                   config=vars(args),
                   name=run_name,
                   entity='sandra-cichocka2000-gdansk-university-of-technology')

    train_loader, val_loader = get_cifar_dataloader(
        os.path.join(args.data_dir),
        args.batch_size,
        args.num_workers,
        augment=args.augment
    )

    model_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101
    }

    if args.model_name not in model_map:
        raise ValueError(f"Unsupported model: {args.model_name}")

    model = model_map[args.model_name](weights="DEFAULT")
    if args.dropout:
        append_dropout(model, args.dropout)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    
    activation_map = {
        "relu": nn.ReLU(inplace=True),
        "relu6": nn.ReLU6(inplace=False),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(inplace=True)
    }
    
    if args.activation not in activation_map:
        raise ValueError(f"Unsupported activation: {args.activation}")

    new_activation = activation_map[args.activation]
    replace_activation(model, new_activation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.load_model_path is not None:
        if os.path.isfile(args.load_model_path):
            checkpoint = torch.load(args.load_model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {args.load_model_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.load_model_path}")

    if args.prune:
        model = apply_pruning(model, amount=args.prune_amount)

    criterion = nn.CrossEntropyLoss()
    
    base_params = [param for name, param in model.named_parameters() if not name.startswith("fc.")]
    fc_params = model.fc.parameters()

    if args.optimizer == "adam":
        optimizer = optim.Adam([
            {'params': base_params, 'lr': args.learning_rate / 10},
            {'params': fc_params, 'lr': args.learning_rate}
        ])
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': args.learning_rate / 10},
            {'params': fc_params, 'lr': args.learning_rate}
        ])
    elif args.optimizer == "sgd":
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.learning_rate / 10},
            {'params': fc_params, 'lr': args.learning_rate}
        ], momentum=0.9)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if epoch == 0:
            epoch_time = time.time() - epoch_start_time
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
            total_params = count_trainable_params(model)

            print(f"\n=== Episode 1 Statistics ===")
            print(f"Duration of episode: {epoch_time:.2f} seconds")
            print(f"Batch size: {args.batch_size}")
            print(f"GPU memory usage: {memory_allocated:.2f} MB")
            print(f"Amount of trainable parameters: {total_params:,}")
            print("============================\n")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    model.load_state_dict(best_model_wts)

    if args.quantization:
        model.cpu()
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        print("Applied dynamic quantization to the model.")

    if args.prune and args.use_wandb:
        sparsity = count_sparsity(model)
        print(f"Final sparsity: {sparsity:.2f}%")
        wandb.log({"final_sparsity": sparsity})
    
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
