import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from torch import nn
from tqdm import tqdm
import torch.multiprocessing as mp

from models.vgg import VGG_A, VGG_A_BatchNorm
from Data.loaders import get_cifar_loader

import torch
import numpy as np
import os
import json

def save_results(model, results, model_name, lr, models_path, metrics_path):

    name_prefix = f"{model_name}_lr_{lr}"

    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(models_path, f"{name_prefix}.pth"))

    # 保存指标
    np.save(os.path.join(metrics_path, f"{name_prefix}_loss.npy"), np.array(results["loss"]))
    np.save(os.path.join(metrics_path, f"{name_prefix}_valacc.npy"), np.array(results["val_acc"]))
    np.save(os.path.join(metrics_path, f"{name_prefix}_gradnorms.npy"), np.array(results["grad_norms"]))
    np.save(os.path.join(metrics_path, f"{name_prefix}_lipschitz.npy"), np.array(results["lipschitz_loss"]))


def set_random_seeds(seed_value=0):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def train(model, optimizer, criterion, train_loader, val_loader, device, scheduler=None, epochs_n=100):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)
    learning_curve = []
    all_losses = []
    all_grads = []
    grad_norm_curve = []
    val_acc_curve = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()

        model.train()
        epoch_loss = 0
        batch_losses = []
        batch_grads = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # Capture gradient
            layer = model.module.classifier[4] if isinstance(model, nn.DataParallel) else model.classifier[4]
            grad = layer.weight.grad.clone().cpu().numpy()
            batch_grads.append(grad)

            optimizer.step()
            batch_losses.append(loss.item())
            epoch_loss += loss.item()

        all_losses.append(batch_losses)
        all_grads.append(batch_grads)
        grad_norm_curve.append(np.mean([np.linalg.norm(g) for g in batch_grads if g is not None]))
        learning_curve.append(epoch_loss / len(train_loader))

        # 计算验证集准确率
        val_acc = get_accuracy(model, val_loader, device)
        val_acc_curve.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs_n} - Loss: {learning_curve[-1]:.4f} - Val Acc: {val_acc:.4f}")

    return all_losses, all_grads, grad_norm_curve, val_acc_curve


def compute_lipschitz_loss_change(model, data_loader, criterion, step_sizes, device, layer_idx=4):
    model.eval()
    X, y = next(iter(data_loader))
    X, y = X.to(device), y.to(device)

    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()

    layer = model.module.classifier if isinstance(model, nn.DataParallel) else model.classifier
    param = layer[layer_idx].weight
    grad = param.grad.clone()

    original_loss = loss.item()
    loss_changes = [original_loss]

    for alpha in step_sizes:
        with torch.no_grad():
            param.add_(-alpha * grad)
            new_loss = criterion(model(X), y).item()
            param.add_(alpha * grad)
            loss_changes.append(new_loss)

    return [0.0] + step_sizes, loss_changes


def plot_gradient_norms(gradnorms_with_bn, gradnorms_without_bn, lr, figures_path):
    plt.figure(figsize=(10, 6))
    plt.plot(gradnorms_with_bn, label='With BN', color='blue')
    plt.plot(gradnorms_without_bn, label='Without BN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title(f'Gradient Norm over Epochs (lr={lr})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'gradient_norm_comparison_lr_{lr}.png'))
    plt.close()


def plot_lipschitz_curve(step_sizes, loss_changes_bn, loss_changes_no_bn, lr, figures_path):
    plt.figure(figsize=(8, 6))
    plt.plot(step_sizes, loss_changes_bn, label='With BN', marker='o')
    plt.plot(step_sizes, loss_changes_no_bn, label='Without BN', marker='o')
    plt.xlabel('Step Size (alpha)')
    plt.ylabel('Loss')
    plt.title(f'Lipschitz Loss Curve (lr={lr})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'lipschitz_curve_lr_{lr}.png'))
    plt.close()


def plot_loss_curve(losses_with_bn, losses_without_bn, learning_rates, figures_path):
    for idx, lr in enumerate(learning_rates):
        plt.figure(figsize=(10, 6))
        plt.plot(losses_with_bn[idx], label='With BN', color='blue')
        plt.plot(losses_without_bn[idx], label='Without BN', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve (lr={lr})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'loss_curve_lr_{lr}.png'))
        plt.close()


def plot_accuracy_curve(acc_with_bn, acc_without_bn, learning_rates, figures_path):
    for idx, lr in enumerate(learning_rates):
        plt.figure(figsize=(10, 6))
        plt.plot(acc_with_bn[idx], label='With BN', color='blue')
        plt.plot(acc_without_bn[idx], label='Without BN', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title(f'Validation Accuracy over Epochs (lr={lr})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'val_accuracy_lr_{lr}.png'))
        plt.close()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    epo = 20
    learning_rates = [1e-2,1e-3, 2e-3, 1e-4, 5e-4]

    # 路径设置
    home_path = os.path.dirname(os.getcwd())
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')
    metrics_path = os.path.join(home_path, 'reports', 'metrics')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    # 数据加载
    train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=4)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=4)

    all_losses_with_bn = []
    all_losses_without_bn = []
    all_gradnorms_with_bn = []
    all_gradnorms_without_bn = []
    all_valaccs_with_bn = []
    all_valaccs_without_bn = []

    for lr in learning_rates:
        print(f"\n=== Training with lr={lr} ===")

        ## 无 BN
        set_random_seeds(2020)
        model_no_bn = VGG_A()
        optimizer = torch.optim.Adam(model_no_bn.parameters(), lr=lr)
        losses, _, grad_norms, val_accs = train(model_no_bn, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, device, epochs_n=epo)
        epoch_losses = [np.mean(bl) for bl in losses]
        all_losses_without_bn.append(epoch_losses)
        all_gradnorms_without_bn.append(grad_norms)
        all_valaccs_without_bn.append(val_accs)

        # Lipschitz loss change（无 BN）
        steps, loss_vals_no_bn = compute_lipschitz_loss_change(model_no_bn, train_loader, nn.CrossEntropyLoss(), [1e-5, 1e-4, 1e-3, 1e-2], device)

        # 保存无 BN 结果
        save_results(model_no_bn, {
            "loss": epoch_losses,
            "val_acc": val_accs,
            "grad_norms": grad_norms,
            "lipschitz_loss": loss_vals_no_bn,
        }, "VGG_A_NoBN", lr, models_path, metrics_path)

        ## 有 BN
        set_random_seeds(2020)
        model_bn = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model_bn.parameters(), lr=lr)
        losses, _, grad_norms, val_accs = train(model_bn, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, device, epochs_n=epo)
        epoch_losses = [np.mean(bl) for bl in losses]
        all_losses_with_bn.append(epoch_losses)
        all_gradnorms_with_bn.append(grad_norms)
        all_valaccs_with_bn.append(val_accs)

        # Lipschitz loss change（有 BN）
        _, loss_vals_bn = compute_lipschitz_loss_change(model_bn, train_loader, nn.CrossEntropyLoss(), [1e-5, 1e-4, 1e-3, 1e-2], device)

        # 保存有 BN 结果
        save_results(model_bn, {
            "loss": epoch_losses,
            "val_acc": val_accs,
            "grad_norms": grad_norms,
            "lipschitz_loss": loss_vals_bn,
        }, "VGG_A_BN", lr, models_path, metrics_path)

        ## 可视化
        plot_gradient_norms(all_gradnorms_with_bn[-1], all_gradnorms_without_bn[-1], lr, figures_path)
        plot_lipschitz_curve(steps, loss_vals_bn, loss_vals_no_bn, lr, figures_path)

    # 整体 Loss 和 Accuracy 曲线绘图
    plot_loss_curve(all_losses_with_bn, all_losses_without_bn, learning_rates, figures_path)
    plot_accuracy_curve(all_valaccs_with_bn, all_valaccs_without_bn, learning_rates, figures_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()




