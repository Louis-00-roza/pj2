import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from itertools import product
import pandas as pd

assert torch.cuda.is_available(), "GPU 不可用，请检查 CUDA 安装或驱动配置"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.makedirs("./saved_models", exist_ok=True)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.use_bn = use_bn
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_bn),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Net(nn.Module):
    def __init__(self, activation=nn.ReLU, dropout=0.5, use_bn=True, use_residual=True, use_logsoftmax=False):
        super(Net, self).__init__()
        self.activation = activation()
        self.in_channels = 32
        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_logsoftmax = use_logsoftmax

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()

        if use_residual:
            self.layer1 = self._make_layer(ResidualBlock, 32, 1, activation, use_bn)
            self.layer2 = self._make_layer(ResidualBlock, 64, 1, activation, use_bn)
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32) if use_bn else nn.Identity(),
                activation(),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64) if use_bn else nn.Identity(),
                activation(),
            )

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1) if use_logsoftmax else nn.Identity()

    def _make_layer(self, block, out_channels, stride, activation, use_bn):
        layer = block(self.in_channels, out_channels, stride, activation, use_bn)
        self.in_channels = out_channels
        return nn.Sequential(layer)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.logsoftmax(x)
        return x


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer_cls, num_epochs=10, lr=0.001, optimizer_kwargs=None):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")

    return model, train_losses, train_accuracies, test_accuracies


if __name__ == '__main__':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    activations = [nn.ReLU, nn.ELU]
    optimizers = [optim.Adam, optim.SGD]
    lrs = [1e-2, 1e-3, 1e-4]
    dropouts = [0.3, 0.5]
    bn_flags = [True]
    res_flags = [True]
    loss_fns = [(nn.CrossEntropyLoss, False), (nn.NLLLoss, True)]

    config_list = list(product(activations, optimizers, lrs, dropouts, bn_flags, res_flags, loss_fns))
    result_summary = []

    for idx, (act_fn, opt_fn, lr, drop, bn, res, (loss_cls, use_logsoftmax)) in enumerate(config_list):
        loss_fn = loss_cls()
        config_name = f"{act_fn.__name__}_{opt_fn.__name__}_lr{lr}_drop{drop}_bn{bn}_res{res}_{loss_cls.__name__}"
        print(f"\n[Config {idx+1}/{len(config_list)}] {config_name}")

        model = Net(activation=act_fn, dropout=drop, use_bn=bn, use_residual=res, use_logsoftmax=use_logsoftmax).to(device)
        model, losses, train_accs, test_accs = train_and_evaluate(
            model=model,
            train_loader=trainloader,
            test_loader=testloader,
            criterion=loss_fn,
            optimizer_cls=opt_fn,
            num_epochs=20,
            lr=lr,
            optimizer_kwargs={'momentum': 0.9} if opt_fn == optim.SGD else {}
        )

        torch.save(model.state_dict(), f"./saved_models/model_{idx+1}_{config_name}.pth")
        np.save(f"./saved_models/losses_{idx+1}.npy", np.array(losses))
        np.save(f"./saved_models/train_accs_{idx+1}.npy", np.array(train_accs))
        np.save(f"./saved_models/test_accs_{idx+1}.npy", np.array(test_accs))

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(losses, label="Train Loss")
        axs[0].set_title("Training Loss")
        axs[0].grid(True)
        axs[1].plot(train_accs, label="Train Acc")
        axs[1].plot(test_accs, label="Test Acc")
        axs[1].legend()
        axs[1].set_title("Accuracy")
        axs[1].grid(True)
        fig.suptitle(config_name)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.savefig(f"./saved_models/curve_{idx+1}_{config_name}.png")
        plt.close()

        result_summary.append({
            "Config": config_name,
            "Final Train Acc": train_accs[-1],
            "Final Test Acc": test_accs[-1],
            "Best Test Acc": max(test_accs)
        })

    df = pd.DataFrame(result_summary)
    df.to_csv("./saved_models/results_summary.csv", index=False)
    print("训练完成，所有结果已保存至 ./saved_models/")
