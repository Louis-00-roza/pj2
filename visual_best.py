# visual.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from cifar_10 import Net  # 使用你定义的模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择一个模型配置文件
model_path = "D:\pythonProject1\PJ2\Codes\saved_models\model_5_ReLU_Adam_lr0.001_drop0.3_bnTrue_resTrue_CrossEntropyLoss.pth"
model = Net().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

# ---------- 1. 可视化卷积核 ----------
def visualize_kernels(model):
    kernels = model.conv1.weight.data.clone().cpu()
    kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())  # 归一化到 0~1

    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        if i < kernels.size(0):
            img = kernels[i].permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
            ax.imshow(img)
            ax.axis("off")
    plt.suptitle("Conv1 Kernels")
    plt.tight_layout()
    plt.savefig("kernels_conv1.png")
    plt.close()

# ---------- 2. 可视化激活图 ----------
def visualize_activations(model, testloader):
    def hook_fn(module, input, output):
        global activations
        activations = output.detach().cpu()

    handle = model.layer1[0].conv1.register_forward_hook(hook_fn)

    images, _ = next(iter(testloader))
    images = images.to(device)
    _ = model(images)

    act = activations[0]  # 取第一张图的激活图 (C, H, W)

    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        if i < act.size(0):
            ax.imshow(act[i], cmap='viridis')
            ax.axis("off")
    plt.suptitle("Activations of layer1[0].conv1")
    plt.tight_layout()
    plt.savefig("activations_layer1_conv1.png")
    plt.close()
    handle.remove()

# ---------- 3. 混淆矩阵 ----------
def plot_confusion_matrix(model, testloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=testset.classes, yticklabels=testset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    print("可视化卷积核 ...")
    visualize_kernels(model)
    print("可视化激活图 ...")
    visualize_activations(model, testloader)
    print("绘制混淆矩阵 ...")
    plot_confusion_matrix(model, testloader)
    print("可视化完成，图片已保存：kernels_conv1.png，activations_layer1_conv1.png，confusion_matrix.png")
