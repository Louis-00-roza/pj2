import os
import numpy as np
import matplotlib.pyplot as plt

def plot_lipschitz_fill_between_all_lrs(lr_list, step_sizes, metrics_path, figures_path):
    step_labels = [0.0] + step_sizes
    os.makedirs(figures_path, exist_ok=True)

    for lr in lr_list:
        path_bn = os.path.join(metrics_path, f'VGG_A_BN_lr_{lr}_lipschitz.npy')
        path_no_bn = os.path.join(metrics_path, f'VGG_A_NoBN_lr_{lr}_lipschitz.npy')

        if not os.path.exists(path_bn) or not os.path.exists(path_no_bn):
            print(f"[!] 缺失文件：lr={lr}，跳过")
            continue

        lipschitz_bn = np.load(path_bn, allow_pickle=True)
        lipschitz_no_bn = np.load(path_no_bn, allow_pickle=True)
        epo = len(lipschitz_bn)

        for idx, alpha in enumerate(step_labels):
            bn_vals = np.array([epoch_curve[idx] for epoch_curve in lipschitz_bn])
            no_bn_vals = np.array([epoch_curve[idx] for epoch_curve in lipschitz_no_bn])

            epochs = np.arange(epo)
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, bn_vals, label='With BN', color='blue', linewidth=2)
            plt.plot(epochs, no_bn_vals, label='Without BN', color='red', linewidth=2)

            # 填充两条线之间的区域
            plt.fill_between(epochs, bn_vals, no_bn_vals, where=(bn_vals > no_bn_vals),
                             interpolate=True, color='blue', alpha=0.2, label='BN > No BN')
            plt.fill_between(epochs, bn_vals, no_bn_vals, where=(bn_vals <= no_bn_vals),
                             interpolate=True, color='red', alpha=0.2, label='No BN > BN')

            plt.xlabel('Epoch')
            plt.ylabel('Lipschitz Loss')
            plt.title(f'Lipschitz Loss Comparison @ Step {alpha} | lr={lr}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_name = f'lipschitz_fill_step_{alpha}_lr_{lr}.png'
            plt.savefig(os.path.join(figures_path, save_name))
            plt.close()


lr_list = [1e-2, 1e-3, 1e-4, 1e-5]
step_sizes = [1e-5, 1e-4, 1e-3, 1e-2]
metrics_path = r'D:\pythonProject1\PJ2\reports\metrics'
figures_path = r'D:\pythonProject1\PJ2\reports\figures'

plot_lipschitz_fill_between_all_lrs(lr_list, step_sizes, metrics_path, figures_path)

