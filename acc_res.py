import os
import numpy as np

metrics_path = r'D:\pythonProject1\PJ2\reports\metrics'

for fname in os.listdir(metrics_path):
    if fname.endswith('_valacc.npy'):
        fpath = os.path.join(metrics_path, fname)
        val_acc = np.load(fpath)
        print(f"{fname}: Final Acc = {val_acc[-1]:.4f}")
