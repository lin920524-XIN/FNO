import os

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import torch
from meep import materials

from FDTD import FDTDSimulator, generate_dataset
import FNO


# 設定頻率數量與 frame 的切分比例
NUM_SAMPLES = 200
RATIO = 0.4


# 多個頻率的模擬
X_all, Y_all, freq_list = generate_dataset(
    n_samples=NUM_SAMPLES,
    freq_min=1.0,
    freq_max=5.0,
    total_meep_time=100,
    num_frames=100,
    resolution=333,
    seed=42,
    ratio=RATIO,
    save_path_X="dataset_X.npy",
    save_path_Y="dataset_Y.npy",
    save_path_freq="dataset_freq.npy",
    verbose=True
)


# 設定 FNO 模型參數
CFG = {
    "path_X": "dataset_X.npy",
    "path_Y": "dataset_Y.npy",
    "val_ratio": 0.15,
    "n_modes": (12, 12),
    "n_layers": 4,
    "hidden_channels": 24,
    "in_channels": None,
    "out_channels": None,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler_step": 60,
    "scheduler_gamma": 0.5,
    "epochs": 100,
    "seed": 42,
    "device": "cpu",
    "save_dir": "checkpoints"
}

# FNO 模型的訓練
model, history, test_loader = FNO.train(CFG)

FNO.plot_loss(history)

# FNO 模型的測試
preds, targets = FNO.test(model, test_loader, CFG)

FNO.plot_prediction(preds, targets, sample_idx=0, frame_idx=-1)
