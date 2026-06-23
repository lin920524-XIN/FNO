import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from neuralop.models import FNO
from neuralop.utils import count_model_params


# 資料的預處理
class FieldDataset(Dataset):

    
    # 初始化與預處理資料
    def __init__(self, X: np.ndarray, Y: np.ndarray, normalize: bool = True):

        X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
        Y = np.transpose(Y, (0, 3, 1, 2)).astype(np.float32)
        
        if normalize:

            mean = X.mean(axis=(1, 2, 3), keepdims=True)
            std = X.std(axis=(1, 2, 3), keepdims=True) + 1e-8
            X = (X - mean) / std
            Y = (Y - mean) / std

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    
    # 取得資料集總數量
    def __len__(self):
        return self.X.shape[0]

    
    # 根據索引值取得單筆特徵與標籤
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# 載入資料函數
def load_data(cfg: dict):

    X_np = np.load(cfg["path_X"])
    Y_np = np.load(cfg["path_Y"])
    
    print(f"Loaded X: {X_np.shape}  Y: {Y_np.shape}")

    cfg["in_channels"] = X_np.shape[-1]
    cfg["out_channels"] = Y_np.shape[-1]

    dataset = FieldDataset(X_np, Y_np, normalize=True)

    B = len(dataset)
    n_train = int(B * cfg["train_ratio"])
    n_val = int(B * cfg["val_ratio"])
    n_test = B - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["seed"])

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    print(f"Split → train: {n_train}  val: {n_val}  test: {n_test}")

    return train_loader, val_loader, test_loader


# FNO 的模型建立
def build_model(cfg: dict) -> FNO:

    model = FNO(
        n_modes=cfg["n_modes"][:2],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        hidden_channels=cfg["hidden_channels"],
        n_layers=cfg["n_layers"],
    )

    print(f"Model params: {count_model_params(model):,}")

    return model
    

# FNO 的模型訓練
def train(cfg: dict):

    torch.manual_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = load_data(cfg)

    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler_step"],
        gamma=cfg["scheduler_gamma"]
    )

    
    # 定義相對損失函數
    def rel_l2_loss(pred, target):

        diff = (pred - target).pow(2).sum(dim=(-1, -2))
        
        denom = target.pow(2).sum(dim=(-1, -2)) + 1e-8

        return (diff / denom).sqrt().mean()

    history = {"train_loss": [], "val_loss": []}

    best_val = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            pred = model(X_batch)
            loss = rel_l2_loss(pred, Y_batch)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        scheduler.step()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for X_batch, Y_batch in val_loader:
                
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                pred = model(X_batch)
                val_loss += rel_l2_loss(pred, Y_batch).item()

        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{cfg['epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg["save_dir"], "best_model.pt"))

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint saved → {cfg['save_dir']}/best_model.pt")

    model.load_state_dict(torch.load(os.path.join(cfg["save_dir"], "best_model.pt"), weights_only=False))

    return model, history, test_loader


# 測試資料的模型測試
def test(model, test_loader, cfg: dict):
    
    device = torch.device(cfg["device"])
    
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():

        for X_batch, Y_batch in test_loader:

            pred = model(X_batch.to(device)).cpu()

            all_preds.append(pred)
            all_targets.append(Y_batch)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    diff = (preds - targets).pow(2).sum(dim=(-1, -2, -3))
    denom = targets.pow(2).sum(dim=(-1, -2, -3)) + 1e-8
    rel_l2 = (diff / denom).sqrt()

    print(f"\nTest Relative L2 — mean: {rel_l2.mean():.4f}  "
          f"std: {rel_l2.std():.4f}  "
          f"max: {rel_l2.max():.4f}")

    return preds, targets


# 繪製損失折線圖
def plot_loss(history: dict):
    
    plt.figure(figsize=(8, 4))

    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")

    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Loss")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig("loss_curve.png", dpi=150)
    plt.show()


# 繪製預測結果
def plot_prediction(preds, targets, sample_idx=0, frame_idx=-1):

    pred_frame = preds[sample_idx, frame_idx].numpy() 
    target_frame = targets[sample_idx, frame_idx].numpy()
    
    error_frame = np.abs(pred_frame - target_frame)

    vmax = max(np.abs(pred_frame).max(), np.abs(target_frame).max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(target_frame.T, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Ground Truth (sample {sample_idx}, frame {frame_idx})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_frame.T, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"FNO Prediction (sample {sample_idx}, frame {frame_idx})")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error_frame.T, cmap="hot", origin="lower")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Hz Field Prediction vs Ground Truth", fontsize=13)
    plt.tight_layout()

    plt.savefig(f"prediction_s{sample_idx}_f{frame_idx}.png", dpi=150)
    plt.show()
