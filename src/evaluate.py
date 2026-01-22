import numpy as np
import torch
import torch.nn as nn

from dataset import load_and_prepare_data
from model import MLPRegressor

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, input_dim = load_and_prepare_data(
        csv_path="data/raw/players.csv",
        batch_size=512
    )

    model = MLPRegressor(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
    model.eval()

    y_true_all = []
    y_pred_all = []

    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy().reshape(-1)
        yb = yb.numpy().reshape(-1)
        y_pred_all.append(preds)
        y_true_all.append(yb)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    _rmse = rmse(y_true, y_pred)
    _r2 = r2(y_true, y_pred)

    print("=== TEST METRICS ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {_rmse:.4f}")
    print(f"R^2  : {_r2:.4f}")

    # Exemples concrets
    idx = np.random.default_rng(42).choice(len(y_true), size=10, replace=False)
    print("\n=== 10 EXEMPLES (vrai -> predit) ===")
    for i in idx:
        print(f"{y_true[i]:6.2f} -> {y_pred[i]:6.2f} | err={abs(y_true[i]-y_pred[i]):.2f}")

    # Pires erreurs
    errs = np.abs(y_true - y_pred)
    worst = np.argsort(-errs)[:20]
    print("\n=== TOP 20 PIRES ERREURS ===")
    for i in worst:
        print(f"true={y_true[i]:6.2f} pred={y_pred[i]:6.2f} err={errs[i]:.2f}")

if __name__ == "__main__":
    main()
