import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import load_and_prepare_data
from model import MLPRegressor


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader, input_dim = load_and_prepare_data(
        csv_path="data/raw/players.csv",
        batch_size=512
    )

    model = MLPRegressor(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
    model.eval()

    y_true_all, y_pred_all = [], []

    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy().reshape(-1)
        yb = yb.numpy().reshape(-1)
        y_true_all.append(yb)
        y_pred_all.append(preds)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    abs_err = np.abs(y_true - y_pred)

    # 1) Scatter vrai vs prédit
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Overall rating (vrai)")
    plt.ylabel("Overall rating (prédit)")
    plt.title("Vrai vs Prédit")
    plt.tight_layout()
    plt.savefig("outputs/logs/true_vs_pred.png", dpi=200)
    plt.close()

    # 2) Histogramme des erreurs absolues
    plt.figure()
    plt.hist(abs_err, bins=30)
    plt.xlabel("|erreur|")
    plt.ylabel("nombre d'exemples")
    plt.title("Distribution des erreurs absolues")
    plt.tight_layout()
    plt.savefig("outputs/logs/abs_error_hist.png", dpi=200)
    plt.close()

    # 3) Courbe cumulative : % des exemples sous un seuil d’erreur
    thresholds = np.linspace(0, max(3.0, abs_err.max()), 50)
    pct = [(abs_err <= t).mean() for t in thresholds]

    plt.figure()
    plt.plot(thresholds, pct)
    plt.xlabel("seuil |erreur|")
    plt.ylabel("proportion d'exemples ≤ seuil")
    plt.title("Couverture d’erreur (cumulative)")
    plt.tight_layout()
    plt.savefig("outputs/logs/error_cdf.png", dpi=200)
    plt.close()

    print("Images enregistrées dans outputs/logs/:")
    print("- true_vs_pred.png")
    print("- abs_error_hist.png")
    print("- error_cdf.png")


if __name__ == "__main__":
    main()
