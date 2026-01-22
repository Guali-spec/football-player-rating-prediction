import numpy as np
import pandas as pd
import torch

from dataset import load_and_prepare_data
from model import MLPRegressor


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # On recharge le CSV juste pour afficher des infos lisibles
    df = pd.read_csv("data/raw/players.csv").reset_index(drop=True)

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

    # Sauvegarde tableau d’analyse simple
    results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": abs_err,
    })

    # Sauvegarder pour le rapport
    results.to_csv("outputs/logs/test_predictions.csv", index=False)

    # Stats globales
    print("=== ERROR SUMMARY ===")
    print("MAE:", abs_err.mean())
    print("Median abs error:", np.median(abs_err))
    print("90th percentile abs error:", np.percentile(abs_err, 90))
    print("95th percentile abs error:", np.percentile(abs_err, 95))

    # Pires cas
    worst = results.sort_values("abs_error", ascending=False).head(20)
    print("\n=== TOP 20 PIRES ERREURS ===")
    print(worst.to_string(index=False))

    print("\nFichier enregistré: outputs/logs/test_predictions.csv")


if __name__ == "__main__":
    main()
