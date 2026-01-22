import pandas as pd
import torch
import numpy as np

from dataset import load_and_prepare_data
from model import MLPRegressor


def interpret_error(error):
    if error < 0.5:
        return "Prédiction très précise"
    elif error < 1.0:
        return "Prédiction précise"
    elif error < 2.0:
        return "Prédiction acceptable"
    else:
        return "Prédiction imprécise"


def rating_category(r):
    if r >= 85:
        return "Joueur d’élite"
    elif r >= 80:
        return "Très bon joueur"
    elif r >= 75:
        return "Bon joueur"
    elif r >= 70:
        return "Joueur moyen"
    else:
        return "Joueur en développement"

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger données + modèle
    df = pd.read_csv("data/raw/players.csv")

    _, _, test_loader, input_dim = load_and_prepare_data(
        csv_path="data/raw/players.csv",
        batch_size=1  # une prédiction à la fois
    )

    model = MLPRegressor(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
    model.eval()

    print("\n=== EXEMPLES DE PRÉDICTIONS LISIBLES ===\n")

    count = 0
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).item()
        true = yb.item()
        error = abs(pred - true)

        interpretation = interpret_error(error)

        print(f"Note réelle du joueur     : {true:.1f}")
        print(f"Note prédite par le modèle: {pred:.1f}")
        print(f"Erreur                    : {error:.2f}")
        print(f"Interprétation            : {interpretation}")
        print("-" * 50)

        count += 1
        if count == 5:
            break


if __name__ == "__main__":
    main()



