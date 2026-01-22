import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader


def load_and_prepare_data(
    csv_path: str,
    batch_size: int = 256,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
):
    # =========================
    # 1) Charger le dataset
    # =========================
    df = pd.read_csv(csv_path)

    TARGET = "overall_rating"

    # =========================
    # 2) Feature engineering
    # =========================
    # Position principale
    df["primary_position"] = (
        df["positions"]
        .astype(str)
        .str.split(",")
        .str[0]
    )

    # =========================
    # 3) Colonnes à garder
    # =========================
    skill_start = df.columns.get_loc("crossing")
    skill_end = df.columns.get_loc("sliding_tackle")
    skill_cols = list(df.columns[skill_start: skill_end + 1])


    numeric_cols = [
        "age",
        "height_cm",
        "weight_kgs",
        "international_reputation(1-5)",
        "weak_foot(1-5)",
        "skill_moves(1-5)",
    ] + skill_cols

    cat_cols = [
        "preferred_foot",
        "primary_position",
    ]

    drop_cols = [
        "name",
        "full_name",
        "birth_date",
        "positions",
        "nationality",
        "value_euro",
        "wage_euro",
        "release_clause_euro",
        "national_team",
        "national_rating",
        "national_team_position",
        "national_jersey_number",
        "potential",
    ]

    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors="ignore"
    )

    df = df[[TARGET] + numeric_cols + cat_cols]

    # =========================
    # 4) Gestion des valeurs manquantes
    # =========================
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # =========================
    # 5) Encodage catégoriel
    # =========================
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # =========================
    # 6) Séparation X / y
    # =========================
    X = df.drop(columns=[TARGET]).values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

    # =========================
    # 7) Split train / val / test
    # =========================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # =========================
    # 8) Normalisation
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # =========================
    # 9) DataLoaders PyTorch
    # =========================
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dim
