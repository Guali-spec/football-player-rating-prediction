import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_and_prepare_data
from model import MLPRegressor


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_mae = 0.0
    n = 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_mae += (preds - yb).abs().mean().item() * bs
            n += bs

    return total_loss / n, total_mae / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ====== Data ======
    train_loader, val_loader, test_loader, input_dim = load_and_prepare_data(
        csv_path="data/raw/players.csv",
        batch_size=256
    )

    # ====== Model ======
    model = MLPRegressor(input_dim=input_dim).to(device)

    # Loss robuste pour régression
    criterion = nn.SmoothL1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Early stopping
    best_val_mae = float("inf")
    patience = 10
    bad_epochs = 0

    epochs = 100
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_mae = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr_loss:.4f} mae={tr_mae:.4f} | "
            f"val loss={va_loss:.4f} mae={va_mae:.4f}"
        )

        # Save best
        if va_mae < best_val_mae - 1e-4:
            best_val_mae = va_mae
            bad_epochs = 0
            torch.save(model.state_dict(), "outputs/models/best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    # ====== Test final ======
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
    te_loss, te_mae = run_epoch(model, test_loader, criterion, None, device)

    print(f"TEST | loss={te_loss:.4f} mae={te_mae:.4f}")


if __name__ == "__main__":
    main()
