from dataset import load_and_prepare_data

train_loader, val_loader, test_loader, input_dim = load_and_prepare_data(
    csv_path="data/raw/players.csv"
)

print("Input dim:", input_dim)
print("Train batches:", len(train_loader))
