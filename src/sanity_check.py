import pandas as pd
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

df = pd.read_csv("data/raw/players.csv")

print("CSV loaded:", df.shape)
print("Columns:", list(df.columns)[:10], "...")
