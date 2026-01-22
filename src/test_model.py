from model import MLPRegressor
import torch

model = MLPRegressor(input_dim=50)  # peu importe la valeur ici
x = torch.randn(8, 50)

y = model(x)
print(y.shape)
