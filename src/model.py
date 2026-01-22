import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)
