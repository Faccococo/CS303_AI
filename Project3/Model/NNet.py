import torch
import torch.nn as nn

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, feature):
        score = self.model(feature)
        return score