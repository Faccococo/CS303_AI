import torch
import torch.nn as nn


class NNet(nn.Module):

    def __init__(self):
        super(NNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def forward(self, feature):
        result = self.classifier(feature)
        return

    def get_result(self, feature):
        return int(torch.round(self.classifier(feature))[0])


# model = ANN()
# model.train(feature_train, label_train)
