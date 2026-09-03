import torch
import torch.nn as nn
import torch.nn.functional as F


class BreastCancerNet(nn.Module):
    def __init__(self):
        super(BreastCancerNet, self).__init__()

        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x