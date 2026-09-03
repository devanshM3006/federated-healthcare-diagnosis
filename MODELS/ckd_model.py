import torch
import torch.nn as nn
import torch.nn.functional as F


class CKDNet(nn.Module):

    def __init__(self):
        super(CKDNet, self).__init__()

        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x