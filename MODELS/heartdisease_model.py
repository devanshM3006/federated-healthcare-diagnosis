import torch
import torch.nn as nn
import torch.nn.functional as F

class HeartDiseaseNet(nn.Module):
    def __init__(self):
        super(HeartDiseaseNet, self).__init__()

        self.fc1 = nn.Linear(13, 16)
        self.fc2 = nn.Linear(16, 12)
        self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x