import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP_P(nn.Module):
    def __init__(self):
        super(MLP_P, self).__init__()
        self.fc1 = nn.Linear(2048, 1024, bias=False)
        # self.fc2 = nn.Linear(4096, 1024, bias=False)
        self.fc3 = nn.Linear(1024, 256, bias=False)
        # self.fc4 = nn.Linear(256, 64, bias=False)
        # self.fc5 = nn.Linear(64, 16, bias=False)
        self.fc6 = nn.Linear(256, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        out = self.fc6(x)
        return out


class MLP_V(nn.Module):
    def __init__(self):
        super(MLP_V, self).__init__()
        self.fc1 = nn.Linear(2048, 256, bias=False)
        # self.fc2 = nn.Linear(4096, 1024, bias=False)
        # self.fc3 = nn.Linear(1024, 256, bias=False)
        self.fc4 = nn.Linear(256, 64, bias=False)
        # self.fc5 = nn.Linear(64, 16, bias=False)
        self.fc6 = nn.Linear(64, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        out = self.fc6(x)
        return out

















