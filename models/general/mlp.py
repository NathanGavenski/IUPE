import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .attention import Self_Attn1D


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.output = nn.Linear(out, out_dim)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.output(x))
        return x


class MlpWithAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MlpWithAttention, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)

        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.fc3 = nn.Linear(out, out)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)
        self.attention2 = Self_Attn1D(out, nn.LeakyReLU)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x, _ = self.attention(x)
        x = self.relu(self.fc(x))
        x, _ = self.attention2(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.output(x))
        return x


class MlpAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MlpAttention, self).__init__()

        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.output = nn.Linear(out, out_dim)
        self.attention = Self_Attn1D(out, nn.LeakyReLU)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.input(x)
        x, att = self.attention(x)
        x = self.relu(self.output(x))
        return x
