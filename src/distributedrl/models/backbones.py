# Backbones map an observation batch to a feature vector. Each exposes `out_dim`.
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """The DQN/Atari CNN (Mnih et al. 2015). Expects uint8-range image observations of shape (C, H, W)."""

    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(conv_out, feature_dim)), nn.ReLU())
        self.out_dim = feature_dim

    def forward(self, x):
        return self.fc(self.conv(x / 255.0))


class MLP(nn.Module):
    """Fully connected network for vector observations."""

    ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}

    def __init__(self, obs_shape, hidden_sizes=(64, 64), activation="tanh"):
        super().__init__()
        act = self.ACTIVATIONS[activation]
        layers, in_dim = [nn.Flatten()], int(np.prod(obs_shape))
        for size in hidden_sizes:
            layers += [layer_init(nn.Linear(in_dim, size)), act()]
            in_dim = size
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x):
        return self.net(x)
