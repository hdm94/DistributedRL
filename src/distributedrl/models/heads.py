# Heads map backbone features to policy logits or a state value.
import torch.nn as nn

from distributedrl.models.backbones import layer_init


class CategoricalActor(nn.Module):
    """Logits over a discrete action space."""

    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.layer = layer_init(nn.Linear(in_dim, n_actions), std=0.01)

    def forward(self, features):
        return self.layer(features)


class ValueHead(nn.Module):
    """Scalar state-value estimate."""

    def __init__(self, in_dim):
        super().__init__()
        self.layer = layer_init(nn.Linear(in_dim, 1), std=1.0)

    def forward(self, features):
        return self.layer(features)
