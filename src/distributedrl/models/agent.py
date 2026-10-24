import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    """Backbone + actor head + critic head. If `critic_backbone` is given the critic gets its own backbone,
    otherwise both heads share `backbone`."""

    def __init__(self, backbone, actor, critic, critic_backbone=None):
        super().__init__()
        self.backbone = backbone
        self.critic_backbone = critic_backbone
        self.actor = actor
        self.critic = critic

    def _critic_value(self, x, features):
        critic_features = features if self.critic_backbone is None else self.critic_backbone(x)
        return self.critic(critic_features)

    def get_value(self, x):
        return self._critic_value(x, self.backbone(x) if self.critic_backbone is None else None)

    def get_action_and_value(self, x, action=None):
        features = self.backbone(x)
        probs = Categorical(logits=self.actor(features))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self._critic_value(x, features)
