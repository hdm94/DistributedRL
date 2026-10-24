from distributedrl.models.agent import ActorCritic
from distributedrl.models.backbones import MLP, NatureCNN
from distributedrl.models.heads import CategoricalActor, ValueHead

BACKBONES = {"nature_cnn": NatureCNN, "mlp": MLP}
ACTORS = {"categorical": CategoricalActor}
CRITICS = {"value": ValueHead}


def _lookup(registry, kind, cfg):
    kwargs = dict(cfg)
    name = kwargs.pop("name")
    if name not in registry:
        raise ValueError(f"unknown {kind} {name!r}, choose from {sorted(registry)}")
    return registry[name], kwargs


def build_agent(model_cfg, envs):
    """Build an ActorCritic from the `model` config (see conf/model/*.yaml)."""
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    backbone_cls, backbone_kwargs = _lookup(BACKBONES, "backbone", model_cfg["backbone"])
    backbone = backbone_cls(obs_shape, **backbone_kwargs)
    critic_backbone = None
    if not model_cfg.get("shared_backbone", True):
        critic_backbone = backbone_cls(obs_shape, **backbone_kwargs)

    actor_cls, actor_kwargs = _lookup(ACTORS, "actor", model_cfg["actor"])
    critic_cls, critic_kwargs = _lookup(CRITICS, "critic", model_cfg["critic"])
    critic_in = (backbone if critic_backbone is None else critic_backbone).out_dim
    return ActorCritic(
        backbone,
        actor_cls(backbone.out_dim, n_actions, **actor_kwargs),
        critic_cls(critic_in, **critic_kwargs),
        critic_backbone,
    )
