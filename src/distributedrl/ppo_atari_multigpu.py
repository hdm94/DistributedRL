# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_multigpupy
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ale_py
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter

from distributedrl.models import build_agent  # isort:skip
from distributedrl.wrappers.atari import (  # isort:skip
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from distributedrl.wrappers.ram import RamInfoEnv, ram_fields_for
from distributedrl.wrappers.rewards import REWARDS, apply_reward
from distributedrl.wrappers.stats import SeaquestStatsEnv

gym.register_envs(ale_py)  # registers the ALE/Atari environments with gymnasium

load_dotenv()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    log_dir: str = "."
    """root directory for tensorboard logs (runs/), videos/, wandb files and hydra outputs/"""
    ram_info: bool = False
    """if toggled, expose the emulator RAM in `info` and log decoded fields (oxygen, depth, ...) as ram/*"""
    game_stats: bool = True
    """if toggled, log Seaquest per-episode stats (surfacings, divers collected/delivered, steps) as game/*"""
    vector_env: str = "sync"
    """how the local envs are stepped: "sync" (serially, in-process) or "async" (one subprocess per env)"""
    reward: str = "clip"
    """reward wrapper applied to the environment, one of the keys of REWARDS in wrappers/rewards.py"""
    model: Dict[str, Any] = field(default_factory=dict)
    """network architecture config, see conf/model/*.yaml and models/__init__.py"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 8
    """the number of parallel game environments (in the local rank)"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    device_ids: List[int] = field(default_factory=lambda: [])
    """the device ids that subprocess workers will use"""
    backend: str = "gloo"
    """the backend for distributed training, one of gloo, nccl, mpi"""

    # to be filled in runtime
    local_batch_size: int = 0
    """the local batch size in the local rank (computed in runtime)"""
    local_minibatch_size: int = 0
    """the local mini-batch size in the local rank (computed in runtime)"""
    num_envs: int = 0
    """the number of parallel game environments (computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""


# register the dataclass as the config schema so hydra validates types and keys
cs = ConfigStore.instance()
cs.store(name="base_ppo_atari_multigpu", node=Args)


def make_env(env_id, idx, capture_video, run_name, log_dir, reward, ram_info=False, game_stats=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"{log_dir}/videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if ram_info:
            env = RamInfoEnv(env, ram_fields_for(env_id))
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = apply_reward(env, reward)
        if game_stats and ram_fields_for(env_id):
            env = SeaquestStatsEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        return env

    return thunk


@hydra.main(version_base=None, config_path="conf", config_name="ppo_atari_multigpu")
def main(cfg: Args) -> None:
    # torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py
    # taken from https://pytorch.org/docs/stable/elastic/run.html
    args: Args = OmegaConf.to_object(cfg)
    assert args.reward in REWARDS, f"unknown reward {args.reward}, choose from {sorted(REWARDS)}"
    assert args.vector_env in ("sync", "async"), f"unknown vector_env {args.vector_env}"
    assert args.backend in ("gloo", "nccl", "mpi"), f"unknown backend {args.backend}"
    os.makedirs(args.log_dir, exist_ok=True)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.num_envs = args.local_num_envs * args.world_size
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=args.world_size)
    else:
        warnings.warn(
            """
Not using distributed mode!
If you want to use distributed mode, please execute this script with 'torchrun'.
E.g., `torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py`
        """
        )
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = None
    if local_rank == 0:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                dir=args.log_dir,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"{args.log_dir}/runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)

    # TRY NOT TO MODIFY: seeding
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if len(args.device_ids) > 0:
        assert len(args.device_ids) == args.world_size, "you must specify the same number of device ids as `--nproc_per_node`"
        device = torch.device(f"cuda:{args.device_ids[local_rank]}" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        device_count = torch.cuda.device_count()
        if device_count < args.world_size:
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # async runs each env in its own subprocess (uses multiple CPU cores); sync steps them serially in this process
    vector_env_cls = gym.vector.AsyncVectorEnv if args.vector_env == "async" else gym.vector.SyncVectorEnv
    envs = vector_env_cls(
        [make_env(args.env_id, i, args.capture_video, run_name, args.log_dir, args.reward, args.ram_info, args.game_stats) for i in range(args.local_num_envs)],
        # SAME_STEP makes the vector env report `final_info`, which the episodic_return logging relies on
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = build_agent(args.model, envs).to(device)
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.local_num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.local_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.local_num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.local_num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    ram_fields = ram_fields_for(args.env_id) if args.ram_info else {}
    stat_names = SeaquestStatsEnv.stat_names if args.game_stats and ram_fields_for(args.env_id) else ()
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.local_num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        ram_values = {name: [] for name in ram_fields}
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if not writer:
                continue

            for name in ram_fields:
                ram_values[name].append(np.mean(infos[name]))

            # gymnasium 1.x vector envs batch final_info as {"episode": {"r": [...], "l": [...]}, "_episode": mask}
            if "final_info" in infos and "episode" in infos["final_info"]:
                episode = infos["final_info"]["episode"]
                for i in np.flatnonzero(infos["final_info"]["_episode"]):
                    print(f"global_step={global_step}, episodic_return={episode['r'][i]}")
                    writer.add_scalar("charts/episodic_return", episode["r"][i], global_step)
                    writer.add_scalar("charts/episodic_length", episode["l"][i], global_step)
                for name in stat_names:
                    if name in infos["final_info"]:
                        for i in np.flatnonzero(infos["final_info"][f"_{name}"]):
                            writer.add_scalar(f"game/{name}", infos["final_info"][name][i], global_step)

        print(
            f"local_rank: {local_rank}, action.sum(): {action.sum()}, iteration: {iteration}, actor param sum: {sum(p.sum() for p in agent.actor.parameters())}"
        )
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.local_batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.local_batch_size, args.local_minibatch_size):
                end = start + args.local_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()

                if args.world_size > 1:
                    # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in agent.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in agent.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / args.world_size
                            )
                            offset += param.numel()

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if local_rank == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            for name, ram_vals in ram_values.items():
                writer.add_scalar(f"ram/{name}", np.mean(ram_vals), global_step)

    envs.close()
    if local_rank == 0:
        writer.close()
        if args.track:
            wandb.finish()


if __name__ == "__main__":
    main()
