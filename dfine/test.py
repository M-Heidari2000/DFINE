import torch
import wandb
import numpy as np
import gymnasium as gym
from argparse import Namespace
from tqdm import tqdm
from .agents import MPCAgent
from .models import (
    Encoder,
    Dynamics,
    CostModel,
)


def test(
    args: Namespace,
    env: gym.Env,
    encoder: Encoder,
    dynamics_model: Dynamics,
    cost_model: CostModel,
) -> np.array:
    
    encoder.eval()
    dynamics_model.eval()
    cost_model.eval()

    # agent
    agent = MPCAgent(
        encoder=encoder,
        dynamics_model=dynamics_model,
        cost_model=cost_model,
        planning_horizon=args.planning_horizon,
        action_noise=args.action_noise_std,
    )

    with torch.no_grad():
        rewards = []
        for _ in tqdm(range(args.num_test_episodes)):
            obs, info = env.reset()
            agent.reset()
            action = env.action_space.sample()
            done = False
            total_reward = 0.0
            while not done:
                planned_actions = agent(y=obs, u=action, explore=False)
                action = planned_actions[0]
                next_obs, reward, terminated, truncated, _ = env.step(action=action)
                done = terminated or truncated
                obs = next_obs
                total_reward += reward
            rewards.append(total_reward)

    wandb.log({"rewards": wandb.Histogram(rewards)})