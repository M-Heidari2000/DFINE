import torch
import wandb
import einops
import numpy as np
import gymnasium as gym
from argparse import Namespace
from tqdm import tqdm
from .agents import ILQRAgent
from .models import (
    Encoder,
    Dynamics,
    CostModel,
    Decoder,
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
    agent = ILQRAgent(
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


def test_prediction(
    args: Namespace,
    encoder: Encoder,
    decoder: Decoder,
    dynamics_model: Dynamics,
    y: torch.Tensor,
    u: torch.Tensor,
):
    with torch.no_grad():
        
        F = y.shape[0]
        B = y.shape[1]
        T = u.shape[0]

        assert T >= F, "the input sequence (u) must be at least the same length as the observation sequence (y)" 

        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=B)

        # initial belief over x0: N(0, I)
        mean = torch.zeros((B, args.x_dim), device=y.device)
        cov = torch.eye(args.x_dim, args.x_dim, device=y.device).repeat([B, 1, 1])

        for t in range(1, F):
            mean, cov = dynamics_model.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t-1],
            )
            mean, cov = dynamics_model.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t],
            )

        pred_y = torch.zeros((T+1-F, B, y.shape[-1]), device=y.device)
        # N(mean, cov) is the posterior for x_{F-1}
        for t in range(F, T+1):
            mean, cov = dynamics_model.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t-1],
            )
            pred_a = dynamics_model.get_a(mean)
            pred_y[t-F] = decoder(pred_a)

        return pred_y
    

def test_A_changes(
    args: Namespace,
    encoder: Encoder,
    dynamics_model: Dynamics,
    y: torch.Tensor,
    u: torch.Tensor,
):
    with torch.no_grad():
        
        F = y.shape[0]
        B = y.shape[1]
        T = u.shape[0]

        assert F == T+1, "the input sequence (u) must be at least the same length as the observation sequence (y)" 

        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=B)

        # initial belief over x0: N(0, I)
        mean = torch.zeros((B, args.x_dim), device=y.device)
        cov = torch.eye(args.x_dim, args.x_dim, device=y.device).repeat([B, 1, 1])

        singular_values = torch.zeros((F-1, B, args.x_dim), device=y.device)

        for t in range(1, F):
            mean, cov = dynamics_model.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t-1],
            )
            mean, cov = dynamics_model.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t],
            )
            A, _, _, _, _ = dynamics_model.get_dynamics(x=mean)
            _, S, _ = A.svd()
            singular_values[t-1] = S

        return singular_values
