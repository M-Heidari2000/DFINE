import os
import json
import torch
import wandb
import einops
import numpy as np
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from datetime import datetime
from .agents import MPCAgent
from .memory import ReplayBuffer
from gymnasium.wrappers import RescaleAction, DtypeObservation
from .env_utils import ActionRepeatWrapper
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)


def train(
    args: Namespace,
):

    # prepare logging
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    
    wandb.init(
        project="Controlling from high-dimensional observations",
        name="DFINE",
        config=vars(args),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("*",step_metric="global_step")

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # make environment
    env = gym.make(id=args.env)
    env = DtypeObservation(env=env, dtype=np.float32)
    env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)
    env = ActionRepeatWrapper(env=env, repeat=args.action_repeat)

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=env.observation_space.shape[0],
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
    ).to(device)

    decoder = Decoder(
        y_dim=env.observation_space.shape[0],
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=args.x_dim,
        u_dim=env.action_space.shape[0],
        a_dim=args.a_dim,
        device=device,
    ).to(device)

    cost_model = CostModel(
        x_dim=args.x_dim,
        u_dim=env.action_space.shape[0],
        device=device,
        hidden_dim=args.hidden_dim,
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder, cost_model], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters()) +
        list(cost_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps)

    # agent
    agent = MPCAgent(
        encoder=encoder,
        dynamics_model=dynamics_model,
        cost_model=cost_model,
        planning_horizon=args.planning_horizon
    )

    # replay buffer
    buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        y_dim=env.observation_space.shape[0],
        u_dim=env.action_space.shape[0],
    )

    # collect seed episodes
    for s in range(1, args.seed_episodes+1):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action=action)
            done = terminated or truncated
            buffer.push(
                y=obs,
                u=action,
                c=-reward,
                done=done
            )
            obs = next_obs

    # train and test loop
    for episode in tqdm(range(args.all_episodes)):
        
        # model fit
        for s in range(args.collect_interval):
            encoder.train()
            decoder.train()
            dynamics_model.train()
            cost_model.train()

            y, u, c, _ = buffer.sample(
                batch_size=args.batch_size,
                chunk_length=args.chunk_length,
            )

            # convert to tensor, transform to device, reshape to time-first
            y = torch.as_tensor(y, device=device)
            y = einops.rearrange(y, "b l y -> l b y")
            a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
            a = einops.rearrange(a, "(l b) a -> l b a", b=args.batch_size)
            u = torch.as_tensor(u, device=device)
            u = einops.rearrange(u, "b l u -> l b u")
            c = torch.as_tensor(c, device=device)
            c = einops.rearrange(c, "b l 1 -> l b 1")

            # initial belief over x0: N(0, I)
            mean = torch.zeros((args.batch_size, args.x_dim), device=device)
            cov = torch.eye(args.x_dim, device=device).repeat([args.batch_size, 1, 1])

            y_pred_loss = 0.0
            y_filter_loss = 0.0
            cost_loss = 0.0

            for t in range(1, args.chunk_length - args.prediction_k):
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
                cost_loss += nn.MSELoss()(cost_model(x=mean, u=u[t]), c[t])
                y_filter_loss += nn.MSELoss()(
                    decoder(mean @ dynamics_model.C.T),
                    y[t],
                )

                # tensors to hold predictions of future ys
                pred_y = torch.zeros((args.prediction_k, args.batch_size, env.observation_space.shape[0]), device=device)

                pred_mean = mean
                pred_cov = cov

                for k in range(args.prediction_k):
                    pred_mean, pred_cov = dynamics_model.dynamics_update(
                        mean=pred_mean,
                        cov=pred_cov,
                        u=u[t+k]
                    )
                    pred_y[k] = decoder(pred_mean @ dynamics_model.C.T)

                true_y = y[t+1: t+1+args.prediction_k]
                true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
                pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
                y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

            y_pred_loss /= (args.chunk_length - args.prediction_k - 1)

            # y filter loss
            y_filter_loss /= (args.chunk_length - 1)

            # cost loss
            cost_loss /= (args.chunk_length - 1)
        
            total_loss = (
                y_pred_loss +
                y_filter_loss +
                cost_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            clip_grad_norm_(all_params, args.clip_grad_norm)
            optimizer.step()

            global_step = episode * args.collect_interval + s
            wandb.log({
                "train/y prediction loss": y_pred_loss.item(),
                "train/y filter loss": y_filter_loss.item(),
                "train/cost loss": cost_loss.item(),
                "global_step": global_step,
            })

        # data collection
        with torch.no_grad():
            obs, info = env.reset()
            agent.reset()
            action = env.action_space.sample()
            done = False
            while not done:
                planned_actions = agent(y=obs, u=action, explore=True)
                action = planned_actions[0]
                next_obs, reward, terminated, truncated, _ = env.step(action=action)
                done = terminated or truncated
                buffer.push(
                    y=obs,
                    u=action,
                    c=-reward,
                    done=done
                )
                obs = next_obs

        # test
        if episode % args.test_interval == 0:
            rewards = []
            print("testing ...")
            for _ in tqdm(range(args.num_test_envs)):
                encoder.eval()
                decoder.eval()
                dynamics_model.eval()
                cost_model.eval()
                with torch.no_grad():
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
                
            avg_mean = np.array(rewards).mean()
            wandb.log({
                "average reward": avg_mean
            })

    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(dynamics_model.state_dict(), log_dir / "dynamics_model.pth")
    torch.save(cost_model.state_dict(), log_dir / "cost_model.pth")
    wandb.finish()

    return {"model_dir": log_dir}