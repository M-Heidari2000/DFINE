import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
from torch.distributions import MultivariateNormal
from torch.distributions import kl_divergence
from .memory import ReplayBuffer
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)


def train_backbone(
    args: Namespace,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
    ).to(device)

    decoder = Decoder(
        y_dim=train_buffer.y_dim,
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=args.x_dim,
        u_dim=train_buffer.u_dim,
        a_dim=args.a_dim,
        dropout_p=args.dropout_p,
        hidden_dim=args.hidden_dim,
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

    # train and test loop
    print(f"training on {device} ...")
    for update in tqdm(range(args.num_updates)):
        
        # train
        encoder.train()
        decoder.train()
        dynamics_model.train()

        y, u, _, _ = train_buffer.sample(
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

        # initial belief over x0: N(0, I)
        mean = torch.zeros((args.batch_size, args.x_dim), device=device)
        cov = torch.eye(args.x_dim, device=device).repeat([args.batch_size, 1, 1])

        y_filter_loss = 0.0
        consistency_loss = 0.0

        for t in range(1, args.chunk_length):
            mean, cov = dynamics_model.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t-1],
            )
            prior = mean if args.consistency_mode == "mean" else MultivariateNormal(loc=mean, covariance_matrix=cov)
            mean, cov = dynamics_model.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t],
            )
            posterior = mean if args.consistency_mode == "mean" else MultivariateNormal(loc=mean, covariance_matrix=cov)
            if args.consistency_mode == "mean":
                consistency = (prior - posterior).norm(dim=1, p=2) / (prior.norm(dim=1, p=2) + 1e-6)
            else:
                consistency = kl_divergence(posterior, prior)

            consistency_loss += consistency.mean()
            filter_a = dynamics_model.get_a(mean)
            y_filter_loss += nn.MSELoss()(decoder(filter_a), y[t])

        # y filter loss
        y_filter_loss /= (args.chunk_length - 1)

        # autoencoder loss
        a_flatten = einops.rearrange(a, "l b a -> (l b) a")
        y_flatten = einops.rearrange(y, "l b y -> (l b) y")
        y_recon = decoder(a_flatten)
        ae_loss = nn.MSELoss()(y_recon, y_flatten)

        # consistency loss
        consistency_loss /= (args.chunk_length - 1)

        total_loss = (
            y_filter_loss +
            args.ae_weight * ae_loss +
            args.consistency_weight * consistency_loss
        )

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()

        wandb.log({
            "train/y filter loss": y_filter_loss.item(),
            "train/ae loss": ae_loss.item(),
            "train/total loss": total_loss.item(),
            "train/consistency loss": consistency_loss.item(),
            "global_step": update,
        })
            
        if update % args.test_interval == 0:
            # test
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                dynamics_model.eval()

                y, u, _, _ = test_buffer.sample(
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

                # initial belief over x0: N(0, I)
                mean = torch.zeros((args.batch_size, args.x_dim), device=device)
                cov = torch.eye(args.x_dim, device=device).repeat([args.batch_size, 1, 1])

                y_filter_loss = 0.0
                consistency_loss = 0.0

                for t in range(1, args.chunk_length):
                    mean, cov = dynamics_model.dynamics_update(
                        mean=mean,
                        cov=cov,
                        u=u[t-1],
                    )
                    prior = mean if args.consistency_mode == "mean" else MultivariateNormal(loc=mean, covariance_matrix=cov)
                    mean, cov = dynamics_model.measurement_update(
                        mean=mean,
                        cov=cov,
                        a=a[t],
                    )
                    posterior = mean if args.consistency_mode == "mean" else MultivariateNormal(loc=mean, covariance_matrix=cov)
                    if args.consistency_mode == "mean":
                        consistency = (prior - posterior).norm(dim=1, p=2) / (prior.norm(dim=1, p=2) + 1e-6)
                    else:
                        consistency = kl_divergence(posterior, prior)

                    consistency_loss += consistency.mean()
                    filter_a = dynamics_model.get_a(mean)
                    y_filter_loss += nn.MSELoss()(decoder(filter_a), y[t])

                # y filter loss
                y_filter_loss /= (args.chunk_length - 1)

                # autoencoder loss
                a_flatten = einops.rearrange(a, "l b a -> (l b) a")
                y_flatten = einops.rearrange(y, "l b y -> (l b) y")
                y_recon = decoder(a_flatten)
                ae_loss = nn.MSELoss()(y_recon, y_flatten)

                # consistency loss
                consistency_loss /= (args.chunk_length - 1)

                total_loss = (
                    y_filter_loss +
                    args.ae_weight * ae_loss +
                    args.consistency_weight * consistency_loss
                )

                wandb.log({
                    "test/y filter loss": y_filter_loss.item(),
                    "test/ae loss": ae_loss.item(),
                    "test/total loss": total_loss.item(),
                    "test/consistency loss": consistency_loss.item(),
                    "global_step": update,
                })

    save_dir = Path(args.log_dir) / args.run_id
    torch.save(encoder.state_dict(), save_dir / "encoder.pth")
    torch.save(decoder.state_dict(), save_dir / "decoder.pth")
    torch.save(dynamics_model.state_dict(), save_dir / "dynamics_model.pth")

    return encoder, decoder, dynamics_model


def train_cost(
    args: Namespace,
    encoder: Encoder,
    decoder: Decoder,
    dynamics_model: Dynamics,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    cost_model = CostModel(
        x_dim=args.x_dim,
        u_dim=train_buffer.u_dim,
        device=device
    ).to(device)

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False

    for p in decoder.parameters():
        p.requires_grad = False

    for p in dynamics_model.parameters():
        p.requires_grad = False

    encoder.eval()
    decoder.eval()
    dynamics_model.eval()

    wandb.watch([cost_model], log="all", log_freq=10)

    all_params = list(cost_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.cost_lr, eps=args.eps, weight_decay=args.weight_decay)

    # train and test loop
    print(f"training on {device} ...")
    for update in tqdm(range(args.num_cost_updates)):    
        # train
        cost_model.train()

        y, u, c, _ = train_buffer.sample(
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

        cost_loss = 0.0

        for t in range(1, args.chunk_length):
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

        cost_loss /= (args.chunk_length - 1)

        optimizer.zero_grad()
        cost_loss.backward()

        clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()

        wandb.log({
            "train/cost loss": cost_loss.item(),
            "global_step": update,
        })
            
        if update % args.test_interval == 0:
            # test
            with torch.no_grad():
                cost_model.eval()

                y, u, c, _ = test_buffer.sample(
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

                cost_loss = 0.0

                for t in range(1, args.chunk_length):
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

                cost_loss /= (args.chunk_length - 1)

                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })

    save_dir = Path(args.log_dir) / args.run_id
    torch.save(cost_model.state_dict(), save_dir / "cost_model.pth")

    return cost_model