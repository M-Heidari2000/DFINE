import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from argparse import Namespace
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
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps)

    # train and test loop
    print("training ...")
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

        y_pred_loss = 0.0
        y_filter_loss = 0.0

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
            filter_a = dynamics_model.get_a(mean)
            y_filter_loss += nn.MSELoss()(decoder(filter_a), y[t])

            # tensors to hold predictions of future ys
            pred_y = torch.zeros((args.prediction_k, args.batch_size, train_buffer.y_dim), device=device)

            pred_mean = mean
            pred_cov = cov

            for k in range(args.prediction_k):
                pred_mean, pred_cov = dynamics_model.dynamics_update(
                    mean=pred_mean,
                    cov=pred_cov,
                    u=u[t+k]
                )
                pred_a = dynamics_model.get_a(pred_mean)
                pred_y[k] = decoder(pred_a)

            true_y = y[t+1: t+1+args.prediction_k]
            true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
            pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
            y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

        y_pred_loss /= (args.chunk_length - args.prediction_k - 1)

        # y filter loss
        y_filter_loss /= (args.chunk_length - 1)

        total_loss = y_pred_loss + y_filter_loss

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, args.clip_grad_norm)
        optimizer.step()

        wandb.log({
            "train/y prediction loss": y_pred_loss.item(),
            "train/y filter loss": y_filter_loss.item(),
            "train/total loss": total_loss.item(),
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

                y_pred_loss = 0.0
                y_filter_loss = 0.0

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
                    filter_a = dynamics_model.get_a(mean)
                    y_filter_loss += nn.MSELoss()(decoder(filter_a), y[t])

                    # tensors to hold predictions of future ys
                    pred_y = torch.zeros((args.prediction_k, args.batch_size, test_buffer.y_dim), device=device)

                    pred_mean = mean
                    pred_cov = cov

                    for k in range(args.prediction_k):
                        pred_mean, pred_cov = dynamics_model.dynamics_update(
                            mean=pred_mean,
                            cov=pred_cov,
                            u=u[t+k]
                        )
                        pred_a = dynamics_model.get_a(pred_mean)
                        pred_y[k] = decoder(pred_a)

                    true_y = y[t+1: t+1+args.prediction_k]
                    true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
                    pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
                    y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

                y_pred_loss /= (args.chunk_length - args.prediction_k - 1)

                # y filter loss
                y_filter_loss /= (args.chunk_length - 1)

                total_loss = y_pred_loss + y_filter_loss

                wandb.log({
                    "test/y prediction loss": y_pred_loss.item(),
                    "test/y filter loss": y_filter_loss.item(),
                    "test/total loss": total_loss.item(),
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
    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps)

    # train and test loop
    print("training ...")
    for update in tqdm(range(args.num_updates)):    
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