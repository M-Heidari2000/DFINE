import torch
import einops
from typing import Optional
from tqdm import tqdm
from argparse import Namespace
from .models import (
    Encoder,
    Dynamics,
    Decoder,
    ZDecoder,
)


def test_prediction(
    args: Namespace,
    encoder: Encoder,
    decoder: Decoder,
    dynamics_model: Dynamics,
    y: torch.Tensor,
    u: torch.Tensor,
):
    with torch.no_grad():

        encoder.eval()
        decoder.eval()
        dynamics_model.eval()
        
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

        encoder.eval()
        dynamics_model.eval()
        
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


def test_k_step_prediction(
    args: Namespace,
    encoder: Encoder,
    decoder: Decoder,
    z_decoder: ZDecoder,
    dynamics_model: Dynamics,
    z: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    prediction_k: Optional[int]=None,
):
    if prediction_k is None:
        prediction_k = args.prediction_k

    with torch.no_grad():

        encoder.eval()
        decoder.eval()
        dynamics_model.eval()
        z_decoder.eval()

        L, B, y_dim = y.shape
        _, _, z_dim = z.shape

        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=B)

        y_pred = torch.zeros(
            (L - prediction_k - 1, B, y_dim),
            device=y.device,
        )

        z_pred = torch.zeros(
            (L - prediction_k - 1, B, z_dim),
            device=y.device,
        )

        # initial belief over x0: N(0, I)
        mean = torch.zeros((B, args.x_dim), device=y.device)
        cov = torch.eye(args.x_dim, device=y.device).repeat([B, 1, 1])

        for t in tqdm(range(1, L - prediction_k)):
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
                
            pred_mean = mean
            pred_cov = cov

            for k in range(prediction_k):
                pred_mean, pred_cov = dynamics_model.dynamics_update(
                    mean=pred_mean,
                    cov=pred_cov,
                    u=u[t+k]
                )
            pred_a = dynamics_model.get_a(pred_mean)
            y_pred[t-1] = decoder(pred_a)
            z_pred[t-1] = z_decoder(pred_mean)

        return y_pred, z_pred