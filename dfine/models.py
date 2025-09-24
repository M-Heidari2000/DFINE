import torch
import torch.nn as nn
from typing import Optional
import torch.nn.init as init


class Encoder(nn.Module):
    """
        y_t -> a_t
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: float=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*y_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, a_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, y):
        return self.mlp_layers(y)
    

class Decoder(nn.Module):
    """
        a_t -> y_t
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: float=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*a_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, y_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, a):
        return self.mlp_layers(a)


class CostModel(nn.Module):
    """
        Learnable quadratic cost function in the latent space
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        device: str,
    ):
        
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        
        self.device = device
        self.A = nn.Parameter(
            torch.eye(x_dim, device=self.device, dtype=torch.float32),
        )
        self.B = nn.Parameter(
            torch.eye(u_dim, device=self.device, dtype=torch.float32)
        )
        self.q = nn.Parameter(
            torch.randn((x_dim, 1), device=self.device, dtype=torch.float32)
        )

    @property
    def Q(self):
        return self.A @ self.A.T
    
    @property
    def R(self):
        L = torch.tril(self.B)
        diagonals = nn.functional.softplus(L.diagonal()) + 1e-4
        X = 1 - torch.eye(self.u_dim, device=self.device, dtype=torch.float32)
        L = L * X + diagonals.diag()
        return L @ L.T
    
    def forward(self, x, u):
        # x: b x
        # u: b u
        cost = 0.5 * x @ self.Q @ x.T + 0.5 * u @ self.R @ u.T
        cost = cost.diagonal().unsqueeze(1) + x @ self.q
        return cost
        

class Dynamics(nn.Module):
    
    """
        KF that obtains belief over x_{t+1} using belief of x_t, u_t, and y_{t+1}
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        a_dim: int,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self._min_var = min_var

        # Dynamics matrices
        self.A = nn.Parameter(
            torch.eye(self.x_dim)
        )
        self.B = nn.Parameter(
            torch.randn(self.x_dim, self.u_dim),
        )
        self.C = nn.Parameter(
            torch.randn(self.a_dim, self.x_dim)
        )

        # Transition noise covariance (diagonal)
        self.nx = nn.Parameter(
            torch.randn(self.x_dim)
        )
        # Observation noise covariance (diagonal)
        self.na = nn.Parameter(
            torch.randn(self.a_dim)
        )

    @property
    def Na(self):
        Na = torch.diag(nn.functional.softplus(self.na) + self._min_var)    # shape: a a
        return Na
    
    @property
    def Nx(self):
        Nx = torch.diag(nn.functional.softplus(self.nx) + self._min_var)    # shape: x x
        return Nx

    def make_psd(self, P, eps=1e-6):
        b = P.shape[0]
        P = 0.5 * (P + P.transpose(-1, -2))
        P = P + eps * torch.eye(P.size(-1), device=P.device).expand([b, -1, -1])
        return P
    
    def get_a(self, x):
        """
        returns emissions (a) based on the input state (x)
        """

        return x @ self.C.T

    def dynamics_update(
        self,
        mean,
        cov,
        u,
    ):
        """
            Single step dynamics update

            mean: b x
            cov: b x x
            u: b u
        """

        next_mean = mean @ self.A.T + u @ self.B.T
        next_cov = self.A @ cov @ self.A.T + self.Nx
        next_cov = self.make_psd(next_cov)

        return next_mean, next_cov
    
    def measurement_update(
        self,
        mean,
        cov,
        a,
    ):
        """
            Single step measurement update
        
            mean: b x
            cov: b x x
            a: b a
        """


        K = cov @ self.C.T @ torch.linalg.pinv(self.C @ cov @ self.C.T + self.Na)
        next_mean = mean + ((a - mean @ self.C.T).unsqueeze(1) @ K.transpose(1, 2)).squeeze(1)
        next_cov = (torch.eye(self.x_dim, device=K.device) - K @ self.C) @ cov
        next_cov = self.make_psd(next_cov)

        return next_mean, next_cov