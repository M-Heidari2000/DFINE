import torch
import torch.nn as nn
from typing import Optional


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
        # TODO: use torch.einsum for efficieny
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
        hidden_dim: Optional[int]=128,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self._min_var = min_var

        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.A_net = nn.Sequential(
            nn.Linear(hidden_dim, x_dim * x_dim),
            nn.Tanh(),
        )
        self.B_net = nn.Sequential(
            nn.Linear(hidden_dim, x_dim * u_dim),
            nn.Tanh(),
        )
        self.C_net = nn.Sequential(
            nn.Linear(hidden_dim, a_dim * x_dim),
            nn.Tanh(),
        )
        self.nx_net = nn.Linear(hidden_dim, x_dim)
        self.na_net = nn.Linear(hidden_dim, a_dim)

    def make_psd(self, P, eps=1e-6):
        b = P.shape[0]
        P = 0.5 * (P + P.transpose(-1, -2))
        P = P + eps * torch.eye(P.size(-1), device=P.device).expand([b, -1, -1])
        return P

    def get_dynamics(self, x):
        """
            get dynamics matrices depending on the state x
        """
        b = x.shape[0]
        hidden = self.backbone(x)
        A = self.A_net(hidden).reshape(b, self.x_dim, self.x_dim)
        B = self.B_net(hidden).reshape(b, self.x_dim, self.u_dim)
        C = self.C_net(hidden).reshape(b, self.a_dim, self.x_dim)
        Nx = torch.diag_embed(nn.functional.softplus(self.nx_net(hidden)) + self._min_var)
        Na = torch.diag_embed(nn.functional.softplus(self.na_net(hidden)) + self._min_var)
        
        return A, B, C, Nx, Na
    
    def get_a(self, x):
        """
        returns emissions (a) based on the input state (x)
        """

        A, B, C, Nx, Na = self.get_dynamics(x=x)
        return torch.einsum('bij,bj->bi', C, x)

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

        A, B, C, Nx, Na = self.get_dynamics(x=mean)

        next_mean = torch.einsum('bij,bj->bi', A, mean) + torch.einsum('bij,bj->bi', B, u)
        next_cov = torch.einsum('bij,bjk,bkl->bil', A, cov, A.transpose(1, 2)) + Nx
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

        A, B, C, Nx, Na = self.get_dynamics(x=mean)

        S = torch.einsum('bij,bjk,bkl->bil', C, cov, C.transpose(1, 2)) + Na
        G = torch.einsum('bij,bjk,bkl->bil', cov, C.transpose(1, 2), torch.linalg.pinv(S))
        innovation = a - torch.einsum('bij,bj->bi', C, mean)
        next_mean = mean + torch.einsum('bij,bj->bi', G, innovation)
        next_cov = cov - torch.einsum('bij,bjk,bkl->bil', G, C, cov)
        next_cov = self.make_psd(next_cov)

        return next_mean, next_cov