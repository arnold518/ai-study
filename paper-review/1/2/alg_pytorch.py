# alg_pytorch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
#      RecNet_PyTorch
# =========================
class RecNet_PyTorch(nn.Module):
    """
    Thin wrapper around nn.RNN (tanh, 1 layer) that matches our naming:

      forward(X_in) returns:
        Y: (T, n)   with Y[t] = h_t  (align with paper y(t+1))

    We use bias=True; initialization uses PyTorch defaults unless set in the trainer.
    """
    def __init__(self, m: int, n: int, device: str = "cpu"):
        super().__init__()
        self.m, self.n = m, n
        self.rnn = nn.RNN(
            input_size=m,
            hidden_size=n,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=False,
            bias=True,
        )
        self.device = torch.device(device)
        self.to(self.device)

    @torch.no_grad()
    def zero_state(self) -> torch.Tensor:
        return torch.zeros(self.n, device=self.device)

    def forward(self, X_in: np.ndarray | torch.Tensor) -> Tuple[torch.Tensor, None]:
        if isinstance(X_in, np.ndarray):
            X = torch.tensor(X_in, dtype=torch.float32, device=self.device)
        else:
            X = X_in.to(self.device)
        X = X.unsqueeze(1)                 # (T, 1, m)
        Y_seq, _ = self.rnn(X)            # (T, 1, n)
        Y = Y_seq.squeeze(1)              # (T, n)
        return Y, None


# =========================
#      Trainer_PyTorch
# =========================
class Trainer_PyTorch:
    """
    Full-sequence BPTT through PyTorch autograd (single update per epoch).
    Public API matches NumPy trainers.
    """
    def __init__(
        self,
        net: RecNet_PyTorch,
        lr: float,
        weight_decay: float,
        visible_idx: List[int],
        data_fn: Callable[[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        scorer: Optional[Callable[..., Dict[str, float]]] = None,
        scorer_kwargs: Optional[Dict] = None,
    ):
        self.net = net
        self.visible_idx = visible_idx
        self.data_fn = data_fn
        self.scorer = scorer
        self.scorer_kwargs = scorer_kwargs or {}

        self.opt = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss(reduction="sum")

    def train_epoch(self, T: int, tau: int) -> float:
        """
        Generate one sequence, forward it through PyTorch RNN,
        compute masked MSE on visible units, single optimizer step.
        """
        X, D, M = self.data_fn(T, tau, self.net.m, len(self.visible_idx))

        X_t = torch.tensor(X, dtype=torch.float32, device=self.net.device)
        D_t = torch.tensor(D, dtype=torch.float32, device=self.net.device)  # (T, n_vis)
        M_t = torch.tensor(M, dtype=torch.float32, device=self.net.device)  # (T, n_vis)

        Y, _ = self.net.forward(X_t)                                        # (T, n)
        Y_vis = torch.stack([Y[:, k] for k in self.visible_idx], dim=1)     # (T, n_vis)

        masked_outputs = Y_vis * M_t
        masked_targets = D_t  * M_t
        loss = self.criterion(masked_outputs, masked_targets)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return float(loss.detach().cpu()) / T

    @torch.no_grad()
    def evaluate(self, num_sequences: int, T: int, tau: int, **scorer_overrides) -> Dict[str, float]:
        if self.scorer is None:
            raise ValueError("No scorer was provided.")
        Ys: List[np.ndarray] = []
        Ds: List[np.ndarray] = []
        Ms: List[np.ndarray] = []

        n_vis = len(self.visible_idx)
        for _ in range(num_sequences):
            X, D, M = self.data_fn(T, tau, self.net.m, n_vis)
            X_t = torch.tensor(X, dtype=torch.float32, device=self.net.device)
            Y, _ = self.net.forward(X_t)                                   # (T, n)
            Y_vis = Y[:, self.visible_idx[0]].unsqueeze(1) if n_vis == 1 else Y[:, self.visible_idx]
            Ys.append(Y_vis.detach().cpu().numpy()); Ds.append(D); Ms.append(M)

        Y_all = np.concatenate(Ys, axis=0)
        D_all = np.concatenate(Ds, axis=0)
        M_all = np.concatenate(Ms, axis=0)
        return self.scorer(Y_all, D_all, M_all, **{**self.scorer_kwargs, **scorer_overrides})
