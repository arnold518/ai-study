# alg_hybrid.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np

from core import Activation, RecNetParams, RecNetCache


def _outer_id_x(x: np.ndarray, n: int) -> np.ndarray:
    """
    Build tensor T of shape (n, n, m+n): T[i,j,k] = δ_{i,j} * x[k]
    Efficient: eye(n)[:, :, None] * x[None, None, :].
    """
    return np.eye(n, dtype=x.dtype)[:, :, None] * x[None, None, :]


# ===================== Model =====================

class RecNet_HYBRID:
    """
    Dynamics (B=1):
      x(t)   = [x_in(t), y(t)] ∈ R^{m+n}
      s(t+1) = W x(t)           ∈ R^{n}
      y(t+1) = f(s(t+1))        ∈ R^{n}

    Provides:
      - zero_state(), step(), forward()
      - hybrid_bucket_bptt_grad(...)           → masked loss + bucket BPTT grad
      - hybrid_update_boundary_over_bucket(...)→ updates internal P_boundary using A,B summaries

    Notes:
      - The boundary eligibility tensor P_boundary (∂y/∂W at bucket boundaries)
        is fully owned by this model and persists across buckets within a sequence.
    """

    def __init__(self, params: RecNetParams, activation: Activation):
        params.check()
        self.p = params
        self.f = activation
        # P_boundary[i,j,k] = ∂y_i(boundary)/∂W_{j,k}, shape (n, n, m+n)
        self.P_boundary = np.zeros((self.p.n, self.p.n, self.p.m + self.p.n), dtype=np.float32)

    # -------- boundary ownership (public) --------

    def reset_boundary(self) -> None:
        """Reset boundary eligibility at the start of a sequence."""
        self.P_boundary.fill(0.0)

    def get_boundary(self) -> np.ndarray:
        """Return a view/copy of the current boundary eligibility tensor."""
        return self.P_boundary

    # -------- core forward pieces --------

    def zero_state(self, dtype=np.float32) -> np.ndarray:
        return np.zeros((self.p.n,), dtype=dtype)

    def step(self, x_in_t: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert x_in_t.shape == (self.p.m,), f"x_in_t shape should be ({self.p.m},)"
        assert y_t.shape == (self.p.n,), f"y_t shape should be ({self.p.n},)"
        x_t = np.concatenate([x_in_t, y_t], axis=0)  # (m+n,)
        s_next = self.p.W @ x_t                       # (n,)
        y_next = self.f.fn(s_next)                    # (n,)
        return s_next, y_next, x_t

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, RecNetCache]:
        assert X.ndim == 2 and X.shape[1] == self.p.m
        T = X.shape[0]
        y_prev = self.zero_state(dtype=X.dtype)

        s_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        x_list: List[np.ndarray] = []
        Y = np.empty((T, self.p.n), dtype=X.dtype)

        for t in range(T):
            s_next, y_next, x_t = self.step(X[t], y_prev)
            s_list.append(s_next)   # s(t+1)
            y_list.append(y_next)   # y(t+1)
            x_list.append(x_t)      # x(t)
            Y[t] = y_next
            y_prev = y_next

        cache = RecNetCache(s_list=s_list, y_list=y_list, x_list=x_list, y_init=self.zero_state(dtype=X.dtype))
        return Y, cache

    # -------- gradient math (inside the model) --------

    @staticmethod
    def _masked_error(y_vec: np.ndarray, D_row: np.ndarray, M_row: np.ndarray,
                      n: int, visible_idx: List[int]) -> np.ndarray:
        """
        E ∈ R^n with nonzeros only on visible units: E[k] = D - y.
        """
        E = np.zeros((n,), dtype=y_vec.dtype)
        for j, k in enumerate(visible_idx):
            if int(M_row[j]) == 1:
                E[k] = D_row[j] - y_vec[k]
        return E

    def hybrid_bucket_bptt_grad(
        self,
        y_list: List[np.ndarray],         # y_list[k] = y(u_k+1)
        s_list: List[np.ndarray],         # s_list[k] = s(u_k+1)
        x_list: List[np.ndarray],         # x_list[k] = x(u_k)
        D_bucket: np.ndarray,             # (L, n_visible)
        M_bucket: np.ndarray,             # (L, n_visible)
        visible_idx: List[int],
        weight_decay: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Standard BPTT restricted to a single bucket (length L).
        Returns:
          - total masked loss over the bucket,
          - gradient dJ/dW accumulated inside the bucket.
        """
        n, m = self.p.n, self.p.m
        W = self.p.W
        W_hh_T = W[:, m:].T

        L = len(y_list)
        total_loss = 0.0
        gW = np.zeros_like(W)

        for k in range(L):  # supervise y_list[k] with D_bucket[k]
            E_k = self._masked_error(y_list[k], D_bucket[k], M_bucket[k], n, visible_idx)
            total_loss += 0.5 * float((E_k * E_k).sum())
            if not np.any(E_k):
                continue

            # seed at local index k
            delta = self.f.deriv(s_list[k]) * E_k
            gW += np.outer(delta, x_list[k])

            # backprop only inside the bucket
            for kk in range(k - 1, -1, -1):
                e_prev_units = W_hh_T @ delta
                delta = self.f.deriv(s_list[kk]) * e_prev_units
                gW += np.outer(delta, x_list[kk])

        if weight_decay > 0.0:
            gW -= weight_decay * W

        return total_loss, gW

    # ----- bucket boundary update (A,B computed internally, P_boundary updated here) -----

    def _compute_A_B_over_bucket(
        self,
        s_list: List[np.ndarray],     # s_list[k] = s(u_k+1)
        x_list: List[np.ndarray],     # x_list[k] = x(u_k)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bucket summaries A and B via forward recurrences:

          A_{t+1} = diag(f'(s_{t+1})) @ W_hh @ A_t,                  with A_0 = I
          B_{t+1} = diag(f'(s_{t+1})) @ (W_hh @ B_t + (δ ⊗ x_t)),    with B_0 = 0

        Shapes:
          - A ∈ R^{n×n}
          - B ∈ R^{n×n×(m+n)} where B[i,j,k] = ∂y_i(end)/∂W_{j,k}
        """
        n, m = self.p.n, self.p.m
        W = self.p.W
        W_hh = W[:, m:]  # (n, n)

        A = np.eye(n, dtype=np.float32)                         # A_0
        B = np.zeros((n, n, m + n), dtype=np.float32)           # B_0

        for s_next, x_t in zip(s_list, x_list):
            fp = self.f.deriv(s_next)                           # (n,)
            # A <- diag(fp) @ (W_hh @ A)
            A = W_hh @ A
            A = fp[:, None] * A

            # B <- diag(fp) @ (W_hh @ B + δ⊗x_t)
            rec = np.einsum('ij,jlk->ilk', W_hh, B, optimize=True)  # (n,n,m+n)
            eye_x = _outer_id_x(x_t, n)                              # (n,n,m+n)
            B = (rec + eye_x)
            B = fp[:, None, None] * B

        return A, B

    def hybrid_update_boundary_over_bucket(
        self,
        s_list: List[np.ndarray],
        x_list: List[np.ndarray],
    ) -> None:
        """
        Update the internal boundary eligibility P_boundary over this bucket:

            P_end = A @ P_start + B

        where (A,B) are computed internally using the bucket's (s_list, x_list).
        """
        A, B = self._compute_A_B_over_bucket(s_list, x_list)
        # P_end = A @ P_start + B
        self.P_boundary = np.einsum('il,ljk->ijk', A, self.P_boundary, optimize=True) + B


# ===================== Trainer =====================

class Trainer_HYBRID:
    """
    Paper-style HYBRID (bucketed) algorithm:

    - Partition time into buckets of length h (last may be shorter).
    - Inside each bucket:
        * Ask the model for the bucket BPTT gradient and loss.
        * Ask the model to update its **internal** boundary eligibility:
              model.hybrid_update_boundary_over_bucket(s_buf, x_buf)
    - Apply exactly ONE weight update per bucket using the bucket's gradient.

    Public API (unchanged):
      - train_epoch(T, tau)  → average loss per time step
      - evaluate(N, T, tau)  → scorer dict (acc or mse)
    """

    def __init__(
        self,
        net: RecNet_HYBRID,
        lr: float,
        weight_decay: float,
        visible_idx: List[int],
        data_fn: Callable[[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        h_bucket: Optional[int],                          # bucket length h (>=1) or None for whole sequence
        grad_clip: Optional[float] = None,
        scorer: Optional[Callable[..., Dict[str, float]]] = None,
        scorer_kwargs: Optional[Dict] = None,
    ):
        assert h_bucket is None or h_bucket >= 1
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.visible_idx = list(visible_idx)
        self.data_fn = data_fn
        self.h = h_bucket
        self.grad_clip = grad_clip
        self.scorer = scorer
        self.scorer_kwargs = scorer_kwargs or {}

    # ---------- training ----------

    def train_epoch(self, T: int, tau: int) -> float:
        """
        One sequence:
          - Roll forward, fill a bucket.
          - When bucket completes (or at sequence end):
              * Model: bucket BPTT grad (loss_b, gW_b)
              * Model: update its internal boundary eligibility over the bucket
              * Single SGD step with gW_b
        Returns average loss per time step.
        """
        X, D, M = self.data_fn(T, tau, self.net.p.m, len(self.visible_idx))

        # Reset model-owned boundary eligibility at sequence start.
        self.net.reset_boundary()

        # rolling buffers for the current bucket
        y_prev = self.net.zero_state()
        s_buf: List[np.ndarray] = []
        y_buf: List[np.ndarray] = []
        x_buf: List[np.ndarray] = []
        D_buf: List[np.ndarray] = []
        M_buf: List[np.ndarray] = []

        total_loss = 0.0

        def _flush_bucket():
            nonlocal total_loss, s_buf, y_buf, x_buf, D_buf, M_buf

            if not y_buf:
                return

            # 1) Gradient & loss INSIDE the bucket (delegated to model)
            D_b = np.stack(D_buf, axis=0)  # (L, n_vis)
            M_b = np.stack(M_buf, axis=0)  # (L, n_vis)
            loss_b, gW_b = self.net.hybrid_bucket_bptt_grad(
                y_list=y_buf, s_list=s_buf, x_list=x_buf,
                D_bucket=D_b, M_bucket=M_b,
                visible_idx=self.visible_idx,
                weight_decay=self.weight_decay,
            )
            total_loss += loss_b

            # 2) Update the model's boundary eligibility over this bucket
            self.net.hybrid_update_boundary_over_bucket(s_buf, x_buf)

            # 3) Clip + SGD step (trainer’s responsibility)
            if self.grad_clip is not None:
                gnorm = float(np.linalg.norm(gW_b))
                if gnorm > self.grad_clip:
                    gW_b *= (self.grad_clip / (gnorm + 1e-12))
            self.net.p.W += self.lr * gW_b

            # 4) reset buffers
            s_buf.clear(); y_buf.clear(); x_buf.clear(); D_buf.clear(); M_buf.clear()

        # Iterate the sequence and fill buckets
        for t in range(T):
            s_next, y_next, x_t = self.net.step(X[t], y_prev)

            s_buf.append(s_next)
            y_buf.append(y_next)
            x_buf.append(x_t)
            D_buf.append(D[t])
            M_buf.append(M[t])

            # bucket boundary?
            if self.h is not None and (len(y_buf) == self.h):
                _flush_bucket()

            y_prev = y_next

        # flush tail
        _flush_bucket()

        return total_loss / T

    # ---------- evaluation ----------

    def evaluate(self, num_sequences: int, T: int, tau: int, **scorer_overrides) -> Dict[str, float]:
        if self.scorer is None:
            raise ValueError("No scorer was provided.")
        Ys: List[np.ndarray] = []
        Ds: List[np.ndarray] = []
        Ms: List[np.ndarray] = []

        n_vis = len(self.visible_idx)
        for _ in range(num_sequences):
            X, D, M = self.data_fn(T, tau, self.net.p.m, n_vis)
            Y, _ = self.net.forward(X)  # (T, n)
            Y_vis = np.stack([Y[:, k] for k in self.visible_idx], axis=1)
            Ys.append(Y_vis); Ds.append(D); Ms.append(M)

        Y_all = np.concatenate(Ys, axis=0)
        D_all = np.concatenate(Ds, axis=0)
        M_all = np.concatenate(Ms, axis=0)
        return self.scorer(Y_all, D_all, M_all, **{**self.scorer_kwargs, **scorer_overrides})
