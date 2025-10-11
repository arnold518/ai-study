# alg_bptt.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict, Iterable
import numpy as np

from core import Activation, RecNetParams, RecNetCache


# ======================== Model ========================

class RecNet_BPTT:
    """
    Discrete-time recurrent network (B=1, paper indexing):

      x(t)   = [x_in(t), y(t)] ∈ R^{m+n}
      s(t+1) = W x(t)           ∈ R^{n}
      y(t+1) = f(s(t+1))        ∈ R^{n}

    Python index t stores:  x(t), s(t+1), y(t+1).
    Supervision at t uses D[t] for y(t+1).

    Public API (unchanged externally):
      - zero_state()
      - step(x_in_t, y_t) -> (s_next, y_next, x_t)
      - forward(X_in)     -> (Y, cache)
      - bptt_grad(cache, times, D, M, visible_idx, horizon, weight_decay)
          -> (loss_sum, gW)
        * This single method powers full BPTT, truncated BPTT, and real-time windows,
          depending solely on 'horizon' and which 'times' you pass in.
    """
    def __init__(self, params: RecNetParams, activation: Activation, truncation_h: Optional[int] = None):
        params.check()
        self.p = params
        self.f = activation
        self.truncation_h = truncation_h  # default horizon (None → full)

    # ----- utilities -----
    def zero_state(self, dtype=np.float32) -> np.ndarray:
        return np.zeros((self.p.n,), dtype=dtype)

    # ----- single step (front pass) -----
    def step(self, x_in_t: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One step:
          x_t = [x_in(t), y(t)]
          s(t+1) = W x_t
          y(t+1) = f(s(t+1))
        """
        assert x_in_t.shape == (self.p.m,), f"x_in_t must be ({self.p.m},)"
        assert y_t.shape == (self.p.n,),     f"y_t must be ({self.p.n},)"

        x_t = np.concatenate([x_in_t, y_t], axis=0)  # (m+n,)
        s_next = self.p.W @ x_t                       # (n,)
        y_next = self.f.fn(s_next)                    # (n,)
        return s_next, y_next, x_t

    # ----- full forward (used for any horizon/update policy) -----
    def forward(self, X_in: np.ndarray) -> Tuple[np.ndarray, RecNetCache]:
        """
        Full forward for T steps.
        Returns:
          Y: (T, n) with Y[t] = y(t+1)
          cache: RecNetCache with s_list[t]=s(t+1), y_list[t]=y(t+1), x_list[t]=x(t)
        """
        assert X_in.ndim == 2 and X_in.shape[1] == self.p.m
        T = X_in.shape[0]
        y_prev = self.zero_state(dtype=X_in.dtype)

        s_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        x_list: List[np.ndarray] = []
        Y = np.empty((T, self.p.n), dtype=X_in.dtype)

        for t in range(T):
            s_next, y_next, x_t = self.step(X_in[t], y_prev)
            s_list.append(s_next)   # s(t+1)
            y_list.append(y_next)   # y(t+1)
            x_list.append(x_t)      # x(t)
            Y[t] = y_next
            y_prev = y_next

        cache = RecNetCache(s_list=s_list, y_list=y_list, x_list=x_list, y_init=self.zero_state(dtype=X_in.dtype))
        return Y, cache

    # ====== unified gradient engine (single method) ======

    @staticmethod
    def _masked_error(y_vec: np.ndarray, D_row: np.ndarray, M_row: np.ndarray,
                      n: int, visible_idx: List[int]) -> np.ndarray:
        """
        Build E ∈ R^n with nonzeros only at visible indices: E[k] = D - y.
        """
        E = np.zeros((n,), dtype=y_vec.dtype)
        for j, k in enumerate(visible_idx):
            if int(M_row[j]) == 1:
                E[k] = D_row[j] - y_vec[k]
        return E

    def bptt_grad(
        self,
        cache: RecNetCache,            # forward history covering at least up to max(times)
        times: Iterable[int],          # collection of t indices to seed δ at y(t+1)
        D: np.ndarray,                 # (T, n_visible)
        M: np.ndarray,                 # (T, n_visible)
        visible_idx: List[int],        # indices into units (length n_visible)
        horizon: Optional[int] = None, # None → full; int ≥1 → truncated window
        weight_decay: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the **sum** of masked losses over 'times' and the corresponding
        gradient wrt W, backpropagating each seed up to 'horizon' steps.

        This single routine powers:
          - full epochwise BPTT:     times = range(T), horizon = None
          - truncated epochwise:     times = range(T), horizon = h
          - continual RT/TBPTT:      times = {t0..tK} on a prefix cache, with horizon = h

        Returns:
          loss_sum : scalar (sum over given times)
          gW       : (n, m+n) gradient accumulated over 'times'
        """
        s_hist, y_hist, x_hist = cache.s_list, cache.y_list, cache.x_list
        T = len(y_hist)
        n, m = self.p.n, self.p.m
        W = self.p.W

        # Normalize horizon
        if horizon is not None:
            assert isinstance(horizon, int) and horizon >= 1, "horizon must be None or integer ≥ 1"

        loss_sum = 0.0
        gW_total = np.zeros_like(W)

        for t in times:
            assert 0 <= t < T, f"time index {t} out of range 0..{T-1}"

            # masked error at t for y(t+1)
            E_t = self._masked_error(y_hist[t], D[t], M[t], n, visible_idx)
            loss_sum += 0.5 * float((E_t * E_t).sum())
            if not np.any(E_t):
                continue

            # seed δ(U=t+1)
            delta = self.f.deriv(s_hist[t]) * E_t

            # contribution at U = t+1
            gW_total += np.outer(delta, x_hist[t])

            # window start
            if horizon is None:
                u_min = 0
            else:
                u_min = max(0, t - horizon + 1)

            # walk back within horizon
            for u_idx in range(t - 1, u_min - 1, -1):
                e_prev_full = W.T @ delta          # (m+n,)
                e_prev_units = e_prev_full[m:]     # (n,)
                delta = self.f.deriv(s_hist[u_idx]) * e_prev_units
                gW_total += np.outer(delta, x_hist[u_idx])

        if weight_decay > 0.0:
            gW_total -= weight_decay * W

        return loss_sum, gW_total


# ======================== Trainer ========================

class Trainer_BPTT:
    """
    Unified BPTT(h, h′) trainer (B=1):

      - Horizon h (self.h): how far back each error can backprop.
        * h=None → ∞ (full BPTT)
        * h=1    → single-step window

      - Update period h′ (self.hprime): how often to apply updates.
        * h′=1     → per-step (continual)
        * h′=k     → every k steps
        * h′=None  → once at end (epochwise)

    The trainer:
      * Generates data
      * Computes gradients by calling **one** model method (bptt_grad)
      * Applies SGD steps according to h′
      * Keeps caches consistent by re-forwarding **only** when weights change
    """
    def __init__(
        self,
        net: RecNet_BPTT,
        lr: float,
        weight_decay: float,
        visible_idx: List[int],
        data_fn: Callable[[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        update_every_hprime: Optional[int] = None,  # h′; None → update at end
        grad_clip: Optional[float] = None,
        scorer: Optional[Callable[..., Dict[str, float]]] = None,
        scorer_kwargs: Optional[Dict] = None,
        truncation_h: Optional[int] = None,         # optional override for net.truncation_h
    ):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.visible_idx = list(visible_idx)
        self.data_fn = data_fn
        self.grad_clip = grad_clip
        self.scorer = scorer
        self.scorer_kwargs = scorer_kwargs or {}

        # policy knobs
        self.h = self.net.truncation_h if truncation_h is None else truncation_h    # horizon h
        self.hprime = update_every_hprime                                           # update period h′

        assert (self.h is None) or (isinstance(self.h, int) and self.h >= 1), "h must be None or >=1"
        assert (self.hprime is None) or (isinstance(self.hprime, int) and self.hprime >= 1), "h′ must be None or >=1"

    # ---------- small helper ----------

    def _apply_update(self, gW_accum: np.ndarray) -> None:
        """Clip and SGD step with the accumulated gradient (decay already applied in model)."""
        if not np.any(gW_accum):
            return
        g_total = gW_accum
        if self.grad_clip is not None:
            gnorm = float(np.linalg.norm(g_total))
            if gnorm > self.grad_clip:
                g_total *= (self.grad_clip / (gnorm + 1e-12))
        self.net.p.W += self.lr * g_total

    # ---------- training (single unified routine) ----------

    def train_epoch(self, T: int, tau: int) -> float:
        """
        One sequence with policy (h, h′) in effect.

        - If h′ is None: make **one** forward cache for whole sequence,
          call model.bptt_grad(times=range(T), horizon=h), single update.

        - If h′ is k (including 1): process the sequence in blocks of size k:
            * For each block [b_start .. b_end], recompute a **prefix** cache
              that covers up to b_end with current weights,
            * Ask model for a gradient for times=that block using the prefix cache,
            * Update once for the block, then continue.
        """
        X, D, M = self.data_fn(T, tau, self.net.p.m, len(self.visible_idx))

        # --- Case 1: epochwise (single update)
        if self.hprime is None:
            Y, cache = self.net.forward(X)
            loss_sum, gW = self.net.bptt_grad(
                cache=cache,
                times=range(T),
                D=D, M=M,
                visible_idx=self.visible_idx,
                horizon=self.h,
                weight_decay=self.weight_decay,
            )
            # one update
            self._apply_update(gW)
            return loss_sum / T

        # --- Case 2: periodic/continual updates (block size = h′)
        total_loss = 0.0
        k = self.hprime
        # Process blocks [0..k-1], [k..2k-1], ..., last partial block if needed
        for b_start in range(0, T, k):
            b_end = min(T - 1, b_start + k - 1)

            # recompute prefix up to b_end under **current** weights (consistency)
            X_prefix = X[: b_end + 1]
            _, cache_prefix = self.net.forward(X_prefix)

            # ask model for grad over this block, with chosen horizon
            times_block = range(b_start, b_end + 1)
            loss_blk, gW_blk = self.net.bptt_grad(
                cache=cache_prefix,
                times=times_block,
                D=D, M=M,
                visible_idx=self.visible_idx,
                horizon=self.h,
                weight_decay=self.weight_decay,
            )
            total_loss += loss_blk

            # single update for the block
            if self.grad_clip is not None:
                gnorm = float(np.linalg.norm(gW_blk))
                if gnorm > self.grad_clip:
                    gW_blk *= (self.grad_clip / (gnorm + 1e-12))
            self.net.p.W += self.lr * gW_blk

        return total_loss / T

    # ---------- evaluation (uniform) ----------

    def evaluate(self, num_sequences: int, T: int, tau: int, **scorer_overrides) -> Dict[str, float]:
        """
        Forward-only evaluation. Collect visible outputs Y_vis[t] = y(t+1)[visible_idx]
        then compute scorer(Y, D, M, ...).
        """
        if self.scorer is None:
            raise ValueError("No scorer was provided.")
        Ys: List[np.ndarray] = []
        Ds: List[np.ndarray] = []
        Ms: List[np.ndarray] = []

        n_vis = len(self.visible_idx)
        for _ in range(num_sequences):
            X, D, M = self.data_fn(T, tau, self.net.p.m, n_vis)
            Y, _ = self.net.forward(X)  # (T, n)
            Y_vis = np.stack([Y[:, k] for k in self.visible_idx], axis=1)  # (T, n_vis)
            Ys.append(Y_vis); Ds.append(D); Ms.append(M)

        Y_all = np.concatenate(Ys, axis=0)
        D_all = np.concatenate(Ds, axis=0)
        M_all = np.concatenate(Ms, axis=0)
        return self.scorer(Y_all, D_all, M_all, **{**self.scorer_kwargs, **scorer_overrides})
