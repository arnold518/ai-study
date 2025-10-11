# alg_rtrl.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict
import numpy as np

from core import Activation, RecNetParams, RecNetCache


# =========================
# RecNet_RTRL (model)
# =========================
class RecNet_RTRL:
    """
    Discrete-time recurrent network (B=1, paper indexing):

      x(t)   = [x_in(t), y(t)] ∈ R^{m+n}
      s(t+1) = W x(t)           ∈ R^{n}
      y(t+1) = f(s(t+1))        ∈ R^{n}

    Provides:
      - zero_state()
      - step(x_in_t, y_t)  -> (s_next, y_next, x_t)
      - forward(X_in)      -> (Y, cache) with Y[t] = y(t+1)
      - rtrl_instant_grad(cache, t, P_prev, D_t, M_t, visible_idx, weight_decay=0.0, grad_clip=None)
          -> (loss_t, gW, P_next)

    Notes:
      - No bias line is injected here; if you want a bias, concatenate a 1.0
        as part of x_in(t) in the data generator (and increase m accordingly).
      - The RTRL sensitivity tensor P is maintained **internally** and reset
        automatically at the beginning of a sequence (t == 0).
    """
    def __init__(self, params: RecNetParams, activation: Activation):
        params.check()
        self.p = params
        self.f = activation
        self._P: Optional[np.ndarray] = None  # (n, n, m+n), initialized lazily

    # ----- utilities -----
    def zero_state(self, dtype=np.float32) -> np.ndarray:
        return np.zeros((self.p.n,), dtype=dtype)

    def rtrl_reset(self) -> None:
        """Optional external reset; usually not needed (we reset on t==0)."""
        self._P = None

    # ----- single step (front pass) -----
    def step(self, x_in_t: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert x_in_t.shape == (self.p.m,), f"x_in_t must be ({self.p.m},)"
        assert y_t.shape == (self.p.n,),     f"y_t must be ({self.p.n},)"
        x_t = np.concatenate([x_in_t, y_t], axis=0)   # (m+n,)
        s_next = self.p.W @ x_t                        # (n,)
        y_next = self.f.fn(s_next)                     # (n,)
        return s_next, y_next, x_t

    # ----- unrolled forward over time (B=1) -----
    def forward(self, X_in: np.ndarray) -> Tuple[np.ndarray, RecNetCache]:
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

        cache = RecNetCache(
            s_list=s_list, y_list=y_list, x_list=x_list,
            y_init=self.zero_state(dtype=X_in.dtype)
        )
        return Y, cache

    # ----- online RTRL instantaneous gradient (PUBLIC API UNCHANGED) -----
    def rtrl_instant_grad(
        self,
        cache: RecNetCache,
        t: int,
        P_prev: Optional[np.ndarray],     # kept for API compatibility; ignored internally
        D_t: np.ndarray,                  # (|visible|,) targets at time t (supervise y(t+1))
        M_t: np.ndarray,                  # (|visible|,) 0/1 mask at time t
        visible_idx: List[int],           # indices of supervised units
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute instantaneous masked loss at index t and its RTRL gradient.

        Paper-time alignment:
          cache.s_list[t] = s(t+1), cache.y_list[t] = y(t+1), cache.x_list[t] = x(t)
          Supervise y(t+1) with D_t and M_t given at Python index t.

        Returns:
          loss_t : scalar  (0.5 * ||masked residual||^2)
          gW     : (n, m+n) gradient wrt W at this step
          P_next : (n, n, m+n) updated sensitivity (also stored internally)
        """
        n, m = self.p.n, self.p.m
        W = self.p.W
        W_hh = W[:, m:]                          # recurrent block

        # Initialize/reset P internally on first step of a sequence
        if self._P is None or t == 0:
            self._P = np.zeros((n, n, m + n), dtype=W.dtype)

        # Pull current step intermediates
        s_t1 = cache.s_list[t]                   # s(t+1)
        y_t1 = cache.y_list[t]                   # y(t+1)
        x_t  = cache.x_list[t]                   # x(t)

        # Local derivatives
        fp = self.f.deriv(s_t1)                  # f'(s(t+1)) ∈ R^n
        I = np.eye(n, dtype=W.dtype)

        # RTRL recursion:
        #   P(t+1) = diag(f'(s(t+1))) @ [ I⊗x(t) + W_hh @ P(t) ]
        base  = I[:, :, None] * x_t[None, None, :]                 # (n,n,m+n)
        recur = np.einsum('ij,jkl->ikl', W_hh, self._P, optimize=True)  # (n,n,m+n)
        P_next = fp[:, None, None] * (base + recur)                # (n,n,m+n)

        # Masked error on supervised units at time t (for y(t+1))
        E = np.zeros((n,), dtype=W.dtype)
        for j, k in enumerate(visible_idx):
            if int(M_t[j]) == 1:
                E[k] = D_t[j] - y_t1[k]

        # Instantaneous loss
        loss_t = 0.5 * float((E**2).sum())

        # Gradient: dJ/dW = sum_i E[i] * P_next[i, :, :]
        gW = np.tensordot(E, P_next, axes=(0, 0))                 # (n, m+n)

        # L2 regularization (weight decay)
        if weight_decay > 0.0:
            gW -= weight_decay * W

        # Optional gradient clipping
        if grad_clip is not None:
            gnorm = float(np.linalg.norm(gW))
            if gnorm > grad_clip:
                gW *= (grad_clip / (gnorm + 1e-12))

        # Commit updated sensitivity internally and return it (API compatibility)
        self._P = P_next
        return loss_t, gW, self._P


# =========================
# Trainer_RTRL (trainer)
# =========================
class Trainer_RTRL:
    """
    RTRL trainer with periodic/epochwise updates:

      - On each time step:
          * Run forward one step (model.step) and cache (x,s,y)
          * Ask model for instantaneous RTRL gradient via model.rtrl_instant_grad(...)
          * Accumulate gradients
      - Every h′ steps (or at the end), apply a single update:
            W ← W + lr * g_accum
        then clear g_accum.

    If update_every_hprime is None → update only at the sequence end
    (epochwise RTRL). If update_every_hprime == 1 → classical online RTRL.

    Public API:
      - train_epoch(T, tau)              → average masked loss per step
      - evaluate(num_sequences, T, tau)  → scorer dict (uses provided scorer)
    """
    def __init__(
        self,
        net: RecNet_RTRL,
        lr: float,
        weight_decay: float,
        visible_idx: List[int],
        data_fn: Callable[[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        update_every_hprime: Optional[int] = 1,   # h′
        grad_clip: Optional[float] = None,
        scorer: Optional[Callable[..., Dict[str, float]]] = None,
        scorer_kwargs: Optional[Dict] = None,
    ):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.visible_idx = visible_idx
        self.data_fn = data_fn
        self.hprime = update_every_hprime  # None → update at end
        self.grad_clip = grad_clip
        self.scorer = scorer
        self.scorer_kwargs = scorer_kwargs or {}

    # ----- training -----
    def train_epoch(self, T: int, tau: int) -> float:
        """
        One sequence:
          - forward step + model.rtrl_instant_grad() each time step
          - weight update every h′ steps (or end), with optional clip/decay
        Returns average loss per time step (masked MSE / 2).
        """
        X, D, M = self.data_fn(T, tau, self.net.p.m, len(self.visible_idx))

        # reset state & fresh cache for this sequence
        y_prev = self.net.zero_state()
        cache = RecNetCache(s_list=[], y_list=[], x_list=[], y_init=y_prev.copy())

        total_loss = 0.0
        g_accum = np.zeros_like(self.net.p.W)
        P_prev = None  # kept for API compatibility; the model ignores it

        for t in range(T):
            # forward step & cache
            s_next, y_next, x_t = self.net.step(X[t], y_prev)
            cache.s_list.append(s_next)
            cache.y_list.append(y_next)
            cache.x_list.append(x_t)

            # instantaneous RTRL gradient from the model
            loss_t, gW_t, P_prev = self.net.rtrl_instant_grad(
                cache=cache,
                t=t,
                P_prev=P_prev,  # ignored internally; preserved for API compatibility
                D_t=D[t].reshape(-1),
                M_t=M[t].reshape(-1),
                visible_idx=self.visible_idx,
                weight_decay=self.weight_decay,
                grad_clip=self.grad_clip,
            )
            total_loss += loss_t
            g_accum += gW_t

            # periodic (epochwise) update?
            do_update = (self.hprime is not None and (t + 1) % self.hprime == 0) or (t == T - 1)
            if do_update:
                # single update
                self.net.p.W += self.lr * g_accum
                g_accum.fill(0.0)

            y_prev = y_next

        return total_loss / T

    # ----- evaluation -----
    def evaluate(self, num_sequences: int, T: int, tau: int, **scorer_overrides) -> Dict[str, float]:
        """
        Algorithm-agnostic evaluation: forward only, then call the provided scorer.
        The scorer receives concatenated (Y, D, M) across sequences.
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
            # Select visible columns in a consistent order
            Y_vis = np.stack([Y[:, k] for k in self.visible_idx], axis=1)  # (T, n_visible)
            Ys.append(Y_vis); Ds.append(D); Ms.append(M)

        Y_all = np.concatenate(Ys, axis=0)
        D_all = np.concatenate(Ds, axis=0)
        M_all = np.concatenate(Ms, axis=0)
        return self.scorer(Y_all, D_all, M_all, **{**self.scorer_kwargs, **scorer_overrides})
