# core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import numpy as np

@dataclass
class Activation:
    fn: Callable[[np.ndarray], np.ndarray]
    deriv: Callable[[np.ndarray], np.ndarray]

@dataclass
class RecNetParams:
    W: np.ndarray  # (n, m+n)
    m: int
    n: int
    def check(self) -> None:
        assert self.W.shape == (self.n, self.m + self.n), \
            f"W must be (n, m+n)=({self.n}, {self.m + self.n}), got {self.W.shape}"

@dataclass
class RecNetCache:
    s_list: List[np.ndarray]  # s(t+1)
    y_list: List[np.ndarray]  # y(t+1)
    x_list: List[np.ndarray]  # x(t)
    y_init: np.ndarray        # y(0)
