# main.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch

from core import Activation, RecNetParams
from helper import tanh_activation, init_weights, xor_delayed
from helper import masked_binary_accuracy, masked_mse

from alg_bptt import RecNet_BPTT, Trainer_BPTT
from alg_rtrl import RecNet_RTRL, Trainer_RTRL
from alg_pytorch import RecNet_PyTorch, Trainer_PyTorch
from alg_hybrid import RecNet_HYBRID, Trainer_HYBRID


# -------- builders (minimal, consistent with earlier code) --------

def _build_bptt(
    *,
    m: int, n: int, lr: float, weight_decay: float,
    truncation_h: int | None, update_every_hprime: int | None,
    grad_clip: float | None, scorer, scorer_kwargs: dict, seed: int,
):
    W0 = init_weights(m, n, seed=seed)
    fn, dfn = tanh_activation()
    act = Activation(fn=fn, deriv=dfn)
    params = RecNetParams(W=W0, m=m, n=n)
    net = RecNet_BPTT(params, act, truncation_h=truncation_h)
    trainer = Trainer_BPTT(
        net=net,
        lr=lr,
        weight_decay=weight_decay,
        visible_idx=[0],
        data_fn=xor_delayed,
        update_every_hprime=update_every_hprime,
        grad_clip=grad_clip,
        scorer=scorer,
        scorer_kwargs=scorer_kwargs,
    )
    return net, trainer


def _build_rtrl(
    *,
    m: int, n: int, lr: float, weight_decay: float,
    rtrl_hprime: int | None,
    grad_clip: float | None, scorer, scorer_kwargs: dict, seed: int,
):
    W0 = init_weights(m, n, seed=seed)
    fn, dfn = tanh_activation()
    act = Activation(fn=fn, deriv=dfn)
    params = RecNetParams(W=W0, m=m, n=n)
    net = RecNet_RTRL(params, act)
    trainer = Trainer_RTRL(
        net=net,
        lr=lr,
        weight_decay=weight_decay,
        visible_idx=[0],
        data_fn=xor_delayed,
        update_every_hprime=rtrl_hprime,
        grad_clip=grad_clip,
        scorer=scorer,
        scorer_kwargs=scorer_kwargs,
    )
    return net, trainer


def _build_pytorch(
    *, m: int, n: int, lr: float, weight_decay: float, scorer, scorer_kwargs: dict,
):
    net = RecNet_PyTorch(m=m, n=n, device="cpu")
    trainer = Trainer_PyTorch(
        net=net,
        lr=lr,
        weight_decay=weight_decay,
        visible_idx=[0],
        data_fn=xor_delayed,
        scorer=scorer,
        scorer_kwargs=scorer_kwargs,
    )
    return net, trainer


def _build_hybrid(
    *, m: int, n: int, lr: float, weight_decay: float,
    h_bucket: int | None,
    grad_clip: float | None, scorer, scorer_kwargs: dict, seed: int,
):
    W0 = init_weights(m, n, seed=seed)
    fn, dfn = tanh_activation()
    act = Activation(fn=fn, deriv=dfn)
    params = RecNetParams(W=W0, m=m, n=n)
    net = RecNet_HYBRID(params, act)
    trainer = Trainer_HYBRID(
        net=net,
        lr=lr,
        weight_decay=weight_decay,
        visible_idx=[0],
        data_fn=xor_delayed,
        h_bucket=h_bucket,
        grad_clip=grad_clip,
        scorer=scorer,
        scorer_kwargs=scorer_kwargs,
    )
    return net, trainer


# -------- runner (kept same shape, only mapping names) --------

def run_experiment(
    alg_name: str,
    *,
    m: int = 2,
    n: int = 8,
    lr: float = 5e-3,
    tau: int = 4,
    Ttrain: int = 20,
    Teval: int = 50,
    eval_seqs: int = 200,
    quick_eval_seqs: int = 50,
    epochs: int = 10_000,
    log_every: int = 500,
    weight_decay: float = 0.0,
    # TBPTT knobs (used for tbptt explicitly)
    truncation_h: int | None = None,         # h
    update_every_hprime: int | None = None,  # h′
    # RTRL knob (only used for rtrl if you override)
    rtrl_hprime: int | None = None,          # h′
    # HYBRID knob
    h_bucket: int | None = 10,
    grad_clip: float | None = None,
    seed: int = 0,
    metric_name: str = "acc",
    metric_threshold: float = 0.5,
    plot: bool = True,
):
    # scorer
    if metric_name == "acc":
        scorer = masked_binary_accuracy
        scorer_kwargs = {"threshold": metric_threshold}
    elif metric_name == "mse":
        scorer = masked_mse
        scorer_kwargs = {}
    else:
        raise ValueError("metric_name must be 'acc' or 'mse'.")

    name = alg_name.lower()

    # --- name → scheduling mapping (as you specified) ---
    # ebptt = bptt(len, INF)
    # ertrl = rtrl(len)
    # rtbptt = bptt(1, INF)
    # tbptt  = bptt(h, h')
    # rtrl   = rtrl(1)
    # pytorch, hybrid as-is
    if name == "ebptt":
        label = "ebptt≡bptt(∞,∞)"
        net, bench = _build_bptt(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            truncation_h=None, update_every_hprime=None,
            grad_clip=grad_clip, scorer=scorer, scorer_kwargs=scorer_kwargs, seed=seed
        )
        header = "EBPTT (h=∞, h′=∞, update at end)"
    elif name == "ertrl":
        label = "ertrl≡rtrl(∞)"
        net, bench = _build_rtrl(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            rtrl_hprime=None,  # update at end of epoch
            grad_clip=grad_clip, scorer=scorer, scorer_kwargs=scorer_kwargs, seed=seed
        )
        header = "ERTRL (RTRL len)"
    elif name == "rtbptt":
        label = "rtbptt≡bptt(∞, 1)"
        net, bench = _build_bptt(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            truncation_h=None, update_every_hprime=1,
            grad_clip=grad_clip, scorer=scorer, scorer_kwargs=scorer_kwargs, seed=seed
        )
        header = "RTBPTT (h=∞, h′=1)"
    elif name == "tbptt":
        label = f"tbptt(h={truncation_h if truncation_h is not None else '∞'},h′={update_every_hprime if update_every_hprime is not None else '∞'})"
        net, bench = _build_bptt(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            truncation_h=truncation_h, update_every_hprime=update_every_hprime,
            grad_clip=grad_clip, scorer=scorer, scorer_kwargs=scorer_kwargs, seed=seed
        )
        header = f"TBPTT (h={truncation_h if truncation_h is not None else '∞'}, h′={update_every_hprime if update_every_hprime is not None else '∞'})"
    elif name == "rtrl":
        label = "rtrl≡rtrl(1)"
        net, bench = _build_rtrl(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            rtrl_hprime=1,  # continual per-step updates
            grad_clip=grad_clip, scorer=scorer, scorer_kwargs=scorer_kwargs, seed=seed
        )
        header = "RTRL (h′=1)"
    elif name == "pytorch":
        label = "pytorch"
        net, bench = _build_pytorch(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            scorer=scorer, scorer_kwargs=scorer_kwargs
        )
        header = "PYTORCH-RNN"
    elif name == "hybrid":
        label = f"hybrid(h={h_bucket if h_bucket is not None else '∞'})"
        net, bench = _build_hybrid(
            m=m, n=n, lr=lr, weight_decay=weight_decay,
            h_bucket=h_bucket, grad_clip=grad_clip,
            scorer=scorer, scorer_kwargs=scorer_kwargs, seed=seed
        )
        header = "HYBRID"
    else:
        raise ValueError(f"Unknown alg_name: {alg_name}")

    print(f"=== {header} | m={m}, n={n}, tau={tau}, Ttrain={Ttrain}, "
          f"lr={lr}, epochs={epochs}, metric={metric_name} ===")

    loss_hist: list[float] = []
    acc_hist: list[float] = []
    epoch_marks: list[int] = []
    w_norm_hist: list[float] = []

    for ep in range(1, epochs + 1):
        loss = bench.train_epoch(Ttrain, tau)
        loss_hist.append(loss)

        if ep % log_every == 0:
            scores = bench.evaluate(quick_eval_seqs, Teval, tau)
            try:
                w_norm = float(np.linalg.norm(net.p.W))  # NumPy models
            except Exception:
                w_norm = float(sum((p.detach().norm()**2 for p in net.parameters())).sqrt().item())  # PyTorch model
            w_norm_hist.append(w_norm)

            if scores.get("name") == "acc":
                acc = scores["metric"]
                acc_hist.append(acc)
                epoch_marks.append(ep)
                print(f"[{label}] Epoch {ep:5d}/{epochs} | loss={loss:.6f} | "
                      f"acc={acc:6.2f}% ({scores.get('correct',0)}/{scores.get('total',0)}) | ||W||={w_norm:.2f}")
            else:
                print(f"[{label}] Epoch {ep:5d}/{epochs} | loss={loss:.6f} | "
                      f"{scores['name']}={scores['metric']:.6f} | ||W||={w_norm:.2f}")

    print("--- Final Evaluation ---")
    final_scores = bench.evaluate(eval_seqs, Teval, tau)
    if final_scores.get("name") == "acc":
        print(f"[{label}] Final: acc={final_scores['metric']:.2f}% "
              f"({final_scores.get('correct',0)}/{final_scores.get('total',0)})")
    else:
        print(f"[{label}] Final: {final_scores['name']}={final_scores['metric']:.6f}")
    print("\n")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(range(1, epochs + 1), loss_hist, label=label)
        ax[0].set_title(f"[{label}] Training Loss (avg per step)")
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()
        if acc_hist:
            ax[1].plot(epoch_marks, acc_hist, label=label)
            ax[1].set_title(f"[{label}] Quick Eval Accuracy")
            ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy (%)"); ax[1].legend()
        else:
            ax[1].text(0.5, 0.5, f"[{label}] Metric snapshots unavailable",
                       ha="center", va="center", transform=ax[1].transAxes)
            ax[1].set_axis_off()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{name}.png", dpi=150)

    return {
        "loss_hist": loss_hist,
        "acc_hist": acc_hist,
        "epoch_marks": epoch_marks,
        "final_scores": final_scores,
        "w_norm_hist": w_norm_hist,
        "label": label,
    }


def run_many(
    alg_names: list[str],
    *,
    m: int = 2, n: int = 8, lr: float = 5e-3,
    tau: int = 4, Ttrain: int = 20, Teval: int = 50,
    eval_seqs: int = 200, quick_eval_seqs: int = 50,
    epochs: int = 10_000, log_every: int = 500,
    weight_decay: float = 0.0,
    truncation_h: int | None = None, update_every_hprime: int | None = None,
    rtrl_hprime: int | None = None,
    h_bucket: int | None = 10,
    grad_clip: float | None = None,
    seed: int = 0,
    metric_name: str = "acc", metric_threshold: float = 0.5,
):
    results = {}
    loss_curves = {}; acc_curves = {}; epoch_marks_map = {}

    for i, alg in enumerate(alg_names):
        res = run_experiment(
            alg_name=alg,
            m=m, n=n, lr=lr, tau=tau,
            Ttrain=Ttrain, Teval=Teval,
            eval_seqs=eval_seqs, quick_eval_seqs=quick_eval_seqs,
            epochs=epochs, log_every=log_every,
            weight_decay=weight_decay,
            truncation_h=truncation_h, update_every_hprime=update_every_hprime,
            rtrl_hprime=rtrl_hprime,
            h_bucket=h_bucket,
            grad_clip=grad_clip,
            seed=seed + i,
            metric_name=metric_name, metric_threshold=metric_threshold,
            plot=False,
        )
        label = res["label"]
        results[label] = res
        loss_curves[label] = res["loss_hist"]
        acc_curves[label] = res["acc_hist"]
        epoch_marks_map[label] = res["epoch_marks"]

    # combined plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for label, loss in loss_curves.items():
        ax[0].plot(range(1, len(loss) + 1), loss, label=label)
    ax[0].set_title(f"Training Loss (avg per step) (tau={tau})")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()

    has_any_acc = False
    for label, accs in acc_curves.items():
        marks = epoch_marks_map[label]
        if accs and marks:
            has_any_acc = True
            ax[1].plot(marks, accs, marker=".", linestyle="-", label=label)
    if has_any_acc:
        ax[1].set_title(f"Quick Eval Accuracy (tau={tau})")
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy (%)"); ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "Accuracy snapshots unavailable",
                   ha="center", va="center", transform=ax[1].transAxes)
        ax[1].set_axis_off()

    plt.tight_layout()
    plt.savefig("compare.png", dpi=150)
    plt.show()
    return results


if __name__ == "__main__":
    # Example comparison using your new names
    results = run_many(
        alg_names=["ebptt", "ertrl", "rtbptt", "tbptt", "rtrl", "pytorch", "hybrid"],
        m=2, n=8, lr=5e-3,
        tau=10, Ttrain=50, Teval=50,
        eval_seqs=200, quick_eval_seqs=50,
        epochs=10000, log_every=250,
        weight_decay=0.0,
        truncation_h=15, update_every_hprime=5,   # used only by tbptt
        rtrl_hprime=1,                            # used only by rtrl (continual)
        h_bucket=10,                              # used only by hybrid
        grad_clip=None,
        seed=0,
        metric_name="acc", metric_threshold=0.5,
    )
