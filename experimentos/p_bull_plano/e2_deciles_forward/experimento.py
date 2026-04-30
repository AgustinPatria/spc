"""E2 — Deciles forward: ¿colapsa el rollout determinista a un punto fijo?

Replica el rollout de Regret_Grid.predict_pbull_rollout (que es el que
alimenta mu_mix/sigma_mix) pero, en lugar de guardar solo p_bull(t),
guarda los 5 deciles predichos en cada paso por activo. Despues:

  1. Grafica los 5 deciles a lo largo de t=1..T_HORIZON.
  2. Grafica max_q |decil[t+1] - decil[t]| en escala log para visualizar
     la velocidad de convergencia al punto fijo.
  3. Mide t_converge (primer paso a partir del cual los deciles dejan
     de cambiar mas que tol=1e-6) por activo.
  4. Vuelca CSV con la trayectoria completa.

Uso:
    python experimentos/p_bull_plano/e2_deciles_forward/experimento.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Raiz del proyecto: 3 niveles arriba (.../experimentos/p_bull_plano/e2_*/file.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import BULL_THRESHOLD, CHECKPOINT_PATH, DATA_DIR, T_HORIZON
from dl.prediccion_deciles import load_checkpoint
from Regret_Grid import load_market_data


_OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Rollout instrumentado
# ---------------------------------------------------------------------

def rollout_with_deciles(model, initial_window: np.ndarray, T: int):
    """Rollout determinista con mediana, registrando los deciles por paso.

    Returns:
        deciles: (T, A, Q) deciles predichos en cada paso forward.
        p_bull:  (T, A)     fraccion de deciles >= BULL_THRESHOLD por paso.
    """
    cfg   = model.config
    med_q = cfg.n_quantiles // 2
    A     = cfg.n_assets
    Q     = cfg.n_quantiles

    window  = initial_window.astype(np.float32).copy()
    deciles = np.empty((T, A, Q), dtype=np.float32)
    p_bull  = np.empty((T, A), dtype=np.float32)

    for t in range(T):
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_tensor).numpy()[0] for net in model.nets]   # K * (A, Q)
        preds = np.mean(np.stack(outs, axis=0), axis=0)                # (A, Q)
        deciles[t] = preds
        p_bull[t]  = (preds >= BULL_THRESHOLD).mean(axis=-1)

        median_r = preds[:, med_q]
        window   = np.concatenate([window[1:], median_r[None, :]], axis=0)

    return deciles, p_bull


# ---------------------------------------------------------------------
# Metricas de convergencia
# ---------------------------------------------------------------------

def measure_convergence(deciles: np.ndarray, tol: float = 1e-6) -> pd.DataFrame:
    """Para cada activo, busca el primer t a partir del cual los deciles
    quedan estables (max_q |delta| < tol para todos los pasos restantes).
    """
    T, A, Q = deciles.shape
    diffs = np.abs(np.diff(deciles, axis=0)).max(axis=2)   # (T-1, A)

    rows = []
    for ai in range(A):
        below = diffs[:, ai] < tol
        t_conv = None
        for t in range(len(below)):
            if below[t:].all():
                t_conv = t + 1
                break
        rows.append({
            "asset_idx":      ai,
            "t_converge":     t_conv if t_conv is not None else -1,
            "max_diff_t1":    float(diffs[0, ai]),
            "max_diff_t10":   float(diffs[min(9,  T - 2), ai]),
            "max_diff_t52":   float(diffs[min(51, T - 2), ai]),
            "max_diff_tlast": float(diffs[-1, ai]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_deciles(deciles: np.ndarray, assets, out_path: Path) -> Path:
    """Un panel por activo con los 5 deciles a lo largo de t."""
    T, A, Q = deciles.shape
    fig, axes = plt.subplots(A, 1, figsize=(10, 3.5 * A), sharex=True)
    if A == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    levels = np.linspace(0, 1, Q + 2)[1:-1]

    for ai, ax in enumerate(axes):
        for q in range(Q):
            ax.plot(
                np.arange(1, T + 1), deciles[:, ai, q],
                color=cmap(levels[q]),
                label=f"q={levels[q]:.2f}",
                linewidth=1.4,
            )
        ax.axhline(BULL_THRESHOLD, color="red", linestyle="--", linewidth=1,
                   label=f"BULL_THRESHOLD={BULL_THRESHOLD}")
        ax.axvline(52, color="gray", linestyle=":", linewidth=1, label="t=H=52")
        ax.set_ylabel(f"{assets[ai]}\nretorno predicho")
        ax.legend(loc="upper right", ncol=Q + 2, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("paso forward t")
    fig.suptitle("Deciles predichos por el LSTM en el rollout determinista (mediana)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_diffs(deciles: np.ndarray, assets, out_path: Path) -> Path:
    """Log-y: max_q |decil[t+1] - decil[t]| por activo, todos los t."""
    T, A, Q = deciles.shape
    diffs = np.abs(np.diff(deciles, axis=0)).max(axis=2)   # (T-1, A)
    fig, ax = plt.subplots(figsize=(10, 4))
    for ai in range(A):
        ax.semilogy(
            np.arange(1, T),
            diffs[:, ai] + 1e-15,            # epsilon para evitar log(0)
            label=assets[ai], linewidth=1.6,
        )
    ax.axvline(52, color="gray", linestyle=":", label="t=H=52")
    ax.set_xlabel("paso forward t")
    ax.set_ylabel("max_q |decil[t+1] - decil[t]|  (log)")
    ax.set_title("Velocidad de convergencia del rollout al punto fijo")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_pbull(p_bull: np.ndarray, assets, out_path: Path) -> Path:
    """Serie de p_bull(t) por activo (replica visual de lo que ve build_dl_context)."""
    T, A = p_bull.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    for ai in range(A):
        ax.plot(np.arange(1, T + 1), p_bull[:, ai],
                label=assets[ai], linewidth=1.6, marker=".")
    ax.set_xlabel("paso forward t")
    ax.set_ylabel("p_bull(t)")
    ax.set_title("p_bull(t) producido por el rollout determinista")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Cargando datos y modelo ...")
    base_ctx = load_market_data(str(DATA_DIR))
    assets   = list(base_ctx["assets"])
    r_hist   = base_ctx["r"]

    model = load_checkpoint(CHECKPOINT_PATH)
    H     = model.config.H
    initial_window = np.stack(
        [r_hist[i].sort_index().values[-H:] for i in assets], axis=1,
    ).astype(np.float32)

    T = T_HORIZON
    print(f"Rollout: T={T}, H={H}, assets={assets}, Q={model.config.n_quantiles}")

    deciles, p_bull = rollout_with_deciles(model, initial_window, T)

    # CSV con la trayectoria completa
    Q = deciles.shape[2]
    rows = []
    for t in range(T):
        for ai, a in enumerate(assets):
            row = {"t": t + 1, "asset": a, "p_bull": float(p_bull[t, ai])}
            for q in range(Q):
                row[f"q{q}"] = float(deciles[t, ai, q])
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = _OUT_DIR / "deciles_forward.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV trayectoria: {csv_path}")

    # Convergencia
    conv = measure_convergence(deciles, tol=1e-6)
    conv["asset"] = [assets[i] for i in conv["asset_idx"]]
    conv = conv[["asset", "t_converge", "max_diff_t1",
                 "max_diff_t10", "max_diff_t52", "max_diff_tlast"]]
    conv_path = _OUT_DIR / "convergencia.csv"
    conv.to_csv(conv_path, index=False)
    print(f"  CSV convergencia: {conv_path}")
    print()
    print(conv.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    print()

    # Plots
    p1 = plot_deciles(deciles, assets, _OUT_DIR / "deciles_forward.png")
    p2 = plot_diffs(deciles,   assets, _OUT_DIR / "diff_consecutivo.png")
    p3 = plot_pbull(p_bull,    assets, _OUT_DIR / "p_bull_forward.png")
    print(f"  Plot deciles:        {p1}")
    print(f"  Plot diferencias:    {p2}")
    print(f"  Plot p_bull(t):      {p3}")

    print()
    print("--- p_bull observado en el rollout ---")
    for ai, a in enumerate(assets):
        col = p_bull[:, ai]
        uniq = np.unique(col)
        print(f"  {a:<8} min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}  valores_unicos={uniq.tolist()}")


if __name__ == "__main__":
    main()
