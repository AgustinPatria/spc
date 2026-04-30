"""Diagnostico visual del colapso de p_bull bajo rollout deterministico.

Produce dos PNGs en este directorio:
  - colapso_pbull_deciles.png : muestra que solo 2 o 3 de los 5 deciles
                                cruzan el BULL_THRESHOLD -> p_bull solo
                                puede tomar 0.4 o 0.6.
  - colapso_pbull_vs_mc.png   : compara la curva determinista (constante)
                                contra la marginal Monte Carlo (variable).
"""

from pathlib import Path
import sys

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR, CHECKPOINT_PATH, T_HORIZON, BULL_THRESHOLD,
    ASSETS, RETURN_CSV, RETURN_COL, DECILES, N_CANDIDATES,
)
from dl.prediccion_deciles import load_checkpoint

OUT_DIR = Path(__file__).resolve().parent


def initial_window(model, data_dir, assets):
    H = model.config.H
    series = []
    for a in assets:
        df = pd.read_csv(data_dir / RETURN_CSV[a])
        df.columns = [c.strip() for c in df.columns]
        col = RETURN_COL[a]
        s = df.sort_values("t")[col].values[-H:]
        series.append(s)
    return np.stack(series, axis=1).astype(np.float32)


def deterministic_rollout(model, iw, T):
    cfg   = model.config
    A, Q  = cfg.n_assets, cfg.n_quantiles
    med_q = Q // 2

    window = iw.astype(np.float32).copy()
    deciles_per_step = np.empty((T, A, Q), dtype=np.float32)
    p_bull           = np.empty((T, A), dtype=np.float32)

    for t in range(T):
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        with torch.no_grad():
            outs = [net(torch.from_numpy(x)).numpy()[0] for net in model.nets]
        preds = np.sort(np.mean(np.stack(outs, 0), 0), axis=-1)        # (A, Q)
        deciles_per_step[t] = preds
        p_bull[t] = (preds >= BULL_THRESHOLD).mean(axis=-1)
        window = np.concatenate([window[1:], preds[:, med_q][None, :]], axis=0)

    return deciles_per_step, p_bull


def mc_rollout_pbull(model, iw, T, N=1000, seed=0):
    cfg   = model.config
    A, Q  = cfg.n_assets, cfg.n_quantiles
    rng   = np.random.default_rng(seed)

    windows = np.tile(iw.astype(np.float32), (N, 1, 1))                # (N, H, A)
    p_bull  = np.empty((N, T, A), dtype=np.float32)

    for t in range(T):
        x = ((windows - model.mean) / model.std).astype(np.float32)
        x_t = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_t).numpy() for net in model.nets]
        preds = np.sort(np.mean(np.stack(outs, 0), 0), axis=-1)        # (N, A, Q)
        p_bull[:, t, :] = (preds >= BULL_THRESHOLD).mean(axis=-1)

        q_idx = rng.integers(low=0, high=Q, size=N)
        r_t   = np.take_along_axis(preds, q_idx[:, None, None], axis=2).squeeze(-1)
        windows = np.concatenate([windows[:, 1:, :], r_t[:, None, :]], axis=1)

    return p_bull   # (N, T, A)


def plot_decile_collapse(deciles_per_step, asset_idx, asset_name, out_path):
    T = deciles_per_step.shape[0]
    snapshots = [0, T // 4, T // 2, T - 1]
    qs        = np.array(DECILES) * 100
    Q         = len(qs)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(14, 4.4), sharey=True)
    for ax, t_idx in zip(axes, snapshots):
        deciles = deciles_per_step[t_idx, asset_idx]
        colors  = ["#d62728" if r < BULL_THRESHOLD else "#2ca02c" for r in deciles]
        ax.scatter(qs, deciles, s=140, c=colors, edgecolors="black", zorder=3)
        for q, r in zip(qs, deciles):
            ax.annotate(f"q{int(q)}", (q, r), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.axhline(BULL_THRESHOLD, color="black", linestyle="--",
                   lw=1.2, alpha=0.8, label=f"BULL_THRESHOLD = {BULL_THRESHOLD}")
        n_bull = int((deciles >= BULL_THRESHOLD).sum())
        ax.set_title(f"t = {t_idx + 1}\np_bull = {n_bull}/{Q} = {n_bull / Q:.1f}",
                     fontsize=10)
        ax.set_xlabel("decil predicho")
        ax.set_xticks(qs)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"retorno semanal predicho — {asset_name}")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Cuello de botella discreto: con 5 deciles y BULL_THRESHOLD=0,\n"
        "p_bull solo puede ser 0/5, 1/5, ..., 5/5  (resolucion = 0.2)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_det_vs_mc(p_bull_det, p_bull_mc, out_path, asset_names):
    T, A    = p_bull_det.shape
    t_axis  = np.arange(1, T + 1)

    fig, axes = plt.subplots(A, 1, figsize=(11, 3.4 * A), sharex=True)
    if A == 1:
        axes = [axes]

    for ai, ax in enumerate(axes):
        path_p          = p_bull_mc[:, :, ai]
        mc_mean         = path_p.mean(axis=0)
        mc_q25, mc_q75  = np.quantile(path_p, [0.25, 0.75], axis=0)
        mc_q05, mc_q95  = np.quantile(path_p, [0.05, 0.95], axis=0)

        ax.fill_between(t_axis, mc_q05, mc_q95, color="#1f77b4", alpha=0.10,
                        label="MC q05-q95 (1000 paths)")
        ax.fill_between(t_axis, mc_q25, mc_q75, color="#1f77b4", alpha=0.22,
                        label="MC q25-q75")
        ax.plot(t_axis, mc_mean, color="#1f77b4", lw=2.2,
                label="MC marginal: E[p_bull(t)] sobre 1000 trayectorias")
        ax.plot(t_axis, p_bull_det[:, ai], color="#d62728", lw=2.2, ls="--",
                label="rollout determinista (q50)")

        # marca los valores discretos posibles del determinista
        for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            ax.axhline(v, color="#aaaaaa", lw=0.5, alpha=0.5)

        ax.axhline(0.5, color="black", lw=0.7, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("p_bull")
        ax.set_title(asset_names[ai])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (paso forward)")
    fig.suptitle(
        "p_bull(t): rollout determinista colapsa a una constante;\n"
        "marginal Monte Carlo varia con t y tiene dispersion real",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Cargando modelo...")
    model = load_checkpoint(CHECKPOINT_PATH)
    iw    = initial_window(model, DATA_DIR, list(ASSETS))

    print("Rollout determinista (q50)...")
    deciles_det, pbull_det = deterministic_rollout(model, iw, T_HORIZON)

    print(f"Rollout Monte Carlo (N={N_CANDIDATES})...")
    pbull_mc = mc_rollout_pbull(model, iw, T_HORIZON, N=N_CANDIDATES, seed=0)

    plot1 = OUT_DIR / "colapso_pbull_deciles.png"
    plot_decile_collapse(deciles_det, asset_idx=0, asset_name=ASSETS[0],
                         out_path=plot1)
    print(f"  guardado: {plot1}")

    plot2 = OUT_DIR / "colapso_pbull_vs_mc.png"
    plot_det_vs_mc(pbull_det, pbull_mc, out_path=plot2,
                   asset_names=list(ASSETS))
    print(f"  guardado: {plot2}")

    # resumen numerico
    print("\nResumen:")
    for ai, a in enumerate(ASSETS):
        det = pbull_det[:, ai]
        mc  = pbull_mc[:, :, ai].mean(axis=0)
        print(f"  {a:7s}  det:  min={det.min():.3f}  max={det.max():.3f}  std={det.std():.4f}")
        print(f"           mc:   min={mc.min():.3f}  max={mc.max():.3f}  std={mc.std():.4f}")
