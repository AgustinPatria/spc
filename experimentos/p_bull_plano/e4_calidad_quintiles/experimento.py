"""E4 — Calidad de los quintiles del LSTM (los 3 gaps de inspeccionar_deciles).

Aborda lo que `inspeccion/prediccion_deciles/inspeccionar_deciles.py` no
mide directamente:

  1. CONDITIONALITY — ¿la prediccion varia con el input?
     Para cada nivel q, std(pred_q) sobre las ventanas, comparada con
     std de los retornos realizados. Si std(pred_q) ~ 0, el LSTM esta
     prediciendo el cuantil incondicional (no aporta nada).

  2. PINBALL VS BASELINE INCONDICIONAL — ¿el LSTM aporta skill?
     Baseline: usar como prediccion los cuantiles empiricos del train,
     constantes en cualquier ventana de test. Comparar pinball loss de
     LSTM vs baseline por nivel y por activo. Si LSTM pinball >= baseline,
     el LSTM no esta aprendiendo nada conditional.

  3. SHARPNESS CONDICIONAL — ¿el ancho q10-q90 cambia con la vol futura?
     Si LSTM predice bandas mas anchas cuando viene una semana volatil,
     captura vol regimes. Si el ancho es constante, es vol-agnostic.

Output en este directorio: PNGs + CSVs + hallazgo.md.

Uso:
    python experimentos/p_bull_plano/e4_calidad_quintiles/experimento.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import ASSETS, CHECKPOINT_PATH, DATA_DIR, DLConfig
from dl.prediccion_deciles import (
    build_windows,
    chrono_split,
    load_checkpoint,
    load_returns,
    predict_deciles_batch,
)


_OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Pinball loss helpers
# ---------------------------------------------------------------------

def pinball_per_q(preds: np.ndarray, y: np.ndarray, qs) -> np.ndarray:
    """preds (N, A, Q), y (N, A), qs (Q,). Returns (Q, A) pinball por nivel."""
    N, A, Q = preds.shape
    out = np.empty((Q, A), dtype=np.float64)
    for qi, q in enumerate(qs):
        diff = y - preds[:, :, qi]                                 # (N, A)
        out[qi] = np.where(diff >= 0, q * diff, (q - 1.0) * diff).mean(axis=0)
    return out


def baseline_unconditional_pinball(y_train: np.ndarray, y_test: np.ndarray,
                                    qs) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: predecir el cuantil empirico del train (constante).
    Returns: (preds_const (Q, A), pinball (Q, A))."""
    Q = len(qs)
    A = y_train.shape[1]
    preds = np.empty((Q, A), dtype=np.float64)
    for qi, q in enumerate(qs):
        preds[qi] = np.quantile(y_train, q, axis=0)
    pinball = np.empty((Q, A), dtype=np.float64)
    for qi, q in enumerate(qs):
        diff = y_test - preds[qi][None, :]
        pinball[qi] = np.where(diff >= 0, q * diff, (q - 1.0) * diff).mean(axis=0)
    return preds, pinball


# ---------------------------------------------------------------------
# Diagnosticos
# ---------------------------------------------------------------------

def conditionality_table(preds: np.ndarray, y: np.ndarray, qs, assets) -> pd.DataFrame:
    """Para cada (q, asset): std(pred_q) vs std(y) y ratio."""
    rows = []
    for qi, q in enumerate(qs):
        for ai, a in enumerate(assets):
            pred_std = float(preds[:, ai, qi].std())
            real_std = float(y[:, ai].std())
            rows.append({
                "q":               q,
                "asset":           a,
                "std_pred":        pred_std,
                "std_real":        real_std,
                "ratio_pred_real": pred_std / real_std if real_std > 0 else np.nan,
                "rng_pred":       f"[{preds[:, ai, qi].min():.4f}, {preds[:, ai, qi].max():.4f}]",
            })
    return pd.DataFrame(rows)


def pinball_comparison_table(p_lstm: np.ndarray, p_base: np.ndarray,
                             qs, assets) -> pd.DataFrame:
    """Pinball LSTM vs baseline incondicional por (q, asset)."""
    rows = []
    for qi, q in enumerate(qs):
        for ai, a in enumerate(assets):
            l_lstm = float(p_lstm[qi, ai])
            l_base = float(p_base[qi, ai])
            rows.append({
                "q":          q,
                "asset":      a,
                "pinball_lstm":   l_lstm,
                "pinball_base":   l_base,
                "ratio_lstm_base": l_lstm / l_base if l_base > 0 else np.nan,
                "ganador":    "LSTM" if l_lstm < l_base else "BASE",
            })
    return pd.DataFrame(rows)


def sharpness_table(preds: np.ndarray, y: np.ndarray, assets) -> pd.DataFrame:
    """Ancho q10-q90 y su correlacion con vol realizada (|y|).
    preds (N, A, Q) — asume Q >= 2 con qs ordenados ascendentes."""
    width = preds[:, :, -1] - preds[:, :, 0]    # (N, A)
    abs_y = np.abs(y)                            # proxy de vol single-period
    rows = []
    for ai, a in enumerate(assets):
        w = width[:, ai]
        v = abs_y[:, ai]
        rho = float(np.corrcoef(w, v)[0, 1]) if w.std() > 0 else np.nan
        rows.append({
            "asset":          a,
            "mean_width":     float(w.mean()),
            "std_width":      float(w.std()),
            "rng_width":      f"[{w.min():.4f}, {w.max():.4f}]",
            "mean_abs_y":     float(v.mean()),
            "std_abs_y":      float(v.std()),
            "corr_width_vol": rho,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_pinball_bars(p_lstm, p_base, qs, assets, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(assets), figsize=(5.5 * len(assets), 4.2),
                             sharey=False)
    if len(assets) == 1: axes = [axes]
    x = np.arange(len(qs))
    width = 0.38
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.bar(x - width / 2, p_lstm[:, ai], width, label="LSTM",  color="#E63946")
        ax.bar(x + width / 2, p_base[:, ai], width, label="Baseline (uncond. q empirical)",
               color="#264653")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q:.2f}" for q in qs])
        ax.set_xlabel("nivel q")
        ax.set_ylabel("pinball loss (test)")
        ax.set_title(a)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("LSTM vs baseline incondicional — pinball loss por nivel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_pred_std(preds, y, qs, assets, out_path: Path) -> Path:
    """Grafico: std de preds por q vs std del realizado (linea horizontal)."""
    fig, axes = plt.subplots(1, len(assets), figsize=(5.5 * len(assets), 4.2))
    if len(assets) == 1: axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        pred_stds = [preds[:, ai, qi].std() for qi in range(len(qs))]
        ax.bar([f"{q:.2f}" for q in qs], pred_stds,
               color="#E63946", alpha=0.85, label="std(pred q)")
        ax.axhline(y[:, ai].std(), color="black", linestyle="--", linewidth=1.2,
                   label=f"std(y) = {y[:, ai].std():.4f}")
        ax.set_xlabel("nivel q")
        ax.set_ylabel("std a lo largo de las ventanas")
        ax.set_title(a)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Conditionality: std de las predicciones por nivel q")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_width_vs_vol(preds, y, assets, t_test, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(assets), figsize=(5.5 * len(assets), 4.2))
    if len(assets) == 1: axes = [axes]
    for ai, a in enumerate(assets):
        w = preds[:, ai, -1] - preds[:, ai, 0]
        v = np.abs(y[:, ai])
        ax = axes[ai]
        ax.scatter(w, v, s=40, alpha=0.7, color="#2E86AB")
        if w.std() > 0:
            slope, intercept = np.polyfit(w, v, 1)
            xs = np.linspace(w.min(), w.max(), 50)
            ax.plot(xs, slope * xs + intercept, color="red", linewidth=1.2,
                    label=f"corr={np.corrcoef(w, v)[0, 1]:.2f}")
            ax.legend()
        ax.set_xlabel("ancho q10-q90 predicho")
        ax.set_ylabel("|retorno realizado|")
        ax.set_title(a)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sharpness condicional: ancho predicho vs vol realizada")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Cargando modelo y datos ...")
    model    = load_checkpoint(CHECKPOINT_PATH)
    cfg      = model.config
    qs       = list(cfg.quantiles)
    H        = cfg.H
    assets   = list(cfg.assets)
    print(f"  cfg: H={H}, assets={assets}, quantiles={qs}")

    returns  = load_returns(DATA_DIR)[list(assets)]
    X, Y, t  = build_windows(returns, H)
    cfg_def  = DLConfig()
    split    = chrono_split(X, Y, t, cfg_def.split)
    print(f"  N total = {len(X)}  train={len(split.X_train)}  "
          f"valid={len(split.X_valid)}  test={len(split.X_test)}")

    # Predicciones por split
    print("Prediciendo deciles ...")
    p_train = predict_deciles_batch(model, split.X_train)
    p_valid = predict_deciles_batch(model, split.X_valid)
    p_test  = predict_deciles_batch(model, split.X_test)

    # ---- 1) Conditionality (test) ----
    cond_test = conditionality_table(p_test, split.Y_test, qs, assets)
    cond_test["split"] = "test"
    cond_train = conditionality_table(p_train, split.Y_train, qs, assets)
    cond_train["split"] = "train"
    cond = pd.concat([cond_train, cond_test], ignore_index=True)
    cond.to_csv(_OUT_DIR / "conditionality.csv", index=False)
    print("\n  --- Conditionality (std de pred / std de y) ---")
    print(cond.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 2) Pinball: LSTM vs baseline incondicional (test) ----
    p_lstm_test = pinball_per_q(p_test, split.Y_test, qs)
    _, p_base_test = baseline_unconditional_pinball(
        split.Y_train, split.Y_test, qs,
    )
    pin_cmp = pinball_comparison_table(p_lstm_test, p_base_test, qs, assets)
    pin_cmp.to_csv(_OUT_DIR / "pinball_vs_baseline.csv", index=False)
    print("\n  --- Pinball LSTM vs baseline incondicional (test) ---")
    print(pin_cmp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    # totales agregados
    sum_lstm = p_lstm_test.sum(axis=0)
    sum_base = p_base_test.sum(axis=0)
    print(f"\n  total pinball (sum sobre q) por activo:")
    for ai, a in enumerate(assets):
        ratio = sum_lstm[ai] / sum_base[ai] if sum_base[ai] > 0 else np.nan
        winner = "LSTM" if sum_lstm[ai] < sum_base[ai] else "BASE"
        print(f"    {a:<8} LSTM={sum_lstm[ai]:.6f}  BASE={sum_base[ai]:.6f}  "
              f"ratio={ratio:.4f}  ganador={winner}")

    # ---- 3) Sharpness condicional (test) ----
    sh = sharpness_table(p_test, split.Y_test, assets)
    sh.to_csv(_OUT_DIR / "sharpness.csv", index=False)
    print("\n  --- Sharpness condicional (test) ---")
    print(sh.to_string(index=False))

    # ---- Plots ----
    p1 = plot_pinball_bars(p_lstm_test, p_base_test, qs, assets,
                           _OUT_DIR / "pinball_vs_baseline.png")
    p2 = plot_pred_std(p_test, split.Y_test, qs, assets,
                       _OUT_DIR / "conditionality.png")
    p3 = plot_width_vs_vol(p_test, split.Y_test, assets, split.t_test,
                           _OUT_DIR / "width_vs_vol.png")
    print(f"\n  Plots:\n    {p1}\n    {p2}\n    {p3}")


if __name__ == "__main__":
    main()
