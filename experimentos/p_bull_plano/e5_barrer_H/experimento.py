"""E5 — D3: Barrido de H (lookback) y comparacion contra baseline incondicional.

Hipotesis a refutar/confirmar (D3 del hallazgo de E4): el lookback H=52
puede ser inadecuado para CMC200 (cripto cambia de regimen mas rapido
que SPX). Si reducimos H, CMC200 deberia mejorar.

Por cada H en {13, 26, 52, 104}:
  1. Reentrena el LSTM con la DLConfig por defecto, sobreescribiendo H.
  2. Computa pinball test (sumado sobre niveles q) por activo.
  3. Computa el baseline incondicional (cuantiles empiricos del train)
     para EL MISMO split y compara.
  4. Reporta tambien rango de la mediana en test y conditionality
     (std_pred / std_real) por activo.

Output: resultados.csv + tabla por consola + grafico comparativo.

Uso:
    python experimentos/p_bull_plano/e5_barrer_H/experimento.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import DLConfig
from dl.prediccion_deciles import (
    LoadedModel,
    QuantileLSTM,
    build_windows,
    chrono_split,
    load_returns,
    predict_deciles_batch,
    train_deciles,
)


_OUT_DIR = Path(__file__).resolve().parent
H_GRID   = [13, 26, 52, 104]


# ---------------------------------------------------------------------
# Pinball helpers (replicados del E4 para que el script sea autocontenido)
# ---------------------------------------------------------------------

def pinball_per_q(preds: np.ndarray, y: np.ndarray, qs) -> np.ndarray:
    Q = preds.shape[2]
    A = preds.shape[1]
    out = np.empty((Q, A), dtype=np.float64)
    for qi, q in enumerate(qs):
        diff = y - preds[:, :, qi]
        out[qi] = np.where(diff >= 0, q * diff, (q - 1.0) * diff).mean(axis=0)
    return out


def baseline_unconditional_pinball(y_train, y_test, qs):
    Q = len(qs); A = y_train.shape[1]
    pinball = np.empty((Q, A), dtype=np.float64)
    for qi, q in enumerate(qs):
        pred = np.quantile(y_train, q, axis=0)
        diff = y_test - pred[None, :]
        pinball[qi] = np.where(diff >= 0, q * diff, (q - 1.0) * diff).mean(axis=0)
    return pinball


# ---------------------------------------------------------------------
# Una corrida por H
# ---------------------------------------------------------------------

def run_one(H: int, base_cfg: DLConfig, returns: pd.DataFrame) -> dict:
    cfg = replace(base_cfg, H=H)

    print(f"  [H={H:>3}] entrenando ...", end=" ", flush=True)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result = train_deciles(cfg)
    dt = time.time() - t0

    # Reconstruir modelo para inferencia
    net = QuantileLSTM(cfg); net.load_state_dict(result.state_dict); net.eval()
    loaded = LoadedModel(nets=[net], config=cfg,
                         mean=np.asarray(result.mean, dtype=np.float32),
                         std=np.asarray(result.std,  dtype=np.float32))

    # Splits coherentes con el entrenamiento
    X, Y, t_idx = build_windows(returns, H)
    split = chrono_split(X, Y, t_idx, cfg.split)

    p_test = predict_deciles_batch(loaded, split.X_test)
    qs     = list(cfg.quantiles)
    Q      = len(qs); med_q = Q // 2

    p_lstm = pinball_per_q(p_test, split.Y_test, qs)
    p_base = baseline_unconditional_pinball(split.Y_train, split.Y_test, qs)

    out = {
        "H":           H,
        "n_train":     int(len(split.X_train)),
        "n_valid":     int(len(split.X_valid)),
        "n_test":      int(len(split.X_test)),
        "best_seed":   int(result.best_seed),
        "valid_loss":  float(result.best_valid),
        "elapsed_sec": round(dt, 1),
    }

    for ai, a in enumerate(cfg.assets):
        med_test = p_test[:, ai, med_q]
        out[f"{a}_pinball_lstm"] = float(p_lstm[:, ai].sum())
        out[f"{a}_pinball_base"] = float(p_base[:, ai].sum())
        out[f"{a}_ratio_l_b"]    = float(p_lstm[:, ai].sum() / p_base[:, ai].sum())
        out[f"{a}_med_min"]      = float(med_test.min())
        out[f"{a}_med_max"]      = float(med_test.max())
        out[f"{a}_med_cruza_0"]  = bool((med_test.min() < 0) and (med_test.max() > 0))
        out[f"{a}_std_med_pred"] = float(med_test.std())
        out[f"{a}_std_y_test"]   = float(split.Y_test[:, ai].std())
        out[f"{a}_ratio_std"]    = (float(med_test.std() / split.Y_test[:, ai].std())
                                    if split.Y_test[:, ai].std() > 0 else np.nan)
        out[f"{a}_width_mean"]   = float((p_test[:, ai, -1] - p_test[:, ai, 0]).mean())
        # corr ancho vs |y| (sharpness condicional)
        w   = p_test[:, ai, -1] - p_test[:, ai, 0]
        ay  = np.abs(split.Y_test[:, ai])
        out[f"{a}_corr_w_voly"] = (float(np.corrcoef(w, ay)[0, 1])
                                   if w.std() > 0 else np.nan)

    print(f"valid={out['valid_loss']:.5f}  ({dt:.1f}s)")
    return out


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_summary(df: pd.DataFrame, assets, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(assets), figsize=(5.5 * len(assets), 4.4))
    if len(assets) == 1: axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.plot(df["H"], df[f"{a}_pinball_lstm"], marker="o",
                label="LSTM",     color="#E63946", linewidth=1.6)
        ax.plot(df["H"], df[f"{a}_pinball_base"], marker="s",
                label="Baseline", color="#264653", linewidth=1.6)
        ax.set_xlabel("H (lookback)"); ax.set_ylabel("pinball test (sum sobre q)")
        ax.set_title(a)
        ax.set_xticks(df["H"])
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Pinball total por activo: LSTM vs baseline incondicional, segun H")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    base_cfg = DLConfig()
    print(f"Sweep H in {H_GRID}")
    print(f"  cfg base: assets={base_cfg.assets}  Q={list(base_cfg.quantiles)}  "
          f"seeds={base_cfg.seeds}  epochs={base_cfg.epochs}")

    returns = load_returns()
    print(f"  T total = {len(returns)}\n")

    rows = []
    for H in H_GRID:
        try:
            rows.append(run_one(H, base_cfg, returns))
        except Exception as e:
            print(f"  [H={H}] ERROR: {e}")
            rows.append({"H": H, "error": str(e)})

    df = pd.DataFrame(rows)
    csv_path = _OUT_DIR / "resultados.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}\n")

    # tabla resumida
    cols = ["H", "n_train", "n_test"]
    for a in base_cfg.assets:
        cols += [f"{a}_pinball_lstm", f"{a}_pinball_base", f"{a}_ratio_l_b",
                 f"{a}_med_cruza_0", f"{a}_ratio_std", f"{a}_corr_w_voly"]
    print("--- Resumen por H ---")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    p1 = plot_summary(df, list(base_cfg.assets),
                      _OUT_DIR / "pinball_vs_H.png")
    print(f"\nPlot: {p1}")


if __name__ == "__main__":
    main()
