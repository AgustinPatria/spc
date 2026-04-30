"""Barrido de hiperparametros del LSTM cuantilico.

Explora un grid de 24 configuraciones sobre los knobs mas relevantes para
los problemas residuales del modelo (cruces de deciles, cobertura ancha,
sesgo de mediana). Para cada combinacion entrena con rolling-origin
(4 folds * seeds), computa metricas OOS agregadas y dumpea un CSV.

Grid:
    H            in {26, 52}
    lstm_hidden  in {24, 32, 48}
    lstm_layers  in {1, 2}
    dropout      in {0.1, 0.3}

Uso:
    PYTHONPATH=. python sweep_lstm.py
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DLConfig, PROJECT_ROOT
from dl.prediccion_deciles import pinball_loss, train_deciles_rolling


def _pinball_np(preds: np.ndarray, Y: np.ndarray, quantiles) -> float:
    p = torch.from_numpy(preds.astype(np.float32))
    y = torch.from_numpy(Y.astype(np.float32))
    return pinball_loss(p, y, quantiles).item()


def compute_metrics(result, cfg: DLConfig) -> dict:
    """Metricas OOS agregadas: pinball, bias, cruces, cobertura por activo."""
    preds     = result.oos_preds
    preds_raw = result.oos_preds_raw
    Y         = result.oos_Y

    Q     = cfg.n_quantiles
    q_med = Q // 2
    q_lo, q_hi = 0, Q - 1

    out: dict = {"pinball_oos": _pinball_np(preds, Y, cfg.quantiles)}
    for ai, asset in enumerate(cfg.assets):
        med = preds[:, ai, q_med]
        lo  = preds[:, ai, q_lo]
        hi  = preds[:, ai, q_hi]
        y   = Y[:, ai]

        cruces    = (np.diff(preds_raw[:, ai, :], axis=-1) < 0).any(axis=-1).mean() * 100.0
        cobertura = ((y >= lo) & (y <= hi)).mean()
        cob_q50   = (y <= med).mean()

        out[f"{asset}_bias_med"]   = float(np.mean(y - med))
        out[f"{asset}_mae_med"]    = float(np.mean(np.abs(y - med)))
        out[f"{asset}_ancho"]      = float(np.mean(hi - lo))
        out[f"{asset}_cruces_pct"] = float(cruces)
        out[f"{asset}_cob_q1090"]  = float(cobertura)
        out[f"{asset}_cob_q50"]    = float(cob_q50)
    return out


def build_configs() -> list[DLConfig]:
    H_grid       = [26, 52]
    hidden_grid  = [24, 32, 48]
    layers_grid  = [1, 2]
    dropout_grid = [0.1, 0.3]

    base = DLConfig()
    configs: list[DLConfig] = []
    for H, h, L, d in product(H_grid, hidden_grid, layers_grid, dropout_grid):
        configs.append(replace(base, H=H, lstm_hidden=h, lstm_layers=L, dropout=d))
    return configs


def main() -> None:
    out_dir = PROJECT_ROOT / "findings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sweep_lstm.csv"

    configs = build_configs()
    n = len(configs)
    print(f"[sweep] {n} configuraciones — estimado 10-20 min total")

    rows: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        tag = (f"H={cfg.H:<2}  hidden={cfg.lstm_hidden:<2}  "
               f"layers={cfg.lstm_layers}  drop={cfg.dropout}")
        print(f"[{i:>2}/{n}] {tag}  ", end="", flush=True)
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_deciles_rolling(cfg)
        m  = compute_metrics(result, cfg)
        dt = time.time() - t0
        rows.append({
            "H":           cfg.H,
            "hidden":      cfg.lstm_hidden,
            "layers":      cfg.lstm_layers,
            "dropout":     cfg.dropout,
            **m,
            "n_oos":       len(result.oos_Y),
            "elapsed_sec": round(dt, 1),
        })
        print(f"pb={m['pinball_oos']:.5f}  "
              f"biasSPX={m['SPX_bias_med']:+.4f}  "
              f"biasCMC={m['CMC200_bias_med']:+.4f}  "
              f"crucesSPX={m['SPX_cruces_pct']:.0f}%  "
              f"({dt:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[sweep] CSV guardado en {out_csv}")

    cols_top = [
        "H", "hidden", "layers", "dropout",
        "pinball_oos", "SPX_bias_med", "CMC200_bias_med",
        "SPX_cruces_pct", "CMC200_cruces_pct",
    ]
    print("\n=== Top 5 por pinball OOS (menor es mejor) ===")
    print(df.nsmallest(5, "pinball_oos")[cols_top].to_string(index=False))

    print("\n=== Top 5 por cruces SPX (menor es mejor) ===")
    print(df.nsmallest(5, "SPX_cruces_pct")[cols_top].to_string(index=False))

    print("\n=== Top 5 por |bias_med CMC200| ===")
    df_abs = df.copy()
    df_abs["abs_bias_CMC"] = df_abs["CMC200_bias_med"].abs()
    print(df_abs.nsmallest(5, "abs_bias_CMC")[cols_top].to_string(index=False))


if __name__ == "__main__":
    main()
