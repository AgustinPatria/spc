"""Inspección del predictor de deciles ya entrenado.

Uso:
    python inspeccion/prediccion_deciles/inspeccionar_deciles.py

Carga el checkpoint en `models/decile_predictor.pt`, reconstruye los splits
cronológicos exactamente como en el entrenamiento y produce:

1. Pinball loss por split (train/valid/test) — el objetivo que se minimizó.
2. Cobertura empírica por decil — debería aproximarse al nominal (0.1, 0.2, ...).
3. % de cruces de deciles — cuántas veces la red viola la monotonicidad antes
   de ordenar la salida.
4. Resumen por activo: MAE de la mediana, bias y ancho promedio de la banda
   q10–q90.
5. Curva de entrenamiento (train vs valid pinball por época).
6. Calibración (cobertura empírica vs nominal).
7. Ejemplos puntuales en test: primer, último y dos intermedios — muestra los
   9 deciles predichos y el retorno realizado.
8. Deciles predichos vs realizados en test: por cada decil q, compara el
   promedio del predicho r̂^(q) contra el cuantil empírico q del realizado.

Todas las figuras se guardan junto a este script en
`inspeccion/prediccion_deciles/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Raíz del proyecto dos niveles arriba: .../SPC_Grid3/inspeccion/prediccion_deciles/<archivo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHECKPOINT_PATH, MODELS_DIR, DLConfig
from dl.prediccion_deciles import (
    ChronoSplit,
    build_windows,
    chrono_split,
    load_checkpoint,
    load_returns,
    pinball_loss,
    plot_fan_chart,
    predict_deciles_batch,
)

import torch


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------

def _pinball_np(preds: np.ndarray, Y: np.ndarray, quantiles) -> float:
    """Pinball loss promediada — mismo cálculo que en entrenamiento, en numpy."""
    p = torch.from_numpy(preds.astype(np.float32))
    y = torch.from_numpy(Y.astype(np.float32))
    return pinball_loss(p, y, quantiles).item()


def cobertura_empirica(preds: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Para cada decil q, fracción de observaciones con Y <= r_hat^(q). Shape (A, Q)."""
    # preds: (N, A, Q); Y: (N, A)
    return (Y[:, :, None] <= preds).mean(axis=0)


def pct_cruces(preds_raw: np.ndarray) -> np.ndarray:
    """% de ventanas donde los deciles NO son monotónicos (antes del sort). Shape (A,)."""
    diffs = np.diff(preds_raw, axis=-1)              # (N, A, Q-1)
    any_cross = (diffs < 0).any(axis=-1)             # (N, A)
    return any_cross.mean(axis=0) * 100.0


def resumen_por_activo(
    preds: np.ndarray, Y: np.ndarray, cfg: DLConfig,
) -> Dict[str, Dict[str, float]]:
    """MAE(mediana), bias(mediana), ancho medio q10-q90 y tasa de hit q10/q90."""
    Q = cfg.n_quantiles
    q_med = Q // 2
    q_lo, q_hi = 0, Q - 1
    nominal_lo = cfg.quantiles[q_lo]
    nominal_hi = cfg.quantiles[q_hi]

    out: Dict[str, Dict[str, float]] = {}
    for ai, asset in enumerate(cfg.assets):
        med = preds[:, ai, q_med]
        lo  = preds[:, ai, q_lo]
        hi  = preds[:, ai, q_hi]
        y   = Y[:, ai]
        dentro = ((y >= lo) & (y <= hi)).mean()
        out[asset] = {
            "mae_mediana":    float(np.mean(np.abs(y - med))),
            "bias_mediana":   float(np.mean(y - med)),
            "ancho_q10_q90":  float(np.mean(hi - lo)),
            "cobertura_central": float(dentro),
            "nominal_central":   float(nominal_hi - nominal_lo),
        }
    return out


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------

def plot_curvas_entrenamiento(history: Dict[str, list], out_path: Path) -> None:
    if not history or not history.get("train"):
        print("[viz] sin historial de entrenamiento — se salta la curva.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train", color="#1f77b4")
    ax.plot(history["valid"], label="valid", color="#E63946")
    best_ep = int(np.argmin(history["valid"]))
    ax.axvline(best_ep, color="grey", linestyle="--", alpha=0.5,
               label=f"best epoch ({best_ep})")
    ax.set_xlabel("época")
    ax.set_ylabel("pinball loss")
    ax.set_title("Curvas de entrenamiento")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] curvas guardadas en: {out_path}")


def plot_curvas_por_fold(folds_payload: list, out_path: Path) -> None:
    """Curvas train/valid por fold en subplots, pinball del mejor epoch marcada."""
    n = len(folds_payload)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False, sharey=True)
    for k, fd in enumerate(folds_payload):
        ax = axes[0][k]
        hist = fd.get("history", {})
        if not hist.get("train"):
            ax.set_title(f"fold {k+1} (sin historial)")
            continue
        ax.plot(hist["train"], label="train", color="#1f77b4")
        ax.plot(hist["valid"], label="valid", color="#E63946")
        best_ep = int(np.argmin(hist["valid"]))
        ax.axvline(best_ep, color="grey", linestyle="--", alpha=0.5,
                   label=f"best ep {best_ep}")
        ax.set_xlabel("época")
        if k == 0:
            ax.set_ylabel("pinball loss")
        ax.set_title(f"fold {k+1}  best_valid={fd['best_valid']:.4f}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] curvas por fold guardadas en: {out_path}")


def plot_calibracion(cov: np.ndarray, cfg: DLConfig, out_path: Path) -> None:
    """cov: (A, Q) con cobertura empírica por activo y decil."""
    fig, ax = plt.subplots(figsize=(6, 6))
    q = np.asarray(cfg.quantiles)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="ideal")
    for ai, asset in enumerate(cfg.assets):
        ax.plot(q, cov[ai], marker="o", label=asset)
    ax.set_xlabel("nominal (q)")
    ax.set_ylabel("empírico  P(Y ≤ r̂^(q))")
    ax.set_title("Calibración de deciles (test)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] calibración guardada en: {out_path}")


def plot_deciles_vs_reales(
    preds: np.ndarray, Y: np.ndarray, cfg: DLConfig, out_path: Path,
) -> None:
    """Deciles promedio predichos vs cuantiles empíricos del realizado (test).

    Por cada decil q y cada activo:
      - predicho  = promedio en test de r̂^(q)_t
      - realizado = cuantil empírico q de Y_{t, activo} en test

    preds: (N, A, Q)
    Y:     (N, A)
    """
    q          = np.asarray(cfg.quantiles)
    pred_mean  = preds.mean(axis=0)                                      # (A, Q)
    pred_lo    = np.quantile(preds, 0.25, axis=0)                        # (A, Q)
    pred_hi    = np.quantile(preds, 0.75, axis=0)                        # (A, Q)
    emp_q      = np.stack(
        [np.quantile(Y, qi, axis=0) for qi in q], axis=-1,
    )                                                                    # (A, Q)

    fig, axes = plt.subplots(
        1, cfg.n_assets, figsize=(5 * cfg.n_assets, 4.5), squeeze=False,
    )
    for ai, asset in enumerate(cfg.assets):
        ax = axes[0][ai]
        ax.fill_between(
            q, pred_lo[ai], pred_hi[ai],
            color="#1f3b73", alpha=0.18, linewidth=0,
            label="predicho (IQR en test)",
        )
        ax.plot(q, pred_mean[ai], marker="o", color="#1f3b73",
                label="predicho (media)")
        ax.plot(q, emp_q[ai], marker="s", color="#E63946",
                label="realizado (cuantil empírico)")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.set_xlabel("decil q")
        ax.set_ylabel("retorno")
        ax.set_title(f"Deciles predichos vs realizados — {asset}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] deciles predichos vs realizados guardado en: {out_path}")


def plot_ejemplos(
    preds: np.ndarray, Y: np.ndarray, t_idx: np.ndarray,
    cfg: DLConfig, out_path: Path, n: int = 4,
) -> None:
    """Muestra los 9 deciles vs el retorno realizado para n periodos del test."""
    N = preds.shape[0]
    if N == 0:
        return
    sel = np.linspace(0, N - 1, num=min(n, N)).astype(int)

    fig, axes = plt.subplots(
        len(sel), cfg.n_assets,
        figsize=(5 * cfg.n_assets, 2.6 * len(sel)),
        squeeze=False,
    )
    q = np.asarray(cfg.quantiles)
    for row, k in enumerate(sel):
        for ai, asset in enumerate(cfg.assets):
            ax = axes[row][ai]
            ax.plot(q, preds[k, ai], marker="o", color="#1f3b73",
                    label="deciles predichos")
            ax.axhline(Y[k, ai], color="#E63946", linestyle="--",
                       label=f"realizado = {Y[k, ai]:+.4f}")
            ax.axhline(0.0, color="grey", linewidth=0.6)
            ax.set_title(f"{asset}  |  t = {int(t_idx[k])}")
            ax.set_xlabel("decil")
            ax.set_ylabel("retorno")
            ax.grid(True, alpha=0.25)
            if row == 0 and ai == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] ejemplos guardados en: {out_path}")


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

OUT_DIR: Path = Path(__file__).resolve().parent


def inspeccionar(
    ckpt_path: Path = CHECKPOINT_PATH,
    out_dir: Path = OUT_DIR,
) -> None:
    print(f"[ckpt] cargando {ckpt_path}")
    model = load_checkpoint(ckpt_path)
    cfg   = model.config

    import torch as _torch
    payload = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_rolling = "oos" in payload and "folds" in payload

    if is_rolling:
        print("[ckpt] modo rolling-origin detectado (OOS agregadas).")
        _inspeccionar_rolling(ckpt_path, payload, model, cfg, out_dir)
    else:
        print("[ckpt] modo split unico (legacy).")
        _inspeccionar_split_unico(ckpt_path, payload, model, cfg, out_dir)


def _inspeccionar_split_unico(ckpt_path, payload, model, cfg, out_dir):
    """Flujo legacy: train/valid/test con un unico corte cronologico."""
    df_ret = load_returns()
    X, Y, t_idx = build_windows(df_ret, cfg.H)
    split: ChronoSplit = chrono_split(X, Y, t_idx, cfg.split)

    print(f"[cfg]  H={cfg.H}  assets={cfg.assets}  deciles={cfg.quantiles}")
    print(f"[data] train={len(split.X_train)}  valid={len(split.X_valid)}  "
          f"test={len(split.X_test)}")

    pbt = _pinball_np(predict_deciles_batch(model, split.X_train), split.Y_train, cfg.quantiles)
    pbv = _pinball_np(predict_deciles_batch(model, split.X_valid), split.Y_valid, cfg.quantiles)
    pbT = _pinball_np(predict_deciles_batch(model, split.X_test),  split.Y_test,  cfg.quantiles)
    print("\n== Pinball loss (menor es mejor) ==")
    print(f"  train = {pbt:.6f}")
    print(f"  valid = {pbv:.6f}")
    print(f"  test  = {pbT:.6f}")

    preds_test = predict_deciles_batch(model, split.X_test)
    _reportar_metricas("test", preds_test, split.Y_test, cfg)

    history = payload.get("history", {})
    print(f"\n[ckpt] best_seed={payload.get('best_seed')}  "
          f"best_valid={payload.get('best_valid')}")

    out_dir = Path(out_dir)
    plot_curvas_entrenamiento(history, out_dir / "curvas.png")
    cov = cobertura_empirica(preds_test, split.Y_test)
    plot_calibracion(cov, cfg, out_dir / "calibracion.png")
    plot_ejemplos(preds_test, split.Y_test, split.t_test, cfg,
                  out_dir / "ejemplos.png", n=4)
    plot_deciles_vs_reales(preds_test, split.Y_test, cfg,
                           out_dir / "deciles_vs_reales.png")
    plot_fan_chart(
        model, split.X_test, split.Y_test, split.t_test,
        out_path=out_dir / "fan_chart_test.png",
        show=False, title_suffix="test",
    )


def _inspeccionar_rolling(ckpt_path, payload, model, cfg, out_dir):
    """Flujo rolling-origin: usa las predicciones OOS agregadas."""
    oos    = payload["oos"]
    folds  = payload["folds"]
    preds  = np.asarray(oos["preds"])
    Y_oos  = np.asarray(oos["Y"])
    t_oos  = np.asarray(oos["t"])
    fid    = np.asarray(oos["fold_id"])

    n_folds = len(folds)
    print(f"[cfg]  H={cfg.H}  assets={cfg.assets}  deciles={cfg.quantiles}")
    print(f"[data] n_folds={n_folds}  N_oos={len(Y_oos)}  "
          f"(vs 22 del split unico legacy)")

    # ---- Pinball por fold + agregada ----
    print("\n== Pinball loss (menor es mejor) ==")
    print(f"  {'fold':<6} {'n_valid':>7}  {'t_range':<12}  {'pinball':>10}  seed  epochs")
    for k, fd in enumerate(folds):
        mask = (fid == k)
        if mask.sum() == 0:
            continue
        pb = _pinball_np(preds[mask], Y_oos[mask], cfg.quantiles)
        t_valid = np.asarray(fd["t_valid"])
        tr_end = fd.get("t_train_end", "?")
        ep_k = len(fd.get("history", {}).get("train", []))
        print(f"  {k+1:<6} {int(mask.sum()):>7}  "
              f"[{int(t_valid[0]):>3}..{int(t_valid[-1]):>3}]  "
              f"{pb:>10.6f}  {fd['best_seed']:>4}  {ep_k:>4}")
    pb_all = _pinball_np(preds, Y_oos, cfg.quantiles)
    print(f"  {'AGREG':<6} {len(Y_oos):>7}  {'(OOS)':<12}  {pb_all:>10.6f}")

    # ---- Cruces, cobertura, resumen sobre las OOS ----
    _reportar_metricas("OOS agregadas", preds, Y_oos, cfg)

    # ---- Graficos ----
    out_dir = Path(out_dir)
    plot_curvas_por_fold(folds, out_dir / "curvas_por_fold.png")
    # Tambien guarda la curva del ultimo fold (el que queda en el checkpoint principal).
    plot_curvas_entrenamiento(payload.get("history", {}), out_dir / "curvas.png")

    cov = cobertura_empirica(preds, Y_oos)
    plot_calibracion(cov, cfg, out_dir / "calibracion.png")
    plot_ejemplos(preds, Y_oos, t_oos, cfg,
                  out_dir / "ejemplos.png", n=4)
    plot_deciles_vs_reales(preds, Y_oos, cfg,
                           out_dir / "deciles_vs_reales.png")

    # Fan chart sobre OOS: reconstruyo X_oos desde los datos crudos usando t_oos.
    df_ret = load_returns()
    X_full, Y_full, t_full = build_windows(df_ret, cfg.H)
    t_to_idx = {int(t): i for i, t in enumerate(t_full)}
    idx_oos  = np.array([t_to_idx[int(t)] for t in t_oos])
    X_oos    = X_full[idx_oos]
    plot_fan_chart(
        model, X_oos, Y_oos, t_oos,
        out_path=out_dir / "fan_chart_oos.png",
        show=False, title_suffix="OOS walk-forward",
    )


def _reportar_metricas(nombre: str, preds: np.ndarray, Y: np.ndarray, cfg: DLConfig) -> None:
    """Imprime cruces, cobertura empirica y resumen por activo sobre `preds`/`Y`."""
    cruces = pct_cruces(preds)
    print(f"\n== Cruces de deciles ({nombre}) ==")
    for ai, asset in enumerate(cfg.assets):
        print(f"  {asset}: {cruces[ai]:5.2f}% de ventanas con al menos un cruce")

    cov = cobertura_empirica(preds, Y)
    print(f"\n== Cobertura empirica ({nombre}) vs nominal ==")
    header = "     nominal  " + "  ".join(f"{q:>5.2f}" for q in cfg.quantiles)
    print(header)
    for ai, asset in enumerate(cfg.assets):
        row = "  ".join(f"{c:>5.2f}" for c in cov[ai])
        print(f"  {asset:<7} empir.   {row}")

    resumen = resumen_por_activo(preds, Y, cfg)
    print(f"\n== Resumen por activo ({nombre}) ==")
    for asset, m in resumen.items():
        print(f"  {asset}:")
        print(f"    MAE mediana       = {m['mae_mediana']:.6f}")
        print(f"    bias mediana      = {m['bias_mediana']:+.6f}")
        print(f"    ancho q10-q90     = {m['ancho_q10_q90']:.6f}")
        print(f"    cobertura central = {m['cobertura_central']:.2%}  "
              f"(nominal {m['nominal_central']:.2%})")


if __name__ == "__main__":
    inspeccionar()
