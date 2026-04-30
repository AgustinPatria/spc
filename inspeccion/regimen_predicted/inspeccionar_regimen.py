"""Inspección del clasificador de régimen bull/bear (PDF sec. 2.4).

Uso:
    python inspeccion/regimen_predicted/inspeccionar_regimen.py

Reusa el checkpoint de deciles (`models/decile_predictor.pt`) y pasa por
`regimen_from_deciles` para obtener `p_bull`/`p_bear`. Reconstruye los splits
cronológicos exactamente como en el entrenamiento y produce:

1. Distribución de p_bull por split (media, std, % cerca de 0/1).
2. Fracción realizada de bull (Y >= 0) por split — el "objetivo".
3. Brier score, log-loss y accuracy @ 0.5, por activo y split.
4. Matriz de confusión @ 0.5 sobre test.
5. Comparación contra baseline trivial (p_bull constante = freq. de bull en train).
6. Serie temporal de p_bull sobre test, con puntos coloreados por realizado.
7. Reliability diagram sobre todo el dataset (train+valid+test).
8. Histograma de p_bull por split.
9. Scatter p_bull vs retorno realizado.

Todas las figuras se guardan junto a este script en
`inspeccion/regimen_predicted/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Raíz del proyecto dos niveles arriba: .../SPC_Grid3/inspeccion/regimen_predicted/<archivo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHECKPOINT_PATH, MODELS_DIR, DLConfig
from dl.prediccion_deciles import (
    ChronoSplit,
    LoadedModel,
    build_windows,
    chrono_split,
    load_checkpoint,
    load_returns,
)
from dl.regimen_predicted import regimen_from_deciles, regimen_probabilities

import torch


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------

def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _accuracy(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    return float(((p >= thr).astype(int) == y.astype(int)).mean())


def _confusion(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> Tuple[int, int, int, int]:
    yhat = (p >= thr).astype(int)
    yt   = y.astype(int)
    tp = int(((yhat == 1) & (yt == 1)).sum())
    tn = int(((yhat == 0) & (yt == 0)).sum())
    fp = int(((yhat == 1) & (yt == 0)).sum())
    fn = int(((yhat == 0) & (yt == 1)).sum())
    return tp, tn, fp, fn


def _metricas_split(p_bull: np.ndarray, Y: np.ndarray, cfg: DLConfig) -> Dict[str, Dict[str, float]]:
    """Devuelve {asset: {brier, log_loss, accuracy, pct_bull_real, pct_bull_pred}}."""
    out: Dict[str, Dict[str, float]] = {}
    for ai, asset in enumerate(cfg.assets):
        p = p_bull[:, ai]
        y = (Y[:, ai] >= 0.0).astype(np.float32)
        out[asset] = {
            "brier":         _brier(p, y),
            "log_loss":      _log_loss(p, y),
            "accuracy":      _accuracy(p, y),
            "pct_bull_real": float(y.mean()),
            "pct_bull_pred": float(p.mean()),
        }
    return out


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------

def plot_serie_probabilidad(
    p_bull: np.ndarray, Y: np.ndarray, t_idx: np.ndarray,
    cfg: DLConfig, out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        cfg.n_assets, 1, figsize=(10, 3 * cfg.n_assets), sharex=True, squeeze=False,
    )
    for ai, asset in enumerate(cfg.assets):
        ax = axes[ai][0]
        ax.plot(t_idx, p_bull[:, ai], color="#1f3b73", linewidth=1.0, label="p_bull")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        is_bull = (Y[:, ai] >= 0.0)
        ax.scatter(t_idx[is_bull],  p_bull[is_bull, ai],
                   color="#2ca02c", s=28, zorder=3, label="bull realizado")
        ax.scatter(t_idx[~is_bull], p_bull[~is_bull, ai],
                   color="#E63946", s=28, zorder=3, label="bear realizado")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"p_bull(t+1) predicho — {asset}")
        ax.set_ylabel("prob. bull")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1][0].set_xlabel("t")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] serie probabilidades guardada en: {out_path}")


def plot_reliability(
    p_bull: np.ndarray, Y: np.ndarray, cfg: DLConfig,
    out_path: Path, n_bins: int = 5,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="ideal")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for ai, asset in enumerate(cfg.assets):
        p = p_bull[:, ai]
        y = (Y[:, ai] >= 0.0).astype(np.float32)
        xs, ys = [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
        ax.plot(xs, ys, marker="o", label=asset)
    ax.set_xlabel("p_bull predicho (promedio por bin)")
    ax.set_ylabel("frecuencia empírica de bull")
    ax.set_title("Reliability diagram (train+valid+test)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] reliability guardado en: {out_path}")


def plot_histograma(
    p_bull_splits: Dict[str, np.ndarray], cfg: DLConfig, out_path: Path,
) -> None:
    splits = list(p_bull_splits.keys())
    fig, axes = plt.subplots(
        cfg.n_assets, len(splits),
        figsize=(4 * len(splits), 3 * cfg.n_assets),
        squeeze=False,
    )
    for ai, asset in enumerate(cfg.assets):
        for si, name in enumerate(splits):
            ax = axes[ai][si]
            ax.hist(
                p_bull_splits[name][:, ai],
                bins=np.linspace(0.0, 1.0, 11),
                color="#1f3b73", alpha=0.8, edgecolor="white",
            )
            ax.set_title(f"{asset} — {name}")
            ax.set_xlim(-0.02, 1.02)
            ax.set_xlabel("p_bull")
            ax.set_ylabel("frecuencia")
            ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] histograma guardado en: {out_path}")


def plot_scatter(
    p_bull: np.ndarray, Y: np.ndarray, cfg: DLConfig, out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1, cfg.n_assets, figsize=(5 * cfg.n_assets, 4), squeeze=False,
    )
    for ai, asset in enumerate(cfg.assets):
        ax = axes[0][ai]
        ax.scatter(p_bull[:, ai], Y[:, ai], alpha=0.75, color="#1f3b73")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.axvline(0.5, color="grey", linewidth=0.6)
        ax.set_xlabel("p_bull predicho")
        ax.set_ylabel("retorno realizado")
        ax.set_title(f"{asset}")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] scatter guardado en: {out_path}")


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

OUT_DIR: Path = Path(__file__).resolve().parent


def _print_tabla_predicciones(
    p_bull: np.ndarray, Y: np.ndarray, t_idx: np.ndarray, cfg: DLConfig,
) -> None:
    """Tabla por consola: por cada t del test, p_bull/p_bear por activo + realizado."""
    head = f"  {'t':>4} "
    for asset in cfg.assets:
        head += f"| {asset+' p_bull':>12}  {asset+' p_bear':>12}  {asset+' ret':>10}  {asset+' real':>7}  ok "
    print(head)
    print("  " + "-" * (len(head) - 2))
    for n in range(len(p_bull)):
        row = f"  {int(t_idx[n]):>4} "
        for ai, _ in enumerate(cfg.assets):
            pb = float(p_bull[n, ai])
            pr = float(Y[n, ai])
            real_bull = pr >= 0.0
            pred_bull = pb >= 0.5
            ok = "OK" if (pred_bull == real_bull) else ".."
            row += (
                f"| {pb:>12.4f}  {1.0 - pb:>12.4f}  {pr:>+10.4f}  "
                f"{'bull' if real_bull else 'bear':>7}  {ok:>2} "
            )
        print(row)


def _guardar_csv_predicciones(
    p_splits: Dict[str, np.ndarray],
    y_splits: Dict[str, np.ndarray],
    t_splits: Dict[str, np.ndarray],
    cfg: DLConfig,
    out_path: Path,
) -> None:
    """Exporta predicciones de todos los splits a un CSV ordenado por t."""
    cols = ["split", "t"]
    for asset in cfg.assets:
        cols += [f"{asset}_p_bull", f"{asset}_p_bear",
                 f"{asset}_ret_real", f"{asset}_regimen_real", f"{asset}_ok"]
    filas = []
    for split_name, p in p_splits.items():
        Y = y_splits[split_name]
        tx = t_splits[split_name]
        for n in range(len(p)):
            fila = [split_name, int(tx[n])]
            for ai, _ in enumerate(cfg.assets):
                pb = float(p[n, ai])
                pr = float(Y[n, ai])
                real_bull = pr >= 0.0
                pred_bull = pb >= 0.5
                fila += [
                    f"{pb:.6f}", f"{(1.0 - pb):.6f}", f"{pr:.6f}",
                    "bull" if real_bull else "bear",
                    int(pred_bull == real_bull),
                ]
            filas.append(fila)
    filas.sort(key=lambda r: r[1])  # ordenar por t

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for fila in filas:
            f.write(",".join(str(x) for x in fila) + "\n")
    print(f"[csv] predicciones guardadas en: {out_path}")


def _print_tabla_metricas(
    metricas_por_split: Dict[str, Dict[str, Dict[str, float]]], cfg: DLConfig,
) -> None:
    header = f"  {'split':<7} {'activo':<8} {'brier':>8} {'logloss':>8} {'acc':>6} {'%bull_pred':>11} {'%bull_real':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for split_name, per_asset in metricas_por_split.items():
        for asset in cfg.assets:
            m = per_asset[asset]
            print(
                f"  {split_name:<7} {asset:<8} "
                f"{m['brier']:>8.4f} {m['log_loss']:>8.4f} {m['accuracy']:>6.2%} "
                f"{m['pct_bull_pred']:>11.2%} {m['pct_bull_real']:>11.2%}"
            )


def inspeccionar(
    ckpt_path: Path = CHECKPOINT_PATH,
    out_dir: Path = OUT_DIR,
) -> None:
    print(f"[ckpt] cargando {ckpt_path}")
    model = load_checkpoint(ckpt_path)
    cfg   = model.config

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_rolling = "oos" in payload and "folds" in payload
    if is_rolling:
        print("[ckpt] modo rolling-origin detectado — uso preds OOS agregadas.")
        _inspeccionar_rolling(payload, model, cfg, out_dir)
        return

    print("[ckpt] modo split unico (legacy).")
    df_ret = load_returns()
    X, Y, t_idx = build_windows(df_ret, cfg.H)
    split: ChronoSplit = chrono_split(X, Y, t_idx, cfg.split)

    print(f"[cfg]  H={cfg.H}  assets={cfg.assets}  deciles={cfg.quantiles}")
    print(f"[data] train={len(split.X_train)}  valid={len(split.X_valid)}  "
          f"test={len(split.X_test)}")

    # ---- Probabilidades por split ----
    p_tr, _ = regimen_probabilities(model, split.X_train)
    p_va, _ = regimen_probabilities(model, split.X_valid)
    p_te, _ = regimen_probabilities(model, split.X_test)

    # ---- Métricas por split ----
    metricas = {
        "train": _metricas_split(p_tr, split.Y_train, cfg),
        "valid": _metricas_split(p_va, split.Y_valid, cfg),
        "test":  _metricas_split(p_te, split.Y_test,  cfg),
    }
    print("\n== Métricas de régimen ==")
    _print_tabla_metricas(metricas, cfg)

    # ---- Predicciones crudas sobre test ----
    print("\n== Predicciones de régimen — TEST ==")
    _print_tabla_predicciones(p_te, split.Y_test, split.t_test, cfg)

    _guardar_csv_predicciones(
        p_splits={"train": p_tr, "valid": p_va, "test": p_te},
        y_splits={"train": split.Y_train, "valid": split.Y_valid, "test": split.Y_test},
        t_splits={"train": split.t_train, "valid": split.t_valid, "test": split.t_test},
        cfg=cfg,
        out_path=OUT_DIR / "predicciones_regimen.csv",
    )

    # ---- Baseline trivial: p_bull constante = freq. de bull en train ----
    print("\n== Baseline trivial (p_bull constante = freq. bull train) ==")
    for ai, asset in enumerate(cfg.assets):
        p_const = np.full_like(p_te[:, ai], metricas["train"][asset]["pct_bull_real"])
        y_te    = (split.Y_test[:, ai] >= 0.0).astype(np.float32)
        print(
            f"  {asset:<8} test_brier_modelo={metricas['test'][asset]['brier']:.4f}  "
            f"test_brier_baseline={_brier(p_const, y_te):.4f}  "
            f"(menor = mejor; modelo debe batir al baseline)"
        )

    # ---- Matriz de confusión en test ----
    print("\n== Matriz de confusión @ 0.5 (test) ==")
    for ai, asset in enumerate(cfg.assets):
        y_te = (split.Y_test[:, ai] >= 0.0).astype(int)
        tp, tn, fp, fn = _confusion(p_te[:, ai], y_te)
        print(f"  {asset}:")
        print(f"    pred bull | real bull  TP={tp:>3}   real bear  FP={fp:>3}")
        print(f"    pred bear | real bull  FN={fn:>3}   real bear  TN={tn:>3}")

    # ---- Gráficos (test + combinado) ----
    p_all = np.concatenate([p_tr, p_va, p_te], axis=0)
    Y_all = np.concatenate([split.Y_train, split.Y_valid, split.Y_test], axis=0)

    out_dir = Path(out_dir)
    plot_serie_probabilidad(
        p_te, split.Y_test, split.t_test, cfg, out_dir / "serie_p_bull_test.png",
    )
    plot_reliability(p_all, Y_all, cfg, out_dir / "reliability.png", n_bins=5)
    plot_histograma(
        {"train": p_tr, "valid": p_va, "test": p_te}, cfg,
        out_dir / "histograma.png",
    )
    plot_scatter(p_te, split.Y_test, cfg, out_dir / "scatter_pbull_vs_retorno.png")


def _inspeccionar_rolling(payload, model, cfg, out_dir):
    """Flujo rolling-origin: usa las preds OOS agregadas + el train del fold 1.

    Las metricas y graficos de OOS se calculan sobre las preds_oos guardadas
    (cada obs predicha por el fold que la validaba — sin fuga). Para el
    baseline 'frecuencia de bull en train' usamos el train del fold 1
    (datos vistos por el primer modelo).
    """
    out_dir = Path(out_dir)
    oos    = payload["oos"]
    folds  = payload["folds"]
    preds  = np.asarray(oos["preds"])         # (N_oos, A, Q) — ya ordenados
    Y_oos  = np.asarray(oos["Y"])             # (N_oos, A)
    t_oos  = np.asarray(oos["t"])             # (N_oos,)
    fid    = np.asarray(oos["fold_id"])       # (N_oos,)

    # p_bull viene directo de los deciles OOS.
    p_oos, _ = regimen_from_deciles(preds)    # (N_oos, A)

    # Frecuencia de bull en el train del fold 1 — base coherente para el baseline.
    df_ret = load_returns()
    X_full, Y_full, t_full = build_windows(df_ret, cfg.H)
    t_train_end_f1 = int(folds[0]["t_train_end"])
    train_mask = t_full <= t_train_end_f1
    Y_train_f1 = Y_full[train_mask]
    pct_bull_train = (Y_train_f1 >= 0.0).mean(axis=0)        # (A,)

    print(f"[cfg]  H={cfg.H}  assets={cfg.assets}  deciles={cfg.quantiles}")
    print(f"[data] N_oos={len(Y_oos)} (vs split unico legacy)  "
          f"folds={len(folds)}  train_fold1={int(train_mask.sum())} obs")

    # ---- Metricas por fold + agregada ----
    print("\n== Metricas de regimen por fold (OOS) ==")
    print(f"  {'fold':<5} {'activo':<8} {'n':>4} {'brier':>8} {'logloss':>8} {'acc':>6} "
          f"{'%bull_pred':>11} {'%bull_real':>11}")
    print("  " + "-" * 66)
    for k in range(len(folds)):
        mask = (fid == k)
        if mask.sum() == 0:
            continue
        m = _metricas_split(p_oos[mask], Y_oos[mask], cfg)
        for asset in cfg.assets:
            mm = m[asset]
            print(f"  {k+1:<5} {asset:<8} {int(mask.sum()):>4} "
                  f"{mm['brier']:>8.4f} {mm['log_loss']:>8.4f} {mm['accuracy']:>6.2%} "
                  f"{mm['pct_bull_pred']:>11.2%} {mm['pct_bull_real']:>11.2%}")

    print("\n== Metricas agregadas (OOS) ==")
    metricas_oos = _metricas_split(p_oos, Y_oos, cfg)
    for asset in cfg.assets:
        m = metricas_oos[asset]
        print(f"  {asset:<8} brier={m['brier']:.4f}  logloss={m['log_loss']:.4f}  "
              f"acc={m['accuracy']:.2%}  "
              f"%bull_pred={m['pct_bull_pred']:.2%}  %bull_real={m['pct_bull_real']:.2%}")

    # ---- Baseline trivial vs modelo (sobre OOS) ----
    print("\n== Baseline trivial (p_bull constante = freq. bull en train fold 1) ==")
    for ai, asset in enumerate(cfg.assets):
        p_const = np.full_like(p_oos[:, ai], pct_bull_train[ai])
        y_oos_a = (Y_oos[:, ai] >= 0.0).astype(np.float32)
        print(
            f"  {asset:<8} p_const={pct_bull_train[ai]:.2%}  "
            f"OOS_brier_modelo={metricas_oos[asset]['brier']:.4f}  "
            f"OOS_brier_baseline={_brier(p_const, y_oos_a):.4f}  "
            f"(menor = mejor)"
        )

    # ---- Matriz de confusion sobre OOS ----
    print("\n== Matriz de confusion @ 0.5 (OOS agregada) ==")
    for ai, asset in enumerate(cfg.assets):
        y_oos_a = (Y_oos[:, ai] >= 0.0).astype(int)
        tp, tn, fp, fn = _confusion(p_oos[:, ai], y_oos_a)
        print(f"  {asset}:")
        print(f"    pred bull | real bull  TP={tp:>3}   real bear  FP={fp:>3}")
        print(f"    pred bear | real bull  FN={fn:>3}   real bear  TN={tn:>3}")

    # ---- Tabla cruda y CSV ----
    print("\n== Predicciones de regimen — OOS agregada ==")
    _print_tabla_predicciones(p_oos, Y_oos, t_oos, cfg)

    # CSV: una fila por t OOS, etiquetada con el fold que la valido.
    fold_label = np.array([f"fold{int(k)+1}" for k in fid])
    p_split_csv = {f: p_oos[fid == k] for k, f in enumerate(np.unique(fold_label))}
    y_split_csv = {f: Y_oos[fid == k] for k, f in enumerate(np.unique(fold_label))}
    t_split_csv = {f: t_oos[fid == k] for k, f in enumerate(np.unique(fold_label))}
    _guardar_csv_predicciones(
        p_splits=p_split_csv, y_splits=y_split_csv, t_splits=t_split_csv,
        cfg=cfg, out_path=out_dir / "predicciones_regimen.csv",
    )

    # ---- Graficos sobre OOS ----
    plot_serie_probabilidad(
        p_oos, Y_oos, t_oos, cfg, out_dir / "serie_p_bull_oos.png",
    )
    plot_reliability(p_oos, Y_oos, cfg, out_dir / "reliability.png", n_bins=5)
    # Histograma por fold (un panel por fold).
    p_por_fold = {f"fold{k+1}": p_oos[fid == k] for k in range(len(folds))
                  if (fid == k).any()}
    plot_histograma(p_por_fold, cfg, out_dir / "histograma.png")
    plot_scatter(p_oos, Y_oos, cfg, out_dir / "scatter_pbull_vs_retorno.png")


if __name__ == "__main__":
    inspeccionar()
