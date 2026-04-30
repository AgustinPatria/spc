"""E3 — p_bull real vs predicha (in-sample), con retornos reales.

Aplica el LSTM cuantilico ENTRENADO sobre ventanas historicas reales
(no rollout determinista) para producir p_bull_pred(t) en t=53..163.
Luego compara estadisticamente contra:

  - p_bull_real(t)  del clasificador previo (prob_*.csv)
  - ret_real(t)     del ret_semanal_*.csv

Bloques:
  1. Series temporales (un panel por activo, con eje twin para retorno).
  2. Distribuciones marginales (histogramas).
  3. p_bull_real vs p_bull_pred: scatter + correlacion + calibracion.
  4. Relacion p_bull vs ret_real(t): scatter + medias condicionales + hit-rate.
  5. Autocorrelacion / persistencia.

Outputs en este directorio: PNGs + metricas.csv + ts_alineadas.csv.

Uso:
    python experimentos/p_bull_plano/e3_real_vs_pred/experimento.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import ASSETS, BULL_THRESHOLD, CHECKPOINT_PATH, DATA_DIR, PROB_CSV
from dl.prediccion_deciles import build_windows, load_checkpoint, load_returns
from dl.regimen_predicted import regimen_probabilities


_OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Carga + alineacion
# ---------------------------------------------------------------------

def load_real_pbull(data_dir: Path) -> pd.DataFrame:
    """DataFrame con columnas <asset>_p_bull indexado por t."""
    out = None
    for a in ASSETS:
        df = pd.read_csv(Path(data_dir) / PROB_CSV[a])
        df.columns = [c.strip() for c in df.columns]
        df["t"] = df["t"].astype(int)
        df = df.rename(columns={"bull": f"{a}_p_bull"})[["t", f"{a}_p_bull"]]
        out = df if out is None else pd.merge(out, df, on="t")
    return out.sort_values("t").set_index("t")


def predict_pbull_in_sample(model, returns: pd.DataFrame):
    """Aplica el LSTM a ventanas historicas reales -> p_bull_pred[t, A].

    Returns: DataFrame con columnas <asset>_p_bull_pred, indexado por t.
    """
    H = model.config.H
    X, _, t_idx = build_windows(returns, H)
    p_bull, _   = regimen_probabilities(model, X)        # (N, A)
    cols = {f"{a}_p_bull_pred": p_bull[:, ai] for ai, a in enumerate(ASSETS)}
    return pd.DataFrame(cols, index=pd.Index(t_idx, name="t"))


# ---------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------

def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _accuracy(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    return float(np.mean((p > thr) == (y > 0.5)))


def _autocorr(x: np.ndarray, lag: int) -> float:
    n  = len(x)
    if lag >= n: return np.nan
    x0 = x[: n - lag] - x[: n - lag].mean()
    x1 = x[lag:]      - x[lag:].mean()
    den = np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2))
    return float(np.sum(x0 * x1) / den) if den > 0 else np.nan


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """df aligned: cols <asset>_p_bull, <asset>_p_bull_pred, <asset>_ret."""
    rows = []
    for a in ASSETS:
        p_pred = df[f"{a}_p_bull_pred"].to_numpy()
        p_real = df[f"{a}_p_bull"].to_numpy()
        r      = df[f"{a}_ret"].to_numpy()
        y_sign = (r >= BULL_THRESHOLD).astype(np.float32)

        # vs p_bull_real (continuo, "imitacion del clasificador previo")
        corr_pred_real = float(np.corrcoef(p_pred, p_real)[0, 1])
        mae_vs_real    = float(np.mean(np.abs(p_pred - p_real)))
        rmse_vs_real   = float(np.sqrt(np.mean((p_pred - p_real) ** 2)))

        # vs y_sign (binario, "skill predictivo del LSTM")
        brier_pred_sign = _brier(p_pred, y_sign)
        ll_pred_sign    = _log_loss(p_pred, y_sign)
        acc_pred_sign   = _accuracy(p_pred, y_sign)
        # baseline 1: p_bull_real como predictor del signo
        brier_real_sign = _brier(p_real, y_sign)
        ll_real_sign    = _log_loss(p_real, y_sign)
        acc_real_sign   = _accuracy(p_real, y_sign)
        # baseline 2: constante (mean(y_sign))
        p_const = float(y_sign.mean())
        brier_const = _brier(np.full_like(y_sign, p_const), y_sign)

        # correlacion con retorno realizado
        corr_pred_ret = float(np.corrcoef(p_pred, r)[0, 1])
        corr_real_ret = float(np.corrcoef(p_real, r)[0, 1])

        # rangos (ya muestra si la pred queda comprimida)
        rng = lambda x: f"[{x.min():.3f}, {x.max():.3f}]  std={x.std():.3f}"

        rows.append({
            "asset": a,
            "n":     len(df),
            "rng_pred":   rng(p_pred),
            "rng_real":   rng(p_real),
            "rng_ret":    rng(r),
            "freq_bull_realizado": p_const,
            "corr(pred, real)":    corr_pred_real,
            "MAE(pred, real)":     mae_vs_real,
            "RMSE(pred, real)":    rmse_vs_real,
            "Brier(pred, sign)":   brier_pred_sign,
            "Brier(real, sign)":   brier_real_sign,
            "Brier(const, sign)":  brier_const,
            "LogLoss(pred, sign)": ll_pred_sign,
            "LogLoss(real, sign)": ll_real_sign,
            "Acc(pred, sign)":     acc_pred_sign,
            "Acc(real, sign)":     acc_real_sign,
            "corr(pred, ret)":     corr_pred_ret,
            "corr(real, ret)":     corr_real_ret,
            "ACF1(pred)":          _autocorr(p_pred, 1),
            "ACF1(real)":          _autocorr(p_real, 1),
            "ACF1(ret)":           _autocorr(r, 1),
            "ACF5(pred)":          _autocorr(p_pred, 5),
            "ACF5(real)":          _autocorr(p_real, 5),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_time_series(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(11, 3.6 * len(ASSETS)),
                             sharex=True)
    if len(ASSETS) == 1: axes = [axes]
    for ai, a in enumerate(ASSETS):
        ax = axes[ai]
        ax.plot(df.index, df[f"{a}_p_bull"],
                label="p_bull real", color="#2E86AB", linewidth=1.4)
        ax.plot(df.index, df[f"{a}_p_bull_pred"],
                label="p_bull pred (in-sample)", color="#E63946", linewidth=1.4,
                marker=".", markersize=3)
        ax.set_ylabel("p_bull")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.bar(df.index, df[f"{a}_ret"],
                color="gray", alpha=0.35, width=1.0, label="ret real")
        ax2.set_ylabel("retorno semanal")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax.set_title(a)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle("Series temporales: p_bull real vs predicha (in-sample) + ret realizado")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_histograms(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(len(ASSETS), 3, figsize=(13, 3.4 * len(ASSETS)))
    if len(ASSETS) == 1: axes = axes[None, :]
    bins_p = np.linspace(0, 1, 21)
    for ai, a in enumerate(ASSETS):
        axes[ai, 0].hist(df[f"{a}_p_bull"], bins=bins_p,
                         color="#2E86AB", alpha=0.85, edgecolor="white")
        axes[ai, 0].set_title(f"{a} — p_bull real")
        axes[ai, 0].set_xlabel("p_bull"); axes[ai, 0].set_ylabel("freq")
        axes[ai, 0].grid(True, alpha=0.3)

        axes[ai, 1].hist(df[f"{a}_p_bull_pred"], bins=bins_p,
                         color="#E63946", alpha=0.85, edgecolor="white")
        axes[ai, 1].set_title(f"{a} — p_bull pred")
        axes[ai, 1].set_xlabel("p_bull"); axes[ai, 1].grid(True, alpha=0.3)

        axes[ai, 2].hist(df[f"{a}_ret"], bins=30,
                         color="gray", alpha=0.85, edgecolor="white")
        axes[ai, 2].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[ai, 2].set_title(f"{a} — ret real")
        axes[ai, 2].set_xlabel("retorno"); axes[ai, 2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_scatter_calibration(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(len(ASSETS), 2, figsize=(11, 4.2 * len(ASSETS)))
    if len(ASSETS) == 1: axes = axes[None, :]
    for ai, a in enumerate(ASSETS):
        p_pred = df[f"{a}_p_bull_pred"].to_numpy()
        p_real = df[f"{a}_p_bull"].to_numpy()

        # Scatter pred vs real
        ax = axes[ai, 0]
        ax.scatter(p_real, p_pred, s=22, alpha=0.55, color="#264653")
        slope, intercept = np.polyfit(p_real, p_pred, 1)
        x = np.linspace(0, 1, 100)
        ax.plot(x, slope * x + intercept, color="red", linewidth=1.2,
                label=f"y = {slope:.2f}x + {intercept:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
                label="y=x")
        corr = np.corrcoef(p_real, p_pred)[0, 1]
        ax.set_title(f"{a} — scatter (corr={corr:.2f})")
        ax.set_xlabel("p_bull real"); ax.set_ylabel("p_bull pred")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Calibracion: bin de p_pred -> mean(p_real)
        ax = axes[ai, 1]
        bins  = np.linspace(0, 1, 11)
        idx   = np.digitize(p_pred, bins) - 1
        idx   = np.clip(idx, 0, len(bins) - 2)
        means_real = []
        means_pred = []
        counts     = []
        for k in range(len(bins) - 1):
            mask = idx == k
            if mask.sum() == 0:
                means_real.append(np.nan); means_pred.append(np.nan); counts.append(0)
            else:
                means_real.append(p_real[mask].mean())
                means_pred.append(p_pred[mask].mean())
                counts.append(int(mask.sum()))
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
                label="diag")
        sizes = [max(c * 6, 8) for c in counts]
        ax.scatter(means_pred, means_real, s=sizes, color="#E63946",
                   alpha=0.85, edgecolor="white", label="bin (size = #obs)")
        ax.set_title(f"{a} — calibracion (bin pred -> mean real)")
        ax.set_xlabel("mean p_pred en bin"); ax.set_ylabel("mean p_real en bin")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_pbull_vs_ret(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(len(ASSETS), 2, figsize=(11, 4.2 * len(ASSETS)))
    if len(ASSETS) == 1: axes = axes[None, :]
    for ai, a in enumerate(ASSETS):
        r = df[f"{a}_ret"].to_numpy()
        for j, kind in enumerate(["p_bull", "p_bull_pred"]):
            ax = axes[ai, j]
            p = df[f"{a}_{kind}"].to_numpy()
            ax.scatter(p, r, s=22, alpha=0.55,
                       color="#2E86AB" if kind == "p_bull" else "#E63946")
            corr = np.corrcoef(p, r)[0, 1]
            # medias condicionales por bin
            bins = np.linspace(0, 1, 6)  # 5 bins
            idx  = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
            xs, ys = [], []
            for k in range(len(bins) - 1):
                mask = idx == k
                if mask.sum() > 0:
                    xs.append(0.5 * (bins[k] + bins[k + 1]))
                    ys.append(r[mask].mean())
            ax.plot(xs, ys, color="black", marker="o", linewidth=1.5,
                    label="E[ret | bin]")
            ax.axhline(0, color="gray", linewidth=0.5)
            label = "real" if kind == "p_bull" else "pred"
            ax.set_title(f"{a} — {label} vs ret  (corr={corr:.2f})")
            ax.set_xlabel(f"p_bull {label}"); ax.set_ylabel("ret")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_acf(df: pd.DataFrame, out_path: Path, max_lag: int = 12) -> Path:
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(10, 3.5 * len(ASSETS)))
    if len(ASSETS) == 1: axes = [axes]
    lags = np.arange(1, max_lag + 1)
    for ai, a in enumerate(ASSETS):
        ax  = axes[ai]
        for col, color, lab in [
            (f"{a}_p_bull",      "#2E86AB", "p_bull real"),
            (f"{a}_p_bull_pred", "#E63946", "p_bull pred"),
            (f"{a}_ret",         "gray",    "ret real"),
        ]:
            x   = df[col].to_numpy()
            acf = [_autocorr(x, k) for k in lags]
            ax.plot(lags, acf, marker="o", label=lab, color=color, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{a} — ACF")
        ax.set_xlabel("lag"); ax.set_ylabel("autocorr")
        ax.set_ylim(-1, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Cargando datos ...")
    returns      = load_returns(DATA_DIR)              # index t, cols=ASSETS
    p_real_df    = load_real_pbull(DATA_DIR)           # index t, cols=<a>_p_bull
    print(f"  retornos: T={len(returns)}, columnas={list(returns.columns)}")
    print(f"  p_bull_real: T={len(p_real_df)}, columnas={list(p_real_df.columns)}")

    print("Cargando LSTM y prediciendo in-sample ...")
    model     = load_checkpoint(CHECKPOINT_PATH)
    p_pred_df = predict_pbull_in_sample(model, returns)
    print(f"  p_bull_pred: rango t = [{p_pred_df.index.min()}, {p_pred_df.index.max()}], "
          f"N = {len(p_pred_df)}  (limitado por H={model.config.H})")

    # Alinear (solo donde tenemos pred + real + ret)
    df = p_real_df.join(p_pred_df, how="inner")
    df = df.join(returns.rename(columns={a: f"{a}_ret" for a in ASSETS}),
                 how="inner")
    print(f"  alineado: T={len(df)}  (t {df.index.min()}..{df.index.max()})")

    df.to_csv(_OUT_DIR / "ts_alineadas.csv")
    print(f"  CSV alineado: {_OUT_DIR / 'ts_alineadas.csv'}")

    # Metricas
    metrics = compute_all_metrics(df)
    metrics_path = _OUT_DIR / "metricas.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\n  Metricas: {metrics_path}\n")
    # Print compacto pero completo
    for _, row in metrics.iterrows():
        print(f"  ===== {row['asset']} (n={row['n']}) =====")
        for k, v in row.items():
            if k == "asset" or k == "n": continue
            if isinstance(v, float):
                print(f"    {k:<24s} {v:.4f}")
            else:
                print(f"    {k:<24s} {v}")
        print()

    # Plots
    p1 = plot_time_series(df,         _OUT_DIR / "time_series.png")
    p2 = plot_histograms(df,          _OUT_DIR / "histogramas.png")
    p3 = plot_scatter_calibration(df, _OUT_DIR / "pred_vs_real.png")
    p4 = plot_pbull_vs_ret(df,        _OUT_DIR / "pbull_vs_ret.png")
    p5 = plot_acf(df,                 _OUT_DIR / "acf.png")
    print(f"  PNGs:")
    for p in [p1, p2, p3, p4, p5]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
