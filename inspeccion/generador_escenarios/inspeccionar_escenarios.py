"""Inspección del generador de escenarios (PDF sección 2.5).

Uso:
    python inspeccion/generador_escenarios/inspeccionar_escenarios.py

Carga el checkpoint en `models/decile_predictor.pt`, toma la última ventana
observada (los H retornos más recientes) como punto de partida, genera N
trayectorias candidatas de largo T, las reduce a n_quintiles representativos
y produce:

1. Resumen de la ventana inicial (t de inicio, H, medias por activo).
2. Estadísticas de los N candidatos:
   - Retorno acumulado del activo resumen (media, std, min, max, quintiles).
   - Por activo, media/std de la trayectoria puntual (paso a paso).
   - Fracción de trayectorias "bull" (retorno acumulado > 0) por activo.
3. Estadísticas de los n_quintiles representativos (retorno acumulado por
   quintil, por activo).
4. Chequeo de reproducibilidad: dos corridas con la misma seed deben coincidir.
5. Figuras:
   - Fan chart de candidatos (bandas por quantiles sobre t) por activo.
   - Trayectorias acumuladas de los representativos por activo.
   - Histograma del retorno acumulado terminal por activo.
   - Scatter 2-D (retorno acumulado SPX vs CMC200) con representativos marcados.

Todas las figuras se guardan junto a este script en
`inspeccion/generador_escenarios/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Raíz del proyecto dos niveles arriba: .../SPC_Grid3/inspeccion/generador_escenarios/<archivo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHECKPOINT_PATH,
    MODELS_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
    DLConfig,
)
from dl.generador_escenarios import (
    generate_candidate_scenarios,
    reduce_to_representatives,
)
from dl.prediccion_deciles import (
    LoadedModel,
    build_windows,
    load_checkpoint,
    load_returns,
)


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def _cumret(traj: np.ndarray) -> np.ndarray:
    """Retorno acumulado a lo largo del tiempo. traj: (..., T, A) -> (..., T, A)."""
    return np.cumprod(1.0 + traj, axis=-2) - 1.0


def _terminal_cumret(traj: np.ndarray) -> np.ndarray:
    """Retorno acumulado terminal. traj: (..., T, A) -> (..., A)."""
    return np.prod(1.0 + traj, axis=-2) - 1.0


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------

def resumen_candidatos(
    scenarios: np.ndarray, summary_asset_idx: int, cfg: DLConfig,
) -> Dict[str, Dict[str, float]]:
    """Retorno acumulado terminal por activo: media/std/min/max/quintiles."""
    term = _terminal_cumret(scenarios)                       # (N, A)
    out: Dict[str, Dict[str, float]] = {}
    for ai, asset in enumerate(cfg.assets):
        col = term[:, ai]
        out[asset] = {
            "mean":   float(col.mean()),
            "std":    float(col.std()),
            "min":    float(col.min()),
            "q20":    float(np.quantile(col, 0.20)),
            "q40":    float(np.quantile(col, 0.40)),
            "q60":    float(np.quantile(col, 0.60)),
            "q80":    float(np.quantile(col, 0.80)),
            "max":    float(col.max()),
            "p_bull": float((col > 0.0).mean()),
            "resumen": float(ai == summary_asset_idx),
        }
    return out


def resumen_representativos(
    reps: np.ndarray, cfg: DLConfig,
) -> Dict[str, Dict[str, float]]:
    """Retorno acumulado terminal por quintil y por activo."""
    term = _terminal_cumret(reps)                            # (n_q, A)
    out: Dict[str, Dict[str, float]] = {}
    for k in range(reps.shape[0]):
        out[f"quintil_{k+1}"] = {
            asset: float(term[k, ai]) for ai, asset in enumerate(cfg.assets)
        }
    return out


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------

def plot_fan_candidatos(
    scenarios: np.ndarray, cfg: DLConfig, out_path: Path,
    quantiles=(0.05, 0.25, 0.50, 0.75, 0.95),
) -> None:
    """Bandas de quantiles sobre t para los N candidatos, por activo."""
    cum = _cumret(scenarios)                                 # (N, T, A)
    T   = cum.shape[1]
    tx  = np.arange(1, T + 1)

    fig, axes = plt.subplots(
        cfg.n_assets, 1, figsize=(10, 3 * cfg.n_assets),
        sharex=True, squeeze=False,
    )
    cmap = plt.get_cmap("Blues")
    for ai, asset in enumerate(cfg.assets):
        ax  = axes[ai][0]
        qs  = np.quantile(cum[:, :, ai], quantiles, axis=0)  # (Q, T)
        n_q = len(quantiles)
        # Bandas pareadas (extremo <-> extremo) hacia la mediana.
        for qi in range(n_q // 2):
            lo_idx, hi_idx = qi, n_q - 1 - qi
            ax.fill_between(
                tx, qs[lo_idx], qs[hi_idx],
                color=cmap(0.3 + 0.5 * qi / max(n_q // 2, 1)),
                alpha=0.35, linewidth=0,
                label=f"q{int(quantiles[lo_idx]*100):02d}–q{int(quantiles[hi_idx]*100):02d}",
            )
        if n_q % 2 == 1:
            ax.plot(tx, qs[n_q // 2], color="#1f3b73", linewidth=1.2,
                    label=f"q{int(quantiles[n_q // 2]*100):02d} (mediana)")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.set_title(f"Retorno acumulado de candidatos — {asset}")
        ax.set_ylabel("retorno acumulado")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("paso t del horizonte")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] fan chart de candidatos guardado en: {out_path}")


def plot_representativos(
    reps: np.ndarray, cfg: DLConfig, out_path: Path,
) -> None:
    """Trayectorias acumuladas de los n_quintiles representativos, por activo."""
    cum = _cumret(reps)                                      # (n_q, T, A)
    n_q, T, _ = cum.shape
    tx  = np.arange(1, T + 1)

    fig, axes = plt.subplots(
        cfg.n_assets, 1, figsize=(10, 3 * cfg.n_assets),
        sharex=True, squeeze=False,
    )
    cmap = plt.get_cmap("RdYlGn")
    for ai, asset in enumerate(cfg.assets):
        ax = axes[ai][0]
        for k in range(n_q):
            color = cmap(k / max(n_q - 1, 1))
            ax.plot(tx, cum[k, :, ai], color=color, linewidth=1.4,
                    label=f"quintil {k+1}  (term={cum[k, -1, ai]:+.2%})")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.set_title(f"Escenarios representativos — {asset}")
        ax.set_ylabel("retorno acumulado")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1][0].set_xlabel("paso t del horizonte")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] trayectorias representativas guardadas en: {out_path}")


def plot_histograma_terminal(
    scenarios: np.ndarray, reps: np.ndarray, cfg: DLConfig, out_path: Path,
) -> None:
    """Histograma del retorno acumulado terminal por activo; marca representativos."""
    term_cand = _terminal_cumret(scenarios)                  # (N, A)
    term_reps = _terminal_cumret(reps)                       # (n_q, A)

    fig, axes = plt.subplots(
        1, cfg.n_assets, figsize=(5 * cfg.n_assets, 4), squeeze=False,
    )
    cmap = plt.get_cmap("RdYlGn")
    for ai, asset in enumerate(cfg.assets):
        ax = axes[0][ai]
        ax.hist(
            term_cand[:, ai], bins=40,
            color="#1f3b73", alpha=0.75, edgecolor="white",
            label="candidatos",
        )
        for k in range(term_reps.shape[0]):
            ax.axvline(
                term_reps[k, ai], color=cmap(k / max(term_reps.shape[0] - 1, 1)),
                linestyle="--", linewidth=1.6,
                label=f"q{k+1}={term_reps[k, ai]:+.2%}",
            )
        ax.axvline(0.0, color="grey", linewidth=0.6)
        ax.set_title(f"Retorno acumulado terminal — {asset}")
        ax.set_xlabel("retorno acumulado")
        ax.set_ylabel("frecuencia")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] histograma terminal guardado en: {out_path}")


def plot_scatter_terminal(
    scenarios: np.ndarray, reps: np.ndarray, cfg: DLConfig, out_path: Path,
) -> None:
    """Scatter retorno acumulado terminal entre los primeros dos activos."""
    if cfg.n_assets < 2:
        return
    term_cand = _terminal_cumret(scenarios)                  # (N, A)
    term_reps = _terminal_cumret(reps)                       # (n_q, A)
    a0, a1 = cfg.assets[0], cfg.assets[1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        term_cand[:, 0], term_cand[:, 1],
        s=10, alpha=0.4, color="#1f3b73", label="candidatos",
    )
    cmap = plt.get_cmap("RdYlGn")
    for k in range(term_reps.shape[0]):
        ax.scatter(
            term_reps[k, 0], term_reps[k, 1],
            s=120, color=cmap(k / max(term_reps.shape[0] - 1, 1)),
            edgecolor="black", linewidth=1.0, zorder=3,
            label=f"quintil {k+1}",
        )
    ax.axhline(0.0, color="grey", linewidth=0.6)
    ax.axvline(0.0, color="grey", linewidth=0.6)
    ax.set_xlabel(f"retorno acumulado {a0}")
    ax.set_ylabel(f"retorno acumulado {a1}")
    ax.set_title("Terminal candidatos vs representativos")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] scatter terminal guardado en: {out_path}")


# ---------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------

def _guardar_csv_representativos(
    reps: np.ndarray, cfg: DLConfig, out_path: Path,
) -> None:
    """Exporta las n_q trayectorias representativas a CSV (una fila por (k, t))."""
    n_q, T, _ = reps.shape
    cum = _cumret(reps)
    cols = ["quintil", "t"]
    for asset in cfg.assets:
        cols += [f"{asset}_ret", f"{asset}_cumret"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for k in range(n_q):
            for t in range(T):
                fila = [str(k + 1), str(t + 1)]
                for ai, _ in enumerate(cfg.assets):
                    fila += [f"{reps[k, t, ai]:.6f}", f"{cum[k, t, ai]:.6f}"]
                f.write(",".join(fila) + "\n")
    print(f"[csv] representativos guardados en: {out_path}")


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

OUT_DIR: Path = Path(__file__).resolve().parent


def _print_resumen_candidatos(
    resumen: Dict[str, Dict[str, float]], summary_asset: str,
) -> None:
    print("\n== Retorno acumulado terminal — candidatos ==")
    header = (
        f"  {'activo':<8} {'mean':>8} {'std':>8} {'min':>8} "
        f"{'q20':>8} {'q40':>8} {'q60':>8} {'q80':>8} {'max':>8} {'p_bull':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asset, m in resumen.items():
        tag = "  <-- resumen" if asset == summary_asset else ""
        print(
            f"  {asset:<8} {m['mean']:>+8.2%} {m['std']:>8.2%} {m['min']:>+8.2%} "
            f"{m['q20']:>+8.2%} {m['q40']:>+8.2%} {m['q60']:>+8.2%} "
            f"{m['q80']:>+8.2%} {m['max']:>+8.2%} {m['p_bull']:>8.2%}{tag}"
        )


def _print_resumen_representativos(
    reps: np.ndarray, cfg: DLConfig, summary_asset_idx: int,
    label: str = "representativos",
) -> None:
    term = _terminal_cumret(reps)                            # (n_q, A)
    print(f"\n== Retorno acumulado terminal — {label} ==")
    header = f"  {'quintil':<8} " + " ".join(f"{a:>10}" for a in cfg.assets)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(reps.shape[0]):
        vals = "  ".join(f"{term[k, ai]:>+8.2%}" for ai, _ in enumerate(cfg.assets))
        tag  = f"  (ordenado por {cfg.assets[summary_asset_idx]})"
        suf  = tag if k == 0 else ""
        print(f"  q{k+1:<7} {vals}{suf}")


def inspeccionar(
    ckpt_path: Path = CHECKPOINT_PATH,
    out_dir: Path = OUT_DIR,
    N: int = N_CANDIDATES,
    T: int = T_HORIZON,
    n_quintiles: int = N_SCENARIOS,
    summary_asset: str = SUMMARY_ASSET,
    seed: int = SCENARIO_SEED,
) -> None:
    print(f"[ckpt] cargando {ckpt_path}")
    model: LoadedModel = load_checkpoint(ckpt_path)
    cfg = model.config

    # Ventana inicial = últimos H retornos observados.
    df_ret = load_returns()
    X, _, t_idx = build_windows(df_ret, cfg.H)
    initial_window = df_ret.to_numpy(dtype=np.float32)[-cfg.H:]
    t_start = int(df_ret.index.to_numpy()[-1])
    assets = tuple(cfg.assets)
    if summary_asset not in assets:
        raise ValueError(f"summary_asset {summary_asset!r} no está en {assets}")
    summary_idx = assets.index(summary_asset)

    print(f"[cfg]  H={cfg.H}  assets={cfg.assets}  deciles={cfg.quantiles}")
    print(f"[data] ventana inicial termina en t={t_start}  "
          f"(T_ventanas_entrenamiento={len(X)})")
    print(f"[data] media por activo en la ventana inicial: "
          + "  ".join(f"{a}={initial_window[:, ai].mean():+.4f}"
                      for ai, a in enumerate(cfg.assets)))
    print(f"[gen]  N={N}  T={T}  n_quintiles={n_quintiles}  "
          f"summary_asset={summary_asset}  seed={seed}")

    # ---- Generar candidatos y reducir ----
    candidates = generate_candidate_scenarios(
        model, initial_window, N=N, T=T, seed=seed,
    )                                                         # (N, T, A)
    reps = reduce_to_representatives(
        candidates, summary_asset_idx=summary_idx, n_quintiles=n_quintiles,
        position="min",
    )                                                         # (n_q, T, A)
    # Comparativa contra el default del PDF (mediano del quintil).
    reps_median = reduce_to_representatives(
        candidates, summary_asset_idx=summary_idx, n_quintiles=n_quintiles,
        position="median",
    )

    # ---- Estadísticas de candidatos ----
    resumen = resumen_candidatos(candidates, summary_idx, cfg)
    _print_resumen_candidatos(resumen, summary_asset)

    # ---- Estadísticas por paso del activo resumen ----
    cum_summary = _cumret(candidates)[:, :, summary_idx]     # (N, T)
    print(f"\n== Retorno acumulado de {summary_asset} por paso (bandas) ==")
    print(f"  {'t':>4}  {'q05':>8} {'q25':>8} {'q50':>8} {'q75':>8} {'q95':>8}")
    for t in (0, T // 4, T // 2, 3 * T // 4, T - 1):
        col = cum_summary[:, t]
        q05, q25, q50, q75, q95 = np.quantile(col, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"  {t+1:>4}  {q05:>+8.2%} {q25:>+8.2%} {q50:>+8.2%} "
              f"{q75:>+8.2%} {q95:>+8.2%}")

    # ---- Representativos: comparativa min vs median ----
    _print_resumen_representativos(reps_median, cfg, summary_idx,
                                   label="representativos (median del quintil, default PDF)")
    _print_resumen_representativos(reps,        cfg, summary_idx,
                                   label="representativos (min del quintil, MAS PESIMISTA)")

    # ---- Reproducibilidad ----
    cand2 = generate_candidate_scenarios(
        model, initial_window, N=N, T=T, seed=seed,
    )
    equal = np.allclose(candidates, cand2)
    print(f"\n[check] reproducibilidad seed={seed}: {'OK' if equal else 'FALLA'} "
          f"(max|diff|={float(np.max(np.abs(candidates - cand2))):.2e})")

    # ---- CSV + figuras ----
    out_dir = Path(out_dir)
    _guardar_csv_representativos(reps, cfg, out_dir / "escenarios_representativos.csv")
    plot_fan_candidatos(candidates, cfg, out_dir / "fan_candidatos.png")
    plot_representativos(reps, cfg, out_dir / "representativos.png")
    plot_histograma_terminal(candidates, reps, cfg, out_dir / "histograma_terminal.png")
    plot_scatter_terminal(candidates, reps, cfg, out_dir / "scatter_terminal.png")


if __name__ == "__main__":
    inspeccionar()
