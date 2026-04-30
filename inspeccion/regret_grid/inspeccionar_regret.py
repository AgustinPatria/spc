"""Inspección del regret-grid (PDF sección 3).

Uso:
    python inspeccion/regret_grid/inspeccionar_regret.py

Construye el contexto DL (mu_hat/sigma_hat históricos + p(t) forward por
walking-window sobre el histórico + 5 escenarios representativos), corre el regret-grid sobre
G = Λ × M (5·3 = 15 puntos) y produce:

1. Resumen de contexto DL:
   - p_bull(t) forward por activo: min/mean/max.
   - Retorno acumulado terminal de cada escenario representativo por activo.
2. Tabla V[g, s] — capital terminal por política y escenario.
3. Tabla R[g, s] = V_best_s − V[g, s] (non-negative, un cero por columna).
4. Resumen por g: mean_regret y worst_regret.
5. Selección g*_mean (ec. 23) y g*_worst (ec. 24).
6. Turnover acumulado (Σ_t Σ_i (u+v)) por política.
7. Figuras:
   - p_bull(t) forward por activo.
   - Retorno acumulado de los 5 escenarios por activo.
   - Heatmap de regret promedio sobre G = Λ × M.
   - Heatmap de regret peor caso sobre G = Λ × M.
   - Pesos w(i, t) bajo g*_mean y g*_worst.
   - Capital por escenario bajo g*_mean y g*_worst.

Todas las salidas (CSVs y PNGs) se guardan junto a este script en
`inspeccion/regret_grid/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Raíz del proyecto dos niveles arriba: .../SPC_Grid3/inspeccion/regret_grid/<archivo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    T_HORIZON,
)
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    run_regret_grid,
    simulate_capital_on_scenario,
)


OUT_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------

def resumen_escenarios(scenarios: np.ndarray, assets) -> pd.DataFrame:
    """Retorno acumulado terminal por escenario y activo."""
    term = np.prod(1.0 + scenarios, axis=1) - 1.0          # (S, A)
    rows = []
    for s in range(scenarios.shape[0]):
        fila = {"escenario": f"s{s + 1}"}
        for ai, asset in enumerate(assets):
            fila[asset] = float(term[s, ai])
        rows.append(fila)
    return pd.DataFrame(rows).set_index("escenario")


def turnover_por_politica(policies, assets, T_vals) -> pd.DataFrame:
    """Σ_t Σ_i (u(i,t) + v(i,t)) como proxy de actividad de la política."""
    rows = []
    for (lam, cm), (_w, u, v, z) in policies.items():
        total = sum(u[i, t] + v[i, t] for i in assets for t in T_vals)
        rows.append({"lambda": lam, "m": cm, "z": z, "turnover": float(total)})
    return pd.DataFrame(rows).set_index(["lambda", "m"])


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------

def plot_p_bull_forward(p_dl, T_vals, out_path: Path) -> None:
    """Serie p_bull(t) forward por activo (walking-window sobre el histórico)."""
    assets = list(p_dl.keys())
    fig, axes = plt.subplots(
        len(assets), 1, figsize=(10, 3 * len(assets)),
        sharex=True, squeeze=False,
    )
    for ai, asset in enumerate(assets):
        ax = axes[ai][0]
        serie = p_dl[asset]["bull"].values
        ax.plot(T_vals, serie, color="#1f3b73", linewidth=1.3, label="p_bull")
        ax.fill_between(T_vals, 0.0, serie, color="#1f3b73", alpha=0.15)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"p_bull(t) forward — {asset}  "
                     f"(min={serie.min():.3f}  mean={serie.mean():.3f}  "
                     f"max={serie.max():.3f})")
        ax.set_ylabel("prob. bull")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1][0].set_xlabel("t (forward)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] p_bull forward guardado en: {out_path}")


def plot_scenarios_cumret(scenarios: np.ndarray, assets, out_path: Path) -> None:
    """Retorno acumulado de los 5 escenarios representativos por activo."""
    cum = np.cumprod(1.0 + scenarios, axis=1) - 1.0        # (S, T, A)
    S, T, A = cum.shape
    tx = np.arange(1, T + 1)

    fig, axes = plt.subplots(
        A, 1, figsize=(10, 3 * A), sharex=True, squeeze=False,
    )
    cmap = plt.get_cmap("RdYlGn")
    for ai, asset in enumerate(assets):
        ax = axes[ai][0]
        for s in range(S):
            ax.plot(tx, cum[s, :, ai],
                    color=cmap(s / max(S - 1, 1)),
                    linewidth=1.4,
                    label=f"s{s + 1}  (term={cum[s, -1, ai]:+.2%})")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.set_title(f"Escenarios representativos — {asset}")
        ax.set_ylabel("retorno acumulado")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1][0].set_xlabel("t (forward)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] escenarios representativos guardados en: {out_path}")


def plot_regret_heatmaps(
    regret_summary: pd.DataFrame, g_mean, g_worst, out_path: Path,
) -> None:
    """Heatmaps de mean_regret y worst_regret sobre Λ × M."""
    mean_tbl = regret_summary["mean_regret"].unstack("m")
    worst_tbl = regret_summary["worst_regret"].unstack("m")

    # Orden ascendente por legibilidad.
    mean_tbl = mean_tbl.sort_index(axis=0).sort_index(axis=1)
    worst_tbl = worst_tbl.sort_index(axis=0).sort_index(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tbl, title, g_star in [
        (axes[0], mean_tbl, "mean_regret(g)",  g_mean),
        (axes[1], worst_tbl, "worst_regret(g)", g_worst),
    ]:
        im = ax.imshow(tbl.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(tbl.columns)))
        ax.set_xticklabels([f"m={c:.1f}" for c in tbl.columns])
        ax.set_yticks(range(len(tbl.index)))
        ax.set_yticklabels([f"λ={r:.2f}" for r in tbl.index])
        for i in range(len(tbl.index)):
            for j in range(len(tbl.columns)):
                ax.text(j, i, f"${tbl.values[i, j]:,.0f}",
                        ha="center", va="center",
                        color="white", fontsize=8)
        lam_s, m_s = g_star
        i_star = list(tbl.index).index(lam_s)
        j_star = list(tbl.columns).index(m_s)
        ax.add_patch(plt.Rectangle(
            (j_star - 0.5, i_star - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2.5,
        ))
        ax.set_title(f"{title}   g* = (λ={lam_s:.2f}, m={m_s:.1f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] heatmaps de regret guardados en: {out_path}")


def plot_weights(
    w_sol, assets, T_vals, title: str, out_path: Path,
) -> None:
    """Pesos w(i, t) en área apilada a lo largo del horizonte."""
    W = np.array([[w_sol[i, t] for t in T_vals] for i in assets])   # (A, T)
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab10")
    ax.stackplot(
        T_vals, W,
        labels=assets,
        colors=[cmap(ai) for ai in range(len(assets))],
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("t (forward)")
    ax.set_ylabel("peso w(i, t)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] pesos guardados en: {out_path}")


def plot_capital_por_escenario(
    w_sol, u_sol, v_sol, ctx, title: str, out_path: Path,
) -> None:
    """Capital por escenario bajo una política."""
    assets    = ctx["assets"]
    T_vals    = ctx["T_vals"]
    c_base    = ctx["c_base"]
    C0        = ctx["Capital_inicial"]
    scenarios = ctx["scenarios"]
    S         = scenarios.shape[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("RdYlGn")
    finales = []
    for s in range(S):
        cap = simulate_capital_on_scenario(
            w_sol, u_sol, v_sol, scenarios[s],
            assets, c_base, C0, T_vals,
        )
        serie = [cap[t] for t in T_vals]
        finales.append(serie[-1])
        ax.plot(T_vals, serie,
                color=cmap(s / max(S - 1, 1)),
                linewidth=1.4,
                label=f"s{s + 1}  (V=${serie[-1]:,.0f})")
    ax.axhline(C0, color="#666", linestyle="--", linewidth=0.8,
               label=f"Capital inicial (${C0:,.0f})")
    ax.set_title(f"{title}\nV promedio=${np.mean(finales):,.0f}  "
                 f"V peor=${np.min(finales):,.0f}")
    ax.set_xlabel("t (forward)")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] capital por escenario guardado en: {out_path}")


# ---------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------

def _guardar_p_bull_forward(p_dl, T_vals, out_path: Path) -> None:
    """Exporta p_bull(t)/p_bear(t) por activo a CSV."""
    df = pd.DataFrame({"t": T_vals})
    for asset, tbl in p_dl.items():
        df[f"{asset}_p_bull"] = tbl["bull"].values
        df[f"{asset}_p_bear"] = tbl["bear"].values
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[csv] p_bull forward guardado en: {out_path}")


# ---------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------

def _print_resumen_escenarios(df: pd.DataFrame) -> None:
    print("\n== Retorno acumulado terminal — escenarios representativos ==")
    header = f"  {'escenario':<10} " + "  ".join(f"{c:>10}" for c in df.columns)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, row in df.iterrows():
        vals = "  ".join(f"{row[c]:>+10.2%}" for c in df.columns)
        print(f"  {idx:<10} {vals}")


def _print_tabla_V(V_table: pd.DataFrame) -> None:
    print("\n== V[g, s] — capital terminal por (lambda, m) y escenario ==")
    print(V_table.to_string(float_format="${:,.2f}".format))


def _print_tabla_R(R_table: pd.DataFrame) -> None:
    print("\n== R[g, s] = V_best_s - V[g, s] ==")
    print(R_table.to_string(float_format="${:,.2f}".format))
    min_por_s = R_table.min(axis=0)
    print("\n  Mínimo de R por escenario (debería ser 0 en algún g):")
    for s, val in min_por_s.items():
        print(f"    s={s}: min_g R = ${val:,.2f}")


def _print_resumen_regret(summary: pd.DataFrame, g_mean, g_worst,
                          g_mean_metric, g_worst_metric,
                          V_table: pd.DataFrame, C0: float) -> None:
    print("\n== Resumen de regret por g ==")
    print(summary.to_string(float_format="${:,.2f}".format))

    lam_m, m_m = g_mean
    lam_w, m_w = g_worst
    V_mean  = V_table.loc[(lam_m, m_m)]
    V_worst = V_table.loc[(lam_w, m_w)]

    print("\n== Seleccion de g* ==")
    print(f"  g*_mean  (ec. 23): lambda={lam_m:.2f}  m={m_m:.1f}  "
          f"mean_regret=${g_mean_metric:,.2f}")
    print(f"      V: mean=${V_mean.mean():>12,.2f}  "
          f"worst=${V_mean.min():>12,.2f}  "
          f"best=${V_mean.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno promedio sobre escenarios = {V_mean.mean()/C0 - 1:+.2%}")

    print(f"  g*_worst (ec. 24): lambda={lam_w:.2f}  m={m_w:.1f}  "
          f"worst_regret=${g_worst_metric:,.2f}")
    print(f"      V: mean=${V_worst.mean():>12,.2f}  "
          f"worst=${V_worst.min():>12,.2f}  "
          f"best=${V_worst.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno en el peor escenario     = {V_worst.min()/C0 - 1:+.2%}")

    if g_mean == g_worst:
        print("  [nota] g*_mean == g*_worst: una misma politica domina "
              "en promedio y en el peor caso.")
    else:
        print("  [nota] g*_mean != g*_worst: trade-off entre promedio y "
              "peor caso.")


def _print_turnover(tbl: pd.DataFrame) -> None:
    print("\n== Turnover acumulado por politica (sum_t sum_i (u+v)) ==")
    print(tbl.to_string(float_format="{:.6f}".format))


def _check_sanity(R_table: pd.DataFrame, V_table: pd.DataFrame) -> None:
    print("\n== Sanity checks ==")
    neg = (R_table.values < -1e-9).sum()
    print(f"  R[g, s] >= 0 en todas las celdas: "
          f"{'OK' if neg == 0 else f'FALLA ({neg} negativos)'}")
    ceros_por_s = (R_table.abs() < 1e-6).any(axis=0)
    ok_ceros = bool(ceros_por_s.all())
    print(f"  Al menos un cero por columna s (g que logra V_best_s): "
          f"{'OK' if ok_ceros else 'FALLA'}")
    pos = (V_table.values > 0).all()
    print(f"  V[g, s] > 0 en todas las celdas: "
          f"{'OK' if pos else 'FALLA (capital no-positivo)'}")


def inspeccionar(
    ckpt_path: Path = CHECKPOINT_PATH,
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    T: int = T_HORIZON,
    N_candidates: int = N_CANDIDATES,
    n_scenarios: int = N_SCENARIOS,
    lambda_grid: Tuple[float, ...] = LAMBDA_GRID,
    m_grid: Tuple[float, ...] = M_GRID,
    seed: int = SCENARIO_SEED,
) -> None:
    print(f"[ckpt] cargando {ckpt_path}")
    print(f"[data] {data_dir}")
    print(f"[cfg]  T={T}  N_candidates={N_candidates}  n_scenarios={n_scenarios}  "
          f"seed={seed}")
    print(f"[grid] lambda={list(lambda_grid)}  m={list(m_grid)}  "
          f"-> {len(lambda_grid) * len(m_grid)} puntos")

    # ---- Contexto DL ----
    ctx = build_dl_context(
        data_dir=data_dir,
        checkpoint_path=ckpt_path,
        T=T,
        N_candidates=N_candidates,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    assets    = ctx["assets"]
    T_vals    = ctx["T_vals"]
    scenarios = ctx["scenarios"]
    p_dl      = ctx["p_dl"]

    print(f"\n[ctx]  assets={assets}  T={ctx['nT']}  "
          f"scenarios={scenarios.shape}")

    print("\n== p_bull(t) forward (walking-window) ==")
    for i in assets:
        col = p_dl[i]["bull"]
        print(f"  {i:<8} min={col.min():.3f}  mean={col.mean():.3f}  "
              f"max={col.max():.3f}  std={col.std():.3f}")

    df_esc = resumen_escenarios(scenarios, assets)
    _print_resumen_escenarios(df_esc)

    # ---- Regret-grid ----
    theta = {a: 1.0 for a in assets}
    print("\n" + "-" * 70)
    print("Corriendo regret-grid")
    print("-" * 70)
    V_df, policies = run_regret_grid(ctx, list(lambda_grid), list(m_grid), theta)

    res = compute_regret_and_select(V_df)
    _print_tabla_V(res["V_table"])
    _print_tabla_R(res["R_table"])
    _print_resumen_regret(
        res["regret_summary"], res["g_mean"], res["g_worst"],
        res["g_mean_metric"], res["g_worst_metric"],
        res["V_table"], ctx["Capital_inicial"],
    )

    turnover = turnover_por_politica(policies, assets, T_vals)
    _print_turnover(turnover)

    _check_sanity(res["R_table"], res["V_table"])

    # ---- CSV ----
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_esc.to_csv(out_dir / "resumen_escenarios.csv", float_format="%.6f")
    print(f"\n[csv] resumen escenarios guardado en: "
          f"{out_dir / 'resumen_escenarios.csv'}")

    _guardar_p_bull_forward(p_dl, T_vals, out_dir / "p_bull_forward.csv")

    res["V_table"].to_csv(out_dir / "tabla_V.csv", float_format="%.6f")
    print(f"[csv] tabla V guardada en: {out_dir / 'tabla_V.csv'}")

    res["R_table"].to_csv(out_dir / "tabla_R.csv", float_format="%.6f")
    print(f"[csv] tabla R guardada en: {out_dir / 'tabla_R.csv'}")

    res["regret_summary"].to_csv(out_dir / "resumen_regret.csv",
                                 float_format="%.6f")
    print(f"[csv] resumen regret guardado en: "
          f"{out_dir / 'resumen_regret.csv'}")

    turnover.to_csv(out_dir / "turnover_por_politica.csv", float_format="%.6f")
    print(f"[csv] turnover guardado en: "
          f"{out_dir / 'turnover_por_politica.csv'}")

    # ---- Figuras ----
    plot_p_bull_forward(p_dl, T_vals, out_dir / "p_bull_forward.png")
    plot_scenarios_cumret(scenarios, assets, out_dir / "escenarios_cumret.png")
    plot_regret_heatmaps(
        res["regret_summary"], res["g_mean"], res["g_worst"],
        out_dir / "regret_heatmap.png",
    )

    w_m, u_m, v_m, _ = policies[res["g_mean"]]
    w_w, u_w, v_w, _ = policies[res["g_worst"]]
    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]

    plot_weights(
        w_m, assets, T_vals,
        title=f"Pesos w(i, t) — g*_mean (λ={lam_m:.2f}, m={m_m:.1f})",
        out_path=out_dir / "pesos_g_mean.png",
    )
    plot_weights(
        w_w, assets, T_vals,
        title=f"Pesos w(i, t) — g*_worst (λ={lam_w:.2f}, m={m_w:.1f})",
        out_path=out_dir / "pesos_g_worst.png",
    )
    plot_capital_por_escenario(
        w_m, u_m, v_m, ctx,
        title=f"Capital por escenario — g*_mean (λ={lam_m:.2f}, m={m_m:.1f})",
        out_path=out_dir / "capital_g_mean.png",
    )
    plot_capital_por_escenario(
        w_w, u_w, v_w, ctx,
        title=f"Capital por escenario — g*_worst (λ={lam_w:.2f}, m={m_w:.1f})",
        out_path=out_dir / "capital_g_worst.png",
    )


if __name__ == "__main__":
    inspeccionar()
