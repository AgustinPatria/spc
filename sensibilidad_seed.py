"""Sensibilidad al SCENARIO_SEED.

Corre el pipeline regret-grid completo con varios seeds y compara:
  - g*_mean (argmin mean_regret)
  - g*_worst (argmin worst_regret)
  - metricas asociadas
  - capital promedio/peor bajo g*_mean

Uso: python sensibilidad_seed.py
"""
from __future__ import annotations

import pandas as pd

from config import (
    ASSETS,
    CHECKPOINT_PATH,
    DATA_DIR,
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import build_dl_context, compute_regret_and_select, run_regret_grid


SEEDS = (0, 1, 2, 42)


def main() -> pd.DataFrame:
    theta = {a: 1.0 for a in ASSETS}
    rows = []

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"SEED = {seed}")
        print("=" * 70)
        ctx = build_dl_context(
            data_dir=DATA_DIR,
            checkpoint_path=CHECKPOINT_PATH,
            T=T_HORIZON,
            N_candidates=N_CANDIDATES,
            n_scenarios=N_SCENARIOS,
            seed=seed,
            summary_asset=SUMMARY_ASSET,
        )
        V_df, _ = run_regret_grid(ctx, list(LAMBDA_GRID), list(M_GRID), theta)
        res = compute_regret_and_select(V_df)

        lam_m, m_m = res["g_mean"]
        lam_w, m_w = res["g_worst"]
        V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
        V_worst_row = res["V_table"].loc[(lam_w, m_w)]
        C0 = ctx["Capital_inicial"]

        rows.append({
            "seed":           seed,
            "lam_mean":       lam_m,
            "m_mean":         m_m,
            "mean_regret":    res["g_mean_metric"],
            "V_mean_avg":     V_mean_row.mean(),
            "V_mean_worst":   V_mean_row.min(),
            "ret_mean_avg":   V_mean_row.mean() / C0 - 1,
            "lam_worst":      lam_w,
            "m_worst":        m_w,
            "worst_regret":   res["g_worst_metric"],
            "V_worst_avg":    V_worst_row.mean(),
            "V_worst_worst":  V_worst_row.min(),
            "ret_worst_min":  V_worst_row.min() / C0 - 1,
        })
        print(f"\n  g*_mean : lambda={lam_m:.2f}  m={m_m:.1f}"
              f"  mean_regret=${res['g_mean_metric']:,.2f}"
              f"  V_mean=${V_mean_row.mean():,.2f}")
        print(f"  g*_worst: lambda={lam_w:.2f}  m={m_w:.1f}"
              f"  worst_regret=${res['g_worst_metric']:,.2f}"
              f"  V_worst=${V_worst_row.min():,.2f}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("RESUMEN POR SEED")
    print("=" * 70)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False,
                       formatters={
                           "mean_regret":   "${:,.2f}".format,
                           "worst_regret":  "${:,.2f}".format,
                           "V_mean_avg":    "${:,.2f}".format,
                           "V_mean_worst":  "${:,.2f}".format,
                           "V_worst_avg":   "${:,.2f}".format,
                           "V_worst_worst": "${:,.2f}".format,
                           "ret_mean_avg":  "{:+.2%}".format,
                           "ret_worst_min": "{:+.2%}".format,
                       }))

    # Estabilidad de la seleccion
    g_mean_set  = df[["lam_mean",  "m_mean"]].apply(tuple, axis=1).unique()
    g_worst_set = df[["lam_worst", "m_worst"]].apply(tuple, axis=1).unique()
    print("\n--- Estabilidad de la seleccion ---")
    print(f"  g*_mean  distintos en {len(SEEDS)} seeds: {len(g_mean_set)}  -> {list(g_mean_set)}")
    print(f"  g*_worst distintos en {len(SEEDS)} seeds: {len(g_worst_set)} -> {list(g_worst_set)}")

    # Rango de metricas
    print("\n--- Rango absoluto ---")
    for col in ("mean_regret", "worst_regret", "V_mean_avg", "V_worst_worst"):
        lo, hi = df[col].min(), df[col].max()
        print(f"  {col:<14}: [{lo:>14,.2f} .. {hi:>14,.2f}]  "
              f"spread=${hi-lo:,.2f}  ({(hi-lo)/abs(lo)*100 if lo else 0:.1f}% del min)")

    return df


if __name__ == "__main__":
    df = main()
    out = DATA_DIR.parent / "resultados" / "sensibilidad_seed.csv"
    df.to_csv(out, index=False)
    print(f"\nGuardado en: {out}")
