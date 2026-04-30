"""Mide cuanto condiciona la LSTM a la ventana de input.

Hipotesis a falsar:
  H0: la LSTM aprendio a emitir aproximadamente los cuantiles incondicionales,
      casi sin sensibilidad al input -> p_bull queda atrapado en {0.4, 0.6}
      por construccion del modelo, no del rollout.

Tres mediciones:
  1. std_pred(q) vs std_real         -> si pred_std/real_std << 1, no condiciona.
  2. correlaciones (mean/std/last de la ventana) vs q50 predicho.
  3. snapshots: deciles predichos para ventanas bearish vs bullish extremas.
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
    DATA_DIR, CHECKPOINT_PATH, ASSETS, RETURN_CSV, RETURN_COL, DECILES,
)
from dl.prediccion_deciles import load_checkpoint, load_returns, build_windows

OUT_DIR = Path(__file__).resolve().parent


def predict_all(model, windows):
    x = ((windows - model.mean) / model.std).astype(np.float32)
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        outs = [net(x_t).numpy() for net in model.nets]
    preds = np.mean(np.stack(outs, 0), 0)
    return np.sort(preds, axis=-1)                                     # (N, A, Q)


def main():
    print("Cargando modelo + datos...")
    model     = load_checkpoint(CHECKPOINT_PATH)
    H         = model.config.H
    rets      = load_returns(DATA_DIR)
    X, Y, _   = build_windows(rets, H)
    N         = X.shape[0]
    print(f"  N ventanas = {N}, H = {H}, activos = {list(ASSETS)}")

    print("Inferencia masiva (batch unico)...")
    preds = predict_all(model, X)                                      # (N, A, Q)

    # ============================================================
    # 1. Std de cada decil predicho vs std del retorno realizado
    # ============================================================
    print("\n=== Test 1: dispersion de las predicciones ===")
    print("Si pred_std << real_std, el modelo casi no usa la ventana.\n")
    print(f"{'asset':8s} {'cuantil':6s} {'pred_std':>10s} {'real_std':>10s} {'ratio':>8s}")
    diag1 = []
    for ai, a in enumerate(ASSETS):
        real_std = float(Y[:, ai].std())
        for qi, q in enumerate(DECILES):
            pred_q   = preds[:, ai, qi]
            pred_std = float(pred_q.std())
            ratio    = pred_std / real_std if real_std > 0 else float("nan")
            diag1.append({
                "asset": a, "q": q,
                "pred_std": pred_std, "real_std": real_std, "ratio": ratio,
            })
            print(f"{a:8s} q{int(q*100):02d}    {pred_std:10.5f} {real_std:10.5f} {ratio:8.3f}")
    pd.DataFrame(diag1).to_csv(OUT_DIR / "diag_dispersion.csv", index=False)

    # ============================================================
    # 2. Correlaciones de la ventana con q50 predicho
    # ============================================================
    print("\n=== Test 2: correlacion ventana <-> q50 predicho ===")
    print("Si todas las correlaciones son ~0, el modelo no esta condicionando.\n")
    diag2 = []
    for ai, a in enumerate(ASSETS):
        win_a    = X[:, :, ai]
        feats = {
            "mean":  win_a.mean(axis=1),
            "std":   win_a.std(axis=1),
            "last":  win_a[:, -1],
            "ewm12": (win_a[:, -12:].mean(axis=1)),
            "y_real": Y[:, ai],
        }
        q50 = preds[:, ai, 2]
        for name, feat in feats.items():
            c = float(np.corrcoef(feat, q50)[0, 1])
            diag2.append({"asset": a, "feature": name, "corr_with_q50": c})
            print(f"  {a:8s} corr({name:6s}, q50_pred) = {c:+.4f}")
    pd.DataFrame(diag2).to_csv(OUT_DIR / "diag_correlaciones.csv", index=False)

    # ============================================================
    # 3. Bench: que tan bien predeciria un modelo trivial (q empirico)
    # ============================================================
    print("\n=== Test 3: pinball loss vs baseline incondicional ===")
    print("Si la LSTM no es mucho mejor que emitir el q empirico, el aprendizaje\n"
          "se quedo en la marginal.\n")
    quants = np.array(DECILES)
    pinball_results = []
    for ai, a in enumerate(ASSETS):
        # baseline: emitir siempre el cuantil empirico de Y_train
        empirical_q = np.quantile(Y[:, ai], quants)                    # (Q,)
        # pinball loss del modelo
        e_model = Y[:, ai:ai+1] - preds[:, ai, :]                      # (N, Q)
        pin_model = np.maximum(quants * e_model, (quants - 1.0) * e_model).mean()
        # pinball loss del baseline
        e_base = Y[:, ai:ai+1] - empirical_q[None, :]
        pin_base = np.maximum(quants * e_base, (quants - 1.0) * e_base).mean()
        improvement = (pin_base - pin_model) / pin_base
        print(f"  {a:8s} pinball(LSTM)={pin_model:.5f}  "
              f"pinball(emp_q)={pin_base:.5f}  "
              f"mejora={improvement*100:+.2f}%")
        pinball_results.append({
            "asset": a, "pin_model": pin_model, "pin_baseline": pin_base,
            "improvement_pct": improvement * 100,
        })
    pd.DataFrame(pinball_results).to_csv(OUT_DIR / "diag_pinball.csv", index=False)

    # ============================================================
    # PLOT 1: scatter q50 predicho vs retorno realizado
    # ============================================================
    fig, axes = plt.subplots(1, len(ASSETS), figsize=(11, 4.4))
    for ai, (a, ax) in enumerate(zip(ASSETS, axes)):
        ax.scatter(Y[:, ai], preds[:, ai, 2], alpha=0.45, s=18, color="#1f77b4")
        med_uncond = float(np.median(Y[:, ai]))
        mean_uncond = float(np.mean(Y[:, ai]))
        ax.axhline(med_uncond, color="#d62728", lw=1.4, ls="--",
                   label=f"mediana incondicional Y = {med_uncond:.4f}")
        ax.axhline(mean_uncond, color="#2ca02c", lw=1, ls=":",
                   label=f"media incondicional Y = {mean_uncond:.4f}")
        ax.set_xlabel("retorno realizado Y")
        ax.set_ylabel("q50 predicho por la LSTM")
        ax.set_title(f"{a}\npred_std={preds[:,ai,2].std():.5f}  "
                     f"real_std={Y[:,ai].std():.5f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Test 1 visual: si la LSTM condiciona, q50 deberia desplazarse "
                 "verticalmente en respuesta a Y\nSi q50 esta pegado a una "
                 "horizontal, el modelo aprendio el cuantil incondicional.",
                 fontsize=11)
    fig.tight_layout()
    p1 = OUT_DIR / "diag_q50_vs_realizado.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  guardado: {p1}")

    # ============================================================
    # PLOT 2: deciles predichos para ventanas bear vs bull extremas
    # ============================================================
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(11, 4.0 * len(ASSETS)))
    if len(ASSETS) == 1:
        axes = [axes]
    qs = np.array(DECILES) * 100
    for ai, (a, ax) in enumerate(zip(ASSETS, axes)):
        feat_mean = X[:, :, ai].mean(axis=1)
        order = np.argsort(feat_mean)
        bear_idx = order[:5]
        bull_idx = order[-5:]
        for k, idx in enumerate(bear_idx):
            label = (f"5 ventanas mas bearish (mean<{feat_mean[bear_idx].max()*100:+.2f}%)"
                     if k == 0 else None)
            ax.plot(qs, preds[idx, ai], marker="o", color="#d62728",
                    alpha=0.55, lw=1.5, label=label)
        for k, idx in enumerate(bull_idx):
            label = (f"5 ventanas mas bullish (mean>{feat_mean[bull_idx].min()*100:+.2f}%)"
                     if k == 0 else None)
            ax.plot(qs, preds[idx, ai], marker="o", color="#2ca02c",
                    alpha=0.55, lw=1.5, label=label)
        # cuantiles empiricos como referencia
        for q, qv in zip(qs, np.quantile(Y[:, ai], np.array(DECILES))):
            ax.scatter([q], [qv], marker="s", s=80, color="black", zorder=4,
                       label="cuantil empirico de Y" if q == qs[0] else None)
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"{a}: deciles predichos para ventanas extremas")
        ax.set_xlabel("decil")
        ax.set_ylabel("retorno predicho")
        ax.set_xticks(qs)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Test 2 visual: si la LSTM condiciona, las lineas rojas y verdes\n"
                 "deberian estar SEPARADAS y los puntos negros (incondicional)\n"
                 "deberian estar entre ellas, no encima.",
                 fontsize=11)
    fig.tight_layout()
    p2 = OUT_DIR / "diag_deciles_extremos.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardado: {p2}")

    # ============================================================
    # Veredicto
    # ============================================================
    print("\n=== Veredicto ===")
    ratios = [d["ratio"] for d in diag1]
    avg_ratio = float(np.nanmean(ratios))
    if avg_ratio < 0.10:
        verdict = "LSTM CASI NO CONDICIONA -> emite cuantiles ~ marginales"
    elif avg_ratio < 0.30:
        verdict = "LSTM CONDICIONA POCO -> dispersion << dispersion real"
    else:
        verdict = "LSTM condiciona razonablemente"
    print(f"  ratio promedio pred_std / real_std = {avg_ratio:.3f}  =>  {verdict}")


if __name__ == "__main__":
    main()
