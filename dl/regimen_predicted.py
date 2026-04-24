"""Probabilidades de régimen bull/bear desde los deciles predichos (PDF sec. 2.4).

El GAMS base usa K = {bear, bull} con la regla:
    bull ⇔ retorno >= 0
    bear ⇔ retorno <  0

Con los deciles que emite `prediccion_deciles`, la aproximación de la ec. (15)
del PDF es:
    p_bull(t+1) ≈ (1/|Q|) * Σ_{q ∈ Q} 1{ r_hat^(q)(t+1) >= 0 }
    p_bear(t+1) = 1 - p_bull(t+1)

Es decir, la probabilidad de bull es la fracción de deciles predichos que son
no-negativos. Esto se hace por activo y por periodo, y cumple automáticamente
p_bull + p_bear = 1 como exige ps.gms."""

from typing import Tuple

import numpy as np

from config import BULL_THRESHOLD
from .prediccion_deciles import LoadedModel, predict_deciles_batch


def regimen_from_deciles(decile_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    decile_preds: (..., n_deciles)  — retornos predichos por decil.
    return: (p_bull, p_bear) con la misma shape de entrada menos la última dim.
    """
    p_bull = (decile_preds >= BULL_THRESHOLD).mean(axis=-1).astype(np.float32)
    p_bear = (1.0 - p_bull).astype(np.float32)
    return p_bull, p_bear


def regimen_probabilities(
    model: LoadedModel,
    windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inferencia + conversión a régimen en un paso.

    windows: (N, H, n_assets) — ventanas de entrada del LSTM.
    return:  (p_bull, p_bear), ambos de shape (N, n_assets).
    """
    preds = predict_deciles_batch(model, windows)     # (N, A, Q)
    return regimen_from_deciles(preds)
