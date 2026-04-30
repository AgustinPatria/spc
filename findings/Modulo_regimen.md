# Modulo de regimen — diagnostico y experimentos

## Como funciona

`dl/regimen_predicted.py` implementa la **ec. 15 del PDF (sec. 2.4)**:

```
p_bull(t+1) ≈ (1/|Q|) * Σ_{q ∈ Q} 1{r̂^(q)(t+1) >= 0}
```

= la fraccion de los cuantiles predichos que superan 0.

`p_bear = 1 - p_bull`.

Esta formula es una aproximacion de step-function de la CDF predicha: cuenta cuantos cuantiles caen sobre el threshold y divide.

## Estado actual (con la config final del modulo de cuantiles)

Sobre las 45 obs OOS del walk-forward, con quintiles `Q = {0.1, 0.3, 0.5, 0.7, 0.9}` segun ec. 12 del PDF, ensemble de 3 seeds, hidden=24, layers=2:

| Metric | SPX | CMC200 |
|---|---|---|
| **Brier OOS modelo** | **0.2311** | **0.2222** |
| Brier baseline trivial | 0.2810 | 0.2631 |
| Mejora vs baseline | -18% | -16% |
| **Logloss** | 0.6550 | 0.6370 |
| **Accuracy @ 0.5** | **64.4%** | **68.9%** |
| `%bull_pred` promedio | 55.1% | 55.1% |
| `%bull_real` | 66.7% | 57.8% |

**Diagnostico:**
- El modelo bate al baseline trivial en ambos activos por margen significativo (16-18% en Brier).
- Accuracy razonable: SPX 64%, CMC200 69%.
- `p_bull` toma solo 2 valores discretos: `0.4` y `0.6`. Distribucion: 11 obs en `0.4`, 34 obs en `0.6` (en ambos activos).

## Por que la grilla discreta de p_bull = {0.4, 0.6} resulto **mejor** que {0.444, 0.556, 0.667}

Hallazgo clave de esta sesion:

Con **9 deciles**, los valores posibles eran `{0/9, ..., 9/9}` y los predichos quedaban atrapados en `{0.444, 0.556, 0.667}` (cerca del umbral 0.5 de decision). El mas minimo ruido en los cuantiles centrales cruzaba el umbral en cualquier direccion.

Con **5 quintiles**, los valores posibles son `{0/5, ..., 5/5}` = `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`:
- Predichos quedaron en `{0.4, 0.6}`, **20pp del umbral 0.5** (vs ~5pp con deciles).
- El modelo se "compromete" con una clase de forma mas robusta.
- Cada cuantil "pesa" 0.2 en `p_bull` (vs 0.11 con deciles), y la grilla cubre mejor el rango [0, 1].

Es decir: **menos resolucion en la probabilidad pero mejor decision binaria**.

## Comparacion deciles vs quintiles

| Metric | Deciles (9) | Quintiles (5) | Cambio |
|---|---|---|---|
| SPX Brier | 0.2587 | **0.2311** | -11% |
| SPX Accuracy | 53.3% | **64.4%** | +11pp |
| CMC200 Brier | 0.2395 | **0.2222** | -7% |
| CMC200 Accuracy | 62.2% | **68.9%** | +7pp |
| SPX TP / FN | 19 / 11 | 24 / 6 | mejor recall |
| CMC200 TP / FN | 16 / 10 | 23 / 3 | mejor recall |

Y como bonus: el cambio a quintiles **alinea el codigo con la ec. 12 textual del PDF**, que especifica `Q = {0.1, 0.3, 0.5, 0.7, 0.9}`.

## Experimento descartado: interpolacion lineal

Antes del switch a quintiles, se probo reemplazar la formula discreta por una **interpolacion lineal**: dado el threshold (0), encontrar el cuantil `q_thr` exacto donde la CDF predicha cruza 0, y devolver `p_bull = 1 - q_thr`. Misma intencion (estimar `P(Y >= 0)` desde los cuantiles) pero produce valores continuos en [0, 1].

Justificacion teorica: el simbolo `≈` en la ec. 15 del PDF habilita aproximaciones equivalentes de la misma cantidad. La interpolacion lineal entre cuantiles es practica estandar en pronostico probabilistico cuantilico (Gneiting, Koenker, etc.).

### Resultados de la comparativa (con deciles)

| Metodo | Activo | Brier | Accuracy | Valores unicos |
|---|---|---|---|---|
| **discrete** | SPX | 0.2587 | 53.3% | 3 |
| **interp** | SPX | 0.2470 (-4.5%) | 53.3% | 44 |
| **discrete** | CMC200 | 0.2395 | 62.2% | 2 |
| **interp** | CMC200 | 0.2445 (+2%) | 62.2% | 41 |

### Por que se descarto

1. **Mejora asimetrica**: SPX mejoraba un poco, CMC200 empeoraba. En agregado no daba ganancia clara.
2. **No resolvia el problema fundamental**: las predicciones seguian concentradas cerca de 0.5 (la interpolacion las achicaba aun mas).
3. **Introducia desviacion respecto a la spec del PDF** sin un beneficio claro a cambio.
4. **El switch a quintiles hizo lo que la interpolacion no logro**: ampliar la separacion entre clases (de ~5pp a 20pp del umbral) y mejorar Brier ~10% en ambos activos. El cambio mas chico (cantidad de cuantiles) tuvo mucho mayor impacto.

## Que NO se puede mejorar mas con esta arquitectura

1. **Granularidad de p_bull**: con 5 quintiles, p_bull toma solo 6 valores posibles. La formula del PDF no permite resolucion fina.
2. **Sesgo de base rate**: %bull predicho 55% vs realizado 67% (SPX) — el modelo subestima el rate de bull. Esto es heredado del bias residual de los cuantiles del LSTM, que viene del drift estructural (CMC200) o del dataset chico.
3. **Confianza extrema**: con 5 quintiles, p_bull rara vez llega a 0.0 / 0.2 / 0.8 / 1.0. La razon: para que llegue a 0.8 tienen que estar 4 de 5 cuantiles sobre 0, lo que requiere distribuciones muy asimetricamente positivas, raras en datos semanales.

## Decision

**Se mantiene la formula discreta del PDF (ec. 15)** sin modificaciones. El experimento de interpolacion fue revertido. El estado actual del codigo es el original (PDF ec. 15) con los cuantiles correctos (PDF ec. 12).

## Implicaciones para el reporte

1. **Reportar la mejora del switch a quintiles**: -11% Brier en SPX, -7% en CMC200, +11pp accuracy SPX.
2. **El modulo bate al baseline trivial por buen margen** (-16-18% Brier).
3. **Documentar el experimento de interpolacion** como exploracion descartada — muestra rigor empirico.
4. **El cambio mas impactante fue alinear con la spec del PDF (quintiles)**, no las modificaciones mas elaboradas.

## Archivos relacionados

- `dl/regimen_predicted.py` — implementacion original (formula PDF ec. 15)
- `inspeccion/regimen_predicted/` — diagnostico OOS sobre el LSTM final con quintiles
- `findings/Sesgo_deciles_correccion.md` — modulo de cuantiles, contexto del cambio a quintiles
