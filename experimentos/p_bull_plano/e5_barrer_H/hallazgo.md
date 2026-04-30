# E5 — D3: Barrido de H. ¿Mejora CMC200 cambiando lookback?

## TL;DR

**No.** El lookback H no es el problema. Para CMC200, **ningun H probado
lo rescata**: en todos, el LSTM es peor que el baseline incondicional, la
mediana nunca cruza 0, y la conditionality es <10%.

## Tabla de resultados

| H   | n_train | n_test | SPX LSTM | SPX BASE | SPX ratio | SPX cruza 0 | std_ratio SPX | corr w/voly SPX | CMC LSTM | CMC BASE | CMC ratio | CMC cruza 0 | std_ratio CMC | corr w/voly CMC |
|-----|--------:|-------:|---------:|---------:|----------:|:-----------:|--------------:|----------------:|---------:|---------:|----------:|:-----------:|--------------:|----------------:|
| 13  | 105     | 23     | 0.0277   | 0.0276   | 1.007     | ❌          | 0.08          | 0.49            | 0.0945   | 0.0871   | **1.085** | ❌          | 0.04          | 0.04            |
| 26  | 95      | 22     | 0.0285   | 0.0288   | 0.990     | ❌          | 0.09          | 0.48            | 0.0948   | 0.0874   | **1.084** | ❌          | 0.05          | 0.04            |
| 52  | 77      | 18     | 0.0297   | 0.0306   | **0.972** | ✅          | 0.12          | **0.73**        | 0.0974   | 0.0965   | 1.009     | ❌          | 0.06          | 0.26            |
| 104 | 41      | 10     | 0.0255   | 0.0174   | 1.464     | ❌          | 0.18          | 0.24            | 0.0760   | 0.0747   | 1.018     | ❌          | 0.09          | 0.14            |

(`ratio_l_b = pinball_LSTM / pinball_baseline`. <1 → LSTM aporta. ≥1 → no aporta.)

## Lecciones

### Para SPX

- **H=52 es objetivamente el mejor punto** del grid: ratio 0.972 (LSTM
  gana al baseline +2.85%), corr ancho-vol 0.73, mediana cruza 0.
- H=13 y H=26 dan empate con el baseline (ratios ~1.0). El modelo no
  aporta skill cuando el lookback es corto.
- H=104 da ratio 1.46 — el LSTM SUFRE con n_test=10 (test diminuto, el
  baseline ajusta casi por cuentas y el LSTM queda mal). Es un artefacto
  de muestra chica, no de capacidad.

→ El default H=52 es el correcto para SPX. No hay que tocarlo.

### Para CMC200

- En **todos los H** la ratio LSTM/baseline esta por encima de 1
  (LSTM peor o, en el mejor caso, empate ajustado en H=52: 1.009).
- La mediana **nunca cruza 0** en ningun H. El LSTM permanentemente
  predice CMC200 en bear.
- conditionality (std_ratio) <0.10 en todos los H — el LSTM nunca llega
  a expresar variabilidad significativa en sus predicciones de mediana
  para CMC200.
- corr ancho-vol oscila entre 0.04 y 0.26 — capacidad de captar vol
  regimes nula a debil.

→ **D3 (lookback) refutada**. Cambiar H no logra que el LSTM aprenda
nada utilizable para CMC200.

## Por que falla la hipotesis

Pensaba que H=52 podia ser demasiado largo para crypto (regimenes
rapidos). El experimento dice que aunque acortemos a H=13 o H=26, el
LSTM sigue sin captar la dinamica de CMC200. La ventana no es el problema.

El problema esta en otra parte:
- **D1**: posiblemente la red comparte capacidad entre SPX y CMC200, y
  la pinball total es dominada por SPX (varianza menor → loss menor en
  magnitud absoluta, pero en proporcion hace que el optimizador "ya tiene
  baja perdida" antes de mejorar CMC200).
- **D2**: la normalizacion mean/std podria estar dominada por outliers
  de CMC200, comprimiendo las observaciones tipicas.
- **D4**: el peso de cada activo en la perdida total no esta balanceado.

## Estado de hipotesis post-E5

| Hip | Estado |
|-----|--------|
| D1 modelo separado por activo  | ⏳ pendiente, **mas prometedora** |
| D2 estandarizacion robusta     | ⏳ pendiente |
| D3 lookback H inadecuado       | ❌ refutada (ningun H ayuda a CMC200) |
| D4 pinball reweighted          | ⏳ pendiente |

## Recomendacion

Pasar a **D1**: entrenar dos LSTMs independientes, uno por activo. Si
CMC200 dedicado tampoco mejora, sabemos que el problema es genuinamente
estructural del dato (cripto es predecible-en-vol pero no en signo) y
podemos discutir mover el alcance del trabajo. Si mejora, tenemos una
solucion concreta.

## Outputs

- `resultados.csv` — tabla completa por H
- `pinball_vs_H.png` — visual de las dos series LSTM vs Baseline
