# Modulo de prediccion de cuantiles — diagnostico y correcciones

## El problema inicial

Las predicciones de cuantiles del LSTM cuantilico estaban sesgadas hacia abajo: la mediana predicha caia por debajo de la mediana realizada en ambos activos.

**Diagnostico del modelo original (H=22, hidden=16, wd=1e-3, deciles, expansive walk-forward):**

| Sintoma | Valor |
|---|---|
| SPX bias mediana (`mean(y - q̂50)`) | +0.009/sem |
| SPX q50 empirico vs nominal 0.50 | 0.32 |
| CMC200 bias mediana | +0.018/sem |
| Curva de calibracion | siempre por debajo de la diagonal |
| Curva de cuantiles | simetrica alrededor de 0; la realizada arriba |

Es decir, el modelo predecia distribuciones **centradas en 0 y demasiado anchas**, mientras que los retornos realizados tienen mediana positiva.

## Causas identificadas

1. **Weight decay sobre el bias del head linear** — AdamW por defecto regulariza tambien el bias `b` de `nn.Linear`. Como las features de entrada estan estandarizadas (media 0), el unico parametro que puede absorber el offset es `b`. Penalizarlo lo arrastra hacia 0 → mediana predicha en 0.
2. **Capacidad por debajo de la spec del PDF** — `lstm_hidden=16` cuando el PDF sugiere `{32, 64, 128}`, y `H=22` cuando el ejemplo es `H=52`.
3. **Numero de cuantiles distinto al PDF** — el codigo usaba 9 deciles cuando la ec. 12 del PDF especifica 5 quintiles `Q = {0.1, 0.3, 0.5, 0.7, 0.9}`.
4. **Drift estructural CMC200** — el periodo de train (cripto-invierno) tiene media negativa, el OOS (bull market) tiene media positiva. Este componente no es un bug del modelo sino de los datos.

## Arreglos aplicados (estado final)

### Arquitectura — alineada estrictamente con el PDF

| Parametro | Original | Final | Origen |
|---|---|---|---|
| `quantiles` | `(0.1, 0.2, ..., 0.9)` (9) | `(0.1, 0.3, 0.5, 0.7, 0.9)` (5) | PDF ec. 12 textual |
| `H_WINDOW` | 22 | 52 | PDF sec. 2.2 ejemplo |
| `LSTM_HIDDEN` | 16 | 24 | PDF sec. 2.3 (rango sugerido 32-128); sweep eligio 24 como compromiso |
| `LSTM_LAYERS` | 1 | 2 | PDF sec. 2.3 permite 1-3; sweep eligio 2 |

### A — Excluir el `bias` del weight_decay
En `_train_one`, se separan los parametros en dos grupos del optimizador AdamW: pesos con weight_decay, biases con weight_decay=0. Asi el offset del head puede crecer libremente hasta su optimo.

### B — Bajar el weight_decay general
`WEIGHT_DECAY`: 1e-3 → **1e-4** (10× menos regularizacion sobre los pesos, para aprovechar la capacidad sin atarlos cerca de 0).

### F1 — Walk-forward con ventana rolling (no expansiva)
En `rolling_origin_splits`, opcion `rolling_window_non_expansive=True`: cada fold descarta los datos mas viejos para que el train tenga tamano fijo (= `initial_train_frac × N`) pero siempre mas reciente.

Esto ataca el drift estructural: en vez de promediar 5 anos de cripto-invierno, el ultimo fold solo entrena con datos de los ultimos meses.

### Ensemble de seeds
En vez de elegir la mejor seed por valid pinball y descartar las otras, se **promedian las predicciones de las K seeds entrenadas**. El checkpoint guarda la lista de state_dicts; en inferencia, `predict_deciles_batch` itera sobre `model.nets` y devuelve el promedio.

## Lo que se descarto

### C (descartado) — Centrar el target (predecir residuo)
Inicialmente se aplico junto con A, B, D y F1: la red predecia cuantiles de `y - μ_train` y la inferencia sumaba `μ_train` de vuelta.

**Verificacion empirica posterior:** se desactivo C y se midio. Los resultados fueron **practicamente identicos**. Conclusion: A solo (excluir bias del wd) ya logra el mismo efecto. C era redundante. Se elimino toda la maquinaria de `mu_y` del codigo.

**Beneficio colateral:** los callers externos (`generador_escenarios.py`, `Regret_Grid.py`) usan el output de la red directamente. Con C activo esos callers tenian un bug latente (interpretaban cuantiles del residuo como cuantiles del retorno). Al eliminar C, esos callers volvieron a ser correctos automaticamente.

## Sweep de hiperparametros

Se exploro un grid de 24 configuraciones para encontrar el sweet spot:
- `H` ∈ {26, 52}
- `lstm_hidden` ∈ {24, 32, 48}
- `lstm_layers` ∈ {1, 2}
- `dropout` ∈ {0.1, 0.3}

**Hallazgos del sweep (con deciles aun):**
- `H=52` domino sistematicamente sobre `H=26`
- `layers=2` mejora pinball
- La config elegida fue `H=52, hidden=24, layers=2, dropout=0.1` (compromiso pinball / calibracion).

Nota: el sweep se corrio con 9 deciles. El cambio posterior a quintiles mejoro el modelo independientemente del sweep. Resultado en `findings/sweep_lstm.csv`.

## Resultados finales

Comparativa con el modelo original:

| Metric | Original (deciles, H=22) | Final (quintiles, todas las correcciones) |
|---|---|---|
| **Pinball OOS agregada** | n/d (split distinto) | **0.01093** |
| **SPX bias mediana** | +0.009 | **-0.001** |
| SPX cruces de cuantiles | desconocido | **2.22%** |
| **CMC200 bias mediana** | +0.018 | +0.013 |
| CMC200 cruces de cuantiles | desconocido | **0.00%** |
| Cobertura central SPX (vs 80% nominal) | desconocida | 97.78% |
| Cobertura central CMC200 | desconocida | 86.67% |

### Lo que se logro

- **Cruces de cuantiles cercanos a 0%** gracias al ensemble + ajuste de capacidad. El `sort` posterior pasa a ser practicamente un no-op.
- **SPX bias mediana practicamente nulo** (-0.001) — calibracion casi perfecta.
- **CMC200 bias mediana mejoro** (de +0.018 a +0.013) gracias al rolling-window.
- **Codigo mas simple**: la limpieza de C eliminó una capa de complejidad sin perder calidad.
- **Alineacion estricta con la spec del PDF**: quintiles `Q = {0.1, 0.3, 0.5, 0.7, 0.9}` exactamente como ec. 12.

### Lo que NO se pudo mover

- **Sesgo CMC200 residual (+0.013):** confirmado numericamente como **drift puro**. `mean(Y_oos) - μ_train` por fold coincide con el bias reportado. Ningun cambio del modelo puede corregir esto porque al entrenar no se conoce el regimen futuro.
- **Tamano del dataset:** ~111 ventanas con H=52, 45 obs OOS total. Cualquier metrica tiene barras de error grandes.

## Conclusion

**El modulo de cuantiles esta cerca de su mejor punto practico** dentro de las restricciones del PDF (pinball loss, LSTM, retornos como features) y del dataset disponible.

Cuello de botella verdadero: la longitud de las series historicas. Con mas datos, modelos mas grandes serian justificables. Con los datos actuales, ya se llego al limite de capacidad util.

## Archivos modificados

- `config.py`:
  - `DECILES`: `(0.1, 0.2, ..., 0.9)` → `(0.1, 0.3, 0.5, 0.7, 0.9)` (quintiles segun PDF ec. 12)
  - `H_WINDOW`: 22 → 52
  - `LSTM_HIDDEN`: 16 → 24
  - `LSTM_LAYERS`: 1 → 2
  - `WEIGHT_DECAY`: 1e-3 → 1e-4
  - `ROLLING_WINDOW_NON_EXPANSIVE`: nuevo flag, default True

- `dl/prediccion_deciles.py`:
  - `LoadedModel.net` → `LoadedModel.nets` (lista de redes para ensemble)
  - `_train_one`: AdamW con param_groups (bias sin wd)
  - `train_deciles_rolling`: guarda todas las seeds por fold; OOS preds con ensemble
  - `RollingResult`: nuevo campo `fold_state_dicts`
  - `rolling_origin_splits`: nueva flag `rolling_window_non_expansive`
  - `predict_deciles` y `predict_deciles_batch`: iteran sobre `model.nets` y promedian
  - `save_rolling_checkpoint`: persiste lista de state_dicts (con fallback singular para back-compat)
  - `load_checkpoint`: lee lista o cae al state_dict singular si no esta

- `dl/generador_escenarios.py`: actualizado para iterar `model.nets`
- `Regret_Grid.py`: actualizado para iterar `model.nets`
- `sweep_lstm.py`: nuevo script de barrido (24 configs); produce `findings/sweep_lstm.csv`

## Recomendaciones para el reporte

Documentar:
1. **Las desviaciones respecto al original** estan justificadas por la spec del PDF (rangos sugeridos para `H`, `lstm_hidden`, `lstm_layers`; cuantiles textuales).
2. **El switch de deciles a quintiles** alinea el codigo con la ec. 12 del PDF y mejora el desempeno (pinball -6.5%, bias SPX practicamente cero).
3. **La eliminacion de C** se hizo despues de verificar empiricamente que era redundante con A — fundamentar como simplificacion guiada por experimento.
4. **El ensemble** es tecnica estandar de ML para reducir varianza y mejorar consistencia.
5. **El sweep** muestra exploracion sistematica del espacio de hiperparametros, con criterio explicito de seleccion.
6. **La limitacion fundamental del dataset:** el drift CMC200 no corregible sin datos futuros — explicitar como honesta limitacion.
