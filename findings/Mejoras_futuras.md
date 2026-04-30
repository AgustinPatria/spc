# Mejoras futuras — fieles a la spec del PDF

Propuestas para futuras sesiones, ordenadas por modulo. Todas mantienen alineacion con el PDF (ec. 12-24, sec. 2-3). No incluye opciones que requieran desviarse del spec — esas estan documentadas en los archivos de cada modulo como "no aplicado".

---

## Modulo de cuantiles del LSTM

### A. Aumentar el ensemble (5-10 seeds en vez de 3)
- **Spec**: sin desviacion (el PDF no menciona seeds).
- **Esfuerzo**: trivial (`SEEDS` en `config.py`).
- **Valor esperado**: marginal (-1-2% pinball, menos cruces, mas reproducibilidad).
- **Costo**: 3-4× tiempo de entrenamiento.

### B. Probar Transformer temporal en lugar de LSTM
- **Spec**: explicitamente habilitado por el PDF sec. 2.3 *("una red recurrente tipo LSTM o un Transformer temporal")*.
- **Esfuerzo**: medio-alto (nueva clase, refactor de `_train_one`).
- **Valor esperado**: incierto. Con dataset chico (66 ventanas/fold) puede sobreajustar mas que el LSTM.
- **Recomendable**: solo con dataset mas grande.

### C. Sweep mas fino centrado en el optimo actual
- **Spec**: sin desviacion.
- **Esfuerzo**: trivial (modificar `sweep_lstm.py`).
- **Valor esperado**: chico — el sweep de 24 ya cubrio el rango. Un sweep fino centrado en `(hidden=32, layers=2, drop=0.1)` podria ajustar marginalmente.

### D. Validacion final con datos out-of-sample fuera del walk-forward
- **Spec**: sin desviacion (es practica estandar de evaluacion).
- **Esfuerzo**: medio (necesita re-particionar el dataset).
- **Valor esperado**: alto en credibilidad — demuestra que el g* elegido por el regret-grid funciona en datos verdaderamente nuevos.

---

## Modulo de regimen

Sin propuestas — la formula esta fija por la ec. 15 del PDF y se experimento con interpolacion (descartada, ver `findings/Modulo_regimen.md`).

---

## Modulo de generador de escenarios

### E. Aumentar `N_CANDIDATES` a 10000 o mas
- **Spec**: sin desviacion (`N=1000` es ejemplo del PDF).
- **Esfuerzo**: trivial (`N_CANDIDATES` en config).
- **Valor esperado**: chico. Reduce ruido en los representativos pero el bias bull es estructural — N mayor no lo arregla.
- **Costo**: 10× tiempo de generacion.

### F. Cambiar `summary_asset` a CMC200 (o promediar)
- **Spec**: sin desviacion (`SPX` es ejemplo del PDF, ec. 17).
- **Esfuerzo**: trivial (`SUMMARY_ASSET` en config).
- **Valor esperado**: cambia como se ordenan los representativos, dando otra distribucion. Vale la pena explorarlo para entender mejor la sensibilidad del regret-grid al activo de resumen.

### G. Generar desde multiples ventanas iniciales (con dataset mas grande)
- **Spec**: zona gris defendible. Probado con dataset actual y no aporto (ver `findings/Generador_escenarios.md`). Con mas datos historicos podria.
- **Esfuerzo**: medio.
- **Valor esperado**: condicional a tener mas datos.

---

## Modulo Regret-Grid

### H. Refinar la grilla `Λ × M`
- **Spec**: sin desviacion (la grilla actual `{0.05,1,3,5,10} × {0.5,3,5}` es eleccion).
- **Esfuerzo**: trivial (`LAMBDA_GRID, M_GRID` en config).
- **Valor esperado**: chico — la asimetria del regret minimax con escenarios skewed bull seguira favoreciendo λ=0.05. Pero un sweep mas fino daria un mapa mas detallado del trade-off.

### I. Backtest historico riguroso
- **Spec**: sin desviacion (es practica estandar).
- **Esfuerzo**: medio. La pipeline ya tiene `run_historical_backtest`, pero seria interesante:
  - Entrenar el LSTM y elegir g* con datos hasta `t_split`
  - Aplicar `w^g*` sobre `t_split..T_max` real
  - Comparar contra naive 50/50, OPT, y otras politicas de la grilla
- **Valor esperado**: **alto**. Es la prueba real de "¿el pipeline DL aporta sobre baselines triviales?".

### J. Probar otras `theta` (ponderaciones por activo)
- **Spec**: sin desviacion (`theta` es input al optimizador).
- **Esfuerzo**: trivial.
- **Valor esperado**: chico — `theta=1` para todos es el caso estandar.

---

## Ideas estructurales (cuello de botella del dataset)

### K. Conseguir mas datos historicos
- **Spec**: sin desviacion.
- **Esfuerzo**: depende de acceso a datos.
- **Valor esperado**: **el mas alto**. El cuello de botella verdadero del proyecto es **111 ventanas con H=52**. Con 500+ ventanas, todo el pipeline mejoraria sustancialmente:
  - Sweep mas fino con menos overfit
  - Multi-ventana del generador (G) funcionaria
  - `p_bull` podria capturar dinamica temporal real
  - Regret-grid mas robusto a la varianza muestral

### L. Anadir mas activos al universo
- **Spec**: sin desviacion (cambia `ASSETS`).
- **Esfuerzo**: medio (verificar que no rompe nada en GAMS).
- **Valor esperado**: alto en interpretacion (portafolios mas ricos), neutro en metodologia.

---

## Ranking valor / esfuerzo

| # | Idea | Esfuerzo | Valor | Comentario |
|---|---|---|---|---|
| K | Mas datos historicos | depende | **alto** | el unico cambio que puede romper limites estructurales |
| I | Backtest historico riguroso | medio | **alto** | demuestra valor del pipeline DL |
| L | Mas activos | medio | medio-alto | si tiempo |
| A | Mas seeds | trivial | bajo | "limpieza" defendible |
| D | OOS fuera del walk-forward | medio | medio | credibilidad metodologica |
| E | N=10000 escenarios | trivial | bajo | menos ruido, no arregla bias |
| H | Grilla `Λ × M` mas fina | trivial | bajo | mapa mas detallado |
| C | Sweep LSTM mas fino | trivial | bajo | opcional |
| F | Cambiar `summary_asset` | trivial | medio | sensibilidad interesante |
| B | Transformer | alto | incierto | solo con mas datos |
| G | Multi-ventana inicial | medio | bajo (sin K) | condicional a K |
| J | Otras `theta` | trivial | bajo | opcional |

---

## Sugerencia de proxima sesion

### Si el objetivo es fortalecer la documentacion / reporte

1. **I (backtest historico riguroso)** — demuestra empiricamente que el pipeline aporta
2. **A (mas seeds)** + **E (mas N)** — limpieza de ruido, ambos defendibles
3. **D (OOS fuera del WF)** — credibilidad metodologica

### Si el objetivo es mejorar performance real

1. **K (mas datos)** — cuello de botella estructural
2. **I (backtest)** + **L (mas activos)** — mejor evaluacion y mas riqueza del problema

### Para exploracion abierta

- **F (otro `summary_asset`)** — entender sensibilidad
- **B (Transformer)** — solo si K es viable

---

## Lo que NO esta aqui (desviaciones del PDF)

Las opciones que requeririan desviarse del PDF estan documentadas en cada modulo como "no aplicado":

- Recalibracion post-hoc de cuantiles → `findings/Sesgo_deciles_correccion.md`
- Penalty de monotonicidad en pinball → idem
- Interpolacion lineal en regimen → `findings/Modulo_regimen.md`
- Sampling no-uniforme de cuantiles en escenarios → `findings/Generador_escenarios.md`
- Bucketing asimetrico → `findings/Modulo_regret_grid.md`
- Regret porcentual / max-min absoluto → idem

Si en el futuro el director acepta desviaciones justificadas, esas opciones quedan como camino mas potente.
