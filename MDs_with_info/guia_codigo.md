# Guía del repositorio SPC_Grid3

**Para qué sirve este archivo:** mapeo rápido de qué hace cada pieza del repo, cómo se conectan entre sí, y por dónde entrar para reproducir el pipeline completo. Sirve como "tabla de contenidos" del zip.

---

## Cómo correr el pipeline de punta a punta

```
python main.py
```

Eso ejecuta dos pasos en serie: (1) reentrena el LSTM cuantílico y guarda el checkpoint, y (2) corre el regret-grid completo (DL → optimizador → simulación → selección de `g*`). Los resultados se escriben en `resultados/` y los gráficos del DL en `inspeccion/`.

Para reproducir sin reentrenar (más rápido), saltear el primer paso comentando `train_and_save()` en `main.py:173`. El checkpoint `models/decile_predictor.pt` ya viene incluido en el zip.

Dependencias en `requirements.txt`. El optimizador necesita además una instalación local de GAMS con licencia para IPOPT.

---

## Archivos en la raíz del proyecto

| archivo | qué hace |
|---|---|
| `main.py` | Punto de entrada único del pipeline. Llama a entrenamiento del LSTM y luego al regret-grid completo. Imprime tablas resumen y guarda CSVs y gráficos en `resultados/`. |
| `config.py` | Configuración centralizada del proyecto. Todas las perillas (paths, universo de activos, hiperparámetros del DL, escenarios, optimizador, grilla de regret) viven acá. Cualquier otro módulo importa de este archivo. Define los dataclasses `DLConfig`, `ScenarioConfig`, `OptConfig`, `RegretGridConfig`. |
| `Regret_Grid.py` | El optimizador media-varianza (con DL acoplado) y la lógica del regret-grid. Contiene: `build_dl_context` (mezcla los momentos históricos con `p_dl(t)`), `solve_portfolio` (FO con GAMSPy + IPOPT), `simulate_capital_*` (simulaciones ex-post), `run_regret_grid` (barrido de la grilla `(λ, m)`), y `compute_regret_and_select` (elige `g*_mean` y `g*_worst` según ec. 23 y 24 del PDF). **Es donde pasa el problema de las soluciones esquina.** |
| `CLAUDE.md` | Documento de contexto del proyecto. Describe qué hace el pipeline, las convenciones, los puntos de entrada y las particularidades (PDF de referencia, idioma español en docstrings, etc.). |
| `requirements.txt` | Dependencias Python (pandas, numpy, torch, gamspy, matplotlib). |
| `.gitignore` | Archivos excluidos del control de versiones. |
| `sensibilidad_seed.py` | Script utilitario: corre el LSTM con varias semillas y reporta dispersión de la pérdida pinball. Sirvió para auditar la sensibilidad del modelo a la inicialización aleatoria. |
| `sweep_lstm.py` | Script utilitario: barre combinaciones de hiperparámetros del LSTM (capas, hidden, dropout) y deja el resultado en `findings/sweep_lstm.csv`. Es lo que se usó para elegir la configuración por defecto de `DLConfig`. |

---

## Carpeta `dl/` — la capa de Deep Learning

Tres módulos que componen el predictor de régimen y el generador de escenarios.

| archivo | qué hace |
|---|---|
| `dl/__init__.py` | Aliasea `dl.config` a `config` para que checkpoints viejos (que pickleaban `dl.config.DLConfig`) sigan deserializando. |
| `dl/prediccion_deciles.py` | LSTM cuantílica. Toma ventanas históricas de retornos de tamaño `H` y predice los deciles forward de la distribución para cada activo. Entrena con pinball loss, validación rolling-origin (walk-forward) y promedio de seeds para robustez. Funciones clave: `train_deciles` (entrenamiento completo), `save_checkpoint`/`load_checkpoint` (persistencia), `forecast_deciles` (inferencia). |
| `dl/regimen_predicted.py` | Convierte los deciles predichos en una probabilidad de régimen `p_bull(t) = fracción de deciles ≥ BULL_THRESHOLD` por activo y por período. Esa salida `p_dl(t)` es la que entra al optimizador. Mantiene `p_bull + p_bear = 1` por construcción. |
| `dl/generador_escenarios.py` | Generador de escenarios futuros de retornos en dos pasos. Primero genera `N` trayectorias candidatas rodando la ventana del LSTM hacia adelante y muestreando el mismo decil entre activos por step. Luego las reduce a 5 escenarios representativos: ranquea por retorno acumulado de `SUMMARY_ASSET` (SPX por default) y elige las medianas de los 5 quintiles. Estos 5 escenarios se usan ex-post para calcular `V[g, s]`. |

---

## Carpeta `data/` — entradas históricas

CSVs con datos semanales del universo `(SPX, CMC200)`. Estos archivos los lee `Regret_Grid.py` (y el `Legacy/basemodelGAMS.py`) para construir los momentos históricos `(μ̂, Σ̂)` por régimen.

| archivo | contenido |
|---|---|
| `ret_semanal_spx.csv` | Retornos semanales de SPX, indexados por `t`. |
| `ret_semanal_cmc200.csv` | Retornos semanales de CMC200, indexados por `t`. |
| `prob_spx.csv` | Probabilidades históricas de régimen bull/bear para SPX, indexadas por `t`. |
| `prob_cmc200.csv` | Probabilidades históricas de régimen bull/bear para CMC200. |
| `sensitivity_results_gams.csv` | Resultados del análisis de sensibilidad del modelo GAMS original, conservados como referencia. |

---

## Carpeta `models/` — checkpoints

| archivo | contenido |
|---|---|
| `decile_predictor.pt` | Checkpoint del LSTM cuantílico entrenado: state dict, configuración usada (`DLConfig`), media y desvío del estandarizador. Permite cargar el modelo y predecir sin reentrenar. |
| `decile_predictor.pt.bak` | Backup del checkpoint anterior. |

---

## Carpeta `Legacy/` — el modelo base sin DL

Contiene la versión sin la capa DL, conservada para referencia cruzada con la versión actual.

| archivo | qué hace |
|---|---|
| `Legacy/basemodelGAMS.py` | Port directo del modelo GAMS (`ps.gms`) a GAMSPy + IPOPT. Implementa el media-varianza con costos pero sin DL: `μ̂, σ̂` se calculan desde los CSVs históricos con probabilidades históricas (no predichas). Tiene su propio `main` para reproducir el caso base, el caso bullish SPX, y la grilla de sensibilidad `(λ, m)` original. Usado para validar que el port a Python da los mismos resultados que el GAMS original. |
| `Legacy/verify_optimum.py` | Sanity check del óptimo: como IPOPT devuelve `OptimalLocal`, este script evalúa la FO en varias políticas naive (50/50 buy & hold, 50/50 rebalanceado, etc.) y compara contra `z` reportado por IPOPT. Sirve para confirmar que el local óptimo no es claramente subóptimo respecto a alternativas obvias. |

---

## Carpeta `inspeccion/` — diagnósticos por componente

Cada subcarpeta tiene un script `inspeccionar_*.py` que corre diagnósticos sobre un componente específico del pipeline y deja CSVs + PNGs en la misma carpeta. Sirven para auditar visualmente que cada pieza está funcionando como se espera.

| subcarpeta | qué inspecciona |
|---|---|
| `inspeccion/prediccion_deciles/` | Calidad de los deciles predichos por el LSTM: pinball por decil, calibración, fan charts en train/test/oos, dispersión, condicionamiento por seed. Incluye `inspeccionar_deciles.py` (script principal) y `diagnostico_condicionamiento.py` (sub-diagnóstico de sensibilidad por seed). |
| `inspeccion/regimen_predicted/` | Calidad de la conversión deciles → `p_bull`. Histogramas, reliability, scatter `p_bull` vs retorno realizado, series temporales en train/test/oos. Incluye `diagnostico_colapso_pbull.py` para investigar el colapso de la distribución de `p_bull` cuando el LSTM se vuelve plano. |
| `inspeccion/generador_escenarios/` | Calidad de los escenarios generados: fan chart de los `N` candidatos, los 5 representativos, histograma del retorno terminal, scatter terminal vs trayectoria. |
| `inspeccion/regret_grid/` | Resultados del regret-grid: heatmap de regret, capital simulado por escenario para `g*_mean` y `g*_worst`, evolución de pesos, turnover, tabla `V[g,s]`, tabla `R[g,s]`, resumen por escenario. **Acá se ven directamente las soluciones esquina** en los gráficos de pesos. |

Para correr cualquiera de estos diagnósticos:

```
python inspeccion/<subcarpeta>/inspeccionar_*.py
```

Cada script vuelve a abrir el contexto y reescribe sus PNGs y CSVs.

---

## Carpeta `experimentos/` — investigaciones puntuales

Experimentos enfocados a entender un comportamiento específico del LSTM (en particular, el "p_bull plano" — cuando la red predice probabilidades casi constantes en lugar de seguir el régimen). Cada experimento tiene su propio `experimento.py` y un `hallazgo.md` con la conclusión.

| experimento | foco |
|---|---|
| `e1_auditoria/` | Auditoría inicial: por qué `p_bull(t)` sale casi plano. Solo `hallazgo.md`. |
| `e2_deciles_forward/` | Inspección de los deciles forward: convergencia, diferencias consecutivas, gráficos de la trayectoria forward. |
| `e3_real_vs_pred/` | Comparación entre `p_bull` predicho y el régimen "real" (basado en signo del retorno realizado). Métricas, ACF, scatter, time series. |
| `e4_calidad_quintiles/` | Calidad cuantílica: pinball vs baseline ingenuo, sharpness (anchos de banda) y condicionalidad respecto a la volatilidad. |
| `e5_barrer_H/` | Barrido del tamaño de ventana `H` del LSTM. Pinball vs `H`, búsqueda del óptimo. |
| `README.md` | Resumen general de la línea de trabajo "p_bull plano". |

---

## Carpeta `findings/` — diagnósticos previos consolidados

Documentos en Markdown que registran problemas detectados antes y diagnósticos asociados. Útiles para entender la historia del proyecto.

| archivo | tema |
|---|---|
| `Resumen_problemas.md` | Mapa general de problemas pendientes y resueltos. |
| `problemas.md` | Lista detallada de problemas detectados. |
| `Cadena_de_fallos.md` | Cómo se propagan los fallos del LSTM al regret-grid (relevante: el "doble canal" DL → optimizador y DL → escenarios). |
| `Problema_LSTM.MD` | Diagnóstico específico del comportamiento plano del LSTM. |
| `Sesgo_deciles_correccion.md` | Investigación sobre sesgo en los deciles predichos y propuesta de corrección. |
| `Modulo_regimen.md` | Análisis del módulo de conversión deciles → p_bull. |
| `Modulo_regret_grid.md` | Análisis del módulo de regret-grid. |
| `Generador_escenarios.md` | Análisis del generador de escenarios. |
| `Mejoras_futuras.md` | Lista de mejoras propuestas. |
| `sweep_lstm.csv` | Resultados crudos del sweep de hiperparámetros del LSTM (output de `sweep_lstm.py`). |

---

## Carpeta `MDs_with_info/` — documentación principal

| archivo | contenido |
|---|---|
| `summary.md` | Resumen ejecutivo del proyecto. |
| `presentacion_profesor.md` | Documento tipo "slides" preparado para una presentación previa al profesor guía. |
| `ex-ante(ex-post).md` | Explicación del doble rol de DL (ex-ante en `μ_mix, σ_mix` y ex-post en escenarios). |
| `problema_soluciones_esquina.md` | **Diagnóstico ordenado del problema de soluciones esquina** (escrito específicamente para esta consulta). Cubre: contexto, síntoma, diagnóstico estático del rango interior `(λ_low, λ_high)`, diagnóstico dinámico con DL, por qué la opción del regulador V no resuelve el trasfondo. |
| `conversacion_LLM_diagnostico_esquinas.md` | **Transcripción completa del ida y vuelta con Claude** donde se fue formulando y debatiendo el diagnóstico, los 23 turnos del usuario con las respuestas del modelo. |
| `guia_codigo.md` | Este archivo. |

---

## Carpeta `resultados/` — outputs del pipeline

Archivos generados por `main.py`. Se regeneran cada vez que se corre el pipeline; el contenido actual del zip es un snapshot de la última ejecución.

| archivo | contenido |
|---|---|
| `regret_grid_results.csv` | Tabla larga `V[g, s]`: capital terminal por punto de la grilla y por escenario. |
| `regret_table.csv` | Tabla `R[g, s]`: regret = mejor capital por escenario menos `V[g, s]`. |
| `regret_summary.csv` | Resumen por `g`: regret promedio y peor caso. Insumo para elegir `g*_mean` (ec. 23) y `g*_worst` (ec. 24). |
| `regret_capital_curves.png` | Curvas de capital simulado bajo `g*_mean` para los 5 escenarios. |
| `evolucion_capital.png` | Backtest histórico: capital evolucionando bajo OPT (`g*_mean`), naive 50/50 buy & hold, naive 50/50 con rebalanceo. |

---

## Cómo se conecta todo (mapa de dependencias)

```
   data/*.csv                          config.py
       │                                   │
       ▼                                   │ (todos importan)
   build_dl_context  ◄────────  dl/prediccion_deciles
   (Regret_Grid.py)              dl/regimen_predicted
       │                          dl/generador_escenarios
       │                                   ▲
       │                                   │
       │                          models/decile_predictor.pt
       ▼
   solve_portfolio  ──────►  IPOPT (vía GAMSPy)
       │
       ▼
   simulate_capital_opt  ──── usa los 5 escenarios
       │
       ▼
   run_regret_grid  ──► V[g,s]
       │
       ▼
   compute_regret_and_select  ──► g*_mean, g*_worst
       │
       ▼
   resultados/*.csv y *.png
```

Punto de fricción: `solve_portfolio` recibe `μ_mix(t), σ_mix(t)` desde `build_dl_context`, que las arma a partir de `p_dl(t)`. El hiperparámetro `λ` que se le pasa al solver es estático (viene de `LAMBDA_GRID` en `config.py`), pero los momentos sobre los que opera son dinámicos. Esa asimetría es el origen del problema descrito en `problema_soluciones_esquina.md`.
