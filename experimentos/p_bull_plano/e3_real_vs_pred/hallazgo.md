# E3 — p_bull real vs predicha (in-sample) + retornos: cambio de diagnostico

## TL;DR

El problema **no es solo el rollout determinista** (E2). El LSTM
**in-sample con ventanas reales** ya produce p_bull comprimido en
{0.4, 0.6}. Es el LSTM mismo, no el rolling forward.

Peor: en CMC200 la pred esta **anti-correlacionada con el clasificador
real (corr = -0.32, p < 0.001)**.

## Numeros principales (n=111, t=53..163)

| metrica                  | SPX             | CMC200          |
|--------------------------|-----------------|-----------------|
| rango p_bull pred        | [0.400, 0.600]  | [0.400, 0.600]  |
| rango p_bull real        | [0.056, 0.940]  | [0.114, 0.883]  |
| std pred / std real      | 0.075 / 0.322   | 0.049 / 0.151   |
| corr(pred, real)         | +0.12           | **-0.32**       |
| MAE(pred, real)          | 0.293           | 0.408           |
| Brier(pred, sign)        | 0.2573          | 0.2573          |
| Brier(real, sign)        | 0.3229          | 0.3413          |
| **Brier(const, sign)**   | **0.2495 (mejor)** | **0.2500 (mejor)** |
| corr(pred, ret)          | -0.05           | +0.09           |
| corr(real, ret)          | +0.06           | +0.03           |
| ACF1(pred)               | 0.36            | 0.24            |
| ACF1(real)               | 0.41            | 0.04            |

## Lecciones criticas

### 1. La compresion {0.4, 0.6} no es del rollout — es del LSTM

E2 mostro que el rollout determinista con mediana converge a un punto
fijo donde p_bull = 0.6/0.4. Mi hipotesis era: "in-sample con ventanas
reales el LSTM sí varia, lo que aplasta es el feedback de medianas".

E3 lo refuta: aplicando el LSTM a 111 ventanas historicas reales (todas
distintas), el output sigue concentrado en {0.4, 0.6}. El rollout solo
**fija** el atractor; la compresion estaba ahi de antes.

### 2. Skill: ni la pred ni la "real" superan a constante

Brier de un predictor constante = mean(y) * (1 - mean(y)) ≈ 0.25 en ambos
activos. El LSTM (0.2573) es marginalmente peor. El clasificador "real"
(0.32-0.34) es **mucho** peor — predice bullish con alta confianza, pero
si lo usas como p(bull next week) te equivocas mas que si predices la
frecuencia base.

→ La señal predictiva de "esta semana sera bull (ret >= 0)" es
extremadamente debil en este dataset. NINGUN modelo accesible (incluido
el "real") la captura.

### 3. Reinterpretacion: la salida del LSTM ES optima para low-SNR

Con pinball loss y casi nula señal predictiva, el optimo es predecir los
quantiles **incondicionales** del retorno. Si el ~55% de los retornos de
SPX historicos son ≥ 0, los 5 deciles aprendidos se centran de modo que
**siempre** 3 quedan ≥ 0 (p_bull=0.6) o 2 (p_bull=0.4) segun tiny shifts
contextuales. Nunca 0/1/4/5 porque eso requiere shifts mucho mayores que
la varianza condicional aprendida.

→ La constante 0.6/0.4 NO es un bug — es la salida ML mas razonable
dada la informacion disponible.

### 4. CMC200: anti-correlacion sistematica

corr(pred, real) = -0.32 con n=111 es estadisticamente robusto. Visto en
los histogramas: el clasificador "real" dice CMC200 esta en bull casi
siempre (mass en 0.7-0.9), y el LSTM dice bear casi siempre (mass en
0.4). Razones plausibles:

- El "real" clasificador puede estar usando features que el LSTM no ve
  (volumen, momentum largo, on-chain).
- El LSTM solo ve los retornos semanales pasados de SPX y CMC200 — su
  vista del mundo es limitada.
- El "real" clasificador puede estar mal calibrado para CMC200 (su
  Brier 0.34 es muy alto).
- Posiblemente ambos estan capturando "alguna cosa" pero en direcciones
  opuestas porque la "cosa" es fundamentalmente ruido.

## Implicancias para el plan original

| Plan original | Reevaluacion |
|---|---|
| E6: Monte Carlo del rollout para evitar el atractor | **Limitado**: si la respuesta del LSTM por-ventana ya es {0.4, 0.6}, promediar M trayectorias estocasticas dara p_bull ≈ 0.5 con muy poca dinamica. El MC ataca un sintoma, no la causa. |
| E4: sensibilidad al BULL_THRESHOLD | Aun util — moviendo el threshold podria activarse el bin {0.6 ↔ 0.4} mas dinamicamente, pero no rescatara el rango [0.06, 0.94] de la real. |
| E5: ablacion del rolling forward | Caja descartada — E3 prueba que el problema persiste sin rollout. |

## Nuevas direcciones (orden tentativo de impacto)

### Opcion A — Aceptar el LSTM y usar la "real" en su lugar
Usar `p_bull_real(t)` (del clasificador previo en `prob_*.csv`) directamente para construir mu_mix(t)/sigma_mix(t) en el horizonte forward. Pero `p_bull_real` solo existe historicamente — no hay clasificador para t > T_obs. Habria que predecir p_bull_real con OTRO modelo (autoregresivo simple con la propia serie persistente, ACF1=0.41). Eso si daria mu_mix(t) dinamico.

### Opcion B — Aumentar la capacidad del LSTM o cambiar features
Probable causa de skill bajo: H=52 no captura ciclos largos, no hay
features cross-asset, los retornos semanales son demasiado ruidosos.
Probar:
  - Lookback mas corto (H=26) o mas largo (H=104).
  - Agregar features (volatilidad rolling, momentum de N semanas).
  - Loss alternativo (CRPS, distributional regression con mixture of Gaussians).
  - Modelo mas chico para evitar overfit (los pinball train ~0.008 son sospechosamente bajos para tan poca data).

### Opcion C — Aceptar que el LSTM es ruido y bajar las expectativas
Si la conclusion es que la señal predictiva semana-a-semana es nula, el
pipeline DL no agrega valor sobre baselines simples. Vale honesto
documentarlo y discutir si el TFG/ trabajo academico se enfoca en:
  - La **metodologia** (regret-grid, robustez) usando un forecast simple
    pero dinamico (p.ej. regimen markoviano de 2 estados con transicion
    estimada por MLE sobre p_bull_real).
  - La capacidad del DL es un componente, pero el aporte real es el
    pipeline regret-grid + robustez.

### Opcion D — Investigar el clasificador "real"
La p_bull real bimodal es muy estructurada — claramente viene de un HMM
o GMM sobre alguna feature. Si conseguimos replicar/recrear ese
clasificador con metodos simples (HMM con 2 estados sobre la propia
serie de retornos), tendriamos un baseline robusto que SI varia, sin
depender del LSTM.

## Hipotesis post-E3

| Hip | Estado |
|-----|--------|
| H1 deciles casi constantes forward | ✅ confirmada (consecuencia del LSTM, no del rollout) |
| H7 rollout colapsa | ✅ confirmada — pero **es un agravante, no la causa raiz** |
| H3 discretizacion | ✅ confirmada — Q=5 hace que cualquier shift sub-0.2 sea invisible |
| H5 LSTM sub-ajustado / low-SNR | ✅ **NUEVA causa raiz**. Pero "sub-ajustado" puede ser optimo dada la señal disponible — el problema podria ser estructural del dataset. |
| H4 BULL_THRESHOLD | ⏳ pendiente, prioridad baja |

## Estado de la investigacion

- [x] E1 auditoria de codigo
- [x] E2 deciles forward
- [x] E3 real vs pred + retornos. **Causa raiz movida: del rollout al LSTM mismo.**
- [ ] Decidir entre Opcion A/B/C/D antes de continuar.
