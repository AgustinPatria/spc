# E2 — Deciles forward: el rollout colapsa antes del paso 30

## TL;DR

H7 (rollout determinista colapsa a un punto fijo) **confirmada con datos
duros**:

- **`t_converge` (max_q |Δ| < 1e-6) = 28 para SPX, 26 para CMC200** —
  i.e., en menos del 17% del horizonte (T=163) los deciles del LSTM
  dejan de moverse.
- La caida de la diferencia entre pasos consecutivos es exponencial:
  6 ordenes de magnitud entre t=1 y t=52. Firma de fixed point estable.
- `p_bull(t)` tiene **un solo valor unico** en los 163 pasos (0.6 SPX,
  0.4 CMC200). No es "aproximadamente constante" — es matematicamente
  constante post-convergencia.

## Numeros

| asset  | t_converge | max_diff t=1 | max_diff t=10 | max_diff t=52 | max_diff t=163 |
|--------|-----------:|-------------:|--------------:|--------------:|---------------:|
| SPX    |         28 |     1.78e-03 |      1.70e-04 |      7.45e-09 |       3.73e-09 |
| CMC200 |         26 |     1.43e-03 |      1.76e-04 |      7.45e-09 |       7.45e-09 |

A `t=52` los deciles ya estan en territorio de epsilon de maquina
(1e-8). Despues de `t=52` no hay variacion observable.

## Por que tan rapido (analisis dinamico)

Esperaba que la convergencia recien arrancara cuando la ventana se
hubiera llenado de medianas (t >= H = 52), pero los datos muestran que
el sistema cae mucho antes. Razon:

- Aunque la ventana tenga historia real al principio, **cada paso de
  rollout reemplaza la entrada mas vieja por una mediana sintetica**.
  A t=10 ya hay 10 entradas sinteticas en la ventana — y son las mas
  recientes, que son las que mas pesan en un LSTM.
- El LSTM, al ver que las ultimas observaciones son todas similares
  (todas medianas previas), pierde rapidamente la senal historica y
  produce deciles cada vez mas parecidos a los anteriores.
- Cuando la nueva mediana es casi identica a la anterior, el sistema
  esta en un punto fijo de facto. Eso ocurre en ~25-30 pasos.

Caida exponencial confirma autovalor dominante del jacobiano del
sistema dinamico < 1 (estable).

## Implicancias para el pipeline

1. **`mu_mix(t)` y `sigma_mix(t)` son constantes** desde t≈30 en
   adelante. → el optimizador, al integrar la FO sobre t=1..163, ve
   esencialmente un problema **estacionario**. Pierde toda la ventaja
   de tener un horizonte forward dinamico.

2. **Los 5 escenarios DL no comparten esta patologia**: ellos usan
   muestreo estocastico de q por paso (`generate_candidate_scenarios`),
   por eso si tienen variabilidad. Pero la POLITICA optima es contra
   un mu_mix/sigma_mix plano → la politica termina siendo "estatica" y
   los escenarios solo afectan la simulacion ex-post, no la decision
   ex-ante.

3. **El feature de la regret-grid (regret = $0 con dominancia uniforme)
   es consecuencia directa**: si el optimizador resuelve un problema
   estacionario, todas las ventanas producen la misma w aproximada, y
   la unica diferencia entre puntos de la grilla es el peso relativo
   varianza vs. retorno → λ alto siempre gana en escenarios moderados,
   λ bajo siempre pierde, sin trade-off escenario-dependiente.

## Hipotesis post-E2

| Hip | Estado |
|-----|--------|
| H1 — deciles casi constantes forward | ✅ CONFIRMADA con t_converge < 30 |
| H7 — rollout determinista colapsa | ✅ CONFIRMADA con caida exponencial |
| H3 — discretizacion aplasta variaciones | ✅ AGRAVANTE (con Q=5, granularidad 0.2 hace que cualquier fluctuacion sub-pasos sea invisible) |
| H4 — BULL_THRESHOLD | Aun pendiente — pero hasta el threshold no rescata si los deciles son constantes |
| H5 — LSTM sub-ajustado | Pendiente — E3 |

## Caminos abiertos (en orden de impacto esperado)

1. **Reemplazar `predict_pbull_rollout` por una version Monte Carlo**:
   correr K=1000 trayectorias estocasticas (como `generate_candidate_scenarios`),
   pero registrar p_bull empirico promediado paso a paso. Esa es la
   "esperanza marginal" del LSTM sobre el regimen — y por construccion
   no colapsa (porque cada trayectoria salta de q en q random). Si MC
   p_bull(t) varia, podemos plug-it-in en `build_dl_context` y todo
   el resto del pipeline se beneficia. **Esto es E6**.

2. **Cambiar la regla de roll**: en lugar de inyectar la mediana,
   inyectar un sample (igual que `generate_candidate_scenarios`). Eso
   convierte a `predict_pbull_rollout` en una sola realizacion
   estocastica. Conceptualmente debil (depende del seed), pero ya rompe
   el atractor.

3. **Aumentar el numero de deciles** (p.ej. Q=11 o 21) para no perder
   resolucion en p_bull. Mejora marginal — sin atacar el rollout, p_bull
   sigue siendo casi constante.

4. **Re-entrenar el LSTM con mas regularizacion / dropout / lookback
   distinto** para que la mediana no sea tan "estable". Caro y de
   dudoso impacto si la causa raiz es estructural.

→ Recomendacion: pasar directo a **E6 (Monte Carlo)**. Los otros
caminos son band-aids; el (1) ataca la raiz.

## Outputs guardados en este directorio

- `deciles_forward.csv` — trayectoria completa (T x A x Q + p_bull)
- `convergencia.csv` — t_converge y diffs en t=1, 10, 52, last
- `deciles_forward.png` — 5 deciles vs t por activo
- `diff_consecutivo.png` — log-y de la diferencia consecutiva (la
  curva de convergencia exponencial)
- `p_bull_forward.png` — la "linea horizontal" de p_bull(t)
