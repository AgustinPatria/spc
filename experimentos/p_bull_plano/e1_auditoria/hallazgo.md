# E1 — Auditoria de codigo: como se calcula p_bull(t)

## Cañeria reconstruida

```
Regret_Grid.build_dl_context  (Regret_Grid.py:355)
  │
  └── predict_pbull_rollout(model, initial_window, T=163)   (line 302)
       │   inputs:
       │     - model: ensemble de K=3 LSTMs (seeds 0,1,2)
       │     - initial_window: (H=52, A=2) ultimos 52 retornos observados
       │
       ├── for t in range(163):
       │     1) normaliza window con (mean, std) del checkpoint
       │     2) forward del ensemble, promedio -> preds shape (A=2, Q=5)
       │     3) p_bull_step = regimen_from_deciles(preds)
       │           = (preds >= BULL_THRESHOLD).mean(axis=-1)        # (A,)
       │     4) p_bull[t] = p_bull_step
       │     5) ROLL: window <- concat([window[1:], MEDIAN_DECILE])
       │              ↑ se inyecta la MEDIANA (q=0.5) como "siguiente retorno"
       │
       └── retorna (T=163, A=2)
```

## Veredicto sobre las hipotesis del README

| Hip | Estado | Comentario |
|---|---|---|
| H2 — rolling-window mal feedeado | ❌ DESCARTADA | El roll esta bien implementado: `np.concatenate([window[1:], median_r[None, :]], axis=0)` |
| H6 — p_dl es escalar replicado | ❌ DESCARTADA | `predict_pbull_rollout` retorna `(T, A)` y se mete en un DataFrame indexado por t. Es serie real por t. |
| H1 — deciles casi constantes | ✅ MUY PROBABLE | Por el motivo que se discute abajo: el rollout colapsa a un punto fijo. |
| H3 — discretizacion de p_bull aplasta variaciones | ✅ AGRAVANTE | Con n_quantiles=5, p_bull ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}. Variaciones sub-0.2 son invisibles. |
| H4 — BULL_THRESHOLD mal calibrado | Pendiente | Para E4. |
| H5 — LSTM sub-ajustado | Pendiente | Para E3. |

## Causa raiz (hallazgo principal)

El rollout determinista de `predict_pbull_rollout` **colapsa a un punto fijo**.

En cada paso futuro se inyecta a la ventana **la mediana** del decil
predicho (`med_q = n_quantiles // 2`, q=0.5). Pasos a tener en cuenta:

1. La ventana tiene H=52 entradas. A partir del paso t=52, **el 100% del
   contenido de la ventana son medianas sinteticas previas** — no queda
   nada de la historia observada original.

2. La mediana de un LSTM bien entrenado tiende a ser estable
   (el modelo aprende un "valor central de retorno semanal" cercano al
   promedio historico). Inyectar siempre ese valor central a la ventana
   converge rapidamente a un punto fijo del sistema dinamico:
   `f(window_estable) = window_estable[1:] + median(LSTM(window_estable))`.

3. Una vez en el punto fijo, el LSTM siempre recibe el mismo input, asi
   que produce siempre los mismos deciles. La fraccion de deciles >= 0
   queda CONGELADA en el conteo del fixed-point.

4. Como `regimen_from_deciles` es **discreto** (con Q=5 da p_bull en
   incrementos de 0.2), aunque el LSTM tuviera variaciones pequenas
   alrededor del punto fijo (digamos cruzar el threshold por +/- 1 decil),
   la salida snap-ea al mismo escalon discreto.

→ Esto es por DISEÑO del rollout, no un bug. Pero el resultado practico
  es que `p_dl(t)` no aporta dinamica temporal alguna. La probabilidad
  reportada (0.6 / 0.4) es esencialmente el "atractor" del sistema, no
  una creencia condicional al horizonte.

## Asimetria con `generate_candidate_scenarios`

Conviene notar que el OTRO rollout que hace el pipeline,
`generate_candidate_scenarios` (dl/generador_escenarios.py:30), funciona
**distinto**:

- En cada paso muestrea aleatoriamente un nivel `q ∈ {0..Q-1}` y inyecta
  el decil `q` de ese paso a la ventana (no la mediana).
- Hace eso N=1000 veces en paralelo con seeds independientes.
- Resultado: 1000 trayectorias que SI varian, porque el rollout es
  estocastico y NO converge a un atractor.

→ Hay una incoherencia metodologica: los **escenarios** del simulador
  ex-post tienen variabilidad, pero la **probabilidad de regimen** que
  alimenta al optimizador (mu_mix, sigma_mix) viene de un rollout
  determinista que la colapsa.

Esto ademas explica un sintoma ya observado: los V[g, s] en la columna
de cada escenario varian linda y monotonamente (porque los escenarios
estocasticos cubren un rango), pero la POLITICA elegida por el
optimizador es la misma en todos los escenarios DL — porque el ctx que
ve el solver tiene mu_mix/sigma_mix sin dinamica temporal de regimen.

## Implicancias para los siguientes experimentos

- **E2 (plot deciles forward)** sigue siendo util: confirma con datos
  duros que los deciles del rollout son casi constantes y mide la
  velocidad de convergencia al fixed-point (¿en cuantos pasos colapsa?).

- **E3 (in-sample vs out-of-sample)** sigue util: si en
  in-sample (donde el LSTM ve historia real, no medianas sinteticas) el
  p_bull tampoco varia, el problema es el LSTM mismo (H5). Si en
  in-sample SI varia y en forward NO, el problema es el rollout
  determinista (lo que sospechamos aca).

- **E4 (sensibilidad threshold)** util pero secundario: aunque movamos
  el threshold, si el rollout sigue siendo determinista converge a un
  punto fijo distinto pero igualmente plano.

- **NUEVA pregunta abierta**: ¿que pasaria si reemplazamos
  `predict_pbull_rollout` por una version ESTOCASTICA — por ejemplo,
  monte-carlo con M trayectorias usando `generate_candidate_scenarios`,
  y promediando el p_bull empirico paso a paso? Esa seria la
  "expectativa marginal" del LSTM sobre p_bull, y deberia tener
  variacion temporal real.

## Estado de hipotesis (post-E1)

- [x] H2 descartada
- [x] H6 descartada
- [x] H1 explicada estructuralmente (pero queda pendiente confirmarla con E2)
- [x] H3 documentada como agravante
- [ ] H4 pendiente (E4)
- [ ] H5 pendiente (E3)
- [ ] **Nueva H7**: el rollout determinista con mediana es la causa raiz.
      Test: comparar p_bull determinista vs p_bull estocastico (Monte
      Carlo sobre escenarios candidatos). Si varia en MC, H7 confirmada.
