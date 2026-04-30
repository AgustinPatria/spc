# Modulo Regret-Grid — diagnostico

## Como funciona

`Regret_Grid.py` implementa la **seccion 3 del PDF**:

1. **Construye el contexto DL** (`build_dl_context`):
   - `mu_hat / sigma_hat` por regimen (historicos)
   - `p_bull(t)` forward via `predict_pbull_rollout` (mediana iterada)
   - `mu_mix(t) / sigma_mix(t)` mezclando con `p_dl(t)`
   - 5 escenarios representativos via `generate_representative_scenarios`

2. **Corre el regret-grid** (`run_regret_grid`):
   - Para cada `g = (λ, m)` en `Λ × M = {0.05, 1, 3, 5, 10} × {0.5, 3.0, 5.0}` (15 combinaciones)
   - Resuelve `solve_portfolio` para obtener `w^g, u^g, v^g`
   - Para cada escenario `s`, simula capital → `V[g, s]`

3. **Calcula regret y selecciona `g*`** (`compute_regret_and_select`):
   - `V_best_s = max_g V[g, s]` (ec. 21)
   - `R[g, s] = V_best_s - V[g, s]` (ec. 22)
   - `g*_mean = argmin_g E_s[R]` (ec. 23)
   - `g*_worst = argmin_g max_s R` (ec. 24)

## Estado del modulo (con la config final)

Sobre 5 escenarios representativos generados con quintiles iguales + `position="min"`:

```
escenario      SPX      CMC200
s1 (peor)   -5.51%    -64.15%   ← bear
s2          +67.01%   +45.55%   ← bull mid
s3         +100.49%  +127.13%
s4         +132.45%  +264.67%
s5 (mejor) +174.11%  +421.39%
```

Resultado del regret-grid:

```
g*_mean = (λ=0.05, m=0.5)   mean_regret = $1,678
g*_worst = (λ=0.05, m=0.5)  worst_regret = $5,635

Capital esperado sobre escenarios = +166.51%
Capital peor escenario             = -65.00%
```

**La politica mas agresiva (λ=0.05) gana en ambos criterios.**

## Problemas observados

### 1. `p_bull(t)` casi constante

El rollout deterministico (con la mediana q3 en cada paso) produce `p_bull` constante en 0.6 para SPX y casi constante para CMC200.

**Causa**: el LSTM predice cuantiles cuya mediana es ligeramente positiva en cualquier ventana. Con quintiles `Q={0.1, 0.3, 0.5, 0.7, 0.9}`, exactamente 3 de 5 cuantiles caen sobre el threshold 0 → `p_bull = 3/5 = 0.6`.

**Es propiedad real del modelo**, no del rollout: los retornos semanales tienen mediana positiva pequeña que es robusta al contexto reciente. Aun reemplazando el rollout deterministico por el promedio de N rollouts estocasticos (probado, revertido), `p_bull` sigue casi constante.

**Impacto**: el optimizador ve mu_mix(t) y sigma_mix(t) sin variacion temporal — esencialmente trabaja con un mundo "estacionario" en momentos esperados.

### 2. El veredicto del regret-grid favorece la politica mas agresiva

Independientemente del bucketing y del `position`, `g*_mean = g*_worst = (λ=0.05, m=0.5)`. Esa politica:
- Captura +75K en escenario bull (vs +29K del conservador)
- Pierde -65% en escenario bear (vs -8% del conservador)

**Causa**: el regret se mide en dolares absolutos. La asimetria de los retornos compuestos a 3 años hace que la "ganancia perdida" en escenarios bull (cola derecha muy alta) supere a la "perdida absoluta" en escenarios bear (cola izquierda mas chica). El regret minimax favorece al que captura el bull market aunque catastrofique en bear.

**Es un sesgo estructural del regret minimax en dolares con escenarios skewed bull.**

### 3. Capital -65% en peor escenario del g*

Como consecuencia directa del problema #2, la politica seleccionada por el regret-grid **NO protege downside**. Pierde 65% del capital en el escenario bear. El termino "robusto" de la spec significa "minimo regret" no "minimo riesgo absoluto".

## Experimentos descartados

### Experimento 1: rollout estocastico para `p_bull(t)`

Reemplazo `predict_pbull_rollout` (deterministico, mediana iterada) por el promedio de p_bull sobre N=1000 rollouts estocasticos del generador.

**Resultado**: SPX siguio constante en 0.6 (std=0). CMC200 paso de constante a `[0.543, 0.600]` con std=0.005. Mejora marginal solo en CMC200. El regret-grid eligio la misma politica con V casi identicos para λ=0.05 (politica corner-solution insensible a cambios chicos en p_dl).

**Decision**: revertido. La estabilidad de `p_bull` es propiedad del LSTM, no del rollout — no hay variacion temporal real que extraer.

### Experimento 2: bucketing asimetrico para escenarios

Edges `(0, 0.05, 0.30, 0.70, 0.95, 1.0)` para enfasis en colas: q1 captura el 5% mas bear, q5 el 5% mas bull.

**Resultado**:
- Escenarios mas dispersos: q5 paso de +218% a +253% (SPX), q2 paso de bull mid (+67%) a casi flat (+26%).
- s2 ahora es bear/flat en CMC200 (-14% vs +46% anterior).
- **PERO el regret-grid sigue eligiendo λ=0.05** con `mean_regret = $2,136`, `worst_regret = $5,635`.

**Decision**: revertido. Aunque mejora la dispersion de los escenarios, **no cambia el veredicto** y agrega complejidad. La asimetria del regret en dolares es estructural y no se resuelve cambiando el bucketing.

## Aplicado y mantenido

### `position="min"` en `reduce_to_representatives`

El PDF dice "se elige 1 escenario representativo por quintil **(por ejemplo, el escenario mediano dentro de ese quintil)**". El "por ejemplo" habilita otras elecciones.

Con `position="median"` (PDF default), q1 representativo era SPX +47% (percentil 10 de los 1000) — todos los representativos eran bull, sin proteccion downside real.

Con `position="min"` (aplicado), q1 representativo es SPX -5.5%, CMC200 -64% — el peor escenario absoluto. Esto **da un escenario bear real** contra el cual el regret-grid puede medir la proteccion.

**Mantenido**: el cambio es defendible por el "por ejemplo" del PDF y aporta valor practico al regret-grid sin cambiar la mecanica.

## Limitaciones documentadas

1. **`p_bull(t)` casi constante** — limitacion estructural del LSTM con retornos semanales i.i.d. con mediana positiva pequeña. El optimizador trabaja con info temporal degradada.

2. **El regret minimax favorece la politica agresiva** — propiedad del metodo del PDF cuando los escenarios son skewed bull (lo cual es propiedad del generador, ya documentada en `findings/Generador_escenarios.md`).

3. **El "g* robusto" del regret-grid no protege downside absoluto** — perder 65% del capital en el peor escenario es la decision optima por la metrica del PDF, pero no satisface una intuicion de "robustez" entendida como "evitar perdidas grandes".

## Que podria mejorar (no aplicado)

Cambiar la metrica de regret a algo que normalice las colas. Tres opciones:

1. **Regret porcentual** (`R_pct = (V_best - V) / V_best`) — la "ganancia perdida" en s5 deja de pesar 27K USD y pasa a ~0.5 (50%). Balancea la asimetria entre s1 y s5.
2. **Max-min de capital absoluto** (`g* = argmax_g min_s V[g, s]`) — directamente protege downside. Es una metrica diferente, no compatible con la del PDF.
3. **Regret normalizado por capital inicial** (`R_pct = (V_best - V) / C0`) — todas las metricas en la misma escala porcentual respecto al capital de partida.

Cualquiera de las tres seria una **desviacion explicita de la spec del PDF (sec. 3, ec. 22-24)**. Si el reporte permite proponer alternativas, la opcion 1 es la mas natural.

## Decisiones del modulo

| Decision | Estado |
|---|---|
| Mantener mecanica del regret-grid | si (PDF sec. 3 textual) |
| Mantener `predict_pbull_rollout` deterministico | si (no hay alternativa que aporte) |
| Mantener bucketing equal | si (cambio asimetrico no aporto al veredicto) |
| Aplicar `position="min"` | si (defendible por "por ejemplo" del PDF) |
| Cambiar metrica de regret | no (seria desviacion seria de la spec) |

## Implicaciones para el reporte

1. **El regret-grid esta implementado segun la spec del PDF** y produce resultados internamente consistentes.
2. **El "g* robusto" elegido por la metrica del PDF es la politica agresiva** debido a la asimetria de los retornos compuestos. Esto es un hallazgo — no un bug — del metodo cuando se aplica a este dataset.
3. **La proteccion downside del g* es limitada** (-65% en peor escenario). Documentar como caracteristica de la metrica, no como falla.
4. **`p_bull(t)` casi constante refleja el comportamiento real del LSTM** sobre retornos semanales i.i.d. con mediana positiva pequeña.

## Archivos relacionados

- `Regret_Grid.py` — implementacion del modulo (sin cambios netos significativos)
- `inspeccion/regret_grid/` — diagnostico OOS y artefactos (CSVs + PNGs)
- `findings/Generador_escenarios.md` — modulo upstream que produce los escenarios skewed bull
- `findings/Sesgo_deciles_correccion.md` — modulo del LSTM cuantilico que alimenta a todo
