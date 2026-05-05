# Presentación al profesor guía
**Tema:** Resultados del pipeline SPC_Grid3 y problemas estructurales identificados
**Foco:** régimen bull/bear y solución esquina del optimizador

---

## Slide 1 — Punto de partida

**Qué se planteó construir (según el PDF de referencia):**

1. Un optimizador media-varianza con costos de transacción y rebalanceo semanal (port GAMS → GAMSPy + IPOPT).
2. Una capa DL que prediga *régimen* `p_bull(t)` y genere *escenarios* de retornos.
3. Un mecanismo de **regret-grid** que elija el par `(λ, m)` óptimo evaluando ex-post sobre los escenarios.

**Universo de activos:** 2 — SPX y CMC200, frecuencia semanal.

---

## Slide 2 — Pipeline completo

```
   datos históricos ──► μ̂, σ̂ por (i, j, régimen)
                              │
                              ▼
   LSTM cuantílico ──► p_dl(t)  ─────► μ_mix(t), σ_mix(t)   (ex-ante)
        │                                        │
        │                                        ▼
        │                              solve_portfolio (IPOPT)
        │                                        │
        │                                        ▼
        └────► 5 escenarios   ─────►   simulate_capital  ─►  V[g, s]
                                                              │
                                                              ▼
                                                   regret R[g,s] = V*_s − V[g,s]
                                                              │
                                                              ▼
                                                    g*_mean / g*_worst
```

**Observación clave:** el LSTM alimenta el pipeline por **dos canales** distintos:
- Canal 1 (ex-ante): vía `p_dl(t)` → entra a la FO como `μ_mix`, `σ_mix`.
- Canal 2 (ex-post): vía escenarios → realizaciones contra las que se simula `V[g, s]`.

> Si el LSTM falla, el daño es doble.

---

## Slide 3 — Lo que sí funciona

| Componente | Estado | Evidencia |
|---|---|---|
| Port GAMS → GAMSPy + IPOPT | ✓ | `verify_optimum.py`: `z(IPOPT) ≥` toda política naive |
| LSTM cuantílico (pinball loss) | ✓ entrena | Pinball valid ≈ 0.0083 |
| Generador de escenarios (N → 5 quintiles) | ✓ | 5 trayectorias coherentes por quintil de SPX |
| Regret-grid (ec. 23 y 24 del PDF) | ✓ | `g*_mean` y `g*_worst` se calculan |
| Inspección / diagnóstico | ✓ | 4 scripts produciendo PNGs y CSVs |

**Aporte metodológico:** el pipeline está **modular y diagnosticable**. Cada caja tiene un script propio que la audita.

---

## Slide 4 — Resultado principal del backtest histórico

| Política | Capital final |
|---|---|
| OPT (λ=1.00, m=1.0, w₀ ≈ 50/50) | **+45.49%** |
| Regret-Grid `g*_mean` (λ=50, all SPX) | +27.16% |
| Naive 50/50 buy & hold | +22.58% |
| Naive 50/50 rebalanceo | +17.65% |

**Lectura:**
- La política seleccionada por la grilla supera a los naive.
- Pero **queda por debajo** del baseline interior λ=1 (cercano a `w₀`).
- **Síntoma:** la grilla está eligiendo un *corner* (todo SPX) cuando el óptimo histórico era una mezcla balanceada.

> Esto motiva los dos problemas que vienen.

---

## Slide 5 — Problema 1: solución esquina del optimizador

### Qué se observa

Pesos `w*` resueltos por IPOPT a lo largo de la grilla `Λ = (0.05, 1, 10, 20, 50)`:

| λ | w(SPX) | w(CMC200) | Interpretación |
|---|---|---|---|
| 0.05 | 0.000 | 1.000 | corner CMC200 (chasing high μ_mix) |
| 1.00 | 0.502 | 0.498 | interior ≈ w₀ (turnover ≈ 0) |
| 10 | 0.968 | 0.032 | quasi-corner SPX |
| 20 | 0.994 | 0.006 | corner SPX |
| 50 | 1.000 | 0.000 | corner SPX (mínima varianza) |

> Solo en una franja muy estrecha de λ aparece una mezcla interior. Fuera de eso, *corner*.

---

## Slide 6 — Problema 1: por qué sucede (matemática)

Con **2 activos**, la frontera de Pareto media-varianza es **unidimensional**.

$$ \max_w \quad \sum_t \big[ w^\top \mu_{mix}\theta - \lambda\, w^\top \Sigma\, w - c\, (u + v) \big] $$

- λ → 0: domina el término lineal `w·μ` → activo de mayor `μ_mix`.
- λ → ∞: domina el término cuadrático `λ·w'Σw` → activo de menor varianza.
- Mezcla interior solo en una zona estrecha intermedia.

> **El optimizador no está roto.** Está respondiendo correctamente. El corner es la geometría del problema con K=2.

---

## Slide 7 — Problema 1: por qué la grilla colapsa

Para que la regret-grid discrimine entre celdas, los **5 escenarios deben ser heterogéneos**: distintas celdas deben ganar en distintos escenarios.

Lo que pasa en la práctica:

| Run | Escenarios | Quién gana | Resultado |
|---|---|---|---|
| Single split | bear-CMC200 | λ=50 en los 5 escenarios | `mean_regret = worst_regret = $0` |
| Rolling no-expansivo | bull-CMC200 | λ=0.05 en los 5 escenarios | corner opuesto, también degenerado |

**Causa raíz:**
Los 5 escenarios son **homogéneos en dirección** porque **todos heredan el mismo sesgo del LSTM**. Sin diversidad, no hay tradeoff que la regret discrimine.

> El corner no es bug del optimizador, es **síntoma** de un input degenerado.

---

## Slide 8 — Problema 2: discretización y colapso de p_bull

### Definición (ec. 15 del PDF)

`p_bull(t)` = fracción de deciles ≥ `BULL_THRESHOLD`.
Con 5 deciles y threshold = 0, el rango teórico es **{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}** → 6 niveles.

### Lo que el LSTM produce en la práctica

- Solo emite **{0.4, 0.6}**.
- En CMC200 test: **constante en 0.4 durante 18 semanas seguidas**, mientras 13 de esas 18 fueron bull realizado.

### Métricas de calibración

|  | SPX | CMC200 |
|---|---|---|
| Test accuracy | 61.1% | **27.8%** (peor que tirar moneda) |
| %bull_pred vs %bull_real | 56% vs 83% | 40% vs 72% |
| Brier modelo | 0.238 | **0.304** |
| Brier baseline trivial | 0.282 | **0.265** |

> En CMC200 el predictor LSTM es **peor que un constante** = frecuencia histórica.

---

## Slide 9 — Problema 2: consecuencia mecánica para el optimizador

$$ \mu_{mix}(t) = p_{bull}(t)\cdot \mu_{bull} + p_{bear}(t)\cdot \mu_{bear} $$

Si `p_bull(t)` es esencialmente **constante** ⇒ `μ_mix(t)` también ⇒ los pesos óptimos `w(i, t)` son **estacionarios** ⇒ **no hay reasignación temporal**.

Eso explica los plots: `w(SPX, t) = 1.00` para *todo t* en λ=50.

> **Síntesis del Problema 2:** un predictor sin discriminación temporal degrada el pipeline en sus dos canales:
> - Canal ex-ante: μ_mix plano → optimizador no rebalancea.
> - Canal ex-post: 5 escenarios apuntando al mismo activo ganador → regret no discrimina.

---

## Slide 10 — Problema 2: por qué se queda en {0.4, 0.6}

Hipótesis ordenadas por probabilidad:

1. **Capacidad limitada por dataset chico**
   Solo 77 ventanas de train, 24 hidden units. El modelo aprende la frecuencia base y oscila tímidamente sin cruzar el siguiente decil.
2. **`BULL_THRESHOLD = 0` es indiferenciado**
   Con bull = "retorno > 0", la mediana cae cerca de la frontera. El modelo se ancla en el centro.
3. **Distribution shift train → test**
   Train: 2021-2023 (~50/50 régimen). Test: 2024+ (~75% bull). El modelo aprende del régimen viejo.

---

## Slide 11 — Cómo se conectan los dos problemas

```
                  LSTM colapsado
                   ({0.4, 0.6})
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
       p_bull plano          5 escenarios
            │                con mismo sesgo
            ▼                       │
      μ_mix(t) plano                │
            │                       │
            ▼                       ▼
     Optimizador no         Regret-grid no
     rebalancea             discrimina entre celdas
            │                       │
            └───────────┬───────────┘
                        ▼
                 Solución esquina
              elegida por g*_mean
```

> Mismo modelo, dos canales de daño. **Los dos problemas no son independientes.**

---

## Slide 12 — Experimentos realizados

| # | Cambio | Hipótesis testeada | Resultado |
|---|---|---|---|
| 1 | Baseline (PDF defaults) | Pipeline end-to-end | g* corner; backtest +27.16% |
| 2 | `position="min"` → `"median"` | Suavizar pesimismo de escenarios | Mismos pesos elegidos. **V cambia, ranking no.** |
| 3 | Rolling-origin no-expansivo | Reducir distribution shift | Predictor flip de bear-CMC200 a bull-CMC200; **g\* salta a corner opuesto**; backtest -20.91% |
| 4 | Rolling-origin expansivo | Combinar todo el historial | p_bull constante (0.8 SPX, 0.6 CMC200); g* vuelve a all-SPX; backtest +27.16% |

**Lectura cruzada:** ningún cambio de validación rompe la limitación.

> El framework de regret **no es** el cuello de botella. **El LSTM sí lo es.**

---

## Slide 13 — Hallazgos secundarios

**13.1 — Asimetría teacher-forcing vs rollout autoregresivo (exposure bias)**
- Train: el LSTM siempre ve ventanas reales.
- Generación de escenarios: rollout — después de H=52 pasos la ventana es 100% sintética.
- Probablemente explica drawdowns extremos (-88% en CMC200 con `position="min"`).

**13.2 — Forward `p_dl(t)` es in-sample para >70% del horizonte**
`predict_pbull_walking` recorre el histórico real, pero la mayoría de las ventanas estuvieron en train.
Aun así el modelo emite p_bull plano → **descarta overfitting** como causa raíz.

**13.3 — Sensibilidad al esquema de validación**
Cambia el *nivel base* del predictor pero **no su capacidad de discriminar**.

---

## Slide 14 — Conclusiones metodológicas

1. **El framework regret-grid funciona** *condicional* a tener escenarios diversos. La crítica honesta no es al framework, es a sus inputs.
2. **El cuello de botella está en la capa DL de régimen.** Con la arquitectura/dataset/features actuales, el LSTM no aprende a discriminar régimen weekly. Esta es una conclusión válida y publicable: no todos los activos/horizontes admiten un predictor LSTM estructural simple.
3. **La discretización de p_bull por deciles** es un cuello de botella adicional: aunque el modelo discriminara más, solo puede producir 6 niveles.
4. **Solución esquina:** con 2 activos y predictor sin señal, el corner es la respuesta correcta del optimizador. Con K > 2 y un predictor con discriminación, la frontera se enriquece y los corners se diluyen.
5. **Asimetría teacher-forcing / rollout** es una crítica metodológica al pipeline original que no estaba documentada en el PDF.

---

## Slide 15 — Agenda de próximas iteraciones

| Prioridad | Intervención | Costo | Impacto esperado |
|---|---|---|---|
| **A** | Features adicionales al LSTM (vol realizada rolling, momentum 4/12 sem, drawdown desde peak, correlación SPX–CMC200) | medio | Alto si la falla es por features; bajo si es por dataset chico |
| **B** | `BULL_THRESHOLD` adaptativo (mediana móvil) en vez de 0 | bajo | Medio — fuerza al modelo a discriminar régimen no trivial |
| **C** | Reemplazar predictor DL de régimen por baseline empírico (frecuencia rolling de `r > 0`); usar DL solo para deciles de escenarios | bajo | **Aísla el framework del problema de régimen.** Permite reportar performance del pipeline con un input "honesto" |
| **D** | Ampliar universo a K > 2 activos | medio | Estructural — diluye corners, obliga a decisiones de cartera no triviales |
| **E** | Scheduled sampling para entrenar con rollout autoregresivo | alto | Solo arregla colas de escenarios, no la discriminación |

**Sugerencia de orden para discutir:**
- A y B → experimentos rápidos.
- **C → la más honesta** para evaluar el framework aisladamente.
- D → cambia la naturaleza del problema (más rico).
- E → académicamente interesante pero no resuelve el cuello principal.

---

## Slide 16 — Preguntas para el profesor

1. ¿Vale la pena seguir invirtiendo en el predictor LSTM o pasamos a un baseline empírico (opción C) para aislar el framework?
2. ¿Tiene sentido ampliar el universo (K > 2) ahora, o mantener K = 2 y completar primero el diagnóstico del régimen?
3. ¿La asimetría teacher-forcing / rollout autoregresivo merece ser parte del aporte metodológico documentado, o queda como nota al pie?
4. ¿`BULL_THRESHOLD` adaptativo respeta el espíritu de la ec. 15 del PDF o lo consideramos una desviación?

---

### Anexos sugeridos para llevar impresos / mostrar en pantalla

- **Reliability diagram** de p_bull (SPX y CMC200) — `inspeccion/regimen_predicted/`
- **Histograma de p_bull** mostrando colapso a {0.4, 0.6}
- **Heatmap de regret por (g, s)** mostrando degeneración (todas las celdas iguales en una columna)
- **Curvas de capital** `OPT vs g*_mean vs naive` — `resultados/evolucion_capital.png`
- **Tabla de los 4 experimentos** (Slide 12)
