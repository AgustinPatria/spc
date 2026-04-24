Resumen ejecutivo — Problemas del módulo de Deep Learning                                                                                                                                             
                                                                                                                                                                                                        
  Hay cuatro etapas en el pipeline DL, y los problemas cascadean: cada etapa hereda y amplifica las fallas de la anterior.                                                                              
                                                                                                                                                                                                        
  ---                                                                                                                                                                                                   
  Etapa 1 — Predicción de deciles (LSTM cuantílica)                                                                                                                                                     
                                                 
  Dónde vive: dl/prediccion_deciles.py

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┐
  │  #  │                                                                                    Problema                                                                                    │ Severidad │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 1.1 │ Test set ridículamente chico (22 ventanas) para evaluar 9 deciles. Barras de error enormes, cualquier métrica es estadísticamente frágil.                                      │ Alta      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 1.2 │ Bandas de cuantiles sobre-anchas. Ancho q10–q90 = 6.8% semanal (SPX) y 21% semanal (CMC200). Cobertura central 80% nominal → empírica 95% (SPX) / 91% (CMC200). Calibración en │ Crítica   │
  │     │  S alrededor de la diagonal → sobre-dispersión.                                                                                                                                │           │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 1.3 │ Bias de mediana sistemático, especialmente CMC200 (+2.9% semanal). Toda la curva predicha corrida hacia abajo vs los cuantiles empíricos.                                      │ Alta      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 1.4 │ Cruces de deciles no manejados: 18% de ventanas en SPX violan monotonicidad. No hay restricción estructural en el modelo.                                                      │ Media     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 1.5 │ valid < train indica split fortuito (tramo de validación menos volátil), no mejor generalización. Early stopping optimista.                                                    │ Media     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 1.6 │ Sin baseline de comparación (EWMA, cuantiles rolling). No sabemos si el LSTM le gana a algo trivial.                                                                           │ Media     │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┘

  ---
  Etapa 2 — Régimen bull/bear (p_bull)

  Dónde vive: dl/regimen_predicted.py

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┐
  │  #  │                                                                                    Problema                                                                                    │ Severidad │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 2.1 │ p_bull colapsa a ~0.5: solo aparecen 3 valores (0.333, 0.444, 0.556) en todo el dataset. Consecuencia directa de 1.2 — si las bandas siempre cruzan cero, "contar deciles ≥ 0" │ Crítica   │
  │     │  da siempre 4 o 5 de 9. La DL no aporta información de régimen.                                                                                                                │           │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 2.2 │ Apenas le gana al baseline constante (Brier 0.244 vs 0.256). Mejora no distinguible del ruido.                                                                                 │ Alta      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 2.3 │ SPX: 0 true negatives en test. De los 5 bears reales, predice bull en los 5. El 64% de accuracy es por "siempre bull", no por señal.                                           │ Alta      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 2.4 │ CMC200: reliability con pendiente negativa — bins con p_bull baja tienen mayor frecuencia real de bull. Señal invertida, peor que ruido.                                       │ Alta      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 2.5 │ Mapeo grueso decil → probabilidad: solo 10 valores posibles, ignora magnitudes. Un decil en −0.0001 vs +0.0001 aporta lo mismo que en ±10%.                                    │ Media     │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┘

  ---
  Etapa 3 — Generador de escenarios

  Dónde vive: dl/generador_escenarios.py

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────┐
  │  #  │                                                                            Problema                                                                            │        Severidad         │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
  │ 3.1 │ Comonotonicidad forzada (bug en una línea): q_idx = rng.integers(..., size=N) se aplica a TODOS los activos → SPX y CMC200 con correlación = 1 por             │ Crítica (y barata de     │
  │     │ construcción. El optimizador no puede diversificar.                                                                                                            │ arreglar)                │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
  │ 3.2 │ Sesgo bajista masivo: 80% de los 5000 escenarios terminan en pérdida después de 163 semanas. Consecuencia de 1.2 + volatility drag (E[Π(1+r)] ≠ Π(1+E[r])).    │ Crítica                  │
  │     │ Con σ ≈ 7%/paso en CMC200, el drag es catastrófico.                                                                                                            │                          │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
  │ 3.3 │ Los 5 representativos son todos bajistas o neutrales: [−48%, −34%, −22%, −9%, +13%] en SPX. El regret grid nunca ve un escenario alcista serio.                │ Alta                     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
  │ 3.4 │ Reducción 1-D por SPX: ordena por retorno acumulado de SPX y parte en quintiles → pierde toda la variabilidad ortogonal. Candidatos con CMC200 terminal >      │ Alta                     │
  │     │ +500% se concentran en un solo quintil y desaparecen.                                                                                                          │                          │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
  │ 3.5 │ Colas absurdas en CMC200: max terminal = +915%. Implausible en 163 semanas. Cola derecha larga por deciles sobre-anchos compuestos.                            │ Media                    │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────┘

  ---
  Etapa 4 — Regret grid

  Dónde vive: Regret_Grid.py

  ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┐
  │  #  │                                                                                    Problema                                                                                    │ Severidad │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 4.1 │ p_bull(t) forward es literalmente constante (std=0 en SPX, std=0.015 en CMC200). mu_mix(t) y sigma_mix(t) son planas → el optimizador resuelve un problema estático, la capa   │ Crítica   │
  │     │ DL no inyecta dinámica temporal.                                                                                                                                               │           │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 4.2 │ Política óptima = ~97% SPX constante. Los pesos no se mueven en todo el horizonte. Cualquier minimizador de varianza trivial daba el mismo resultado.                          │ Crítica   │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 4.3 │ g*_mean == g*_worst: la misma celda gana ambos criterios con regret ≈ $1.38. El grid se degenera — no hay trade-off que arbitrar.                                              │ Alta      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 4.4 │ Dimensión m ignorada: variación dentro de una fila de λ es <10%; entre filas de λ es >100×. El grid 5×3 se comporta como 5×1.                                                  │ Media     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 4.5 │ FO del óptimo da z negativo: el propio modelo admite que la política recomendada tiene retorno esperado < 0.                                                                   │ Media     │
  ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
  │ 4.6 │ Turnover invertido: mayor λ (más aversión al riesgo) → más rebalanceo, contra la intuición. La formulación actual hace que costo_mult pierda peso cuando λ domina.             │ Baja      │
  └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┘

  ---
  Causa raíz y orden de ataque recomendado

  El único problema realmente autónomo es 3.1 (comonotonicidad) — es un bug independiente, una línea, alto impacto. Los demás cascadean desde 1.2 (deciles sobre-anchos) y 1.3 (bias). Arreglando eso,
  2.1, 3.2, 4.1 y 4.2 se desbloquean automáticamente.

  Orden sugerido:

  1. 3.1 — Comonotonicidad (bug trivial, 1 línea, ganancia inmediata en diversificación).
  2. 1.2 + 1.3 — Deciles (problema raíz; sin esto el resto no tiene remedio). Antes de tocar el modelo, correr un baseline EWMA+σ para saber si el LSTM le gana a algo simple.
  3. 2.5 — p_bull continua por interpolación (desbloquea resolución sin tocar el LSTM).
  4. 3.4 — Reducción multivariable (k-medoids o similar).
  5. 4.4 — Revisar si m aporta algo a la grilla dada la formulación actual.

  El pipeline tiene el esqueleto correcto (sanity checks pasan, la infra GAMSPy+IPOPT funciona, la persistencia es reproducible). El problema está en que el componente predictivo entrega señal ≈ 0 y
  el componente generador introduce sesgos propios.