Resumen consolidado del análisis de punta a punta del módulo DL. Los problemas están ordenados por impacto en el resultado final, no por módulo.
                        
  1. Discretización del régimen con 1 bit de resolución — el cuello de botella
                        
  regimen_from_deciles cuenta cuántos de los 9 deciles están ≥ umbral y divide entre 9. Como los deciles están ordenados y el umbral es ~0, la salida casi siempre es 4/9 o 5/9. El LSTM puede predecir 
  lo que quiera: aguas abajo solo llega "cruzó 0 o no cruzó".
                                                                                                                                                                                                        
  Evidencia: histograma de p_bull en train/valid/test con toda la masa entre 0.3 y 0.6, ningún valor cerca de 0 o 1. p_bull forward plano (std=0.000 en SPX) durante los 163 pasos.                     
                                                                                                                                                                                                      
  2. Inestabilidad estructural a H (ventana del LSTM)                                                                                                                                                   
                                                                                                                                                                                                        
  Un cambio de ±5 pasos en H hace que el valor final del portafolio salte de −29% a +50% sin pasar por valores intermedios. La causa: el problema 1 actúa como función escalón sobre H, así que pequeños
   cambios bastan para bascular qué lado del umbral toca la ventana inicial.
                                                                                                                                                                                                      
  ┌─────┬────────────┐
  │  H  │ ret V_mean │
  ├─────┼────────────┤
  │ 20  │       −16% │
  ├─────┼────────────┤
  │ 25  │       +50% │
  ├─────┼────────────┤
  │ 30  │       −29% │
  └─────┴────────────┘

  La métrica de entrenamiento (pinball) no correlaciona con el resultado final — H=30 gana en pinball y pierde en portafolio.

  3. 63% de cruces de deciles en SPX antes del sort

  La salida cruda del LSTM no es monotónica en 2/3 de las ventanas SPX. El sort post-hoc tapa el síntoma visualmente pero colapsa deciles adyacentes al mismo valor (por eso q=0.8 y q=0.9 en SPX dan
  cobertura idéntica de 0.95). Es síntoma de que las 9 cabezas cuantílicas se comportan como independientes — la pinball loss no las consistenta lo suficiente.

  4. Distribución predicha sobre-dispersa y sesgada a la baja

  - q10 predicho casi nunca se viola (cobertura empírica = 0/22 en ambos activos).
  - q50 predicho queda por debajo del q50 real → bias negativo en la mediana.
  - Banda q10–q90 cubre 95% en SPX y 91% en CMC200 (nominal 80%).
  - Ancho predicho ≈ 2× el empírico.

  Consecuencia: al componer sobre 163 semanas, la cola derecha de CMC200 explota a +2900% y la izquierda toca −96%. Los 5 candidatos extremos dominan cualquier estadístico agregado.

  5. Reducción de escenarios colapsa los extremos del segundo activo

  reduce_to_representatives ordena los 5000 candidatos por retorno acumulado de SUMMARY_ASSET=SPX y toma medianas por quintil. Como los quintiles altos/bajos de SPX no coinciden con los de CMC200, los
   representativos colapsan:
  escenario    SPX      CMC200
  s1         -35%     -51%
  s2         -14%     -50%    ← igual que s1 en CMC200
  s3          +3%     -14%
  s4         +25%    +133%
  s5         +62%    +140%    ← igual que s4 en CMC200
  Los 5 escenarios declarados son efectivamente 3 para CMC200.

  6. LSTM produce bandas casi constantes en t (no aprende condicionalidad)

  El fan chart en test muestra bandas casi horizontales — el modelo no cambia la forma de la distribución según la ventana de entrada; predice ~la distribución marginal. Esto explica por qué el drift
  positivo de la ventana inicial (SPX +0.59%, CMC200 +2.44%) se pierde al primer paso del rollout forward.

  7. Régimen pierde al baseline constante en CMC200

                  brier_modelo   brier_baseline
  SPX              0.2480         0.2589         (modelo gana por 0.01)
  CMC200           0.2581         0.2481         (modelo PIERDE)
  El baseline es simplemente p_bull = frecuencia de bull en train. En CMC200, ignorar el modelo es mejor que usarlo. Accuracy en test: SPX 55%, CMC200 45% (peor que moneda).

  8. Eje m del regret grid es prácticamente inerte

  Con λ=0.05 el turnover es 1.0 (rebalanceo único al inicio); con λ=1.0 es 0.0 (nunca rebalancea). Las tres columnas del heatmap del regret (m=0.5, 3.0, 5.0) son casi indistinguibles. Se resuelven 15
  optimizaciones cuando con 5 bastaba — y el mecanismo que m debería capturar no está activo.

  9. g*_mean = g*_worst → el minimax no discrimina

  En H=25 la misma política gana en promedio y en peor caso (ambos eligen λ=0.05, m=0.5). Con los 5 escenarios realmente siendo 3 y p_bull casi constante, la FO reduce a "maximizar retorno esperado";
  el grid elige el λ mínimo y el concepto de trade-off entre media y peor caso desaparece. El regret-grid, en la práctica, no está haciendo el trabajo que se diseñó para hacer.

  10. Contexto DL ≈ contexto histórico

  mu_mix(t) = p_bull(t)·mu_bull + (1−p_bull(t))·mu_bear. Con p_bull(t) casi constante, mu_mix(t) degenera a un promedio ponderado fijo — muy parecido a lo que darías al optimizador usando la p
  histórica del CSV. El LSTM no está aportando información condicional al optimizador.

  11. Régimen de pocos datos

  - Train = 95 ventanas, Valid = 20, Test = 22.
  - Cada punto de calibración tiene resolución de 1/22 ≈ 4.5 pp.
  - El seed ganador del ensemble cambia entre corridas (seed 0 en H=20/30, seed 1 en H=25) — el ensemble no converge.
  - Sobre 22 puntos cualquier métrica tiene varianza enorme.

  Esto es contexto, no una falla del código — pero amplifica todos los problemas anteriores y hace que las conclusiones por split individual sean frágiles.

  ---
  Mapa de causalidad

    pocos datos (11)
         │
         ▼
    LSTM no aprende condicionalidad (6) + cabezas independientes con cruces (3)
         │
         ▼
    distribución sobre-dispersa y sesgada (4)
         │
         ├──► conteo con umbral convierte esto en 1 bit de resolución (1)  ──► p_bull ≈ const
         │                                                                        │
         │                                                                        ▼
         │                                                               contexto DL ≈ histórico (10)
         │                                                                        │
         ▼                                                                        │
    escenarios sin drift + cola CMC200 explosiva + reducción que colapsa (5)      │
         │                                                                        │
         └────────────────────────────────┬───────────────────────────────────────┘
                                          ▼
                         regret-grid degenera: m inerte (8), g*_mean=g*_worst (9)
                                          │
                                          ▼
                         inestabilidad catastrófica a H (2)

  Punto único de máximo apalancamiento: el problema 1 (discretización con 1 bit). Si p_bull fuera continuo (ej. interpolación lineal entre los deciles adyacentes que cruzan cero), las fluctuaciones
  del problema 2 quedarían absorbidas como cambios pequeños y continuos en vez de saltos de régimen. Los problemas 6, 10 y 9 también se aliviarían al recibir una señal más rica.