Por qué el LSTM está sesgado — el diagnóstico detallado                                                                                                                                               
                                                                                                                                                                                                        
  Lo que "ve" el modelo durante el entrenamiento                                                                                                                                                        
                                                                                                                                                                                                        
  t:   1 ............ 27 ................................. 121 ....... 141 ....... 163                                                                                                                  
          └── H=26 ──┘└──────── TRAIN (n=95) ─────────────┘└─ VALID ─┘└─ TEST ─┘                                                                                                                        
  regimen:  BULLISH              BAJISTA (tercio 2)           mixto     BULLISH                                                                                                                         
                                                                                                                                                                                                        
  Tomando medias por tercio del dataset completo:                                                                                                                                                       

  ┌────────┬─────────┬──────────────┬─────────────────┐
  │ tercio │    t    │ SPX mean/sem │ CMC200 mean/sem │
  ├────────┼─────────┼──────────────┼─────────────────┤
  │ 1      │ 1–54    │       +0.40% │          +1.22% │
  ├────────┼─────────┼──────────────┼─────────────────┤
  │ 2      │ 55–108  │       −0.25% │          −1.27% │
  ├────────┼─────────┼──────────────┼─────────────────┤
  │ 3      │ 109–163 │       +0.41% │          +1.29% │
  └────────┴─────────┴──────────────┴─────────────────┘

  El split cronológico 70/15/15 hace que el train esté casi enteramente dentro del tercio 2 más el arranque del 3 (t=27..121). Medias del target del train: SPX −0.045%/sem, CMC200 −0.256%/sem.
  Negativas ambas.

  Entonces el LSTM no está "equivocado": está fiel a su evidencia. Lo que aprendió como "estado estacionario" es el régimen del tercio 2 (un mercado lateral/bajista), y eso es lo que colapsa cuando la
   ventana se llena de retornos sintéticos.

  ¿Por qué se nota tanto?

  Dos elementos específicos lo empeoran:

  1. Pocos datos + capacidad alta: 95 ventanas de entrenamiento, LSTM_HIDDEN=32, seed averaging sobre 3 semillas. El best_valid que termina guardado suele ser el que más memoriza el régimen bajista
  (porque valid también está sesgado: VALID CMC200 mean/sem = −0.606%). O sea, el criterio de selección premia modelos pesimistas.
  2. La calibración de test confirma el sesgo: en TEST (bullish real +0.6%/+2.9%), "frac de realizados ≤ q50 predicho" da 31.8%/27.3% (ideal 50%). El realizado queda mayormente por encima del q50 → el
   q50 predicho está bajo.

  Y encima — importante para la operativa del pipeline — este LSTM luego se usa para forward forecasting partiendo de la última ventana, que es precisamente la zona bullish (tercio 3). El
  rolling-window borra esa condición inicial en ~10 pasos y el modelo "vuelve a casa" a su régimen de train.

  ---
  Soluciones (ordenadas de más pragmáticas a más invasivas)

  A. Post-calibración del sesgo (parche, <1 día)

  Antes de samplear, corregir los deciles predichos con un shift que alinee el promedio de predicciones con la media histórica:

  bias_a = mean_dataset(Y_a) - mean_dataset(pred_q50_a)    # por activo
  preds_calibrados[a, q] = preds[a, q] + bias_a

  Ventajas: trivial, no requiere reentrenar, preserva la forma de la distribución predicha.
  Desventajas: es una curita — el bias varía con el régimen; esta constante sólo funciona si asumes que el régimen "promedio" histórico es el régimen forward. Además aplicar el mismo shift a los 9
  deciles destruye algo de la calibración quantílica. Variante más honesta: re-estimar la escala y el shift minimizando pinball post-hoc.

  B. Reducir capacidad + regularización más fuerte (medio día, recomendado)

  El modelo con 95 ventanas y 32 hidden units está holgadamente sobreparametrizado. Probar:
  - LSTM_HIDDEN = 8 o 16
  - dropout = 0.3 y/o weight_decay = 1e-3
  - patience más chico (15), o epochs más chico, porque seguir entrenando hace que la red ajuste cada vez mejor al régimen bajista del train.

  Casi seguro disminuye el sesgo porque un modelo menos flexible no puede memorizar con tanta precisión el "mundo bajista" del tercio 2 — queda más cerca de un promedio global.

  C. Predecir residuos (1 día, estructural)

  Entrenar el LSTM para predecir r_t - μ̂_t donde μ̂_t es un baseline (p.ej. media móvil de los últimos H semanas, o media histórica global). En inferencia se suma la media de referencia elegida. Esto
  desacopla el aprendizaje de la forma (dispersión, asimetría) del régimen del train, que es justo lo que queremos del componente cuantílico; el nivel lo aporta un modelo explícito que podés elegir
  tan simple o tan sofisticado como quieras (media histórica vs media rolling vs régimen bull/bear del regimen_predicted).

  D. Cambiar el split o hacer walk-forward (1 día)

  Si la serie histórica es todo lo que hay, un split cronológico 70/15/15 va a sesgar siempre (TRAIN queda viejo, TEST queda reciente → régimen distinto). Opciones:
  - Re-entrenar en el dataset completo después de seleccionar hiperparámetros con walk-forward.
  - Usar purged k-fold cronológico: bloques disjuntos de train/test que muestreen varios regímenes.

  Esto no corrige el sesgo per se, pero ajusta la capacidad y evalúa honestamente.

  E. ¿Más datos? (depende de disponibilidad)

  163 semanas ≈ 3 años. Un LSTM cuantílico sobre retornos semanales con cola fat-tail necesita más. Si podés extender la serie a 10+ años (extendiendo los CSV de data/), el LSTM va a ver varios ciclos
   y su estado estacionario va a regularizar hacia un promedio más neutral. Pregunta concreta: ¿tenés acceso a los retornos históricos de SPX y del CMC200 (o un proxy) más atrás? Esto solo dispararía
  un reentrenamiento trivial.

  F. Reemplazar el LSTM por algo más honesto para este tamaño de muestra (1–2 días, radical)

  Los retornos semanales tienen autocorrelación muy baja. Para una serie corta, un modelo sin memoria puede ser mejor: por ejemplo un quantile MLP sobre resúmenes de la ventana (media, varianza,
  momento, régimen), o un bootstrap condicional (muestrear retornos históricos del régimen predicho). Esto elimina el problema del colapso del rolling-window de un saque.

  ---
  Mi recomendación concreta

  Empezaría combinando B + C:
  1. Bajar capacidad y subir regularización → menos sobreajuste al régimen del train.
  2. Predecir residuos respecto de la media histórica global (o respecto de la media por régimen que ya calcula build_dl_context) → decoupla nivel de forma.

  Si después de eso el sesgo sigue, agregar A (post-calibración) como safety net.