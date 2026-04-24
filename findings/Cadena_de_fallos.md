Cadena de fallos (de causa a efecto)

  ┌─────┬──────────────────────┬─────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐
  │  #  │        Módulo        │                             Síntoma                             │                               Causa probable                               │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ prediccion_deciles   │ Train=95, valid=20, test=22                                     │ Datos insuficientes para un LSTM con H=26                                  │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 2   │ prediccion_deciles   │ 63% de ventanas SPX con cruces de deciles                       │ 9 cabezas independientes, sin restricción de monotonicidad                 │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ prediccion_deciles   │ Distribución ~2× más ancha que la empírica                      │ Sobre-dispersión inducida por entrenamiento con pocos datos + pinball loss │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 4   │ prediccion_deciles   │ Mediana con bias negativo, cola izquierda vacía                 │ Mismo                                                                      │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ prediccion_deciles   │ Bandas casi constantes en t (fan chart)                         │ Poco aprendizaje condicional                                               │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 6   │ regimen_predicted    │ p_bull ∈ {4/9, 5/9} — 1 bit de resolución                       │ Discretización por conteo de deciles con umbral (estructural)              │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 7   │ regimen_predicted    │ CMC200 modelo pierde vs baseline constante                      │ Efecto combinado de 5+6                                                    │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 8   │ generador_escenarios │ Mediana forward plana pese a ventana inicial con drift positivo │ Efecto de 5                                                                │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 9   │ generador_escenarios │ Cola CMC200 terminal a +2900%                                   │ Ancho de deciles inflado se amplifica al componer                          │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 10  │ generador_escenarios │ 5 escenarios efectivos = 3                                      │ Reducción por SUMMARY_ASSET colapsa extremos del otro                      │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 11  │ Regret_Grid          │ p_bull(t) forward = constante                                   │ Efecto de 6                                                                │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 12  │ Regret_Grid          │ g*_mean == g*_worst = λ mínimo                                  │ El grid no tiene información útil para discriminar                         │
  ├─────┼──────────────────────┼─────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ 13  │ Regret_Grid          │ Eje m inerte                                                    │ Turnover casi-nulo con λ bajo + dominancia de CMC200                       │
  └─────┴──────────────────────┴─────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘

  Dónde intervenir (ordenado por impacto × esfuerzo)

  1. Régimen (#6) — alto impacto, bajo esfuerzo. Cambiar la discretización por conteo a una interpolación continua entre deciles (fracción del intervalo que queda sobre 0), o pasar directamente a un
  clasificador binario. Recupera resolución continua de p_bull.
  2. Reducción de escenarios (#10) — medio impacto, bajo esfuerzo. Ordenar por una función conjunta (ej. retorno acumulado medio, o PCA) en vez de por SPX. Restaura la diversidad de los 5 escenarios.
  3. Monotonicidad (#2) — medio impacto, medio esfuerzo. Reparametrizar la salida como cumsum(softplus(deltas)) en vez de sort post-hoc. Evita que el sort colapse dos deciles al mismo valor.
  4. Datos (#1) — alto impacto, alto esfuerzo / potencialmente fuera de control. Con 22 periodos de test cualquier conclusión es frágil. Rolling-origin eval ayuda; más datos cambia fundamentalmente.
  5. Grid de (λ, m) — bajo esfuerzo. El eje m actual no agrega; vale la pena reducirlo a 1-2 valores y ampliar λ en valores intermedios (0.1, 0.2, 0.5, 2) donde podría haber señal.

  El análisis del módulo DL está completo: tenemos evidencia de punta a punta de que el cuello de botella principal es la discretización del régimen (#6), y los demás problemas (sobre-dispersión, cola
   en CMC200, reducción por SPX) agravan pero no causan la inercia del contexto DL.