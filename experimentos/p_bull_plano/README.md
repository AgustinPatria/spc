# p_bull plano — investigacion

## Sintoma

En la corrida actual del pipeline (`main.py`), el contexto DL reporta:

```
p_bull SPX    : min=0.600  max=0.600  mean=0.600
p_bull CMC200 : min=0.400  max=0.400  mean=0.400
```

`p_bull(t)` es **constante** en todo el horizonte forward (t=1..163). Eso implica
que el LSTM cuantilico, una vez convertido a probabilidad de regimen, no
distingue entre "ahora" y "163 semanas en el futuro": le asigna la misma
probabilidad a todo.

## Por que importa

Cadena de consecuencias rio abajo:

1. `mu_mix(t) = p_bull(t) * mu_bull + (1 - p_bull(t)) * mu_bear` queda **fijo
   en t** salvo por los componentes historicos. → el optimizador resuelve
   contra una FO donde el "regimen" es una constante, no una dinamica.
2. Los 5 escenarios representativos generados por `generador_escenarios` heredan
   esa estructura: pueden diferir en magnitud (sampling decil) pero comparten
   el mismo "shape" de regimen → **escenarios homogeneos**.
3. La regret-grid no encuentra trade-offs (g*_mean = g*_worst con regret=$0)
   porque las 5 trayectorias son demasiado parecidas en lo que importa para
   discriminar politicas.
4. El argumento academico de "robustez frente a regimen variable" se cae:
   el modelo, en la practica, no esta protegiendo contra cambios de regimen,
   solo contra magnitudes diferentes del mismo regimen.

## Hipotesis a investigar

H1. **Los deciles predichos son casi constantes en el horizonte forward**.
    El LSTM aprende un "regimen promedio" estable, y la conversion a p_bull
    refleja eso. Test: graficar los 5 deciles de cada activo a lo largo de
    t=1..163 y ver dispersion.

H2. **El input del LSTM no se actualiza correctamente en el rolling forward**.
    Si el codigo de `generador_escenarios` o el forecast usa siempre el mismo
    window historico H y no inserta los retornos sinteticos al avanzar t,
    entonces por construccion el LSTM ve el mismo input → mismo output. Test:
    auditar `generate_representative_scenarios` y `regimen_from_deciles` para
    ver que window se feedea en cada paso.

H3. **`regimen_from_deciles` colapsa la dispersion**. La conversion decil →
    p_bull es discreta (count de deciles >= BULL_THRESHOLD / n_deciles). Si
    los deciles fluctuan **dentro** del rango [decil_threshold-1, decil_threshold],
    el conteo siempre da lo mismo y se pierde toda la variacion. Test: comparar
    los deciles brutos vs el p_bull resultante; ver si los deciles varian pero
    el conteo discreto los aplasta.

H4. **El BULL_THRESHOLD esta mal calibrado**. Si el threshold queda muy lejos
    del centro de la distribucion de deciles predichos (siempre por encima
    o por debajo), p_bull queda saturado en 0 o en 1 — lo cual NO es lo que
    pasa aca (estamos en 0.6/0.4), pero algo similar puede pasar si el
    threshold cae justo en una zona insensible. Test: barrer BULL_THRESHOLD
    y ver como cambia min/max/mean de p_bull.

H5. **El LSTM esta sub-ajustado / aprendio una constante**. Pinball loss
    medio (~0.008) suena chico pero podria estar siendo dominado por una
    prediccion de mediana plana que satisface "razonablemente" la quantile
    loss. Test: comparar deciles predichos vs deciles empiricos en el
    train+valid set; ver si en train-set tambien sale constante o si solo
    aplana en out-of-sample.

H6. **El p_dl se calcula una sola vez en `build_dl_context`** (sin re-evaluar
    a lo largo del horizonte). Test: leer `build_dl_context` y verificar que
    `p_dl[i]["bull"]` sea efectivamente una serie indexada por t, no un
    escalar replicado.

## Experimentos planificados (orden tentativo)

> Antes de cada experimento, decidir el alcance con el usuario; esta lista
> es un mapa, no un plan rigido.

- E1. **Auditoria de codigo**: leer `dl/regimen_predicted.py`,
      `dl/generador_escenarios.py` y `Regret_Grid.build_dl_context` para
      mapear como se calcula p_bull(t) y donde podria estar el cuello de
      botella. (Cubre H2, H6.)

- E2. **Plot deciles forward**: para los 163 pasos forward, dump de los 5
      deciles por activo y grafico de su dispersion en t. (Cubre H1, H3.)

- E3. **Comparacion in-sample vs out-of-sample**: aplicar el mismo LSTM
      al train+valid y graficar p_bull historico — ¿varia ahi tambien?
      (Cubre H1, H5.)

- E4. **Sensibilidad a BULL_THRESHOLD**: barrer el threshold y ver si
      p_bull empieza a moverse. (Cubre H3, H4.)

- E5. **Ablacion del rolling forward**: forzar diferentes inputs en cada
      paso y verificar que el LSTM responda con outputs distintos. (Cubre
      H2.)

## Outputs esperados

Cada experimento que produzca PNGs/CSVs los guarda en este mismo
directorio bajo subcarpetas `e1_*`, `e2_*`, etc. Cada subcarpeta lleva un
`hallazgo.md` corto con conclusion + evidencia.

Al cierre de la investigacion, consolidar findings en
`findings/Regimen_constante.md` (o equivalente) y, si aplica, abrir
fixes en `dl/`.

## Estado

- [x] Carpeta creada, hipotesis listadas.
- [x] E1 — auditoria de codigo. **Hallazgo principal**: el rollout
      determinista con mediana en `predict_pbull_rollout` colapsa a un
      punto fijo y explica el p_bull plano. Ver
      `e1_auditoria/hallazgo.md`. Hipotesis H2 y H6 descartadas.
      Nueva H7 abierta: el rollout determinista es la causa raiz.
- [x] E2 — plot deciles forward. **H7 confirmada**: rollout converge en
      t ~ 28 (SPX) / 26 (CMC200), con caida exponencial de diferencias
      (1.78e-3 en t=1 → 7.45e-9 en t=52). p_bull tiene un solo valor
      unico en los 163 pasos. Ver `e2_deciles_forward/hallazgo.md`.
- [x] E3 — real vs pred (in-sample) + retornos. **Causa raiz movida**:
      el LSTM in-sample TAMBIEN produce p_bull ∈ {0.4, 0.6} con ventanas
      reales. El rollout es agravante, no causa raiz. Ademas:
      corr(pred, real) = -0.32 en CMC200 (anti-correlacion). Ni pred ni
      real superan a constante en Brier sobre el signo del retorno.
      Reinterpretacion: el LSTM da el optimo pinball-loss en regimen
      low-SNR. Ver `e3_real_vs_pred/hallazgo.md`.
- [x] E4 — calidad de los quintiles (3 gaps de inspeccionar_deciles).
      **Diagnostico actualizado**: el problema es asimetrico por activo.
      SPX: skill modesto pero real (LSTM gana al baseline incondicional
      +2.85%, captura vol regimes con corr 0.73). **CMC200: el LSTM es
      peor que predecir constante (-0.88% en pinball total)**, mediana
      siempre negativa, banda casi constante. Ver `e4_calidad_quintiles/hallazgo.md`.
- [x] E5 — D3 (barrido de H ∈ {13, 26, 52, 104}). **D3 refutada**.
      Ningun H rescata a CMC200: ratio LSTM/baseline >= 1 en todos,
      mediana nunca cruza 0. Para SPX, H=52 ya es el optimo. Ver
      `e5_barrer_H/hallazgo.md`.
- [ ] (proxima) D1 — modelo separado por activo.
