# Problema de soluciones esquina en el optimizador

**Foco:** explicar de punta a punta por qué el optimizador está produciendo carteras concentradas al 100 % en un activo, qué relación tiene eso con la grilla de λ que se está usando, y cómo la capa de Deep Learning hace que el problema sea peor de lo que parece a primera vista.

---

## 1. Contexto

El pipeline SPC_Grid3 implementa un optimizador media-varianza con costos de transacción y rebalanceo semanal (port del modelo GAMS original a GAMSPy + IPOPT). Encima de ese optimizador hay dos capas adicionales:

- **Capa DL:** una LSTM cuantílica que predice los deciles de retorno semana a semana, y a partir de ellos arma una probabilidad de régimen bull / bear (`p_dl(t)`) por activo y por período del horizonte forward.
- **Regret-grid:** una grilla de hiperparámetros `(λ, m)` sobre la que se resuelve el problema de optimización, se evalúa cada solución contra escenarios futuros y se elige la combinación que minimiza el regret esperado o el del peor caso.

La función objetivo del optimizador es:

```
max  Σ_t [ Σ_i w_i,t · μ_i,t · θ_i  −  λ · Σ_{i,j} w_i,t · w_j,t · Σ_{i,j,t}  −  Σ_i c_i^eff · (u_i,t + v_i,t) ]
```

con las restricciones habituales `Σ_i w_i,t = 1`, `w ∈ [0,1]`, identidades de rebalanceo, y costos por compra (`u`) y venta (`v`).

`λ` es el hiperparámetro que regula cuánto pesa el término de varianza frente al de retorno: λ chico → "no me importa el riesgo", λ grande → "el riesgo me aterra".

---

## 2. El síntoma

Con la grilla actual `LAMBDA_GRID = (0.05, 1, 10, 20, 30)`, el solver está entregando soluciones esquina en cuatro de los cinco puntos. Es decir, carteras concentradas al 100 % en un activo, alternando entre SPX = 100 % y CMC200 = 100 % según el λ. Solo `λ = 1` produce una solución interior, y aún así está pegada cerca del borde (≈ 89 % SPX).

No se trata de una falla numérica del solver. IPOPT está resolviendo correctamente. El problema es estructural: con esa grilla y este universo de activos, las esquinas son la respuesta matemáticamente correcta del optimizador.

---

## 3. Por qué hay soluciones esquina (parte estática, sin DL)

### 3.1 Las dos fuerzas que pelean por la cartera

La FO se puede leer como un tira y afloja entre dos imanes que tiran a la cartera hacia direcciones opuestas:

| fuerza | qué hace | hacia dónde tira en este universo |
|---|---|---|
| **retorno (lineal en w)** | tira hacia el activo con mayor μ medio | hacia CMC200 (mayor retorno semanal medio) |
| **varianza (cuadrática en w)** | tira hacia el portafolio de mínima varianza global (GMV) | hacia SPX (porque el GMV cae fuera del segmento factible, en `w_SPX ≈ 1.02`, por la altísima varianza de CMC200) |

El parámetro `λ` actúa como **megáfono** del segundo imán: λ grande hace que la fuerza de varianza grite más fuerte, λ chico la silencia.

### 3.2 Cuándo aparece una solución interior

Para que el óptimo sea interior (alguna mezcla de SPX y CMC200), las dos fuerzas tienen que equilibrarse. Si una aplasta a la otra, el solver es empujado a la esquina del lado ganador.

Las "fuerzas" cuantitativamente son:

```
fuerza retorno  =  |μ_SPX − μ_CMC|       (constante, no depende de w)
fuerza varianza =  λ · |∂Var/∂w_SPX|     (depende de la posición y de λ)
```

La pendiente `|∂Var/∂w_SPX|` no es constante en el segmento `[0, 1]`. Con los datos históricos:

- En el extremo CMC (`w_SPX = 0`): `|∂Var/∂w_SPX| = 0.01775` (pendiente empinada)
- En el extremo SPX (`w_SPX = 1`): `|∂Var/∂w_SPX| = 0.000334` (pendiente casi plana)

Y la diferencia de retornos `|μ_SPX − μ_CMC| = 0.00232` (semanal).

### 3.3 El rango interior de λ

Para que la fuerza de varianza `λ · |∂Var/∂w_SPX|` pueda igualar a la fuerza de retorno `0.00232` en *algún* punto interior del segmento, λ tiene que estar en el rango:

```
λ_low  = |Δμ| / |∂Var/∂w|_max  =  0.00232 / 0.01775   ≈  0.131
λ_high = |Δμ| / |∂Var/∂w|_min  =  0.00232 / 0.000334  ≈  6.94
```

- **Si λ < 0.131:** incluso aplicando λ a la pendiente más empinada, la fuerza de varianza no llega a 0.00232. El retorno gana en *todos* los puntos. Solver corre a `w_SPX = 0` → **esquina CMC**.
- **Si λ > 6.94:** incluso con la pendiente más suave, la fuerza de varianza supera a 0.00232. La varianza gana en *todos* los puntos. Solver corre a `w_SPX = 1` → **esquina SPX**.
- **Si λ ∈ [0.131, 6.94]:** existe un punto interior donde las dos fuerzas se igualan → solución no-esquina.

### 3.4 Por qué la grilla actual produce esquinas

```
        λ_low=0.131               λ_high=6.94
            │                        │
   0.05    │   1                    │  10    20    30
    ●------│----●---------|---------│---●-----●-----●
   esquina │    interior            │     esquina SPX
    CMC    │ (única solución real)  │
```

De los cinco puntos de `LAMBDA_GRID`, **cuatro caen fuera del rango interior**. Las esquinas no son una preferencia económica del modelo; son consecuencia matemática de elegir λ fuera del rango donde el trade-off entre retorno y varianza realmente existe.

### 3.5 Conclusión parcial

`λ_low` y `λ_high` no son hiperparámetros que el usuario elige: son **propiedades de los datos**. Salen de:

- el gap de retorno entre activos (numerador)
- las pendientes mínima y máxima de la varianza en el segmento factible (denominador)

Cualquier λ fuera del intervalo así definido empuja al solver a una esquina por construcción matemática. La grilla actual no está calibrada al rango interior del universo SPX / CMC200.

---

## 4. Por qué se complica con la capa de DL (parte dinámica)

Hasta ahora el análisis usó los momentos históricos planos. Pero el optimizador no recibe esos momentos. Recibe momentos *mezclados* período a período, modulados por la salida de DL.

### 4.1 Las dos personalidades de cada activo

Del histórico, una sola vez, se calculan momentos condicionados por régimen:

```
SPX en bull   →  retorno medio +0.18%/sem,  vol 1.96%/sem
SPX en bear   →  retorno medio +0.21%/sem,  vol 2.76%/sem
CMC en bull   →  retorno medio +0.28%/sem,  vol 9.70%/sem
CMC en bear   →  retorno medio +0.87%/sem,  vol 9.91%/sem    ← bear gana más en CMC
```

Estos perfiles `(μ̂(i, k), Σ̂(i, j, k))` son los ingredientes fijos: no se tocan después.

### 4.2 Qué es `p_dl(t)`

`p_dl(t)` es la probabilidad de régimen que predice la LSTM cuantílica para el período `t` y el activo `i`. Concretamente, la LSTM produce los deciles de la distribución forward de retornos; `regimen_from_deciles` los convierte en `p_bull(t) = fracción de deciles ≥ BULL_THRESHOLD` (y `p_bear = 1 − p_bull`).

Es decir, para cada `t` del horizonte forward y cada activo, la red dice algo como: *"yo creo que SPX está al 70 % en bull y al 30 % en bear este período"*.

### 4.3 La mezcla `μ_mix(t), σ_mix(t)`

Lo que el optimizador efectivamente recibe son los momentos *mezclados* por período:

```
μ_mix(i, t)        = Σ_k  p_dl(t, i, k) · μ̂(i, k)
σ_mix(i, j, t)     = Σ_{k_i, k_j}  p_dl(t, i, k_i) · p_dl(t, j, k_j) · Σ̂(i, j, k_i)
```

Cada período `t` tiene su propio `(μ_mix(t), σ_mix(t))`, definido por la receta de mezcla que armó DL.

### 4.4 Cómo eso mueve `λ_low, λ_high`

Volvamos a la fórmula del rango interior:

```
λ_low(t)  = |μ_mix_SPX(t) − μ_mix_CMC(t)| / pendiente_max(σ_mix(t))
λ_high(t) = |μ_mix_SPX(t) − μ_mix_CMC(t)| / pendiente_min(σ_mix(t))
```

Tanto el numerador como el denominador dependen de los momentos mezclados, que dependen de `p_dl(t)`. Por lo tanto los umbrales **no son fijos**: cambian con `t` y con cada reentrenamiento de DL.

Para cuantificar la magnitud, calculé `λ_low, λ_high` para cinco perfiles distintos de `p_dl`:

| caso | μ_SPX | μ_CMC | σ²_SPX | σ²_CMC | λ_low | λ_high |
|---|---|---|---|---|---|---|
| Hist promedio (p=0.5/0.5)        | +0.0019 | +0.0057 | 0.00057 | 0.00962 | **0.217** | **10.900** |
| Bull fuerte ambos (0.9/0.9)      | +0.0018 | +0.0034 | 0.00042 | 0.00946 | 0.092 | 7.384 |
| Bear fuerte ambos (0.1/0.1)      | +0.0020 | +0.0081 | 0.00072 | 0.00979 | 0.341 | 15.230 |
| SPX bull, CMC bear (0.9/0.1)     | +0.0018 | +0.0081 | 0.00042 | 0.00979 | 0.341 | **23.769** |
| SPX bear, CMC bull (0.1/0.9)     | +0.0020 | +0.0034 | 0.00072 | 0.00946 | **0.082** | 2.643 |

Observaciones clave:

- **`λ_high` recorre de 2.6 a 23.8** según el perfil de p_dl: un factor 9 ×.
- **`λ_low` recorre de 0.08 a 0.34**: factor 4 ×.
- El "rango seguro" — λ que cae en `(λ_low(t), λ_high(t))` para *todo* `t` del horizonte — es la intersección de todos esos rangos. En el caso extremo es `(0.341, 2.643)`, mucho más estrecho que cualquier rango individual.

### 4.5 La consecuencia

Aún si recalibrara la grilla al rango histórico `(0.131, 6.94)` (el de la sección estática), el optimizador internamente está usando momentos que tienen un rango interior **distinto, variable y a priori desconocido**. La grilla queda mal calibrada de una forma que se mueve cada vez que DL cambia su predicción:

1. **Variación intra-solve:** dentro de un mismo solve, distintos `t` pueden tener rangos interiores muy distintos. Un λ fijo puede caer interior en algunos períodos y esquina en otros, dentro del mismo problema.
2. **Variación entre corridas:** cada vez que se reentrena la LSTM, `p_dl(t)` cambia, los momentos mezclados cambian, y los umbrales se desplazan. Una grilla fija no puede acompañar.

El problema no es solo calibrar mejor la grilla: es que estoy apuntando a un blanco que se mueve con la salida de DL.

---

## 5. Por qué la opción del regulador V no resuelve el trasfondo

En conversaciones previas surgió la idea de agregar `V` como restricción dura sobre la varianza, en línea con la formulación lagrangiana del PDF (ec. 2–4):

```
Σ_{i,j} w_i,t · w_j,t · Σ_{i,j,t}  ≤  V
```

Con esta restricción, las carteras cuya varianza supera `V` se vuelven infactibles, y `λ` deja de ser un input del usuario para pasar a ser un multiplicador KKT calculado por el solver.

### 5.1 Lo que V sí arregla

Geométricamente, sí rompe las esquinas cuando la restricción está activa. Si `V < σ²_CMC`, la esquina CMC se vuelve infactible. Si además `V < σ²_de_la_otra_esquina`, ambas se descartan, y el óptimo cae forzosamente en algún punto interior del elipsoide de varianza recortado por la simplex.

Esto es una solución **estructural** al síntoma estático: las esquinas dejan de ser una opción válida para el solver.

### 5.2 Lo que V no arregla

V tiene su propio "rango útil" para que la restricción muerda sin volverse infactible:

```
σ²_min(t)  ≤  V  ≤  σ²_max(t)
```

- `σ²_min(t)` ≈ varianza de la cartera de mínima varianza dado `(μ_mix(t), σ_mix(t))`.
- `σ²_max(t)` ≈ varianza de la cartera de máxima varianza.

Y ambos extremos **dependen de los momentos vigentes**, que son los que mezcla DL. Por lo tanto V tiene exactamente el mismo problema dinámico que λ:

- Si DL predice un régimen de baja volatilidad → `σ²_max(t)` baja → un V fijo puede pasar a ser **no-vinculante** → la restricción no muerde y vuelve la esquina CMC.
- Si DL predice un régimen muy volátil → `σ²_min(t)` sube → un V fijo puede pasar a ser **infactible** → IPOPT no resuelve.

V cambia la unidad del hiperparámetro (de "1/varianza" a "varianza") pero no cambia la estructura del problema: tanto `λ` como `V` son escalares fijos enfrentados a momentos que se mueven con DL. El blanco sigue moviéndose, solo que ahora se expresa en otras coordenadas.

---

## 6. Resumen del problema

Hay dos niveles anidados:

### Nivel estático (sin DL)

La FO solo admite óptimo interior cuando λ cae en un rango `(λ_low, λ_high)` determinado por los momentos del universo. Con SPX / CMC200 históricos, el rango es `(0.13, 6.94)`. La grilla actual tiene cuatro de cinco puntos fuera de ese rango, así que las esquinas son consecuencia matemática de la elección de λ, no respuesta económica del modelo.

### Nivel dinámico (con DL)

El optimizador no usa los momentos históricos planos. Usa `μ_mix(t), σ_mix(t)`, mezclas convexas de los `(μ̂, Σ̂)` por régimen ponderadas por `p_dl(t)`. Eso significa:

- `λ_low(t), λ_high(t)` varían período a período.
- Cambian cada vez que se reentrena la LSTM.
- La intersección segura (λ interior para todo `t`) puede ser muy estrecha y no se conoce a priori.

Una grilla fija de λ no puede acompañar ese blanco móvil.

### Por qué V no resuelve el trasfondo

La idea del regulador V (PDF ec. 2–4) rompe las esquinas geométricamente cuando está activa, pero V hereda el mismo problema: su rango útil también depende de los momentos vigentes, que se mueven con DL. Un V fijo puede pasar a no-vinculante en regímenes calmos (vuelve la esquina CMC) o a infactible en regímenes volátiles. Cambia la coordenada del problema sin cambiar su estructura.

### La duda de fondo

El planteo actual usa un hiperparámetro escalar fijo (`λ` o `V`) sobre momentos que son dinámicos y vienen de DL. Esa incoherencia estructural — input estático del usuario versus datos que se mueven con la red — es lo que produce el problema, y no se resuelve solo con recalibrar la grilla.

No tengo claro si la causa última es:

- un error de mi parte en cómo conecté la salida de la LSTM con el optimizador (la mezcla `μ_mix, σ_mix`), o
- algo del planteo del modelo en sí que conviene repensar.
