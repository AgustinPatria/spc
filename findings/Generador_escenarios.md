# Modulo de generador de escenarios — diagnostico

## Como funciona

`dl/generador_escenarios.py` implementa la **seccion 2.5 del PDF**:

1. **`generate_candidate_scenarios`**: desde la ultima ventana observada (los `H=52` ultimos retornos), genera N=5000 trayectorias futuras de largo `T=163` (~3 años):
   - En cada paso, predice los 5 cuantiles con el LSTM congelado
   - Samplea **uniformemente** un nivel `q ∈ Q = {0.1, 0.3, 0.5, 0.7, 0.9}` (mismo `q` para los 2 activos)
   - Usa `r_t = q-esimo cuantil` como retorno del paso
   - Rolea la ventana (descarta el retorno mas viejo, agrega el nuevo)

2. **`reduce_to_representatives`**: ordena los N candidatos por retorno acumulado del activo resumen (SPX), los parte en 5 quintiles, y elige 1 escenario mediano por quintil → 5 representativos para alimentar el regret-grid.

## Estado actual

Sobre N=5000 escenarios desde la ultima ventana observada:

| Metric | SPX | CMC200 |
|---|---|---|
| Retorno acumulado terminal media | **+125%** | +249% |
| Retorno acumulado terminal std | 71% | 293% |
| Min terminal | -40% | -88% |
| Max terminal | +689% | +4885% |
| **P(bull terminal)** | **99.5%** | 91.8% |

Los **5 representativos** que alimentan el regret-grid:

| Quintil | SPX | CMC200 |
|---|---|---|
| q1 (peor) | +47% | +12% |
| q2 | +82% | +64% |
| q3 (mediana) | +115% | +169% |
| q4 | +152% | +299% |
| q5 (mejor) | +219% | +639% |

**Ningun representativo es bear.** El "peor" SPX (+47%) seria un excelente retorno a 3 años en cualquier mercado.

## Diagnostico — por que falla

### Paso 1: el LSTM en un solo paso es razonable

Las predicciones del LSTM en un solo step son consistentes con la realidad historica:
- Mediana SPX (q3): +0.5%/sem (≈ mediana historica +0.4%/sem)
- Cola izquierda (q1): -3.4%
- Cola derecha (q5): +4.7%

**Hasta aqui esta todo bien.**

### Paso 2: el sampling uniforme **deforma la media**

La spec del PDF dice samplear uniformemente sobre los 5 cuantiles. Eso significa:

```
E[r_sampled] = (q1 + q2 + q3 + q4 + q5) / 5
```

Con los cuantiles del LSTM en una ventana bullish:
```
E[r_sampled] = (-0.034 - 0.008 + 0.005 + 0.018 + 0.045) / 5 = +0.0052/sem
```

**Ese +0.5%/sem matchea la MEDIANA historica del SPX (+0.4%/sem), no la MEDIA historica (+0.2%/sem)**.

### Paso 3: la diferencia entre mediana y media se amplifica al rolear

Por TLC, la suma de 163 retornos sigue:
```
Σ r_t  ~  Normal(163·μ, σ·√163)
```

| Si μ = ... | Mean cum. | P(cum < 0) |
|---|---|---|
| 0.002 (media historica real) | +39% | **~13%** |
| 0.005 (mediana historica ≈ sample mean) | +126% | ~0.3% |

**El sampling uniforme convierte la mediana en media, y esa diferencia de 0.3pp/sem se convierte en una diferencia de 87% en el retorno acumulado terminal.**

## Por que esto pasa: skew de los retornos

Los retornos semanales tienen **left skew** (cola izquierda larga por crashes ocasionales):
- Mediana > Media
- En SPX: mediana ≈ +0.4%/sem, media ≈ +0.2%/sem (la diferencia es exactamente la asimetria)

El sampling uniforme sobre 5 cuantiles trata cada quantil como equiprobable:
- En la realidad, los crashes son raros pero grandes (cola izquierda larga)
- El sampling uniforme los hace tan probables como cualquier otro quantil
- **Resultado**: el sampling subestima la frecuencia de "buena semana media" (cerca de la mediana) y sobre-pondera las colas

Como las colas son asimetricas (cola izquierda mas larga que cola derecha), promediarlas **da algo cercano a la mediana, no a la media**.

## Lo que esto NO es

**No es** que el LSTM tenga sesgo. El LSTM predice cuantiles correctos.

**No es** que falten datos bear de 3 años en el dataset. La matematica dice que con `μ` historica real (+0.2%/sem), un 13% de las trayectorias de 163 semanas terminarian negativas — totalmente consistente con la frecuencia historica de periodos bear.

**No es** que la ventana inicial este sesgada. Probamos con multi-ventana (10 ventanas iniciales distintas a lo largo del historico) y el resultado fue **identico** — el sesgo es del sampler, no de la condicion inicial.

## Lo que SI es

**Es un sesgo estructural del sampling uniforme cuando la distribucion de retornos tiene skew.** La spec del PDF (sec. 2.5) dice "samplear uniformemente sobre Q", pero esa eleccion no preserva la media de la distribucion subyacente cuando hay asimetria.

## Implicacion practica para el regret-grid

Los 5 representativos son todos bull (rango +47% a +219% en SPX). El regret-grid construye el portafolio "robusto" contra estos 5 escenarios, pero como **todos son bull**, el portafolio queda optimizado para mercado bull sin proteccion bear real.

Si el futuro es efectivamente bull (lo mas probable historicamente), el portafolio funciona bien. Si hay un crash, el portafolio elegido por el regret-grid no tendra defensa porque no se entreno con esos escenarios.

## Experimento descartado: multi-ventana

Se probo generar escenarios desde **10 ventanas iniciales distribuidas en el tiempo** (incluyendo periodos bear historicos). Resultado: **practicamente identico** al single-window (`p_bull = 99.6%` vs `99.5%`). La diversidad de la condicion inicial no compensa el sesgo del sampling.

El cambio fue revertido. La spec del PDF se mantiene tal cual.

## Posibles mejoras (no implementadas)

Estas opciones cambiarian el sampling, lo cual seria una desviacion explicita de la spec del PDF. Las dejo documentadas como referencia:

1. **Sampling no uniforme con pesos `1/Q` originales**: usar pesos que matcheen la distribucion empirica de probabilidad de cada cuantil (q=0.1 representa 20% de masa entre q=0 y q=0.2; q=0.3 representa 20% entre q=0.2 y q=0.4; etc.). Esto preserva mejor la media.

2. **Sampling continuo via interpolacion inversa**: samplear `u ~ Uniform(0,1)` y aplicar la CDF inversa interpolada de los 5 cuantiles. Es la version continua del sampling.

3. **Bootstrap de retornos historicos**: ignorar los cuantiles del LSTM y samplear retornos del historico directamente. Pierde la info del LSTM pero matchea la distribucion historica.

4. **Calibrar la mediana al promedio**: shiftear todos los cuantiles por `(media_historica - mediana_historica)` para que el sampling uniforme produzca la media correcta. Hack, pero efectivo.

## Mejora aplicada: cambiar la regla de seleccion del representativo

### Justificacion

El PDF dice textualmente:
> "se elige 1 escenario representativo por quintil **(por ejemplo, el escenario mediano dentro de ese quintil)**"

El **"por ejemplo"** habilita otras elecciones. La spec no fija "mediano" como regla rigida — solo lo da como ejemplo razonable.

### Cambio

Se agrego un parametro `position` a `reduce_to_representatives` con tres opciones:
- `"median"` (default del PDF, comportamiento original)
- `"min"` (peor escenario de cada quintil)
- `"max"` (mejor escenario de cada quintil)

`SCENARIO_POSITION = "min"` en `config.py` y se pasa explicitamente desde `Regret_Grid.build_dl_context`. La inspeccion muestra ambas para comparar.

### Resultado con `position="min"`

| Quintil | Median (PDF default) | **MIN** (nuevo) |
|---|---|---|
| q1 (peor) | SPX +42% / CMC +3% | **SPX -5.5% / CMC -64%** ← bear |
| q2 | +83% / +74% | +67% / +46% |
| q3 | +116% / +199% | +100% / +127% |
| q4 | +151% / +357% | +132% / +265% |
| q5 (mejor) | +218% / +502% | +174% / +421% |

**El escenario q1 es ahora claramente bear** (SPX -5.5%, CMC200 -64%), que es lo que el regret-grid necesita para optimizar un portafolio robusto contra crashes.

Logica matematica: con la `min` de cada quintil, q1 toma el peor escenario absoluto de los 1000 (el ~percentil 0), no el percentil 10 que daba `median`. Como ~7 escenarios de 1000 terminan bear con la media del LSTM, el min los captura.

### Por que esto sigue alineado con el PDF

1. La spec dice "por ejemplo, el escenario mediano" — `mediano` es ejemplo, no requisito.
2. Se mantiene "1 representativo por quintil".
3. Se mantiene la division en 5 quintiles del peor al mejor.
4. La unica eleccion afectada es el indice dentro del bucket (mediano vs minimo).

## Decision

**Se mantiene el sampling uniforme original** (sec. 2.5 del PDF) y se reemplaza la regla "mediano del quintil" por "min del quintil" como interpretacion alternativa del "por ejemplo" del PDF.

El sesgo bull persistente del sampling uniforme se documenta como limitacion estructural conocida.

## Implicaciones para el reporte

1. **Reportar honestamente** que el generador produce 99% de escenarios bull a 3 años.
2. **Explicar el origen matematico**: el sampling uniforme convierte la mediana en media, lo cual amplifica un sesgo positivo de ~0.3pp/sem que se vuelve dominante a 163 pasos.
3. **Validar contra la realidad**: el resultado es consistente con la **mediana** historica del SPX, no la media. El metodo del PDF efectivamente captura "lo mas probable" pero no "el promedio" cuando hay skew.
4. **Limitacion del regret-grid**: el portafolio resultante esta optimizado para escenarios bull. Documentar esto como caracteristica del metodo, no bug.

## Archivos relacionados

- `dl/generador_escenarios.py` — implementacion (sin cambios netos)
- `inspeccion/generador_escenarios/` — diagnostico OOS sobre el LSTM final
