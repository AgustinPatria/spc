# E4 — Calidad de los quintiles del LSTM (3 gaps de inspeccionar_deciles)

## TL;DR

**El problema es asimetrico por activo**: el LSTM tiene skill modesto
para SPX y **estructuralmente inservible para CMC200**.

| Diagnostico                                  | SPX           | CMC200          |
|----------------------------------------------|---------------|-----------------|
| ¿Le gana al baseline incondicional?          | Si (+2.85%)   | **No (-0.88%)** |
| ¿Capta vol regimes? (corr ancho-vs-vol real) | Si (+0.73)    | Apenas (+0.26)  |
| ¿La mediana cruza 0 en test?                 | Si            | **No (-0.019..-0.008)** |
| std(pred) / std(real) (mediana)              | 12%           | **6%**          |
| Skill condicional global                     | Modesto       | Casi nulo       |

→ Diagnostico actualizado: ya NO es "el LSTM no funciona" en general.
**Es "el LSTM no funciona para CMC200"**. SPX esta razonable.

## Numeros clave

### Conditionality (test, n=18)

Para cada nivel q, std(pred_q) sobre las ventanas / std(retorno realizado):

| q   | SPX  | CMC200 |
|-----|------|--------|
| 0.1 | 0.17 | 0.08   |
| 0.3 | 0.11 | 0.04   |
| 0.5 | 0.12 | 0.06   |
| 0.7 | 0.22 | 0.03   |
| 0.9 | 0.29 | 0.03   |

- **SPX**: 11-29% de la variabilidad real. Mejor en colas (q=0.9 con 29%
  capta picos altos), peor en el centro. Razonable.
- **CMC200**: solo 3-8%. La mediana en test queda en
  **[-0.019, -0.008]** — siempre negativa, nunca cruza 0. El LSTM
  esencialmente decidio "CMC200 es bear permanente" y oscila tantito
  alrededor de eso.

### Pinball LSTM vs baseline incondicional (test)

Baseline: predecir los cuantiles empiricos del train, constantes para
toda ventana de test.

**Total sumado sobre niveles q**:

| asset  | LSTM     | Baseline | ratio  | ganador |
|--------|----------|----------|--------|---------|
| SPX    | 0.0297   | 0.0306   | 0.972  | LSTM    |
| CMC200 | 0.0974   | 0.0965   | 1.009  | **BASE** |

→ SPX: LSTM gana por **2.85%**. Aporte real pero marginal.
→ CMC200: LSTM **pierde por 0.88%**. **Es peor que predecir constante**.

Por nivel q:

| q   | SPX LSTM | SPX BASE | SPX gana? | CMC200 LSTM | CMC200 BASE | CMC200 gana? |
|-----|----------|----------|-----------|-------------|-------------|--------------|
| 0.1 | 0.0042   | 0.0043   | ✅ LSTM   | 0.0138      | 0.0142      | ✅ LSTM      |
| 0.3 | 0.0084   | 0.0085   | ≈         | 0.0223      | 0.0221      | ❌ BASE      |
| 0.5 | 0.0079   | 0.0088   | ✅ LSTM   | 0.0288      | 0.0270      | ❌ BASE      |
| 0.7 | 0.0060   | 0.0052   | ❌ BASE   | 0.0227      | 0.0236      | ✅ LSTM      |
| 0.9 | 0.0032   | 0.0038   | ✅ LSTM   | 0.0098      | 0.0097      | ❌ BASE      |

### Sharpness condicional (test)

¿La banda q10-q90 predicha covaria con la vol realizada (|y|)?

| asset  | mean width | std width | corr(width, \|y\|) |
|--------|------------|-----------|--------------------|
| SPX    | 0.0711     | 0.0024    | **+0.73**          |
| CMC200 | 0.1922     | 0.0040    | +0.26              |

→ SPX: 0.73 es **fuerte**. El LSTM SI capta vol regimes — predice bandas
mas anchas cuando viene una semana volatil.
→ CMC200: 0.26 es debil. El ancho es esencialmente constante.

## Causas probables de la asimetria SPX vs CMC200

H_a — **Estandarizacion compartida con outliers**: la normalizacion usa
un solo (mean, std) calculado sobre los retornos del train. CMC200 tiene
retornos con cola larga (-36% a +25%); algunos outliers pueden estar
empujando el mean/std de modo que las observaciones "tipicas" quedan
muy comprimidas en el espacio normalizado, y el LSTM aprende menos.

H_b — **Misma red para los dos activos**: el LSTM toma como input las
ventanas (H, A=2) y predice (A=2, Q=5). Una sola red comparte pesos
entre activos. Si SPX domina la perdida (porque tiene varianza menor
y por tanto pinball loss menor en magnitud absoluta), CMC200 podria
estar siendo subentrenado.

H_c — **Lookback H=52 incorrecto para crypto**: cripto cambia de regimen
mas rapido que equity. H=52 semanas (1 año) puede ser demasiado largo
para CMC200 — la red ve historia que ya no es relevante.

H_d — **Tamaño del dataset insuficiente**: 77 ventanas de train para
aprender un mapeo (52, 2) -> (2, 5) es muy poca data. SPX puede salirse
con baja varianza condicional aprendida; CMC200 necesita mas para
capturar la dinamica de vol.

H_e — **El LSTM esta sub-ajustado para CMC200 porque pinball loss
penaliza menos los grandes desvios cuando la vol es alta**. Pinball
es asimetrico pero LINEAL en el desvio — un retorno de +13% se penaliza
con peso q=0.5 igual que un +2.6%, escalado. La perdida por activo se
suma sin reweighting; CMC200 contribuye mas al loss total y la red
intenta minimizar esa parte con la solucion mas conservadora (cuantil
incondicional). Eso explicaria por que CMC200 *converge* al baseline.

## Hipotesis post-E4

Las hipotesis del README original quedan reemplazadas por:

| Hip nuevo | Estado |
|-----------|--------|
| El LSTM funciona razonable para SPX | ✅ confirmada (skill modesto, capta vol) |
| El LSTM no aporta para CMC200 | ✅ confirmada (peor que baseline en pinball) |
| La estandarizacion compartida o el modelo compartido daña a CMC200 | a investigar |
| Lookback H=52 puede ser inadecuado para crypto | a investigar |

## Direcciones concretas (orden tentativo de impacto / esfuerzo)

### D1 — Modelo separado por activo (alto impacto, bajo esfuerzo)
Entrenar dos LSTMs independientes, cada uno con su propio
(mean, std), sus propios pesos, su propio early stopping.
Hipotesis a refutar: si CMC200 sigue siendo flat con su modelo
dedicado, el problema NO es interferencia entre activos.

### D2 — Estandarizacion robusta (bajo impacto, bajo esfuerzo)
Reemplazar mean/std por mediana/MAD (o quantile-based) para no
ser arrastrado por outliers en CMC200. Tambien probar normalizacion
por activo (lo cual es equivalente a D1 si el modelo es por activo).

### D3 — Barrer H (medio impacto, bajo esfuerzo)
Entrenar con H ∈ {13, 26, 52, 104} y comparar pinball test por activo.
Si CMC200 mejora con H mas chico, el problema es lookback.

### D4 — Pinball loss reweighted por activo (alto impacto, bajo esfuerzo)
Escalar la perdida de cada activo por 1/std(y_train_a) para que
CMC200 no domine simplemente por tener mas magnitud. Eso fuerza
al modelo a invertir capacidad equilibradamente.

### D5 — Reentrenar con mas data si disponible (medio impacto, esfuerzo
incierto). 77 train obs es poquito para aprender 5 deciles condicionales
de 2 activos. Si hay datos adicionales (mas historia o mas activos
similares), ayuda.

### D6 — Cambiar arquitectura (medio impacto, alto esfuerzo)
Probar un MLP con feature engineering manual (vol rolling, momentum)
en lugar de LSTM. La cantidad de data sugiere que el LSTM puede ser
overkill para esto.

## Recomendacion concreta

**D1 (modelo separado por activo) primero**. Es la prueba mas limpia
y de menor riesgo. Si CMC200 sigue mal con modelo dedicado, sabemos
que el problema no es interferencia, y ahi pasamos a D3/D4.

Si D1 ayuda significativamente a CMC200, eso valida la hipotesis y
nos da una version mejorada del LSTM con minimo cambio arquitectonico.

## Outputs en este directorio

- `conditionality.csv`        — std(pred)/std(real) por (q, asset, split)
- `pinball_vs_baseline.csv`   — comparacion LSTM vs incondicional
- `sharpness.csv`             — ancho medio, std, corr con vol
- `pinball_vs_baseline.png`   — barras side-by-side
- `conditionality.png`        — std de predicciones por nivel
- `width_vs_vol.png`          — scatter ancho vs vol realizada
