# Banco de pruebas del analizador de platos

[P1-PLAN-LOTE-570..573 · 2026-09-27] Spec: `docs/superpowers/specs/2026-09-27-analizador-banco-de-pruebas-design.md`
(raíz del workspace). Plan: `docs/superpowers/plans/2026-09-27-analizador-banco-de-pruebas.md`.

## Qué mide

El analizador REAL (`vision_agent.process_image_with_vision`, el mismo de `/api/diary/upload`) sobre 150 platos de
Nutrition5k con su verdad pesada, siempre los mismos (`data/banco_analizador/manifest.json`, sha256 de cada PNG). Por
plato: error relativo `|estimado − verdad| / max(verdad, piso)` de calorías (piso 100 kcal) y de cada macro (piso
10 g); y qué componentes principales (≥ 15 % de la masa) reconoce, por CLASE (arroz, pollo, huevo…), con el «other» de
los dos lados medido. Agregado: mediana y p90, % de platos con calorías a ≤ 20 %, fallos, latencia y coste.

## Cómo repetirlo (en el VPS: la clave de visión solo vive allí)

```bash
cd /opt/mealfit/backend
/home/ubuntu/miniforge3/envs/mealfit/bin/python scripts/banco_analizador_preparar.py \
    --cache /home/ubuntu/banco_analizador/cache --solo-cache          # verifica (o rellena) las 150 fotos
/home/ubuntu/miniforge3/envs/mealfit/bin/python scripts/banco_analizador_correr.py \
    --cache /home/ubuntu/banco_analizador/cache --salida /home/ubuntu/banco_analizador/<nombre>.json \
    --guardar --notas "<qué cambió>"
```

`--guardar` escribe una fila en `analyzer_benchmark_runs` (el panel `/admin` la muestra). La línea que imprime al final
(`n=… ok=… valida=… kcal_med=… prot_med=… recall=… coste_usd=…`) es el resumen. Sale 0 si la corrida es válida (fallos
≤ 10 %), 2 si la caché no cuadra o la visión está apagada (no gasta), 3 si no es válida.

## Línea base (27-sep-2026, 19:06 UTC)

`gemini-3.8-flash` · `prompt_sha 30be2fac4b17` · `codigo_sha 3f9ac03f35a2` · `manifest_sha 94638f856c30` (ver la fila en
`analyzer_benchmark_runs` y `/home/ubuntu/banco_analizador/linea_base.json`).

| Métrica | Mediana | p90 | Meta (spec §0) |
|---|---|---|---|
| Error de calorías (kcal_med) | **26,2 %** | 52,2 % | ≤ 15 % |
| Error de proteína | **28,5 %** | 62,5 % | ≤ 15 % |
| Error de carbohidratos | 23,8 % | 62,0 % | — |
| Error de grasa | 31,7 % | 70,5 % | — |
| Platos con calorías a ≤ 20 % | 38,7 % | | — |
| Componentes principales reconocidos | **83,5 %** | | ≥ 90 % |

Fallos 0 de 150 · latencia p50 6,0 s / p90 7,9 s · «other» sin clasificar: 1,5 % en la verdad, 5,1 % en lo que dice el
analizador · 411.900 tokens de entrada + 92.235 de salida = **US$0,65** por corrida.

**Lo que dicen los números (para elegir la primera mejora):** el analizador SUBESTIMA de forma sistemática —sesgo
mediano de calorías −17 % (85 platos por debajo del 10 %, 35 por encima), grasa −20 %, proteína −14 %, carbohidratos
−3,5 %— y el sesgo crece con el plato: 100-300 kcal +5 % (error 21 %), 300-600 kcal −15 % (21 %), > 600 kcal −38 %
(38 %). Porciones comprimidas hacia la media y grasa oculta (aceite, aderezos) son las dos hipótesis a probar primero.

## Regla de aceptación (spec §4) — ahora pareada (P1-PLAN-LOTE-601)

La regla original («mejora la mediana de calorías o de proteína sin empeorar otra más de 2 puntos») comparaba UNA
corrida contra otra, y eso no distingue nada: ver «Ruido del banco». Desde el lote 601 un cambio entra solo si, plato a
plato contra la media de **al menos 2 corridas de la base** (errores topados al 100 %, intervalo bootstrap al 90 %):

- calorías o proteína mejoran **≥ 2 puntos de media con el intervalo entero por debajo de 0**;
- ninguna macro empeora más de 2 puntos con el intervalo entero por encima de 0;
- no hay más fallos que en la peor base, y la latencia p50 no sube más de un 25 %.

```bash
python scripts/banco_analizador_comparar.py \
    --base /home/ubuntu/banco_analizador/linea_base.json,/home/ubuntu/banco_analizador/linea_base_r2.json \
    --candidata /home/ubuntu/banco_analizador/<corrida>.json        # sale 0 si entra, 1 si no
```

Función pura: `banco_analizador.comparar_pareado`. Cada cambio sigue siendo un lote con su corrida y su comparación
anotadas en el commit.

## Ruido del banco (medido el 27-sep, P1-PLAN-LOTE-601)

La misma base (mismo `prompt_sha 30be2fac4b17` y `codigo_sha 3f9ac03f35a2`) corrida dos veces:

| Métrica | Corrida 1 | Corrida 2 |
|---|---|---|
| Error de calorías (mediana) | 26,2 % | **28,8 %** |
| Error de proteína | 28,5 % | 30,8 % |
| Error de grasa | 31,7 % | 36,2 % |
| Platos con calorías a ≤ 20 % | 38,7 % | 35,3 % |

Entre corridas, la lectura de un plato cambia una mediana de 6,8 % en calorías (p90 22,5 %). Con ese ruido la regla de
una sola corrida rechazaba la base contra sí misma. Promediar las dos lecturas apenas baja el error (calorías 26,2 %,
grasa 33,3 %): el error del analizador es sobre todo **sesgo**, no azar.

## Variantes del prompt probadas y rechazadas (P1-PLAN-LOTE-601)

Leídos plato a plato, los platos de > 600 kcal que salían un 38 % cortos no eran «porciones grandes» sino comida densa
en montón: ~120 g de almendras (≈ 690 kcal) leídos como ~50 g, 8-10 tiras de tocineta como 2, 45 g de aceite en una
ensalada. Dos variantes, cada una con su corrida guardada en `analyzer_benchmark_runs`:

| Variante | Sesgo kcal | Diferencia pareada kcal (IC 90 %) | Proteína | Veredicto |
|---|---|---|---|---|
| v1 · peso de cada componente con la escala del plato + grasa que no se ve (`prompt 6b77e1dc4259`) | −14,4 % | +0,8 [−0,7, +2,3] | −0,5 [−1,8, +0,7] | rechazada |
| v2 · lo denso se cuenta (frutos secos, tocineta, aceite visible; `prompt 424f4d0bee4d`) | −11,5 % | 0,0 [−1,4, +1,5] | −0,9 [−2,3, +0,5] | rechazada |

v2 reduce el sesgo (los platos grandes pasan de −38 % a −35 %) pero empuja los pequeños a sobreestimar (+11 %): el error
no baja. Coincide con el estudio de 40 modelos sobre Nutrition5k citado en `vision_luna.md`: el prompt no mueve la
precisión; la mueven el **modelo** y la **foto**. Palancas siguientes, por impacto esperado: probar en este banco un
modelo de visión mejor (cambiar `MEALFIT_VISION_MODEL`, sin código), calcular las macros desde el catálogo cuando el
ítem se resuelve, y las correcciones reales de los usuarios (`scan_outcome` y, con la política nueva, la capa 2).

## Límites

Nutrition5k es comida de cafetería de EE. UU.: no hay mangú, moro ni sancocho. El sesgo dominicano llegará con las
fotos cedidas por usuarios (capa 3 del panel). Los tests de platos criollos siguen siendo la red para eso.

## Atribución

Datos: **Nutrition5k** (Thames et al., Google Research, 2021), `gs://nutrition5k_dataset`, licencia CC BY 4.0. Se usan
150 fotos cenitales y sus metadatos, sin modificar, solo para medir; las imágenes no se redistribuyen (la caché vive en
el VPS y el repo guarda solo sus sha256).
