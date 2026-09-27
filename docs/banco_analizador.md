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

## Regla de aceptación (spec §4)

Un cambio al analizador entra si mejora la mediana de calorías O de proteína sin empeorar ninguna otra mediana más de
2 puntos ni la tasa de fallos, y sin subir la latencia p50 más de un 25 %. Cada cambio es un lote con su corrida
anotada (antes/después) en su commit.

## Límites

Nutrition5k es comida de cafetería de EE. UU.: no hay mangú, moro ni sancocho. El sesgo dominicano llegará con las
fotos cedidas por usuarios (capa 3 del panel). Los tests de platos criollos siguen siendo la red para eso.

## Atribución

Datos: **Nutrition5k** (Thames et al., Google Research, 2021), `gs://nutrition5k_dataset`, licencia CC BY 4.0. Se usan
150 fotos cenitales y sus metadatos, sin modificar, solo para medir; las imágenes no se redistribuyen (la caché vive en
el VPS y el repo guarda solo sus sha256).
