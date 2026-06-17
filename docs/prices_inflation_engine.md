# Motor de precios + inflación (P2-PRICES-ENGINE-1)

[P2-PRICES-ENGINE-1 · 2026-06-16] Precios reales en RD$ en la lista de compras,
mantenidos vivos por un índice de inflación en vez de re-encuestar el supermercado.

## Por qué esta arquitectura

El **costeo ya existía**: `shopping_calculator.aggregate_and_deduct_shopping_list`
([backend/shopping_calculator.py:7408](../shopping_calculator.py#L7408)) lee
`master_ingredients.price_per_lb`/`price_per_unit` y emite `estimated_cost_rd` por
ítem (acumula `total_estimated_cost`). Lo que faltaba: precios poblados, proveniencia,
y un mecanismo de actualización. Este módulo lo aporta **sin tocar el calculador**.

**Separación base ↔ vivo** (clave anti-compounding):

```
price_per_lb (vivo, lo que lee el calculador)
    = price_per_lb_base × (food_cpi_actual / food_cpi_del_período_base)
```

La base nunca cambia; el cron sólo reescala el vivo. Reaplicar el índice es idempotente.

**Honestidad**: un precio proyectado por índice es ESTIMADO, no exacto. El índice
agregado (subíndice de alimentos del BCRD) modela bien staples estables y peor los
perecederos volátiles → `price_confidence` ('high'/'medium'/'low') deja a la UI
etiquetar honestamente. El re-anclaje periódico (re-captura de la base) corrige drift.

## Schema (migración SSOT [`p2_prices_engine_1_2026_06_16.sql`](../migrations/p2_prices_engine_1_2026_06_16.sql))

`master_ingredients` (+7 columnas, todas NULLABLE → degradación grácil):

| Columna | Tipo | Rol |
|---|---|---|
| `price_per_lb_base` / `price_per_unit_base` | numeric | precio base (pre-inflación) |
| `price_base_period` | text YYYY-MM | período del índice contra el que se capturó la base |
| `price_source` | text | proveniencia ('nacional_online', 'manual', 'crowdsource'…) |
| `price_captured_at` | date | fecha de captura de la base (staleness) |
| `price_confidence` | text | high/medium/low (CHECK) |
| `price_adjusted_at` | timestamptz | última reescala del cron |

`price_inflation_index` (1 fila/mes): `period` (PK, YYYY-MM), `food_cpi` (>0),
`source` (default 'bcrd'), `note`, `ingested_at`.

## Motor ([backend/price_engine.py](../price_engine.py))

| Función | Rol |
|---|---|
| `ingest_inflation_index(period, food_cpi)` | upsert de un punto del IPC-alimentos BCRD |
| `import_base_prices(rows)` | puebla precios base online (UPDATE COALESCE por slug/name) |
| `recompute_adjusted_prices()` | reescala vivo = base × factor (núcleo del cron, idempotente) |
| `price_staleness_report()` | cobertura + staleness (observabilidad) |
| `clamp_factor` / `adjusted_price` | helpers puros (testables sin DB) |

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_PRICES_ENABLED` | `false` | gate maestro; OFF → el cron registra pero no escribe |
| `MEALFIT_PRICE_ADJUST_INTERVAL_H` | `24` | frecuencia del cron de reescala, clamp [1,168] |
| `MEALFIT_PRICE_STALENESS_DAYS` | `180` | umbral "precio viejo → re-ancla", clamp [1,1095] |
| `MEALFIT_PRICE_INFLATION_MAX_FACTOR` | `3.0` | clamp de sanidad del factor [1/N, N], clamp knob [1.1,50] |
| `MEALFIT_PRICE_REGION_DEFAULT` | `nacional_online` | `price_source` default del importer |

## Cron

`price_inflation_adjust` ([cron_tasks.py](../cron_tasks.py) `_price_inflation_adjust_job`),
registrado en `register_plan_chunk_scheduler`. Diario, auto-sanador (sólo escribe filas
cuyo precio efectivo cambia), gateado DENTRO de `recompute_adjusted_prices` → no-op si OFF.
Emite tick a `pipeline_metrics` (`_price_inflation_adjust_job_tick`).

## Adquisición de datos online (probado en vivo 2026-06-16)

| Fuente | Resultado | Método |
|---|---|---|
| **BCRD** (índice alimentos) | ✅ alcanzable | prensa/PDF mensual; valor manual al script |
| **Supermercados Nacional** | ✅ **scrapeable autónomo** | Magento server-rendered: precios en HTML crudo (`data-price-amount`), `requests` puro (sin JS) |
| **PedidosYa** | ❌ bloquea (HTTP 403) | anti-bot |

**Scraper autónomo** [`backend/scripts/fetch_nacional_prices.py`](../scripts/fetch_nacional_prices.py):
fetch de ~30 categorías reales **paginando** `?p=N` (con retry) → parser regex Magento → `MATCH`
(slug→categorías+keywords) → precio representativo = MEDIANA. Corrida en vivo: **96/108 ingredientes
mapeados con precio real** (~85% del catálogo de 113). El `MATCH` es DATA — extender keywords/categorías
mejora cobertura sin tocar código.

Normalización (lo difícil, verificado en vivo):
- `to_per_lb()`: la marca de venta-por-libra (", Lb" / "Por Libra") gana; si no, divide por el peso
  del paquete. Reconoce `Lb`, `Gr`, `Onz`, `Kg` ("Arroz 10 Lb" @426→42.7/lb; "Habichuela 800 Gr" @88→49.9/lb).
- `estimate_prices()` ramifica por unidad del catálogo: **peso** (lb/g) → RD$/libra; **envase**
  (botella/pote/lata/paquete) → el precio del envase ES el por-unidad; **pieza** (unidad/cabeza) →
  deriva la pieza de "N Und/Paq" / "Aprox N Unidades" / per-lb÷(unid/libra) y pobla AMBAS columnas
  cuando puede (manzana → 99.98/lb + 35.18/ud).

12 sin resolver = baja disponibilidad o keyword fino (clara/yema líquida, especias, queso cottage/
ricotta) — extensibles vía `MATCH`. Tests: [`test_p2_prices_nacional_scraper.py`](../tests/test_p2_prices_nacional_scraper.py).

**Política**: el scraper es SCRIPT supervisado, NO cron prod (un cambio de HTML no debe corromper
precios en silencio). El cron de inflación mantiene los vivos entre scrapes.

## ⚠️ Gate de display (sutil pero importante)

`shopping_calculator` lee `price_per_lb`/`price_per_unit` (VIVOS) **sin gate** → poblar los vivos
= mostrar RD$ en la lista de compras de TODOS los usuarios. Por eso:
- Los importers (`import_prices.py`, `fetch_nacional_prices.py --apply`) pueblan **sólo las columnas BASE**.
- Sólo `recompute_adjusted_prices()` escribe los vivos, y está **gateado por `MEALFIT_PRICES_ENABLED`**.
- ⇒ Mientras `MEALFIT_PRICES_ENABLED=false`, los vivos quedan en 0 y NO se muestra ningún precio,
  aunque la base esté poblada. El flip a `true` + el cron es lo que enciende el display.

## Índice BCRD: anclaje relativo

El motor sólo usa el RATIO `food_cpi_actual / food_cpi_base`, así que el valor absoluto del índice
no importa. Convención: anclar el mes de captura de la base en `food_cpi=100.0` y rodar cada mes con
la variación publicada del grupo alimentos (ej. mayo 2026: -0.58% → siguiente = anterior × 0.9942).
Esto evita depender de extraer el valor base-2019 oficial del PDF.

## SOP: arranque online (sin ir al supermercado)

1. **Aplicar la migración** a Neon (additive/idempotente).
2. **Índice**: `PYTHONPATH=backend python backend/scripts/ingest_bcrd_inflation.py 2026-06 100.0`
   (ancla; luego cada mes el valor rodado por la variación BCRD).
3. **Precios base autónomos**: `PYTHONPATH=backend python backend/scripts/fetch_nacional_prices.py --apply`
   (scrapea Nacional + importa BASE; los vivos NO se publican aún si el feature está OFF).
4. **Activar el display**: `MEALFIT_PRICES_ENABLED=true` en el `.env` del VPS + restart. El cron
   publica los vivos = base × índice; re-scrapeas para re-anclar cada 3-6 meses.

## Límites conocidos (no son bugs)

- El índice agregado subestima la volatilidad de perecederos → marcados `low` confidence.
- El scraper resuelve ~40/57 mapeados; charcutería/embutidos, especias y enlatados (atún) requieren
  más categorías en `CATEGORIES`/`MATCH` (data, no código).
- El `total_estimated_cost` se acumula en el calculador pero NO se retorna; el frontend puede sumar
  `estimated_cost_rd` por ítem. Surface del total backend = follow-up.

Test ancla: [`test_p2_prices_engine_1.py`](../tests/test_p2_prices_engine_1.py) + [`test_p2_prices_nacional_scraper.py`](../tests/test_p2_prices_nacional_scraper.py).
