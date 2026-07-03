# Supermercado RD artificial (`supermarket_products`)

[P1-SUPERMARKET-DB · 2026-07-02] Base de datos pública y editable del "supermercado
dominicano" de MealfitRD. Cada fila es **una presentación comprable** de un alimento
verificado: alimento + marca opcional + presentación + porción + duración + precio RD$.
Seed inicial: los +200 alimentos verificados del owner (237 presentaciones, 14 categorías).

**Objetivo de producto**: fuente de verdad del mercado RD para la futura integración con
el sistema de lista de compras — el cliente podrá elegir hasta la **marca** de su yogurt,
carnes, arroz, habichuelas, etc. La conexión se hará vía `master_food_name` (link suave
por nombre al catálogo `master_ingredients`, curado desde la admin UI; NO es FK dura para
que el supermercado pueda contener productos aún sin mapear).

## Piezas

| Pieza | Archivo | Notas |
|---|---|---|
| Migración SSOT | `migrations/p1_supermarket_db_2026_07_02.sql` (ambos dirs, P3-MIGRATIONS-SSOT) | `CREATE TABLE IF NOT EXISTS` + unique index de variante `uq_supermarket_products_variant` (lower(food_name), lower(brand), lower(presentation)) + DO $$ sanity. Aplicada a Neon 2026-07-02. |
| Migración media (fase 2) | `migrations/p1_supermarket_media_2026_07_02.sql` (ambos dirs) | `image_url` (NULL ⇒ placeholder por categoría en el frontend) + `description` (specs del SKU para el modal de detalle). Aplicada a Neon 2026-07-02. |
| Router | `backend/routers/supermarket.py` (prefix `/api/supermarket`) | Ver contrato abajo. |
| Seed base | `backend/scripts/seed_supermarket_2026_07_02.py` | Dry-run default, `--commit` para escribir. Idempotente (`ON CONFLICT DO NOTHING`) — re-ejecutar NO pisa ediciones de la admin UI. 237 filas genéricas ejecutadas 2026-07-02. |
| Seed leches (fase 2) | `backend/scripts/seed_supermarket_leches_2026_07_02.py` | 114 SKUs de leche con MARCA real (catálogo La Sirena 2026-07): 13 tipos de leche × 35 marcas, con `description`. Primera familia con variantes; patrón a replicar para carnes/arroz/etc. |
| Seed pimienta negra (fase 2) | `backend/scripts/seed_supermarket_pimienta_negra_2026_07_02.py` | 11 SKUs (Badia molida/entera/orgánica/molinillo/fina, Wala, Oriente, granel Lb). Los seeds por familia son data-only: NO requieren redeploy — el API lee Neon directo. |
| Página landing | `frontend/src/pages/SupermarketPage.jsx` (+ `Supermarket.module.css`) | Ruta `/supermercado` (App.jsx), link en Footer ("Empresas" → "Supermercado RD"), título/description en RouteTitle. Fase 2: catálogo estilo supermercado online — tarjeta por SKU (imagen o placeholder-emoji por categoría) → modal de detalle con specs + otras variantes del mismo alimento; filtros por categoría/marca, orden por precio, paginación "Mostrar más" (48). |
| Test ancla | `backend/tests/test_p1_supermarket_db.py` | Import+prefix, migraciones dual-dir idénticas/idempotentes (base + media), gate admin por handler, GET sin paywall, frontend backend-only, seed gateado. |

## Contrato de endpoints

| Endpoint | Auth | Notas |
|---|---|---|
| `GET /api/supermarket/products` | Pública + `RateLimiter(60/60s)` per-IP | Solo `active` salvo `include_inactive=1` **con token admin**. Filtros `q`/`category`, `limit` clamp 1000. NO usa `verify_api_quota` (página de marketing, cero costo LLM — misma razón que la historial-quota-exemption). Devuelve `{products, total, categories}`. |
| `POST /api/supermarket/products` | `_verify_admin_token` + `_check_admin_rate_limit` | 409 si colisiona con el unique index de variante. |
| `PATCH /api/supermarket/products/{id}` | admin (igual) | Update parcial whitelisted (`_MUTABLE_FIELDS`), `updated_at=now()`. 409 en colisión de variante. |
| `DELETE /api/supermarket/products/{id}` | admin (igual) | Hard delete. Para ocultar sin borrar: `PATCH active=false` (la UI lo muestra atenuado en modo admin). |

## Modo edición del landing

La página `/supermercado` es pública (navegación read-only). El botón de llave abre un
input de token; el token (= `CRON_SECRET`) se guarda en `sessionStorage`
(`mf_market_admin_token`) y viaja como `Authorization: Bearer` en cada mutación. Token
inválido/stale → 401/403 → la página lo descarta y vuelve al modo público. El cliente
JAMÁS escribe directo a la tabla (simétrica a la invariante I6).

## Convenciones de datos (seed)

- `brand = NULL` ⇒ producto genérico verificado. Variantes de marca = filas adicionales
  con el mismo `food_name`.
- `portion_label` ∈ {Mínima, Mediana, Mayor, Única, Mediano}; `duration_label` ∈
  {7 días, 15 días, 30 días, Relativo}; `notes` canónicas: "Uso exclusivo plan de 7 días",
  "Rinde para planes de 7 y 15 días", "Rinde para todos los planes", "Relativo".
- Transcripción fiel al PDF del owner con 2 normalizaciones: "L" a secas → "Lb";
  "Relativa" → "Relativo". Strings compuestos ("Paquete 2L", "Cartón L", "1.47 L")
  VERBATIM — se curan desde la admin UI.

## Roadmap — estado

1. **Variantes de marca** — ✅ cargadas (capturas La Sirena + Nacional, ~1,800 productos /
   ~128 familias, 2026-07-02). Nuevas marcas: owner vía admin UI.
2. **Conexión lista de compras** — ✅ CERRADA (P1-SUPERMARKET-MATCH + P1-SUPERMARKET-PREFS +
   P1-SUPERMARKET-COSTING · 2026-07-02): `POST /api/supermarket/match` + panel "Marcas del
   súper" (Dashboard) + `user_brand_preferences` + overlay del envase preferido en el costeo
   (`fetch_brand_pref_packages`, knob `MEALFIT_BRAND_PREF_COSTING`).
3. **Sync a `market_packages`** — ✅ (fill-only-empty + reconciliación de precios report-first:
   `scripts/sync_supermarket_to_market_packages_2026_07_02.py` y
   `scripts/sync_supermarket_prices_report_2026_07_02.py`).
4. **Conexión a la GENERACIÓN de platos** — ✅ fase 1 (P1-SUPERMARKET-PERSONALIZATION ·
   2026-07-03, audit v6 · P1-2): las marcas preferidas del usuario se inyectan como señal de
   preferencia POSITIVA al planner/day-gen vía [`backend/brand_personalization.py`](../brand_personalization.py)
   (`build_brand_pref_context`, canal `taste_profile` — mismo patrón del taste aprendido; knob
   `MEALFIT_BRAND_PREF_PERSONALIZATION`, señal suave que jamás fuerza platos ni rompe clínica).
   Gap-report de familias sin master verificado (candidatos de expansión de catálogo):
   `scripts/supermarket_family_gap_report_2026_07_03.py` (read-only). Test ancla:
   `test_p1_supermarket_personalization.py`.

## Roadmap (pendiente)

- Imágenes de producto + cadencia de actualización de precios + más familias (owner-side).
- Fase 2 de generación: derivar masters verificados nuevos desde las familias del gap-report
  (patrón USDA→JSON→owner PRICES→--commit).
