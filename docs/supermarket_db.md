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

## Roadmap (pendiente)

1. **Variantes de marca**: cargar marcas reales por alimento (owner, vía admin UI).
2. **Conexión lista de compras**: resolver `master_food_name` → `master_ingredients`
   (match case-insensitive + aliases) y exponer selección de marca/presentación en la
   lista de compras del plan (preferencia de usuario + recosteo por `price_rd`).
3. Posible sync `supermarket_products` → `master_ingredients.market_packages` para que
   el costeo P1-PKG use los mismos precios que muestra el supermercado.
