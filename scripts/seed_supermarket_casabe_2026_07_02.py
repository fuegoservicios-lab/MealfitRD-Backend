"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de CASABE.

Octava familia con variantes de MARCA del Supermercado RD: 28 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Wala (triangulares/redondo/
chips), Guaraguano (tradicional/extrafino/atípico/K-Sabito's/artesanal/herbs),
El Sabrocito, Buren, Casabi.

Agrupación: todo bajo "Casabe" (sabores en la presentación) SALVO el Wala Albahaca,
que va bajo el food existente "Casabe albahaca" (generico del PDF, mismo precio RD$99).

Notas:
  * Deduplicados 2 listados repetidos de La Sirena (Wala Natural Triangular 11 Oz y
    Wala al Ajillo Triangular 11 Oz aparecen dos veces).
  * Se incluyen 4 SKUs marcados "No disponible" en La Sirena (Casabi snack/ajo,
    Guaraguano Ranchito/Perejil) — son catálogo real con precio de referencia; el
    owner puede ocultarlos con PATCH active=false si prefiere.

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_casabe_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_casabe_2026_07_02.py --commit   # inserta
"""
import os
import sys

try:
    from dotenv import load_dotenv
    for _p in (os.path.join(os.path.dirname(__file__), "..", ".env"),
               os.path.join(os.getcwd(), ".env"), "/opt/mealfit/backend/.env"):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except Exception:
    pass

import psycopg

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

NOTES = "Precio de referencia La Sirena · 2026-07"
CATEGORY = "Panadería y harinas"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Wala ──
    ("Casabe", "Wala", "Triangular Natural 11 Oz", 94, "Casabe natural de yuca, gluten free"),
    ("Casabe", "Wala", "Triangular al Ajillo 11 Oz", 99, "Casabe al ajillo, gluten free"),
    ("Casabe", "Wala", "Triangular Aceite de Oliva 11 Oz", 105, "Casabe sabor aceite de oliva, gluten free"),
    ("Casabe", "Wala", "Triangular Parmesano 11 Oz", 115, "Casabe sabor parmesano, gluten free"),
    ("Casabe", "Wala", "Redondo Pequeño al Ajillo 11 Oz", 166, "Casabe redondo pequeño al ajillo"),
    ("Casabe", "Wala", "Chips Natural 7 Oz", 105, "Chips de casabe natural, gluten free"),
    ("Casabe", "Wala", "Chips Sabor Ajo 7 Oz", 105, "Chips de casabe sabor ajo, gluten free"),
    ("Casabe", "Wala", "Chips Sabor Parmesano 7 Oz", 105, "Chips de casabe sabor parmesano, gluten free"),
    ("Casabe albahaca", "Wala", "Triangular Albahaca 11 Oz", 99, "Casabe sabor albahaca, gluten free"),
    # ── Guaraguano ──
    ("Casabe", "Guaraguano", "Tradicional Natural Grande (unidad)", 215, "Casabe tradicional de yuca, torta grande"),
    ("Casabe", "Guaraguano", "Tradicional Original 8 Oz", 195, "Casabe tradicional original, pan plano de yuca (3 unidades)"),
    ("Casabe", "Guaraguano", "Al Ajillo Grande (unidad)", 225, "Casabe al ajillo (garlic), torta grande"),
    ("Casabe", "Guaraguano", "Extrafino Natural 4 Oz", 165, "Casabe extrafino natural original"),
    ("Casabe", "Guaraguano", "Extrafino Natural 8 Oz", 230, "Casabe extrafino (thin) natural original"),
    ("Casabe", "Guaraguano", "Atípico Natural 6.75 Oz", 85, "Casabe atípico natural (4 pedazos)"),
    ("Casabe", "Guaraguano", "Atípico al Ajillo 7 Oz", 90, "Casabe atípico al ajillo (4 pedazos)"),
    ("Casabe", "Guaraguano", "Caja K-Sabito's al Ajillo 4 Oz", 220, "Casabitos artesanales al ajillo — artesanal de la capital del casabe, Monción"),
    ("Casabe", "Guaraguano", "Artesanal Ranchito 9 Oz", 195, "Casabe artesanal ranchito monseñor, natural original (4 unidades)"),
    ("Casabe", "Guaraguano", "Herbs Perejil Ajillo 6 Oz", 180, "Casabe con hierbas: perejil y ajillo (3 unidades)"),
    # ── El Sabrocito ──
    ("Casabe", "El Sabrocito", "Natural Pequeño (unidad)", 64, "Casabe natural artesanal pequeño"),
    ("Casabe", "El Sabrocito", "Natural Grande (unidad)", 137, "Casabe natural artesanal grande"),
    ("Casabe", "El Sabrocito", "Al Ajillo Grande 4/1", 120, "Casabe al ajillo grande"),
    ("Casabe", "El Sabrocito", "5 Estrellas 40 Oz", 199, "Casabe dietético calidad insuperable (Super Casabe 5 Estrellas)"),
    # ── Buren ──
    ("Casabe", "Buren", "Premium Natural 385 gr", 195, "Pan de yuca premium natural, gluten free (4 pedazos)"),
    ("Casabe", "Buren", "Al Ajillo 300 gr", 245, "Casabe artesanal al ajillo, horneado (4 tortas)"),
    # ── Casabi ──
    ("Casabe", "Casabi", "Caja Casabitos Natural 300 gr", 245, "Casabitos artesanales naturales, pan plano de Monción"),
    ("Casabe", "Casabi", "Caja Casabitos con Ajo 300 gr", 239, "Casabitos artesanales con ajo"),
    ("Casabe", "Casabi", "Funda Snack Horneados 35 gr", 58, "Casabitos snack horneados crujientes, gluten free"),
]

_INSERT = """
INSERT INTO public.supermarket_products
    (food_name, brand, presentation, portion_label, duration_label,
     price_rd, notes, category, master_food_name, description, is_verified, active)
VALUES (%s, %s, %s, NULL, NULL, %s, %s, %s, %s, %s, true, true)
ON CONFLICT (lower(food_name), lower(coalesce(brand,'')), lower(coalesce(presentation,'')))
DO NOTHING
RETURNING id
"""


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    seen = set()
    for (food, brand, pres, *_rest) in ROWS:
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    brands = {r[1] for r in ROWS}
    print(f"Seed casabe: {len(ROWS)} SKUs · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, CATEGORY, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
