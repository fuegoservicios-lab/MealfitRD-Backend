"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de ARROZ BLANCO.

Décima familia con variantes de MARCA del Supermercado RD: 53 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02):

  * "Arroz blanco" (41): Campos, Wala, La Garza, Pimco (incl. Gourmet y precocido
    parboiled), Bisonó, Dos Pinos, El Molino, Luz de Luna — de 1 Lb a sacos de 50 Lb.
  * Arroces especiales como foods propios con `master_food_name="Arroz blanco"`
    (así el selector de la lista de compras puede ofrecerlos como alternativa premium):
    "Arroz basmati" (Goya, Vigo), "Arroz jazmín" (Goya, Roland, Vigo),
    "Arroz arborio" (Riso Scotti), "Arroz valencia" (Goya).
  * "Arroz sazonado" (mezclas preparadas, NO sustituto de arroz plano): Vigo Yellow,
    Goya Primavera.
  * BONUS del search: Pan blanco Wala Viga 820 gr (bajo "Pan blanco familiar") y
    Vinagre Blanco Wala 16 Oz (coincide exacto con el genérico RD$25 del PDF).

EXCLUIDOS a propósito: sobres blancos de papel (papelería) y Loza Crema (lavaplatos).
Deduplicado: Vigo Basmati 2 Lb aparece dos veces en La Sirena (mismo precio); los
integrales La Garza/Goya ya viven en seed_supermarket_arroz_integral.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_arroz_blanco_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_arroz_blanco_2026_07_02.py --commit   # inserta
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
GRANO = "Granos y cereales"
PAN = "Panadería y harinas"
SALSA = "Salsas y aderezos"

# Arroces especiales → alternativa premium del arroz blanco en la lista de compras.
MASTER_OVERRIDE = {
    "Arroz basmati": "Arroz blanco",
    "Arroz jazmín": "Arroz blanco",
    "Arroz arborio": "Arroz blanco",
    "Arroz valencia": "Arroz blanco",
}

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Campos ──
    ("Arroz blanco", "Campos", "Funda Premium 1 Lb", 55, "Arroz premium grano extra largo, fortificado", GRANO),
    ("Arroz blanco", "Campos", "Funda Premium 5 Lb", 235, "Arroz premium grano extra largo — la mejor de la naturaleza", GRANO),
    ("Arroz blanco", "Campos", "Funda Premium 20 Lb", 925, "Arroz premium grano extra largo", GRANO),
    ("Arroz blanco", "Campos", "Saco Premium 30 Lb", 1415, "Arroz premium grano extra largo", GRANO),
    ("Arroz blanco", "Campos", "Saco Premium 50 Lb", 2315, "Arroz premium grano extra largo, fortificado", GRANO),
    # ── Wala ──
    ("Arroz blanco", "Wala", "Funda Selecto 1 Lb", 42, "Arroz selecto", GRANO),
    ("Arroz blanco", "Wala", "Funda Selecto 5 Lb", 169, "Arroz selecto", GRANO),
    ("Arroz blanco", "Wala", "Funda Selecto 10 Lb", 327, "Arroz selecto", GRANO),
    ("Arroz blanco", "Wala", "Funda Selecto 20 Lb", 669, "Arroz selecto", GRANO),
    ("Arroz blanco", "Wala", "Funda Premium 5 Lb", 175, "Arroz premium", GRANO),
    ("Arroz blanco", "Wala", "Funda Premium 10 Lb", 348, "Arroz premium", GRANO),
    # ── La Garza ──
    ("Arroz blanco", "La Garza", "Funda 10 Lb", 475, "Arroz enriquecido", GRANO),
    ("Arroz blanco", "La Garza", "Funda Premium 3 Lb", 150, "Arroz premium enriquecido", GRANO),
    ("Arroz blanco", "La Garza", "Funda Premium 5 Lb", 245, "Arroz premium", GRANO),
    ("Arroz blanco", "La Garza", "Funda Premium 20 Lb", 945, "Arroz premium", GRANO),
    ("Arroz blanco", "La Garza", "Saco Premium 30 Lb", 1475, "Arroz premium", GRANO),
    ("Arroz blanco", "La Garza", "Saco Premium 50 Lb", 2455, "Arroz premium", GRANO),
    # ── Pimco ──
    ("Arroz blanco", "Pimco", "Funda Selecto 3 Lb", 148, "Arroz selecto especial", GRANO),
    ("Arroz blanco", "Pimco", "Funda Selecto 5 Lb", 230, "Arroz selecto especial", GRANO),
    ("Arroz blanco", "Pimco", "Funda Selecto 10 Lb", 445, "Arroz selecto grado 1", GRANO),
    ("Arroz blanco", "Pimco", "Saco Selecto 50 Lb", 2215, "Arroz selecto especial", GRANO),
    ("Arroz blanco", "Pimco", "Funda Premium 3 Lb", 159, "Arroz premium grano extra largo, libre de impurezas", GRANO),
    ("Arroz blanco", "Pimco", "Funda Premium 5 Lb", 245, "Arroz premium super selecto, grano extra largo", GRANO),
    ("Arroz blanco", "Pimco", "Funda Premium 10 Lb", 465, "Arroz premium grano extra largo, libre de impurezas", GRANO),
    ("Arroz blanco", "Pimco", "Saco Premium 50 Lb", 2365, "Arroz premium super selecto", GRANO),
    ("Arroz blanco", "Pimco", "Funda Gourmet 5 Lb", 275, "Arroz gourmet super selecto, grano extra largo", GRANO),
    ("Arroz blanco", "Pimco", "Funda Gourmet 10 Lb", 485, "Arroz gourmet super selecto, grano extra largo", GRANO),
    ("Arroz blanco", "Pimco", "Funda Precocido 5 Lb", 295, "Arroz precocido (parboiled)", GRANO),
    # ── Bisonó ──
    ("Arroz blanco", "Bisonó", "Funda Super Selecto 5 Lb", 220, "Arroz super selecto premium", GRANO),
    ("Arroz blanco", "Bisonó", "Funda Super Selecto 10 Lb", 449, "Arroz super selecto premium", GRANO),
    ("Arroz blanco", "Bisonó", "Saco Super Selecto 20 Lb", 895, "Arroz super selecto", GRANO),
    ("Arroz blanco", "Bisonó", "Saco Super Selecto 30 Lb", 1340, "Arroz super selecto", GRANO),
    ("Arroz blanco", "Bisonó", "Saco Super Selecto 50 Lb", 2240, "Arroz super selecto", GRANO),
    ("Arroz blanco", "Bisonó", "Funda Selecto Enriquecido 5 Lb", 215, "Arroz selecto enriquecido", GRANO),
    ("Arroz blanco", "Bisonó", "Funda Selecto Enriquecido 10 Lb", 405, "Arroz selecto enriquecido", GRANO),
    ("Arroz blanco", "Bisonó", "Funda Selecto Enriquecido 20 Lb", 805, "Arroz selecto enriquecido", GRANO),
    ("Arroz blanco", "Bisonó", "Saco Selecto Enriquecido 50 Lb", 2125, "Arroz selecto enriquecido", GRANO),
    # ── Otras marcas ──
    ("Arroz blanco", "Dos Pinos", "Funda 10 Lb", 430, "Arroz selecto especial (Coopearroz)", GRANO),
    ("Arroz blanco", "Dos Pinos", "Saco 20 Lb", 865, "Arroz selecto especial (Coopearroz)", GRANO),
    ("Arroz blanco", "El Molino", "Funda 10 Lb", 395, "Arroz selecto grano largo, enriquecido", GRANO),
    ("Arroz blanco", "Luz de Luna", "Funda Selecto 10 Lb", 395, "Arroz selecto especial", GRANO),
    # ── Arroces especiales (master → Arroz blanco) ──
    ("Arroz basmati", "Goya", "Caja Aged 12 Oz", 285, "Arroz basmati añejado, importado de India", GRANO),
    ("Arroz basmati", "Vigo", "Paquete 2 Lb", 635, "Arroz basmati de variedad antigua, naturalmente aromático", GRANO),
    ("Arroz jazmín", "Goya", "Paquete 5 Lb", 580, "Arroz jazmín (jasmine rice)", GRANO),
    ("Arroz jazmín", "Roland", "Caja Fragrant 17.5 Oz", 285, "Arroz jazmín fragante premium de Tailandia, gluten free", GRANO),
    ("Arroz jazmín", "Vigo", "Paquete 2 Lb", 395, "Arroz jazmín importado, fragante", GRANO),
    ("Arroz arborio", "Riso Scotti", "Caja 500 gr", 225, "Arroz arborio italiano para risotto", GRANO),
    ("Arroz arborio", "Riso Scotti", "Caja 1 Kg", 505, "Arroz arborio italiano para risotto", GRANO),
    ("Arroz valencia", "Goya", "Paquete California Pearl 24 Oz", 270, "Arroz tipo valenciano (California pearl), grano corto", GRANO),
    # ── Arroz sazonado (mezclas preparadas) ──
    ("Arroz sazonado", "Vigo", "Sobre Yellow Rice 8 Oz", 120, "Arroz amarillo sazonado estilo español, listo para preparar", GRANO),
    ("Arroz sazonado", "Goya", "Caja Primavera 7 Oz", 139, "Arroz primavera con vegetales y cheddar", GRANO),
    # ── BONUS del search (SKUs legítimos) ──
    ("Pan blanco familiar", "Wala", "Viga Mediana 820 gr", 105, "Pan blanco de viga, mediano", PAN),
    ("Vinagre Blanco", "Wala", "Botella 16 Oz", 25, "Vinagre blanco", SALSA),
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

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS}
    print(f"Seed arroz blanco: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category) in ROWS:
                master = MASTER_OVERRIDE.get(food, food)
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, category, master, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
