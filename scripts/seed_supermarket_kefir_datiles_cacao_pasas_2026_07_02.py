"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de KÉFIR + DÁTILES
+ CACAO EN POLVO + PASAS.

Familias 125-128 del Supermercado RD (capturas del owner, 2026-07-02,
La Sirena).

KÉFIR (7):
  * DOBLE CALCE: Okava natural 32 Oz RD$420 = genérico "Pote 32 Oz" y
    Okava fresa 6 Oz RD$120 = genérico "Pote 6 Oz".
  * Okava ×5 (natural/vainilla/arándano/fresa 32 Oz, fresa 6 Oz) +
    Lifeway ×2 (original plain 8 Oz, low fat fresa banana 32 Oz).
  * FLAG al owner: existen 2 filas Lifeway previas bajo "Kefir" SIN
    acento ("Low Fat Fresa 8 Oz" RD$125 y "Low Fat Vainilla 32 Oz"
    RD$395) — mismos precio/tamaño que las capturas de hoy pero sabor
    distinto. Verificar en admin UI si son SKUs reales (renombrar a
    "Kéfir" para unificar familia) o transcripciones erradas (merge).

DÁTILES (1): el "Dates Mejool Lb" RD$340 ES el genérico (calce, no
duplicar). Entra Dynasty doypack 6 Oz RD$225. EXCLUIDO el cereal Post
Great Grains Raisins-Dates-Pecan 16 Oz — product-truth es cereal de
desayuno, no dátiles, y no existe food de cereal de caja en el catálogo
(criterio barras de chocolate).

CACAO EN POLVO (2): CALCE — Dona Jura (Cacau Foods) 55% 200 gr RD$130 =
genérico "Paquete 200 gr". + Dona Jura 70% 200 gr RD$160. El empaque
dice DONA JURA (empaque manda; el título del listing dice Cacau Foods,
la empresa).

PASAS (3): CALCE — Ligo caja 250 gr RD$189 = genérico "Paquete 250 gr".
+ Ligo caja 1 Oz RD$39 y Wala sobre 1 Oz RD$22 (sin semilla todas).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_kefir_datiles_cacao_pasas_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_kefir_datiles_cacao_pasas_2026_07_02.py --commit   # inserta
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
LACTEOS = "Lácteos y huevos"
FRUTAS = "Frutas"
OTROS = "Otros"

KEFIR = "Kéfir"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Kéfir · Okava ──
    (KEFIR, "Okava", "Botella Natural 32 Oz", 420, "Kéfir natural (0 gr azúcar añadida)", LACTEOS),
    (KEFIR, "Okava", "Botella Vainilla 32 Oz", 420, "Kéfir sabor vainilla", LACTEOS),
    (KEFIR, "Okava", "Botella Arándano 32 Oz", 420, "Kéfir sabor arándano (blueberry)", LACTEOS),
    (KEFIR, "Okava", "Botella Fresa 32 Oz", 420, "Kéfir sabor fresa", LACTEOS),
    (KEFIR, "Okava", "Botella Fresa 6 Oz", 120, "Kéfir sabor fresa, tamaño personal", LACTEOS),
    # ── Kéfir · Lifeway ──
    (KEFIR, "Lifeway", "Botella Original Plain 8 Oz", 125, "Kéfir original sin sabor ni azúcar (plain unsweetened)", LACTEOS),
    (KEFIR, "Lifeway", "Botella Low Fat Fresa Banana 32 Oz", 395, "Kéfir bajo en grasa sabor fresa-banana", LACTEOS),
    # ── Dátiles ──
    ("Dátiles", "Dynasty", "Funda 6 Oz", 225, "Dátiles, doypack", FRUTAS),
    # ── Cacao en polvo ──
    ("Cacao en polvo", "Dona Jura", "Caja 55% 200 gr", 130, "Chocolate en polvo 55% cacao (Cacau Foods)", OTROS),
    ("Cacao en polvo", "Dona Jura", "Caja 70% 200 gr", 160, "Chocolate en polvo 70% cacao (Cacau Foods)", OTROS),
    # ── Pasas ──
    ("Pasas", "Ligo", "Caja 250 gr", 189, "Pasas de California sin semilla (seedless raisins)", FRUTAS),
    ("Pasas", "Ligo", "Caja 1 Oz", 39, "Pasas de California sin semilla, tamaño snack", FRUTAS),
    ("Pasas", "Wala", "Sobre 1 Oz", 22, "Pasas sin semilla, tamaño snack", FRUTAS),
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
    for row in ROWS:
        if len(row) != 6:
            print(f"FATAL: fila con {len(row)} campos (esperados 6): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed kéfir + dátiles + cacao + pasas: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
