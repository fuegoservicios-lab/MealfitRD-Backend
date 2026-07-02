"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de MOLONDRONES, PEPINO, REPOLLO, TAYOTA, VAINITAS y ZANAHORIA.

Familias 60-65 del Supermercado RD: 19 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Molondrones" (1): Lucas Pérez baby bandeja RD$84 ("No disponible").
    El "Fresco Lb" RD$54 CALZA EXACTO con el genérico — no se duplica.
  * "Pepino" (3): VMB dulces bandeja Lb RD$40; Essential Everyday kosher baby
    dills 16 Oz RD$260 (pepinillos encurtidos — misma familia, descripción
    honesta, criterio jengibre encurtido); vaso porcionado RD$43. El "Fresco
    Lb" RD$24 calza exacto con el genérico — no se duplica.
  * "Repollo" (2): chino Lb RD$25 y vaso porcionado RD$39. FLAG owner: el
    genérico "Unidad (mitad)" está a RD$56 y hoy La Sirena lista RD$104 —
    posible update de precio (no lo pisa el seed).
  * "Repollo morado" (1): el listing "Repollo Morado" RD$355 muestra un
    RADICCHIO en la foto — entra con descripción honesta (achicoria roja
    importada). El "Morado Lb" RD$59 calza exacto con el genérico.
  * "Tayota" (1): funda porcionada al vacío RD$39 ("No disponible"). La "Lb"
    RD$24 calza exacto con el genérico — no se duplica.
  * "Vainitas" (8): la China Larga Lb RD$44 CALZA EXACTO con el genérico "Lb"
    (tipo sin marca, criterio yautía); italiana Lucas Pérez y VMB Linda;
    Goya en lata (rebanadas 8/14.5 Oz + cortadas 8 Oz — conserva, misma
    familia, criterio habichuelas); española, china chata VMB y orgánicas
    ("No disponible" ×3).
  * "Zanahoria" (2): funda baby petite 12 Oz RD$175 y porcionada al vacío
    RD$49 ("No disponible"). La "Fresca Lb" RD$27 calza exacto con el
    genérico — no se duplica.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_vegetales_6fam_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_vegetales_6fam_2026_07_02.py --commit   # inserta
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
VEG = "Vegetales y verduras"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Molondrones ──
    ("Molondrones", "Lucas Pérez", "Paquete Baby (unidad)", 84, "Molondrones baby frescos en bandeja", VEG),
    # ── Pepino ──
    ("Pepino", "VMB", "Paquete Dulces Lb", 40, "Pepinos dulces (mini) frescos en bandeja, por libra", VEG),
    ("Pepino", "Essential Everyday", "Frasco Kosher Baby Dills 16 Oz", 260, "Pepinillos encurtidos kosher (baby dills)", VEG),
    ("Pepino", None, "Vaso Fresco Porcionado (unidad)", 43, "Pepino fresco porcionado en rodajas", VEG),
    # ── Repollo ──
    ("Repollo", None, "Chino Lb", 25, "Repollo chino (napa) fresco, por libra", VEG),
    ("Repollo", None, "Vaso Porcionado (unidad)", 39, "Repollo rallado porcionado, listo para cocinar", VEG),
    # ── Repollo morado ──
    ("Repollo morado", None, "Radicchio Importado (unidad)", 355, "Radicchio (achicoria roja importada) — listado en La Sirena como repollo morado", VEG),
    # ── Tayota ──
    ("Tayota", None, "Funda Porcionada Lb", 39, "Tayota porcionada empacada al vacío, por libra", VEG),
    # ── Vainitas ──
    ("Vainitas", None, "China Larga Lb", 44, "Vainita china larga fresca, por libra", VEG),
    ("Vainitas", None, "Española Lb", 48, "Vainita española fresca, por libra", VEG),
    ("Vainitas", None, "Orgánicas (paquete)", 190, "Vainitas orgánicas frescas", VEG),
    ("Vainitas", "Lucas Pérez", "Paquete Italiana (unidad)", 79, "Vainita italiana fresca en bandeja", VEG),
    ("Vainitas", "VMB", "Paquete Italiana Linda (unidad)", 60, "Vainita italiana fresca en bandeja", VEG),
    ("Vainitas", "VMB", "Paquete China Chata (unidad)", 60, "Vainita china chata fresca en bandeja", VEG),
    ("Vainitas", "Goya", "Lata Rebanadas 8 Oz", 88, "Habichuelas tiernas rebanadas en conserva (french style green beans)", VEG),
    ("Vainitas", "Goya", "Lata Rebanadas 14.5 Oz", 115, "Habichuelas tiernas rebanadas en conserva (french style green beans)", VEG),
    ("Vainitas", "Goya", "Lata Cortadas 8 Oz", 90, "Habichuelas tiernas cortadas en conserva (cut green beans)", VEG),
    # ── Zanahoria ──
    ("Zanahoria", None, "Funda Baby 12 Oz", 175, "Zanahorias baby (petite carrots) microwaveables", VEG),
    ("Zanahoria", None, "Funda Fresca Porcionada Lb", 49, "Zanahoria fresca pelada porcionada al vacío, por libra", VEG),
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
    print(f"Seed vegetales 6 familias: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
