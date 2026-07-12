"""[P1-PAVO-CATALOG · 2026-07-12] Inserta 'Pechuga de pavo' en master_ingredients (Neon).

Forense del plan vivo df263d1b: el plato "Pavo Desmenuzado Asado sobre Cama de Verduras
Salteadas" lleva `Pechuga de pavo` en ingredients_raw, pero el catálogo solo tenía
'Jamón de pavo' y 'Pavo molido' → `VERIFIED-ONLY-DROP` la excluía de la lista de compras
(fail-safe correcto, lista incompleta: la proteína del plato no se compraba). Cierra el
gap con el pipeline estándar de los add_foods_batch (USDA verificado + precio + densidad).

NUTRICIÓN: USDA SR Legacy fdc_id=174515 "Turkey, retail parts, breast, meat only, raw",
per 100 g crudo (convención del catálogo: carnes CRUDAS). Atwater: 23.34×4 + 2.33×9 =
114.3 ≈ 114 kcal ✓ (0 flags). Folate/omega-3 sin dato en SR → None (el panel los trata
como no-reportados, jamás 0 falso).

PRECIO: RD$285/lb ESTIMADO (entre Jamón de pavo 255/lb y Pavo molido 320/lb del propio
catálogo; pechuga fresca/congelada La Sirena). Ajustar con dato real de mercado cuando el
owner lo tenga — el estimate evita el gate anti-precio-0 sin distorsionar el costeo.

ALIAS: NO incluye 'pavo' a secas (ya es alias de Jamón de pavo — no romper ese mapping ni
canonicalize_pavo del coherence guard).

USO (en el VPS, /opt/mealfit/backend):
  python scripts/seed_pechuga_pavo_2026_07_12.py            # DRY-RUN
  python scripts/seed_pechuga_pavo_2026_07_12.py --commit   # inserta
"""
import datetime
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

import psycopg  # noqa: E402

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

RECORD = {
    "slug": "pechuga_de_pavo",
    "name": "Pechuga de pavo",
    "category": "Proteínas",
    "aliases": ["pechuga de pavo", "filete de pavo", "pechuga de pavo deshuesada",
                "pavo pechuga", "turkey breast"],
    "default_unit": "lb",
    "is_dominican_cultivar": False,
    # Un filete de pechuga de pavo es ~2x el de pollo (170 g) — porción cruda típica.
    "density_g_per_unit": 300,
    "density_g_per_cup": None,
    "nutrition_source": "usda",
    "fdc_id": 174515,
    "price_per_lb": 285,     # ESTIMADO — ver docstring
    "price_per_unit": 285,
    # ── USDA SR Legacy 174515, per 100 g crudo ──
    "kcal_per_100g": 114,
    "protein_g_per_100g": 23.34,
    "carbs_g_per_100g": 0.0,
    "fats_g_per_100g": 2.33,
    "fiber_g_per_100g": 0.0,
    "sugars_g_per_100g": 0.0,
    "saturated_fat_g_per_100g": 0.344,
    "sodium_mg_per_100g": 74,
    "cholesterol_mg_per_100g": 53,
    "calcium_mg_per_100g": 9,
    "iron_mg_per_100g": 0.76,
    "potassium_mg_per_100g": 267,
    "magnesium_mg_per_100g": 27,
    "phosphorus_mg_per_100g": 185,
    "zinc_mg_per_100g": 1.16,
    "vitamin_d_mcg_per_100g": 0.2,
    "vitamin_b12_mcg_per_100g": 1.35,
    "folate_mcg_dfe_per_100g": 7,      # SR "Folate, total" 7 µg (carne sin fortificar: total≡DFE)
    "vitamin_a_mcg_rae_per_100g": 5,
    "vitamin_c_mg_per_100g": 0.0,
    "vitamin_e_mg_per_100g": 0.09,
    "vitamin_k_mcg_per_100g": 0.0,
    "selenium_mcg_per_100g": 22.1,
    "omega3_ala_g_per_100g": 0.015,    # SR "PUFA 18:3 n-3 c,c,c (ALA)"
}


def main():
    if not _NEON:
        print("FATAL: NEON url ausente")
        sys.exit(1)
    with psycopg.connect(_NEON) as conn:
        exists = conn.execute(
            "SELECT 1 FROM public.master_ingredients WHERE name = %s", (RECORD["name"],)
        ).fetchone()
        if exists:
            print(f"~ EXISTE, salto: {RECORD['name']} (idempotente)")
            return
        cols = dict(RECORD)
        cols["nutrition_source_date"] = datetime.date.today()
        colnames = list(cols.keys())
        placeholders = ", ".join(["%s"] * len(colnames))
        if COMMIT:
            conn.execute(
                f"INSERT INTO public.master_ingredients ({', '.join(colnames)}) "
                f"VALUES ({placeholders})",
                [cols[c] for c in colnames],
            )
            conn.commit()
        print(f"{'+ INSERTADO' if COMMIT else '+ (dry) insertaría'}: {RECORD['name']} "
              f"[{RECORD['category']}] ({RECORD['kcal_per_100g']}kcal/"
              f"{RECORD['protein_g_per_100g']}P) lb={RECORD['price_per_lb']} "
              f"fdc={RECORD['fdc_id']}")


if __name__ == "__main__":
    main()
