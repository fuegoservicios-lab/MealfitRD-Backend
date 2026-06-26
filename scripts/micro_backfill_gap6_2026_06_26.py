"""[P1-MICRO-BACKFILL-GAP6 · 2026-06-26] Backfill de micros NULL + densidades g/taza (auditoría gap #6).

ROOT CAUSE (auditoría "¿100%?"): el panel de micros sub-reporta sistemáticamente porque (1) ~28% del
catálogo tenía ≥1 de los 9 micros extendidos NULL (NULL se propaga como 0 sin sumar → falso-bajo) y (2)
frutas/veggies medidos en "taza" SIN density_g_per_cup descartaban TODOS sus micros (modo catastrófico).

FIX (idempotente, NULL-only, NO toca valores existentes):
  - density_g_per_cup (USDA cup-weights, raw/as-eaten) a las frutas/veggies cup-measurables que faltaban
    (Mango/Melón/Piña/Sandía/Lechosa/Guineo + Cebolla/Ají/Papa). Sandía tenía AMBAS densidades NULL.
  - 9 micros extendidos (satfat/zinc/folate/vit_a/vit_c/vit_e/vit_k/selenio/omega3) per 100g (USDA SR) en
    los staples con celdas NULL — SOLO valores de alta confianza.

DELIBERADAMENTE OMITIDO (lado seguro — el panel coverage-aware maneja los NULL restantes):
  - Garnish/seasoning sin cup-density: Perejil/Cilantro (vitK altísima → over-count), Ajo/Jengibre/Limón
    (se usan en cdas, no en taza). Replica la lección del corrective previo.
  - Gandules vit_e/vit_k (valor incierto para leguminosa cocida) → NULL.
  - Plátano/Yuca/Chinola/Guineo verde cup-density (rara vez en taza; medidos por unidad).
  - cholesterol (no es uno de los 9 del panel DRI).

PROVENANCE: USDA FoodData Central / SR Legacy per 100g (cocido/as-eaten según corresponda). Yogures con
satfat=0.1 confirmados nonfat por su fats_g=0.37 en el catálogo (no se inventa). Condimentos (sal/vinagre)
≈0 de todo por composición. El panel es ADVISORY (no gate duro) y NULL-only → riesgo acotado.

    python scripts/micro_backfill_gap6_2026_06_26.py [--commit]      (DRY-RUN sin --commit)
"""
import os, sys
try:
    from dotenv import load_dotenv
    for _p in (os.path.join(os.path.dirname(__file__), "..", ".env"),
               os.path.join(os.getcwd(), ".env"), "/opt/mealfit/backend/.env"):
        if os.path.exists(_p):
            load_dotenv(_p); break
except Exception:
    pass
import psycopg

NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

# USDA cup-weights (g por taza, raw/as-eaten) — frutas/veggies cup-measurables que faltaban.
DENSITY = {
    "Mango": 165, "Melón": 160, "Piña": 165, "Sandía": 152, "Lechosa": 145, "Guineo": 150,
    "Cebolla": 160, "Ají cubanela": 150, "Ají morrón": 149, "Papa": 150,
}

# Columnas de micro (nombres reales del schema).
COL = {
    "satfat": "saturated_fat_g_per_100g", "zinc": "zinc_mg_per_100g", "folate": "folate_mcg_dfe_per_100g",
    "vita": "vitamin_a_mcg_rae_per_100g", "vitc": "vitamin_c_mg_per_100g", "vite": "vitamin_e_mg_per_100g",
    "vitk": "vitamin_k_mcg_per_100g", "sel": "selenium_mcg_per_100g", "o3": "omega3_ala_g_per_100g",
}

# Valores per 100g (USDA). Solo celdas que estaban NULL (UPDATE es WHERE col IS NULL → idempotente).
MICRO_FILL = {
    # Condimentos (≈0 por composición):
    "Sal":                {"satfat": 0, "zinc": 0.1, "folate": 0, "vita": 0, "vitc": 0, "vite": 0, "vitk": 0, "sel": 0.1, "o3": 0},
    "Vinagre blanco":     {"satfat": 0, "zinc": 0, "folate": 0, "vita": 0, "vitc": 0, "vite": 0, "vitk": 0, "sel": 0, "o3": 0},
    "Vinagre balsámico":  {"zinc": 0.08, "folate": 0, "vita": 0, "vitc": 0, "vite": 0, "vitk": 0, "sel": 0.5, "o3": 0},
    "Mostaza":            {"satfat": 0.3, "zinc": 0.6, "folate": 7, "vita": 1, "vitc": 1.5, "vite": 0.3, "vitk": 1.8, "sel": 25, "o3": 0.1},
    # Granos/almidones (cocidos; enriquecidos donde aplica):
    "Arroz blanco":       {"satfat": 0.05, "folate": 58, "vita": 0, "vitc": 0, "vite": 0.04, "vitk": 0, "o3": 0.01},
    "Arroz integral":     {"satfat": 0.2, "folate": 4, "vita": 0, "vitc": 0},
    "Pasta integral":     {"vita": 0, "vitc": 0},
    "Quinoa":             {"vitc": 0},
    "Pan blanco familiar": {"zinc": 0.7, "folate": 85, "vita": 0, "vitc": 0, "vite": 0.2, "vitk": 5, "sel": 28, "o3": 0.1},
    "Tortilla de trigo":  {"zinc": 0.6, "folate": 96, "vita": 0, "vitc": 0, "vite": 0.3, "vitk": 1.5, "sel": 18, "o3": 0.2},
    "Tortilla integral":  {"zinc": 1.0, "folate": 30, "vita": 0, "vitc": 0, "vite": 0.4, "vitk": 2, "sel": 20, "o3": 0.3},
    "Casabe":             {"zinc": 0.3, "folate": 27, "vita": 0, "vitc": 0, "vite": 0, "vitk": 2, "sel": 0.7, "o3": 0},
    "Casabe albahaca":    {"zinc": 0.3, "folate": 27, "vita": 0, "vitc": 0, "vite": 0, "vitk": 2, "sel": 0.7, "o3": 0},
    # Proteínas (cocidas):
    "Ajo":                {"satfat": 0.09, "zinc": 1.16, "folate": 3, "vita": 0},
    "Carne de res":       {"vitc": 0, "sel": 20, "folate": 7, "vita": 0},
    "Cerdo":              {"vitc": 0, "vite": 0.2, "vitk": 0, "sel": 35, "o3": 0.02, "folate": 5, "vita": 2},
    "Jamón de pavo":      {"vite": 0.1, "vitk": 0, "sel": 20, "folate": 4, "vita": 0},
    "Longaniza dominicana": {"zinc": 2.0, "vitc": 0, "vite": 0.2, "vitk": 1, "sel": 18, "o3": 0.1, "folate": 3, "vita": 0},
    "Salami":             {"zinc": 2.6, "vitc": 0, "vite": 0.2, "vitk": 1.5, "sel": 18, "o3": 0.2, "folate": 2, "vita": 0},
    "Queso de hoja":      {"zinc": 2.9, "vitc": 0, "vite": 0.1, "vitk": 2, "sel": 14, "o3": 0.1, "folate": 10, "vita": 150},
    # Lácteos:
    "Leche evaporada":    {"vite": 0.1, "vitk": 0.3, "vita": 65},
    "Yogurt":             {"satfat": 0.1},                # nonfat (fats_g=0.37 en catálogo)
    "Yogurt griego sin azúcar": {"satfat": 0.1},          # nonfat (fats_g=0.37)
    # Frutas/veggies (satfat ≈0):
    "Brócoli":            {"satfat": 0.04},
    "Coliflor":           {"satfat": 0.13},
    "Vainitas":           {"satfat": 0.04},
    "Manzana":            {"satfat": 0.03},
    "Melón":              {"satfat": 0.05},
    "Piña":               {"satfat": 0.02},
    "Plátano maduro":     {"satfat": 0.1},
    "Atún en agua":       {"satfat": 0.2},
    # Semillas:
    "Semillas de chía":   {"vitk": 0, "vita": 0},
}


def main():
    if not NEON:
        print("FATAL: NEON url"); sys.exit(1)
    print(f"commit={COMMIT}")
    dens = micro = 0
    with psycopg.connect(NEON) as conn:
        for nm, dv in DENSITY.items():
            if COMMIT:
                n = conn.execute("UPDATE public.master_ingredients SET density_g_per_cup=%s "
                                 "WHERE name=%s AND density_g_per_cup IS NULL", (dv, nm)).rowcount
            else:
                n = conn.execute("SELECT 1 FROM public.master_ingredients "
                                 "WHERE name=%s AND density_g_per_cup IS NULL", (nm,)).rowcount
            if n:
                print(f"  density {nm} = {dv}")
            dens += n
        for nm, cells in MICRO_FILL.items():
            for short, v in cells.items():
                col = COL[short]
                if COMMIT:
                    n = conn.execute(f"UPDATE public.master_ingredients SET {col}=%s "
                                     f"WHERE name=%s AND {col} IS NULL", (v, nm)).rowcount
                else:
                    n = conn.execute(f"SELECT 1 FROM public.master_ingredients "
                                     f"WHERE name=%s AND {col} IS NULL", (nm,)).rowcount
                if n:
                    print(f"  micro {nm}.{col} = {v}")
                micro += n
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. densities={dens}, micro cells={micro}")
        else:
            print(f"\nDRY-RUN. pendientes: densities={dens}, micro cells={micro}. Re-run con --commit.")


if __name__ == "__main__":
    main()
