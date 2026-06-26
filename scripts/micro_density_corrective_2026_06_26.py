"""[micro-precision corrective II · 2026-06-26] Densidades g/taza + backfill de micros NULL.

ROOT CAUSE (audit del owner, plan 20635f78): el panel mostraba 5 micros "bajos" (Vit D, Omega-3,
Vit K, Vit E, Fibra) pero 2 eran FALSOS-BAJOS por imprecisión del catálogo:

  1. Ingredientes medidos en "taza" SIN `density_g_per_cup` → `to_grams("taza", info)` devuelve None →
     `micros_from_ingredient_string` devuelve None → los micros de ESE ingrediente NO se cuentan.
     El caso grave: BRÓCOLI (vitamin_k=101.6/100g) aparecía 3× como "1 taza de brócoli" → ~277 mcg de
     Vit K silenciosamente descartados → Vit K mostraba 66 cuando el real era ~144 (sobre el piso 120).
     Igual con Fibra (brócoli 2.4 + fresas 2.0 perdidos).
  2. Resueltos con micros NULL: Arroz integral / Carne de res / Ajo tenían vit_e/vit_k/omega3 = NULL.

FIX (idempotente, NULL-only, NO toca valores existentes):
  - density_g_per_cup (USDA cup-weights, raw/as-eaten) a veggies/frutas/carbs comúnmente medidos en taza.
  - vit_e/vit_k/omega3 a Arroz integral, Carne de res, Ajo (valores USDA per 100g).

DELIBERADAMENTE EXCLUIDOS: Perejil (vitK 1640) y Cilantro (vitK 310) — son guarnición ("para decorar",
cdas), y una cup-density errónea ahí sobre-contaría Vit K en ~1000 mcg. Under-count de guarnición = lado
seguro. Si algún día se miden en taza de verdad, añadir con cuidado.

Verificado (plan 20635f78, recompute build_micronutrient_report sex=M): Vit K 66→144 (OK), Fibra 35.9→38
(OK), coverage 68%→73%, gaps 5→3 (quedan Vit D / Vit E / Omega-3, genuinos, con suplemento sugerido).

LECCIÓN: tener fdc_id + macros ≠ panel de micros preciso. Si un ingrediente se mide en "taza" y le falta
density_g_per_cup, TODOS sus micros se pierden (no solo el peso). Verificar densidad para todo veggie/fruta.

    NEON_DATABASE_URL(_POOLED) en .env.  python scripts/micro_density_corrective_2026_06_26.py [--commit]
"""
import os, sys
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass
import psycopg

NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

# USDA cup-weights (g por taza, raw/as-eaten) para veggies/frutas/carbs comúnmente medidos en "taza".
DENSITY = {
    "Brócoli": 91, "Fresas": 152, "Lechuga": 36, "Aguacate": 150, "Manzana": 125, "Naranja": 180,
    "Repollo": 89, "Espinaca": 30, "Coliflor": 107, "Auyama": 116, "Zanahoria": 128, "Molondrones": 100,
    "Tomate": 180, "Pepino": 119, "Berenjena": 82, "Tayota": 132, "Vainitas": 110, "Remolacha": 136,
    # carbs/víveres altos en fibra, forma "taza de cubos"/"taza cocida":
    "Pasta integral": 140, "Ñame": 136, "Batata": 133, "Yautía": 130,
}
# vit_e_mg / vit_k_mcg / omega3_ala_g per 100g (USDA) para los resueltos con NULL.
MICRO_FILL = {
    "Arroz integral": {"vitamin_e_mg_per_100g": 0.06, "vitamin_k_mcg_per_100g": 0.6, "omega3_ala_g_per_100g": 0.014},
    "Carne de res":   {"vitamin_e_mg_per_100g": 0.17, "vitamin_k_mcg_per_100g": 1.2, "omega3_ala_g_per_100g": 0.05},
    "Ajo":            {"vitamin_e_mg_per_100g": 0.08, "vitamin_k_mcg_per_100g": 1.7, "omega3_ala_g_per_100g": 0.02},
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
        for nm, cols in MICRO_FILL.items():
            for col, v in cols.items():
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
