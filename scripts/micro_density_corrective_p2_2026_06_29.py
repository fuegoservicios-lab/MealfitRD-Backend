"""[P2-DENSITY-CUP-FOODS · 2026-06-29] (audit objetivo · P2-5) Backfill de `density_g_per_cup` para los
CUP-FOODS BÁSICOS que el corrector previo (micro_density_corrective_2026_06_26.py) NO cubrió: avena, leche,
harina, yogurt.

ROOT CAUSE (mismo modo de fallo que el gap #6, pero en alimentos de uso DIARIO): un ingrediente medido en
"taza" SIN `density_g_per_cup` → `grams_from_ingredient_string` devuelve None → `micros_from_ingredient_string`
descarta TODOS sus micros (no solo el peso). El corrector de 2026-06-26 cubrió veggies/frutas/víveres, pero
NO avena / leche / harina / yogurt — staples que aparecen como "1 taza de avena", "1 taza de leche" en
desayunos/batidos → sus micros (fibra, calcio, magnesio) se perdían silenciosamente → más falsos
'estimado_bajo' en el panel.

FIX (idempotente, NULL-only, NO toca valores existentes): density_g_per_cup (USDA cup-weights) para los
nombres canónicos más probables de estos staples. EXACT-match por `name` (igual que el corrector previo) →
los nombres que no existan en el catálogo simplemente reportan 0 (sin error). Corre DRY-RUN primero para ver
qué matchea; revisa la lista; luego `--commit`.

VERIFICACIÓN MANUAL REQUERIDA (cooked vs raw): la densidad debe ser COHERENTE con la base de las kcal del
catálogo. Avena y harina van como POLVO SECO (la receta las cocina después) → density seca. Leche y yogurt
van as-eaten (líquido/cremoso). Si tu catálogo guarda la avena/harina ya cocida (kcal ~70-90/100g en vez de
~360-389), NO uses estas densidades secas — ajusta. El DRY-RUN imprime el nombre + kcal_per_100g de cada
match para que confirmes la base ANTES de commitear.

    NEON_DATABASE_URL(_POOLED) en .env.  python scripts/micro_density_corrective_p2_2026_06_29.py [--commit]

ESTADO (verificado en Neon 2026-06-29): el catálogo YA tenía density_g_per_cup poblada para casi todos los
cup-foods (Avena=80, Leche=244, Harina de trigo=125, Yogurt/griego=245, Arroz=185) — el backfill previo ya
los cubrió. El ÚNICO con NULL era "Harina de maíz precocida", corregido a 120 g/taza (kcal 361.8 = polvo seco).
Este script queda como red de seguridad idempotente para futuros INSERTs de cup-foods.
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

# USDA cup-weights (g por taza). Avena/harina = polvo SECO; leche/yogurt = as-eaten. Varias variantes de
# nombre por staple (exact-match; las que no existan reportan 0). Revisa el kcal_per_100g del DRY-RUN para
# confirmar que la base (seca/cocida) es coherente con la densidad ANTES de commitear.
DENSITY = {
    # avena (rolled oats, DRY): ~80 g/taza
    "Avena": 80, "Avena en hojuelas": 80, "Hojuelas de avena": 80,
    # leche (líquida): ~244 g/taza
    "Leche": 244, "Leche entera": 244, "Leche descremada": 245, "Leche semidescremada": 244,
    # harina (de trigo, DRY): ~125 g/taza; maíz precocida ~120
    "Harina": 120, "Harina de trigo": 125, "Harina de maíz precocida": 120, "Harina de maíz": 110,
    # yogurt (griego/natural, as-eaten): ~245 g/taza
    "Yogurt griego": 245, "Yogur griego": 245, "Yogurt": 245, "Yogur": 245, "Yogurt natural": 245,
}


def main():
    if not NEON:
        print("FATAL: falta NEON_DATABASE_URL(_POOLED) en .env"); sys.exit(1)
    print(f"[P2-DENSITY-CUP-FOODS] commit={COMMIT}")
    pending = applied = 0
    with psycopg.connect(NEON) as conn:
        for nm, dv in DENSITY.items():
            # Mostrar el match + kcal_per_100g (para validar base seca/cocida) ANTES de tocar nada.
            row = conn.execute(
                "SELECT density_g_per_cup, kcal_per_100g FROM public.master_ingredients WHERE name=%s",
                (nm,),
            ).fetchone()
            if not row:
                continue  # nombre no existe en el catálogo → skip silencioso
            cur_density, kcal = row
            if cur_density is not None:
                continue  # ya tiene densidad → NULL-only, no tocar
            print(f"  {nm}: density_g_per_cup={dv} (kcal/100g={kcal} ← confirma base seca/cocida)")
            pending += 1
            if COMMIT:
                applied += conn.execute(
                    "UPDATE public.master_ingredients SET density_g_per_cup=%s "
                    "WHERE name=%s AND density_g_per_cup IS NULL", (dv, nm),
                ).rowcount
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. density_g_per_cup seteadas: {applied}")
        else:
            print(f"\nDRY-RUN. {pending} staple(s) con densidad NULL pendientes. "
                  f"Revisa los kcal/100g arriba y re-corre con --commit.")


if __name__ == "__main__":
    main()
