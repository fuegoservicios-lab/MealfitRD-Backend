"""[P2-EXTENDED-MICRO-COVERAGE · 2026-06-29] (audit objetivo · P2-4) Diagnóstico de COBERTURA de las 8 columnas
del panel EXHAUSTIVO (zinc/folato/vitA/C/E/K/selenio/omega3) en master_ingredients.

GAP (audit): a diferencia de las 10 columnas BASE (que tienen un DO-block de sanity que GARANTIZA cero-NULL,
ver p1_micronutrient_catalog_backfill_2026_06_24.sql), las 8 columnas EXTENDIDAS se poblaron por
scripts/backfill_extended_micros.py SOLO `WHERE fdc_id IS NOT NULL` — sin ninguna verificación de completitud.
Los alimentos sin fdc_id (MANUAL_MACROS + cultivares dominicanos + condimentos) quedan NULL → el steering de
esos micros empuja contra datos que no se pueden medir, y el panel da una falsa sensación de exhaustividad.

Este script NO escribe nada: MIDE la cobertura real por columna (global y sobre los alimentos VERIFICADOS por
precio, que son los que importan para los planes), y lista los alimentos con fdc_id pero micro NULL (= gap del
backfill, re-corre backfill_extended_micros.py) vs los SIN fdc_id (= necesitan valor MANUAL USDA). Con esto el
owner decide: si la cobertura verificada está <~90%, completar; documentar el resultado en food_db_integration.md.

    NEON_DATABASE_URL(_POOLED) en .env.  python scripts/check_extended_micro_coverage.py
"""
import os, sys
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass
import psycopg

NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")

_EXTENDED_COLS = [
    "zinc_mg_per_100g", "folate_mcg_dfe_per_100g", "vitamin_a_mcg_rae_per_100g", "vitamin_c_mg_per_100g",
    "vitamin_e_mg_per_100g", "vitamin_k_mcg_per_100g", "selenium_mcg_per_100g", "omega3_ala_g_per_100g",
]
# Umbral de cobertura "sano" sobre los alimentos VERIFICADOS — debajo de esto, el panel extendido es parcialmente
# decorativo y conviene completar el backfill (USDA por fdc_id + MANUAL para los sin fdc_id).
COVERAGE_FLOOR = float(os.environ.get("MEALFIT_EXTENDED_MICRO_COVERAGE_FLOOR", "0.90"))


def main():
    if not NEON:
        print("FATAL: falta NEON_DATABASE_URL(_POOLED) en .env"); sys.exit(1)
    with psycopg.connect(NEON) as conn:
        verified_pred = "((price_per_lb IS NOT NULL AND price_per_lb > 0) OR (price_per_unit IS NOT NULL AND price_per_unit > 0))"
        total = conn.execute("SELECT count(*) FROM public.master_ingredients").fetchone()[0]
        verified = conn.execute(f"SELECT count(*) FROM public.master_ingredients WHERE {verified_pred}").fetchone()[0]
        print(f"[P2-EXTENDED-MICRO-COVERAGE] master_ingredients: total={total}, verificados(precio>0)={verified}\n")
        print(f"{'columna':<28} {'cobertura_global':>16} {'cobertura_verificada':>22}")
        print("-" * 70)
        below = []
        for col in _EXTENDED_COLS:
            g = conn.execute(f"SELECT count(*) FROM public.master_ingredients WHERE {col} IS NOT NULL").fetchone()[0]
            v = conn.execute(
                f"SELECT count(*) FROM public.master_ingredients WHERE {col} IS NOT NULL AND {verified_pred}"
            ).fetchone()[0]
            cov_g = (g / total) if total else 0.0
            cov_v = (v / verified) if verified else 0.0
            flag = "  ⚠️ BAJO" if cov_v < COVERAGE_FLOOR else ""
            print(f"{col:<28} {cov_g:>15.0%} {cov_v:>21.0%}{flag}")
            if cov_v < COVERAGE_FLOOR:
                below.append((col, cov_v))

        # Alimentos VERIFICADOS con fdc_id pero ALGÚN micro extendido NULL → gap del backfill (re-corre backfill).
        null_any = " OR ".join(f"{c} IS NULL" for c in _EXTENDED_COLS)
        with_fdc = conn.execute(
            f"SELECT name FROM public.master_ingredients WHERE {verified_pred} AND fdc_id IS NOT NULL "
            f"AND ({null_any}) ORDER BY name"
        ).fetchall()
        no_fdc = conn.execute(
            f"SELECT name FROM public.master_ingredients WHERE {verified_pred} AND fdc_id IS NULL "
            f"AND ({null_any}) ORDER BY name"
        ).fetchall()

        print(f"\nVerificados con fdc_id pero micro extendido NULL ({len(with_fdc)}) → re-corre backfill_extended_micros.py:")
        print("  " + (", ".join(r[0] for r in with_fdc[:40]) + (" …" if len(with_fdc) > 40 else "") if with_fdc else "(ninguno)"))
        print(f"\nVerificados SIN fdc_id con micro extendido NULL ({len(no_fdc)}) → necesitan valor MANUAL USDA:")
        print("  " + (", ".join(r[0] for r in no_fdc[:40]) + (" …" if len(no_fdc) > 40 else "") if no_fdc else "(ninguno)"))

        print("\n" + ("=" * 70))
        if below:
            print(f"RESULTADO: {len(below)} columna(s) bajo el piso de cobertura {COVERAGE_FLOOR:.0%} sobre verificados.")
            print("Acción: (1) re-corre backfill_extended_micros.py para los con fdc_id; (2) backfill MANUAL para")
            print("los sin fdc_id; (3) documenta la cobertura final en backend/docs/food_db_integration.md.")
            sys.exit(2)
        else:
            print(f"RESULTADO: todas las 8 columnas ≥ {COVERAGE_FLOOR:.0%} sobre verificados. Cobertura sana.")


if __name__ == "__main__":
    main()
