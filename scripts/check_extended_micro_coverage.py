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

[P1-EXTENDED-MICROS-GUARD · 2026-07-01] El gap descrito arriba quedó CERRADO: la migración
p1_extended_micros_zero_null_guard_2026_07_01.sql backfilleó la última fila NULL (Gandules
vitE/vitK, proxy leguminosa conservador) y añadió el DO-block cero-NULL espejo del base.
Este script queda como diagnóstico ops. Dos fixes del mismo P-fix: (a) stdout forzado a
utf-8 (crasheaba en consola Windows cp1252 al imprimir '→'); (b) la tabla muestra el CONTEO
exacto de NULLs por columna — el formato `:.0%` redondeaba 201/202 a "100%" y ocultó la
única fila NULL del audit 2026-07-01.
"""
import os, sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # [P1-EXTENDED-MICROS-GUARD] consola Windows cp1252
except Exception:
    pass
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
        # [P1-EXTENDED-MICROS-GUARD · 2026-07-01] columna `nulls` con el CONTEO EXACTO: el formato `:.0%`
        # redondeaba 201/202 a "100%" y ocultaba filas NULL sueltas (así se escondió Gandules en el audit).
        print(f"{'columna':<28} {'cobertura_global':>16} {'cobertura_verificada':>22} {'nulls':>6}")
        print("-" * 78)
        below = []
        for col in _EXTENDED_COLS:
            g = conn.execute(f"SELECT count(*) FROM public.master_ingredients WHERE {col} IS NOT NULL").fetchone()[0]
            v = conn.execute(
                f"SELECT count(*) FROM public.master_ingredients WHERE {col} IS NOT NULL AND {verified_pred}"
            ).fetchone()[0]
            cov_g = (g / total) if total else 0.0
            cov_v = (v / verified) if verified else 0.0
            n_null = total - g
            flag = "  ⚠️ BAJO" if cov_v < COVERAGE_FLOOR else ("  ⚠️ NULLS" if n_null else "")
            print(f"{col:<28} {cov_g:>15.0%} {cov_v:>21.0%} {n_null:>6}{flag}")
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
        # [P1-EXTENDED-MICROS-GUARD · 2026-07-01] el contrato ya no es "≥90%": es CERO-NULL (DO-block en
        # p1_extended_micros_zero_null_guard_2026_07_01.sql). Cualquier fila NULL = exit 2.
        n_any_null = len(with_fdc) + len(no_fdc)
        if below or n_any_null:
            if below:
                print(f"RESULTADO: {len(below)} columna(s) bajo el piso de cobertura {COVERAGE_FLOOR:.0%} sobre verificados.")
            if n_any_null:
                print(f"RESULTADO: {n_any_null} fila(s) verificada(s) con algún micro extendido NULL (contrato = cero-NULL).")
            print("Acción: (1) re-corre backfill_extended_micros.py para los con fdc_id; (2) backfill MANUAL para")
            print("los sin fdc_id; (3) re-aplica p1_extended_micros_zero_null_guard_2026_07_01.sql (idempotente).")
            sys.exit(2)
        else:
            print(f"RESULTADO: cero-NULL en las 8 columnas extendidas (contrato P1-EXTENDED-MICROS-GUARD). Cobertura sana.")


if __name__ == "__main__":
    main()
