"""[P1-SUPERMARKET-PERSONALIZATION · 2026-07-03] (audit v6 · P1-2c) Reporte de familias del súper
SIN alimento master verificado — candidatos de expansión del catálogo.

Los ~2,000 `supermarket_products` agrupan en ~128 familias (`food_name`). El motor de generación
solo puede usar los masters VERIFICADOS (`master_ingredients` con precio, VERIFIED_ONLY ON): toda
familia del súper sin master es variedad comprable que el planner/day-gen NO puede aprovechar.
Este reporte cierra el loop medir→expandir: enumera esas familias (con # de productos y marcas)
ordenadas por tamaño, para alimentarlas al patrón probado de expansión de catálogo
(USDA→JSON→owner PRICES→--commit, memoria project_catalog_expansion_batch1_2026_06_26).

READ-ONLY (jamás escribe). USO (desde backend/):
  python scripts/supermarket_family_gap_report_2026_07_03.py
"""
import os
import sys

# Consolas Windows cp1252 no codifican ≥/→ del reporte — UTF-8 defensivo.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shopping_calculator import (  # noqa: E402  (normalización SSOT del engine)
    _norm_pref_food,
    _singular_pref_key,
)

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")


def _resolve_master(family: str, master_keys: set):
    """exacto → singular → contención word-boundary (clave más larga gana) — misma escalera
    del engine (POST /api/supermarket/match y _resolve_brand_pref)."""
    key = _norm_pref_food(family)
    if not key:
        return None
    for probe in (key, _singular_pref_key(key)):
        if probe in master_keys:
            return probe
    # contención bidireccional también contra el singular (familias plurales vs master singular)
    for probe in dict.fromkeys((key, _singular_pref_key(key))):
        if len(probe) < 4:
            continue
        padded = f" {probe} "
        best = None
        for k in master_keys:
            if len(k) >= 4 and (f" {k} " in padded or f" {probe} " in f" {k} "):
                if best is None or len(k) > len(best):
                    best = k
        if best:
            return best
    return None


def main():
    if not _NEON:
        print("FALTA NEON_DATABASE_URL(_POOLED) en el entorno/.env — abortando.")
        sys.exit(1)
    with psycopg.connect(_NEON) as conn, conn.cursor() as cur:
        # Familias del súper: agrupadas por master_food_name si existe, si no food_name.
        cur.execute(
            """
            SELECT COALESCE(NULLIF(btrim(master_food_name), ''), food_name) AS family,
                   COUNT(*) AS n_products,
                   COUNT(DISTINCT brand) FILTER (WHERE brand IS NOT NULL) AS n_brands
            FROM public.supermarket_products
            WHERE active
            GROUP BY 1
            ORDER BY 2 DESC, 1
            """
        )
        families = cur.fetchall()
        # Masters VERIFICADOS (mismo criterio del catálogo de generación: tienen precio).
        cur.execute(
            """
            SELECT name FROM public.master_ingredients
            WHERE COALESCE(price_per_lb, 0) > 0 OR COALESCE(price_per_unit, 0) > 0
            """
        )
        # Claves master en AMBAS formas (crudo + singular): "Fresas" master debe matchear la
        # familia "Fresa" del súper — la escalera singulariza la sonda, no el catálogo.
        master_keys = set()
        for (name,) in cur.fetchall():
            k = _norm_pref_food(name)
            if k:
                master_keys.add(k)
                ks = _singular_pref_key(k)
                if ks:
                    master_keys.add(ks)

    gaps, covered = [], 0
    for family, n_products, n_brands in families:
        if _resolve_master(str(family), master_keys):
            covered += 1
        else:
            gaps.append((str(family), int(n_products), int(n_brands or 0)))

    total = len(families)
    print(f"Familias activas en supermarket_products: {total}")
    print(f"  con master verificado (usables por el motor): {covered}")
    print(f"  SIN master verificado (candidatos de expansión): {len(gaps)}")
    if not gaps:
        print("\n✔ Cobertura total: toda familia del súper resuelve a un master verificado.")
        return
    print("\nfamilia · #productos · #marcas   (ordenado por #productos — priorizar arriba)")
    print("-" * 72)
    for family, n_products, n_brands in gaps:
        print(f"{family:<40} {n_products:>6} {n_brands:>8}")
    print("-" * 72)
    print("Siguiente paso: patrón USDA→JSON→owner PRICES→--commit "
          "(memoria project_catalog_expansion_batch1_2026_06_26).")


if __name__ == "__main__":
    main()
