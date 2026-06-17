"""[P2-PRICES-ENGINE-1 · 2026-06-16] Importa precios BASE de alimentos desde un CSV
(datos online: webs de cadenas / encuesta) a master_ingredients.

El CSV se enruta a `price_engine.import_base_prices` (UPDATE COALESCE: sólo pisa lo
que el row provee). Tras importar, opcionalmente reescala los precios vivos con el
último índice de inflación (`--recompute`).

Columnas del CSV (header obligatorio; al menos `slug` O `name`):
    slug, name, price_per_lb_base, price_per_unit_base,
    price_base_period (YYYY-MM), price_source, price_confidence (high|medium|low),
    price_captured_at (YYYY-MM-DD)

Uso:
    PYTHONPATH=backend python backend/scripts/import_prices.py precios.csv \
        --period 2026-06 --source nacional_online --recompute

Plantilla mínima:
    slug,name,price_per_lb_base,price_per_unit_base,price_confidence
    ,Pollo,95,,low
    ,Arroz,38,,high
    ,Huevo,,12,medium
"""
import argparse
import csv
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

# backend/ al path para importar price_engine/db_core sin depender de PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IMPORTANTE: load_dotenv ANTES de importar price_engine/db_core (el pool se
# configura al import leyendo NEON_DATABASE_URL_*).
import price_engine as pe  # noqa: E402


def _clean(v):
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def main():
    ap = argparse.ArgumentParser(description="Importa precios base de alimentos desde CSV.")
    ap.add_argument("csv_path", help="Ruta al CSV de precios.")
    ap.add_argument("--period", default=None, help="price_base_period default (YYYY-MM) si el CSV no lo trae.")
    ap.add_argument("--source", default=None, help="price_source default si el CSV no lo trae.")
    ap.add_argument("--recompute", action="store_true", help="Reescala vivos = base × índice tras importar.")
    args = ap.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"FATAL: no existe {args.csv_path}")
        sys.exit(1)

    rows = []
    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append({k: _clean(v) for k, v in raw.items()})

    # El connection_pool de db_core no se auto-abre en scripts standalone.
    from db_core import connection_pool as _pool
    if _pool is not None:
        try:
            _pool.open(); _pool.wait(timeout=15)
        except Exception as _e:
            print(f"  (pool open: {_e})")

    print(f"Importando {len(rows)} filas desde {args.csv_path}…")
    result = pe.import_base_prices(rows, default_period=args.period, default_source=args.source)
    print(f"  matched={result['matched']}  unmatched={result['unmatched']}  errors={result['errors']}")
    if result["unmatched_keys"]:
        print("  NO MATCHEADOS (revisar slug/name vs master_ingredients):")
        for k in result["unmatched_keys"]:
            print(f"    - {k}")

    if args.recompute:
        # NO force: respeta MEALFIT_PRICES_ENABLED (los precios VIVOS sólo se publican con el feature ON).
        print("Reescalando precios vivos…")
        rc = pe.recompute_adjusted_prices()
        print(f"  {rc}")
        if rc.get("status") == "disabled":
            print("  ℹ Importado a BASE; activa MEALFIT_PRICES_ENABLED=true para publicar los vivos.")

    rep = pe.price_staleness_report()
    print(f"Cobertura: {rep['priced']}/{rep['total']} ({rep['coverage_pct']}), "
          f"stale={rep['stale']}, índice={rep['latest_index_period']}, enabled={rep['enabled']}")


if __name__ == "__main__":
    main()
