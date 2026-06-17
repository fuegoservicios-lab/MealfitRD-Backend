"""[P2-PRICES-ENGINE-1 · 2026-06-16] Ingesta puntos del subíndice IPC "Alimentos y
bebidas no alcohólicas" del Banco Central de RD (BCRD) a price_inflation_index.

El BCRD publica el IPC mensual (~2-3 semanas de rezago) en bancentral.gov.do; no hay
API formal, así que la ingesta es manual/asistida: tomas el valor del subíndice de
alimentos del mes y lo registras. El motor proyecta precio_actual = precio_base ×
(food_cpi_actual / food_cpi_base) — no necesitas re-encuestar el supermercado.

Uso (un punto):
    PYTHONPATH=backend python backend/scripts/ingest_bcrd_inflation.py 2026-06 121.4

Uso (CSV de la serie histórica; header: period,food_cpi[,note]):
    PYTHONPATH=backend python backend/scripts/ingest_bcrd_inflation.py --csv serie.csv

Tras ingerir el mes nuevo, reescala con:
    PYTHONPATH=backend python backend/scripts/import_prices.py --recompute ...
o espera al cron diario `price_inflation_adjust`.
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

import price_engine as pe  # noqa: E402  (tras load_dotenv)


def main():
    ap = argparse.ArgumentParser(description="Ingesta puntos del índice IPC-alimentos del BCRD.")
    ap.add_argument("period", nargs="?", help="Período YYYY-MM (modo punto único).")
    ap.add_argument("food_cpi", nargs="?", type=float, help="Valor del subíndice (modo punto único).")
    ap.add_argument("--csv", default=None, help="CSV con la serie (header: period,food_cpi[,note]).")
    ap.add_argument("--source", default="bcrd")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    # El connection_pool de db_core no se auto-abre en scripts standalone.
    from db_core import connection_pool as _pool
    if _pool is not None:
        try:
            _pool.open(); _pool.wait(timeout=15)
        except Exception as _e:
            print(f"  (pool open: {_e})")

    n = 0
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"FATAL: no existe {args.csv}")
            sys.exit(1)
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            for raw in csv.DictReader(f):
                try:
                    pe.ingest_inflation_index(
                        raw["period"].strip(), float(raw["food_cpi"]),
                        source=args.source, note=(raw.get("note") or args.note),
                    )
                    n += 1
                    print(f"  OK {raw['period'].strip()} = {raw['food_cpi']}")
                except Exception as e:
                    print(f"  ERR {raw.get('period')}: {type(e).__name__}: {e}")
    elif args.period and args.food_cpi is not None:
        pe.ingest_inflation_index(args.period, args.food_cpi, source=args.source, note=args.note)
        n = 1
        print(f"  OK {args.period} = {args.food_cpi}")
    else:
        ap.error("Da `period food_cpi` (punto único) o `--csv archivo.csv` (serie).")

    latest = pe.latest_index()
    print(f"\nDONE: {n} períodos ingeridos. Último índice: {latest}")


if __name__ == "__main__":
    main()
