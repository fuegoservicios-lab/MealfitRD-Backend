"""[P2-SUPERMARKET-PKG-SYNC · 2026-07-02] (audit v4 presupuesto, batch P2-OBJECTIVE-V4-BATCH)
Reconciliación de PRECIOS supermarket_products (genéricos) → master_ingredients.market_packages.

Complementa la parte B del P1-SUPERMARKET-COSTING (fill-only-empty): aquella solo llenaba masters
SIN packages; este script cubre la otra mitad del roadmap #4 — cuando el owner ACTUALIZA precios en
/supermercado, los `market_packages` curados (la fuente del costeo P1-PKG de la lista) se van
quedando stale y la lista costea distinto de lo que muestra el súper. REPORT-FIRST:

  * DRY-RUN por default: imprime la tabla de drifts (master · package · precio curado · precio del
    súper · Δ%) sin tocar nada. `--commit` aplica SOLO los precios (jamás tamaños/labels/unidades).
  * Matching CONSERVADOR: solo productos GENÉRICOS (brand IS NULL, active, price>0) del MISMO
    alimento (food_name/master_food_name normalizado, exacto → contención word-boundary) y con
    tamaño a ±10% del package (size_grams explícito P2-BRANDPREF-SIZE-COLUMN primero; el parser de
    `presentation` como fallback — la "L" suelta ambigua se salta, mismo criterio de la parte B).
  * Guards de commit: precio nuevo > 0 y dentro de 2.5× bidireccional del curado (mismo espíritu
    del guard anti-especializados de la parte B); drift mínimo 5% (ruido de centavos no toca DB).
  * Packages por `units` (huevos/ajo) NO se tocan (sin gramos comparables).
  * Backup del before-state a %TEMP% (rollback = restaurar el JSON). Idempotente: re-correr tras
    un commit produce 0 drifts.

USO (desde backend/):
  python scripts/sync_supermarket_prices_report_2026_07_02.py            # REPORT (dry-run)
  python scripts/sync_supermarket_prices_report_2026_07_02.py --commit   # aplica precios
"""
import json
import os
import sys
import tempfile
import time

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

import psycopg
from psycopg.types.json import Jsonb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shopping_calculator import (  # noqa: E402  (parser/normalización SSOT del engine)
    _norm_pref_food,
    _parse_presentation_grams,
)

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv
SIZE_TOL = 0.10        # ±10% de gramos para considerar "el mismo envase"
DRIFT_MIN = 0.05       # <5% de drift = ruido, no se reporta ni se toca
PRICE_GUARD = 2.5      # precio nuevo dentro de 2.5× bidireccional del curado


def _resolve_food(master_name: str, by_food: dict):
    """exacto → contención word-boundary (clave más larga gana) — misma escalera del engine."""
    key = _norm_pref_food(master_name)
    if not key:
        return None
    if key in by_food:
        return by_food[key]
    padded = f" {key} "
    best = None
    for k in by_food:
        if len(k) >= 4 and (f" {k} " in padded or f" {key} " in f" {k} "):
            if best is None or len(k) > len(best):
                best = k
    return by_food.get(best) if best else None


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    with psycopg.connect(_NEON) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT name, market_packages
            FROM master_ingredients
            WHERE market_packages IS NOT NULL AND jsonb_array_length(market_packages) > 0
        """)
        masters = cur.fetchall()

        cur.execute("""
            SELECT food_name, master_food_name, presentation,
                   price_rd::float8, size_grams::float8
            FROM public.supermarket_products
            WHERE active AND brand IS NULL AND price_rd IS NOT NULL AND price_rd > 0
        """)
        generic_rows = cur.fetchall()

    # Índice de genéricos con tamaño resoluble: {food_norm: [(grams, price, pres)]}
    by_food: dict = {}
    for (food, master_food, pres, price, size_grams) in generic_rows:
        grams = None
        try:
            grams = float(size_grams) if size_grams and 1.0 <= float(size_grams) <= 50000.0 else None
        except (TypeError, ValueError):
            grams = None
        if grams is None:
            grams = _parse_presentation_grams(pres)
        if not grams:
            continue
        for key_src in {(food or ""), (master_food or "")}:
            key = _norm_pref_food(key_src)
            if key:
                by_food.setdefault(key, []).append((grams, float(price), str(pres or "")))

    drifts = []          # (master, idx, label, grams, precio_curado, precio_super, pres, aplicable)
    updates: dict = {}   # master → market_packages nuevo (solo precios cambiados)
    for (master_name, packages) in masters:
        if not isinstance(packages, list):
            continue
        candidates = _resolve_food(master_name, by_food)
        if not candidates:
            continue
        new_packages = json.loads(json.dumps(packages))  # copia profunda JSON-safe
        touched = False
        for idx, pkg in enumerate(new_packages):
            if not isinstance(pkg, dict):
                continue
            try:
                pkg_g = float(pkg.get("grams") or 0)
                pkg_price = float(pkg.get("price") or 0)
            except (TypeError, ValueError):
                continue
            if pkg_g <= 0 or pkg_price <= 0:
                continue  # units-based (huevos/ajo) o sin precio → no comparable
            same_size = [c for c in candidates if abs(c[0] - pkg_g) <= pkg_g * SIZE_TOL]
            if not same_size:
                continue
            super_g, super_price, super_pres = min(same_size, key=lambda c: c[1])
            drift = abs(super_price - pkg_price) / pkg_price
            if drift < DRIFT_MIN:
                continue
            in_guard = (super_price > 0
                        and super_price <= pkg_price * PRICE_GUARD
                        and pkg_price <= super_price * PRICE_GUARD)
            drifts.append((master_name, idx, pkg.get("label") or f"{pkg_g:.0f}g", pkg_g,
                           pkg_price, super_price, super_pres, in_guard))
            if in_guard:
                new_packages[idx]["price"] = round(super_price, 2)
                touched = True
        if touched:
            updates[master_name] = new_packages

    print(f"\n[P2-SUPERMARKET-PKG-SYNC] masters con packages: {len(masters)} · "
          f"genéricos con tamaño: {sum(len(v) for v in by_food.values())} · drifts ≥{DRIFT_MIN:.0%}: {len(drifts)}\n")
    for (m, idx, label, g, old, new, pres, ok) in sorted(drifts, key=lambda d: -abs(d[5] - d[4]) / d[4]):
        flag = "APLICA" if ok else "FUERA-DE-GUARD(2.5x)"
        print(f"  {m:32s} [{label:>14s} ~{g:6.0f}g]  RD${old:8.2f} → RD${new:8.2f} "
              f"({(new - old) / old:+6.1%})  «{pres[:28]}»  {flag}")

    if not COMMIT:
        print(f"\nDRY-RUN — nada escrito. {len(updates)} master(s) recibirían precios nuevos. "
              f"Re-corre con --commit para aplicar.")
        return

    if not updates:
        print("\nNada que aplicar (0 drifts dentro de guard).")
        return

    backup_path = os.path.join(tempfile.gettempdir(),
                               f"pkg_price_sync_before_{int(time.time())}.json")
    with psycopg.connect(_NEON) as conn, conn.cursor() as cur:
        cur.execute("SELECT name, market_packages FROM master_ingredients WHERE name = ANY(%s)",
                    (list(updates.keys()),))
        before = {r[0]: r[1] for r in cur.fetchall()}
        with open(backup_path, "w", encoding="utf-8") as fh:
            json.dump(before, fh, ensure_ascii=False, indent=1)
        applied = 0
        for name, pkgs in updates.items():
            cur.execute(
                "UPDATE master_ingredients SET market_packages = %s WHERE name = %s",
                (Jsonb(pkgs), name),
            )
            applied += cur.rowcount
        conn.commit()
    print(f"\nCOMMIT: {applied} master(s) actualizados. Backup before-state: {backup_path}")
    print("El costeo de planes NUEVOS/recalculados usa los precios frescos (cache master TTL ~300s).")


if __name__ == "__main__":
    main()
