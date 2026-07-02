"""[P1-SUPERMARKET-COSTING · 2026-07-02 · parte B] Sync supermarket_products →
master_ingredients.market_packages (FILL-ONLY-EMPTY).

Cierra el roadmap #4 del Supermercado RD: que el costeo P1-PKG use los mismos
precios que muestra /supermercado. Alcance CONSERVADOR a propósito:

  * Solo master_ingredients SIN market_packages (NULL o lista vacía) — los 113
    curados a mano NO se tocan.
  * Solo productos GENÉRICOS del súper (brand IS NULL, active, price_rd > 0) —
    las variantes de marca quedan para el overlay per-usuario
    (P1-SUPERMARKET-COSTING parte A), así el display default no muestra marcas.
  * Solo presentaciones con tamaño parseable (mismo parser del engine:
    `shopping_calculator._parse_presentation_grams`; la "L" suelta ambigua se
    salta). Dedupe por gramos (mismo tamaño → el más barato). Máx 6 por food.
  * `market_container`/`container_weight_g` se completan SOLO si están NULL
    (requisito del path por packages), con la palabra de envase y el tamaño
    del paquete más pequeño.
  * Idempotente: el UPDATE re-verifica `market_packages IS NULL OR vacío` en
    el WHERE. Backup del before-state a un JSON (rollback = restaurarlo).

USO (desde backend/):
  python scripts/sync_supermarket_to_market_packages_2026_07_02.py            # DRY-RUN
  python scripts/sync_supermarket_to_market_packages_2026_07_02.py --commit   # aplica
"""
import json
import os
import sys
import time

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
from shopping_calculator import (  # noqa: E402  (parser SSOT del engine)
    _PRES_SIZE_RX,
    _norm_pref_food,
    _parse_presentation_grams,
    _pref_container_word,
)

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv
MAX_PKGS = 6


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    with psycopg.connect(_NEON) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT name, market_container, container_weight_g, price_per_lb::float8
            FROM master_ingredients
            WHERE market_packages IS NULL OR jsonb_array_length(market_packages) = 0
        """)
        targets = {r[0]: {"container": r[1], "weight_g": r[2], "price_per_lb": r[3]} for r in cur.fetchall()}

        cur.execute("""
            SELECT food_name, presentation, price_rd::float8
            FROM public.supermarket_products
            WHERE active AND brand IS NULL AND price_rd IS NOT NULL AND price_rd > 0
        """)
        generic_rows = cur.fetchall()

    # Índice de genéricos por food normalizado. SOLO empaques con tamaño
    # EXPLÍCITO ("800 gr", "15 Oz", "1 Lb") — la venta-por-libra a granel
    # ("Lb", "Criolla Lb", "80/20 Lb") se excluye del sync: esos items ya
    # costean bien por price_per_lb y convertirlos a "packages" degradaría
    # el display ("2 libras (Lb)") sin ganancia de precio.
    by_food: dict = {}
    for (food, pres, price) in generic_rows:
        if not _PRES_SIZE_RX.search(str(pres or "")):
            continue
        grams = _parse_presentation_grams(pres)
        if not grams:
            continue
        key = _norm_pref_food(food)
        size_part = (pres or "").split(" ", 1)[1] if (
            pres and _norm_pref_food(pres).split(" ")[0] in
            {"funda", "lata", "paquete", "botella", "frasco", "tarro", "pote", "caja",
             "carton", "brik", "sobre", "bandeja", "clamshell", "cubo", "barra", "tubo",
             "pieza", "malla", "tetra"} and " " in pres
        ) else (pres or "")
        by_food.setdefault(key, []).append({
            "grams": round(grams, 2),
            "price": price,
            "label": size_part.strip(),
            "unit": _pref_container_word(pres),
        })

    plan = {}
    skipped_sanity = []
    for name, meta in sorted(targets.items()):
        pkgs = by_food.get(_norm_pref_food(name))
        if not pkgs:
            continue
        # Guard de cordura de precio: un empaque cuyo precio/gramo supere 2.5×
        # el price_per_lb del master es un producto ESPECIALIZADO listado bajo
        # el genérico (cebollitas perla en malla, mix gourmet de lechuga, baby
        # carrots) — llenarlo como único package dispararía el costeo del
        # staple. Sin price_per_lb de referencia, el guard no aplica.
        ppl = meta.get("price_per_lb") or 0
        base_per_g = (ppl / 453.592) if ppl > 0 else None
        sane = []
        for p in pkgs:
            per_g = p["price"] / p["grams"]
            if base_per_g and per_g > 2.5 * base_per_g:
                skipped_sanity.append(f"{name}: {p['grams']:g}g=RD${p['price']:g} ({per_g/base_per_g:.1f}× granel)")
                continue
            sane.append(p)
        if not sane:
            continue
        # Dedupe por gramos (mismo tamaño → el más barato), orden por tamaño, cap.
        best = {}
        for p in sane:
            g = p["grams"]
            if g not in best or p["price"] < best[g]["price"]:
                best[g] = p
        chosen = sorted(best.values(), key=lambda p: p["grams"])[:MAX_PKGS]
        plan[name] = {"packages": chosen, "meta": meta}

    if skipped_sanity:
        print("Excluidos por guard de precio (especializados bajo el genérico):")
        for s in skipped_sanity:
            print(f"  ✗ {s}")
        print()

    print(f"Sync fill-only-empty: {len(targets)} masters sin packages · "
          f"{len(plan)} con genéricos parseables del súper. Modo: {'COMMIT' if COMMIT else 'DRY-RUN'}\n")
    for name, info in plan.items():
        fills = []
        if not info["meta"]["container"]:
            fills.append(f"container→{info['packages'][0]['unit']}")
        if not info["meta"]["weight_g"]:
            fills.append(f"weight_g→{info['packages'][0]['grams']}")
        extra = f"  (+{', '.join(fills)})" if fills else ""
        sizes = ", ".join(f"{p['grams']:g}g=RD${p['price']:g}" for p in info["packages"])
        print(f"  · {name}: {len(info['packages'])} pkg [{sizes}]{extra}")

    if not COMMIT:
        print("\nDRY-RUN (no escribe). Ejecuta con --commit para aplicar.")
        return

    # Backup del before-state (rollback = restaurar estos valores).
    backup_path = os.path.join(
        os.environ.get("TEMP", "/tmp"),
        f"market_packages_backup_{int(time.time())}.json",
    )
    with open(backup_path, "w", encoding="utf-8") as fh:
        json.dump({n: targets[n] for n in plan}, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\nBackup before-state: {backup_path}")

    updated = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for name, info in plan.items():
                pkgs_json = [{k: v for k, v in p.items()} for p in info["packages"]]
                cur.execute(
                    """
                    UPDATE master_ingredients
                    SET market_packages = %s,
                        market_container = COALESCE(market_container, %s),
                        container_weight_g = COALESCE(container_weight_g, %s)
                    WHERE name = %s
                      AND (market_packages IS NULL OR jsonb_array_length(market_packages) = 0)
                    """,
                    (Jsonb(pkgs_json), info["packages"][0]["unit"],
                     info["packages"][0]["grams"], name),
                )
                updated += cur.rowcount
        conn.commit()
    print(f"OK: {updated} master_ingredients actualizados (fill-only-empty).")


if __name__ == "__main__":
    main()
