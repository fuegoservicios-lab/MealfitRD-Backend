"""[P2-POOL-PRICE-CONTRACT · 2026-07-01] (audit v2 creatividad GAP-3/GAP-4, batch P2-AUDIT-V2-BATCH)
Contrato DATA: TODO item de los 4 pools del planner (DOMINICAN_PROTEINS/CARBS/VEGGIES_FATS/FRUITS,
constants.py) debe resolver a una fila de master_ingredients CON PRECIO (>0 en price_per_lb o
price_per_unit).

Por qué: si un item de pool no resuelve a fila priced, el planner lo asigna como base del día, el
day-gen lo pone en ingredients[], y VERIFIED-ONLY lo DROPEA de la lista de compras EN SILENCIO
(cero divergencia, cero retry, cero aviso — el espejo del coherence guard filtra con el mismo
predicado). El usuario ve "Panqueques de harina" en la receta pero la harina nunca aparece en su
lista. El audit 2026-07-01 verificó en vivo que HOY los 4 pools están sanos (harinas incluidas);
este script protege el contrato para el FUTURO: correr tras cada expansión de pools o de catálogo.

Resolución: usa la MISMA cascada que shopping (`IngredientNutritionDB.lookup` → normalize/fuzzy) —
no exact-match ingenuo, porque los pools usan display names que resuelven vía synonyms.

    NEON_DATABASE_URL(_POOLED) en .env.  python scripts/check_pool_prices.py
    Exit 0 = contrato OK; exit 2 = items sin precio (listados).
"""
import os, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass


def main():
    from constants import (DOMINICAN_PROTEINS, DOMINICAN_CARBS,
                           DOMINICAN_VEGGIES_FATS, DOMINICAN_FRUITS)
    # El ConnectionPool de db_core nace CERRADO (la app lo abre en el lifespan de FastAPI);
    # en un script standalone hay que abrirlo explícito o todo lookup devuelve None.
    try:
        from db_core import connection_pool
        if connection_pool is not None:
            connection_pool.open()
    except Exception as _pool_e:
        print(f"FATAL: no se pudo abrir el pool de Neon: {_pool_e}")
        sys.exit(1)
    from nutrition_db import IngredientNutritionDB

    db = IngredientNutritionDB()
    pools = {
        "DOMINICAN_PROTEINS": DOMINICAN_PROTEINS,
        "DOMINICAN_CARBS": DOMINICAN_CARBS,
        "DOMINICAN_VEGGIES_FATS": DOMINICAN_VEGGIES_FATS,
        "DOMINICAN_FRUITS": DOMINICAN_FRUITS,
    }
    bad = []
    total = 0
    for pool_name, pool in pools.items():
        for item in pool:
            total += 1
            name = str(item).strip()
            row = None
            try:
                # `_match_row` = la resolución interna (regex/synonyms/fuzzy) devolviendo la FILA
                # cruda del master con TODAS las columnas (NutritionInfo no expone precios).
                row = db._match_row(name)
            except Exception as e:
                bad.append((pool_name, name, f"lookup error: {e}"))
                continue
            if not row:
                bad.append((pool_name, name, "NO resuelve en master_ingredients"))
                continue
            plb = float(row.get("price_per_lb") or 0)
            ppu = float(row.get("price_per_unit") or 0)
            if plb <= 0 and ppu <= 0:
                bad.append((pool_name, name, f"resuelve a '{row.get('name', '?')}' pero SIN precio"))

    print(f"[P2-POOL-PRICE-CONTRACT] items de pool verificados: {total}")
    if bad:
        print(f"\n⚠️ {len(bad)} item(s) de pool violan el contrato precio>0 (drop silencioso de VERIFIED-ONLY):")
        for pool_name, name, reason in bad:
            print(f"  - {pool_name}: {name!r} → {reason}")
        print("\nAcción: añadir precio en master_ingredients (batch owner) o synonym que resuelva a fila priced,")
        print("o retirar el item del pool. NO dejarlo: el planner lo asignará y la lista quedará incompleta.")
        sys.exit(2)
    print("RESULTADO: los 4 pools resuelven a filas con precio. Contrato sano.")


if __name__ == "__main__":
    main()
