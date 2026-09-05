"""[P1-ARQ25-F6-DISH-REGISTRY · 2026-09-05] Compila el snapshot del Dish Registry para las 6 bibliotecas.

USO:
    python backend/scripts/compile_dish_registry.py            → data/registry/dish_registry_<lib>_v<versión>.json (×6)
    python backend/scripts/compile_dish_registry.py --check    → recompila en memoria y falla (exit 3) si algún hash difiere
                                                                 del snapshot en disco (reproducibilidad bit a bit)

Necesita el catálogo (`master_ingredients`, Neon): abre el pool fuera de FastAPI (SOP forense).
La versión activa sale del knob `MEALFIT_DISH_REGISTRY_SNAPSHOT` (default "1") o de `--version`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verificar reproducibilidad contra el snapshot en disco")
    ap.add_argument("--version", default=None)
    ap.add_argument("--only", default=None, help="biblioteca única (do|es|mx|co|pr|us)")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BACKEND, ".env"))
    import db_core
    db_core.connection_pool.open()
    import dish_registry as dr
    from shopping_calculator import get_master_ingredients
    rows = list(get_master_ingredients() or [])
    if not rows:
        print("catálogo vacío: ¿pool abierto?", file=sys.stderr)
        return 2
    libs = [args.only] if args.only else list(dr.LIBRARIES.keys())
    drift = 0
    for lib in libs:
        snap = dr.compile_library(lib, catalog_rows=rows, version=args.version)
        st = snap["stats"]
        path = dr.snapshot_path(lib, snap["registry_version"])
        if args.check:
            try:
                with open(path, encoding="utf-8") as f:
                    on_disk = json.load(f)
                same = on_disk.get("snapshot_hash") == snap["snapshot_hash"]
            except FileNotFoundError:
                same = False
            drift += 0 if same else 1
            print(f"{lib}: {'OK' if same else 'DRIFT'} {snap['snapshot_hash'][:12]} · {st['templates']} plantillas · "
                  f"{st['resolved']}/{st['constituents']} resueltos ({st['resolution_pct']} %) · ok/partial/excluded="
                  f"{st['ok']}/{st['partial']}/{st['excluded']}")
        else:
            p = dr.write_snapshot(snap, path)
            print(f"{lib}: escrito {os.path.relpath(p, BACKEND)} · hash {snap['snapshot_hash'][:12]} · {st['templates']} plantillas · "
                  f"{st['resolved']}/{st['constituents']} resueltos ({st['resolution_pct']} %) · ok/partial/excluded="
                  f"{st['ok']}/{st['partial']}/{st['excluded']}")
            for t in snap["templates"]:
                for e in t["excluded"]:
                    print(f"    · excluido [{t['name'][:40]}] {e['name']} ({e['reason']})")
    return 3 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
