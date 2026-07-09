"""[P3-CONFIG-KNOBS-DOC-REFRESH · 2026-07-09] Regeneración desde el registry: vuelca el set VIVO de
knobs `MEALFIT_*` registrados en `_KNOBS_REGISTRY` at import (nombre/tipo/default/current/caller).

Es la fuente autoritativa "regenerada desde el registry" que reemplaza el conteo stale del doc:
`knobs_reference.md` NO mirror-ea los knobs por diseño (drift garantizado) → este script es el listado
vivo on-demand. Nota: el registry captura los knobs leídos AT IMPORT (module-scope). Los knobs leídos
in-function (ej. la cola SSE) se registran al ejecutarse; el universo estático de nombres es mayor
(grep `MEALFIT_[A-Z0-9_]+` sobre el source).

Uso:
    PYTHONPATH=backend python backend/scripts/dump_knobs.py            # tabla legible
    PYTHONPATH=backend python backend/scripts/dump_knobs.py --json     # JSON para tooling
"""
# [P2-LOGGER-EXEMPT: CLI subcommand — salida a stdout es el producto del script]
import os
import sys
import json

# Correr `python backend/scripts/dump_knobs.py` pone scripts/ en sys.path, no backend/ → añadirlo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    if not isinstance(snap, dict):
        print("get_knobs_registry_snapshot() no devolvió dict", file=sys.stderr)
        return 2

    as_json = "--json" in sys.argv[1:]
    rows = []
    for name in sorted(snap):
        meta = snap[name] if isinstance(snap.get(name), dict) else {}
        rows.append({
            "name": name,
            "type": meta.get("type"),
            "default": meta.get("default"),
            "current": meta.get("value"),          # valor efectivo (default u override)
            "is_override": bool(meta.get("is_override")),
        })

    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return 0

    n_over = sum(1 for r in rows if r["is_override"])
    print(f"# _KNOBS_REGISTRY snapshot — {len(rows)} knobs registrados at import ({n_over} con override activo)\n")
    print(f"{'NAME':<52} {'TYPE':<7} {'DEFAULT':<14} {'CURRENT':<14} OVERRIDE")
    print("-" * 100)
    for r in rows:
        print(f"{str(r['name']):<52} {str(r['type']):<7} "
              f"{str(r['default'])[:13]:<14} {str(r['current'])[:13]:<14} {'*' if r['is_override'] else ''}")
    print(f"\n# Total registrados at import: {len(rows)}. "
          f"Universo estático de nombres MEALFIT_* (incl. in-function): grep 'MEALFIT_[A-Z0-9_]+' sobre el source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
