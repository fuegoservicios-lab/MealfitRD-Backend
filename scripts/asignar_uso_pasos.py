# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-25 · 2026-09-12] C4: asignación paso↔ingrediente de la receta congelada (`recipe_usage`), medida y escrita.

    python scripts/asignar_uso_pasos.py                       # mide: estados, hallazgos, alimentos; no escribe nada
    python scripts/asignar_uso_pasos.py --write               # escribe data/registry/recipe_usage_do_v1.json
    python scripts/asignar_uso_pasos.py --verificar           # exit 3 si el snapshot falta, caducó o no reproduce
    python scripts/asignar_uso_pasos.py --json                # el resumen como JSON (para el informe)
    python scripts/asignar_uso_pasos.py --revisar             # las recetas `revisar`, paso a paso (para el dueño)

El índice del catálogo (para «fuera de plantilla») sale del corpus culinario FIJO más reciente en `scripts/data/`
(`--corpus` para otro): así la derivación no depende de la base ni de un `.env`, y reproduce. Sólo lee; lo único que
escribe, con `--write`, es el snapshot de la asignación.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import date
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.append(str(_BACKEND))

import recipe_usage as ru  # noqa: E402


def _indice(corpus_path):
    """`(index, huella)` del catálogo congelado en el corpus; `(None, None)` si no hay corpus."""
    p = corpus_path or (sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "culinary_corpus_*.json"))) or [None])[-1]
    if not p or not os.path.exists(p):
        return None, None
    try:
        c = json.loads(Path(p).read_text(encoding="utf-8"))
        from culinary_coherence import build_culinary_index
        return build_culinary_index(c.get("catalogo_filas") or []), (c.get("catalogo") or {}).get("huella")
    except Exception as e:                                                     # noqa: BLE001
        print(f"[aviso] corpus ilegible ({e!r}): sin detección de «fuera de plantilla»", file=sys.stderr)
        return None, None


def render(snap: dict) -> str:
    r = snap["resumen"]
    out = [f"recetas: {r['recetas']} · estados: " + ", ".join(f"{k} {v}" for k, v in sorted(r["estados"].items())),
           f"constituyentes: {r['constituyentes']} · con Σ = 1: {r['constituyentes_con_suma_1']} · fracciones estimadas: {r['fracciones_estimadas']}",
           "hallazgos: " + (", ".join(f"{k} {v}" for k, v in sorted(r["hallazgos"].items())) or "ninguno")]
    for t in r["top"]:
        out.append(f"  · {t['tipo']:<20} {t['alimento']:<28} ×{t['n']}")
    return "\n".join(out)


def render_revisar(snap: dict) -> str:
    import dish_registry as dr
    tpl = dr.templates_by_id(snap.get("country") or "DO")
    pasos = dr.recipe_steps_index(dr.library_for_country(snap.get("country") or "DO"))
    out = []
    for tid, e in snap["por_id"].items():
        if e.get("estado") != "revisar":
            continue
        out.append(f"\n### {tpl.get(tid, {}).get('name', tid)} ({tid})")
        for h in e.get("hallazgos") or []:
            if h["tipo"] in ("fuera_de_plantilla", "sin_uso_condimento"):
                continue
            out.append(f"  [{h['tipo']}] {h.get('alimento')}: {h['detalle']}")
        for i, p in enumerate(pasos.get(tid) or []):
            out.append(f"  [{i}] {p}")
    return "\n".join(out) or "(ninguna receta en `revisar`)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="DO")
    ap.add_argument("--corpus", help="corpus culinario fijo del que tomar el catálogo (por defecto, el más reciente)")
    ap.add_argument("--write", action="store_true", help="escribe el snapshot data/registry/recipe_usage_<cc>_v1.json")
    ap.add_argument("--verificar", action="store_true", help="exit 3 si el snapshot falta, caducó o no reproduce")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--revisar", action="store_true", help="imprime las recetas `revisar` paso a paso")
    a = ap.parse_args(argv)

    index, huella = _indice(a.corpus)
    snap = ru.derivar_biblioteca(a.country, index, indice_huella=huella, generado_el=date.today().isoformat())

    if a.verificar:
        guardado = ru.cargar_uso(a.country)
        if not guardado:
            print("snapshot AUSENTE: python scripts/asignar_uso_pasos.py --write")
            return 3
        if guardado.get("por_id") != snap["por_id"]:
            dif = [t for t in set(guardado.get("por_id") or {}) | set(snap["por_id"])
                   if (guardado.get("por_id") or {}).get(t) != snap["por_id"].get(t)]
            print(f"snapshot DESFASADO en {len(dif)} receta(s) (texto editado o derivador cambiado): "
                  f"python scripts/asignar_uso_pasos.py --write · p.ej. {sorted(dif)[:3]}")
            return 3
        print("snapshot vigente y reproducible")
        return 0

    if a.json:
        print(json.dumps(snap["resumen"], ensure_ascii=False, indent=1))
    else:
        print(render(snap))
    if a.revisar:
        print(render_revisar(snap))
    if a.write:
        p = ru.escribir_snapshot(snap)
        print(f"escrito: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
