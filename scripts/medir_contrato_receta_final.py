# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-23 · 2026-09-12] (C2 · CUL-P0-03) Cuánto contradicen los pasos a la lista, y cuánto repara el contrato final.

    python scripts/medir_contrato_receta_final.py                       # el corpus fijo del 09-12 (sin DB)
    python scripts/medir_contrato_receta_final.py --vivo 40             # además, los últimos 40 planes (SOLO SELECTs)
    python scripts/medir_contrato_receta_final.py --json --ejemplos 20

Mide ANTES y DESPUÉS de aplicar `recipe_contract.reconcile_days` sobre una COPIA (jamás escribe en la base ni en el
corpus): hallazgos de capa 1 por check (V4 gramos, V6 unidades, V7a lista de más, V7e paso de más), comidas tocadas,
reescrituras por familia, lo que no se repara y por qué (`gramatical`, `reparto`, `conteo_con_gramos`), y la
idempotencia (la segunda pasada tiene que reescribir 0).

Medido al nacer (corpus 087cfc31d3105f79, 64 comidas): V7e 38 → 3, V6 11 → 0, V4 4 → 0 tras corregir la atribución
de gramos; V7a 47 → 44 (lo que queda es número gramatical —«2 claras» y «la clara»— que no se toca a ciegas).
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

CORPUS = _BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json"
CHECKS = ("V4", "V6", "V7a", "V7e")


def medir_filas(filas: list, cat: list, ejemplos: int = 8) -> dict:
    """`filas` = [{"id", "plan_data"}]. Puro: trabaja sobre copias."""
    from culinary_coherence import build_culinary_index, culinary_contract_scan
    import recipe_contract as rc

    index = build_culinary_index(cat)
    antes, despues = collections.Counter(), collections.Counter()
    fams, sinrep = collections.Counter(), collections.Counter()
    comidas = tocadas = reescritas = segunda = 0
    muestra = []
    for f in filas:
        pd = copy.deepcopy(f["plan_data"] or {})
        for v in culinary_contract_scan(pd, cat):
            antes[v["check"]] += 1
        for d in pd.get("days") or []:
            for m in d.get("meals") or []:
                comidas += 1
                r = rc.reconcile_step_quantities(m, index)
                if r["reescritas"]:
                    tocadas += 1
                reescritas += r["reescritas"]
                fams.update(r["familias"])
                sinrep.update(r["sin_reparar"])
                for ch in r["cambios"]:
                    if len(muestra) < ejemplos:
                        muestra.append({"plan": str(f["id"])[:8], "plato": str(m.get("name"))[:40], **ch})
                segunda += rc.reconcile_step_quantities(m, index)["reescritas"]
        for v in culinary_contract_scan(pd, cat):
            despues[v["check"]] += 1
    return {"planes": len(filas), "comidas": comidas, "comidas_tocadas": tocadas, "reescritas": reescritas,
            "familias": dict(fams), "sin_reparar": dict(sinrep), "segunda_pasada_reescribe": segunda,
            "idempotente": segunda == 0,
            "hallazgos_antes": dict(sorted(antes.items())), "hallazgos_despues": dict(sorted(despues.items())),
            "ejemplos": muestra}


def medir_corpus(path: Path, ejemplos: int) -> dict:
    from culinary_corpus import cargar, filas_para_medir
    c = cargar(path)
    r = medir_filas(filas_para_medir(c), c.get("catalogo_filas") or [], ejemplos)
    r["fuente"] = f"corpus fijo {path.name} · huella {c.get('huella')}"
    return r


def medir_vivo(n: int, ejemplos: int) -> dict:
    """Los últimos `n` planes de la base, SOLO lectura (`read_only`): nada de aquí escribe."""
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as conn:
        conn.read_only = True
        cat = [dict(r) for r in conn.execute(
            "SELECT name, aliases, category, ready_to_eat, prep_methods FROM master_ingredients").fetchall()]
        filas = conn.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (int(n),)).fetchall()
    r = medir_filas(filas, cat, ejemplos)
    r["fuente"] = f"ventana viva · últimos {n} planes"
    return r


def render(r: dict) -> str:
    o = [f"{r['fuente']}", f"planes {r['planes']} · comidas {r['comidas']} · comidas tocadas {r['comidas_tocadas']} · "
         f"cantidades reescritas {r['reescritas']} · familias {r['familias']}",
         f"sin reparar (a propósito): {r['sin_reparar']} · idempotente: {'sí' if r['idempotente'] else 'NO (defecto)'}", ""]
    o.append(f"  {'check':6s} {'antes':>6s} {'después':>8s}")
    for k in sorted(set(r["hallazgos_antes"]) | set(r["hallazgos_despues"])):
        marca = "  ←" if k in CHECKS else ""
        o.append(f"  {k:6s} {r['hallazgos_antes'].get(k, 0):6d} {r['hallazgos_despues'].get(k, 0):8d}{marca}")
    if r["ejemplos"]:
        o.append("")
        o.append("  ejemplos (la lista manda):")
        for e in r["ejemplos"]:
            o.append(f"    [{e['familia']:10s}] {e['food'][:18]:18s} {e['de']:g} → {e['a']:g}  «{e['antes']}» ⇒ «{e['despues']}»")
    o += ["", "  ← las cuatro familias del contrato. V7a que queda es número gramatical («2 claras» vs «la clara»):",
          "    no se reescribe a ciegas y sigue en warn."]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--vivo", type=int, default=0, help="además del corpus, los últimos N planes vivos (solo lectura)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ejemplos", type=int, default=8)
    a = ap.parse_args()
    salidas = [medir_corpus(Path(a.corpus), a.ejemplos)]
    if a.vivo:
        salidas.append(medir_vivo(a.vivo, a.ejemplos))
    if a.json:
        print(json.dumps(salidas, ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(render(r) for r in salidas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
