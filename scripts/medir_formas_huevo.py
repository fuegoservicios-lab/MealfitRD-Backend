# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-24 · 2026-09-12] (C3 · CUL-P0-04) Las tres formas del huevo: cuánto se confunden lista y pasos, y cuánto repara el contrato.

    python scripts/medir_formas_huevo.py                 # el corpus fijo del 09-12 (sin DB)
    python scripts/medir_formas_huevo.py --vivo 40       # además, los últimos 40 planes (SOLO SELECTs)
    python scripts/medir_formas_huevo.py --json

Por comida con huevo en la lista: qué formas compra la lista (entero / clara / yema), si salió del tope diario de enteros
(`_egg_day_capped`), y si los pasos la contradicen: «lista con claras y pasos que sólo hablan de huevos enteros» (la forma
cambió en la lista y no en la receta), «lista sólo de claras y pasos que cascan huevos», «claras compradas que ningún paso
usa». Después aplica `recipe_contract.reconcile_meal` sobre una COPIA (jamás escribe) y vuelve a contar; la segunda pasada
tiene que reescribir 0. Lo que NO cuenta como contradicción: una lista de enteros cuyos pasos dicen «hasta que la clara
cuaje» — eso es técnica.

Medido al nacer (corpus 087cfc31d3105f79, 64 comidas): 14 con huevo, 6 salidas del tope diario, y en las 6 los pasos sin
sincronizar (2 de ellas EMPEORADAS por el contrato de cantidades de C2, ciego a la forma: «casca 6 huevos» → «casca 3»);
tras C3: 0 contradicciones, idempotente.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
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
_W = re.compile(r"\bhuevos?\b")
_CL = re.compile(r"\bclaras?\b")
_Y = re.compile(r"\byemas?\b")
_FORMAS_COMPUESTAS = re.compile(r"\bclaras?\s+de\s+huevos?\b|\byemas?\s+de\s+huevos?\b")


def _formas_lista(ings: list) -> set:
    from constants import strip_accents as sa
    out = set()
    for s in ings:
        t = sa(str(s).lower())
        if _CL.search(t):
            out.add("clara")
        elif _Y.search(t):
            out.add("yema")
        elif _W.search(t):
            out.add("entero")
    return out


def _menciones_pasos(rec: list) -> dict:
    from constants import strip_accents as sa
    import recipe_contract as rc
    t = sa(" ".join(str(p) for p in (rec or []) if isinstance(p, str) and not rc._es_nota(p)).lower())
    return {"entero": bool(_W.search(_FORMAS_COMPUESTAS.sub(" ", t))), "clara": bool(_CL.search(t)), "yema": bool(_Y.search(t))}


def _contradicciones(m: dict) -> list:
    fl = _formas_lista([str(x) for x in (m.get("ingredients") or [])])
    if not fl:
        return []
    mp = _menciones_pasos(m.get("recipe") or [])
    out = []
    if fl == {"clara"} and (mp["entero"] or mp["yema"]):
        out.append("lista solo claras / pasos cascan huevos o yemas")
    if "clara" in fl and "entero" in fl and not mp["clara"]:
        out.append("lista enteros+claras / pasos sin las claras")
    if "yema" in fl and "entero" not in fl and mp["entero"]:
        out.append("lista solo yemas / pasos cascan huevos")
    return out


def medir_filas(filas: list, cat: list) -> dict:
    """`filas` = [{"id", "plan_data"}]. Puro: trabaja sobre copias."""
    from culinary_coherence import build_culinary_index
    import recipe_contract as rc

    index = build_culinary_index(cat)
    r = {"planes": len(filas), "comidas": 0, "con_huevo": 0, "formas_lista": collections.Counter(),
         "egg_day_capped": 0, "contradicciones_antes": collections.Counter(), "contradicciones_despues": collections.Counter(),
         "comidas_contradictorias_antes": 0, "comidas_contradictorias_despues": 0,
         "reescritas_forma": 0, "lista_canonizada": 0, "segunda_pasada_reescribe": 0, "ejemplos": []}
    for f in filas:
        pd = copy.deepcopy(f["plan_data"] or {})
        for d in pd.get("days") or []:
            for m in d.get("meals") or []:
                r["comidas"] += 1
                fl = _formas_lista([str(x) for x in (m.get("ingredients") or [])])
                if not fl:
                    continue
                r["con_huevo"] += 1
                r["formas_lista"][",".join(sorted(fl))] += 1
                if m.get("_egg_day_capped"):
                    r["egg_day_capped"] += 1
                antes = _contradicciones(m)
                if antes:
                    r["comidas_contradictorias_antes"] += 1
                    for c in antes:
                        r["contradicciones_antes"][c] += 1
                rep = rc.reconcile_meal(m, index)
                r["reescritas_forma"] += int((rep.get("familias") or {}).get("huevo_forma", 0))
                r["lista_canonizada"] += int(rep.get("lista_reescrita") or 0)
                r["segunda_pasada_reescribe"] += rc.reconcile_meal(m, index)["reescritas"]
                despues = _contradicciones(m)
                if despues:
                    r["comidas_contradictorias_despues"] += 1
                    for c in despues:
                        r["contradicciones_despues"][c] += 1
                for ch in rep.get("cambios") or []:
                    if ch.get("familia") == "huevo_forma" and len(r["ejemplos"]) < 8:
                        r["ejemplos"].append({"plan": str(f["id"])[:8], "plato": str(m.get("name"))[:40],
                                              "antes": ch["antes"][:90], "despues": ch["despues"][:90]})
    for k in ("formas_lista", "contradicciones_antes", "contradicciones_despues"):
        r[k] = dict(r[k])
    r["idempotente"] = r["segunda_pasada_reescribe"] == 0
    return r


def medir_corpus(path: Path) -> dict:
    from culinary_corpus import cargar, filas_para_medir
    c = cargar(path)
    r = medir_filas(filas_para_medir(c), c.get("catalogo_filas") or [])
    r["fuente"] = f"corpus fijo {path.name} · huella {c.get('huella')}"
    return r


def medir_vivo(n: int) -> dict:
    """Los últimos `n` planes de la base, SOLO lectura (`read_only`): nada de aquí escribe."""
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as conn:
        conn.read_only = True
        cat = [dict(x) for x in conn.execute(
            "SELECT name, aliases, category, ready_to_eat, prep_methods FROM master_ingredients").fetchall()]
        filas = conn.execute(
            "SELECT id, plan_data FROM meal_plans WHERE plan_data->'days' IS NOT NULL "
            "ORDER BY created_at DESC LIMIT %s", (int(n),)).fetchall()
    r = medir_filas(filas, cat)
    r["fuente"] = f"ventana viva · últimos {n} planes"
    return r


def render(r: dict) -> str:
    o = [r["fuente"],
         f"planes {r['planes']} · comidas {r['comidas']} · con huevo en la lista {r['con_huevo']} · formas {r['formas_lista']} · "
         f"salidas del tope diario de enteros {r['egg_day_capped']}",
         f"contradicciones lista↔pasos ANTES: {r['comidas_contradictorias_antes']} comidas {r['contradicciones_antes']}",
         f"contradicciones lista↔pasos DESPUÉS del contrato: {r['comidas_contradictorias_despues']} comidas {r['contradicciones_despues']}",
         f"pasos reescritos por forma {r['reescritas_forma']} · líneas de lista canonizadas («huevos sin yema») {r['lista_canonizada']} · "
         f"idempotente: {'sí' if r['idempotente'] else 'NO (defecto)'}"]
    if r["ejemplos"]:
        o.append("  ejemplos (la lista manda):")
        for e in r["ejemplos"]:
            o.append(f"    [{e['plan']}] {e['plato']}: «{e['antes']}» ⇒ «{e['despues']}»")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--vivo", type=int, default=0, help="además del corpus, los últimos N planes vivos (solo lectura)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    salidas = [medir_corpus(Path(a.corpus))]
    if a.vivo:
        salidas.append(medir_vivo(a.vivo))
    if a.json:
        print(json.dumps(salidas, ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(render(r) for r in salidas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
