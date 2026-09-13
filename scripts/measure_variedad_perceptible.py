# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-35 · 2026-09-13] CUL-P2-01: variedad PERCEPTIBLE a 7/15/30 días — nombres nuevos frente a platos nuevos.

«Pollo con arroz y ensalada» renombrado siete veces son siete nombres y un plato. El reporte de variedad del grafo
(`build_variety_report`) cuenta repeticiones por nombre y proteína del MISMO día; nadie contaba, a lo largo de 7, 15 y 30
días, cuántos platos DISTINTOS recibe la persona. Este script separa las dos cosas con una **firma de preparación**:

    firma = `_template_id` si el plato viene de la biblioteca; si no,
            (familias de proteína, técnicas de cocción, bases de carbohidrato) — NADA del nombre

  · familias de proteína: `graph_orchestrator._MAIN_PROTEIN_ALIASES` sobre los ingredientes (por palabra completa:
    «res» no está en «queso fresco»);
  · técnicas: `culinary_coherence.VERB_TO_METHOD` sobre los pasos de la receta;
  · bases: el vocabulario `BASES` de abajo sobre los ingredientes.

Por ventana (los primeros 7, 15 y 30 días; una ventana más larga que el plan no se mide y se dice) y por franja:
nombres distintos, firmas distintas, **renombrados** (nombre nuevo, firma ya servida), y la **compra reusada con otra
preparación** (misma familia de proteína con técnicas distintas: variedad real con la misma lista de compras). Además, en
cada ventana deslizante de 7 días, las firmas servidas por encima del tope de la política
(`horizon.repetition_limits_for("balanced", 7)`).

Solo lectura y sin LLM. Fuentes: el corpus fijo (planes de producción de 3-4 días), los planes del bench real, y —con
`--deterministico N`— N días del día determinista (lee el catálogo de la base, no escribe).

    python scripts/measure_variedad_perceptible.py --corpus --real --deterministico 30 --out scripts/data/variedad_perceptible_<fecha>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

VENTANAS = (7, 15, 30)
#: base de carbohidrato → raíces (sobre ingredientes y nombre, sin acentos, por palabra completa)
BASES = {
    "arroz": ("arroz",), "platano": ("platano", "platanos", "mangu", "tostones", "maduro", "maduros"),
    "yuca": ("yuca", "casabe"), "papa": ("papa", "papas"), "batata": ("batata", "batatas"),
    "pan": ("pan", "tostada", "tostadas", "wrap", "arepa", "arepas", "tortilla de harina"),
    "avena": ("avena",), "pasta": ("pasta", "espagueti", "espaguetis", "coditos", "fideos", "macarrones"),
    "guineo": ("guineo", "guineos", "guineito", "guineitos"), "quinoa": ("quinoa",),
    "vianda": ("yautia", "name", "auyama", "mapuey"), "maiz": ("maiz", "harina de maiz", "polenta"),
}


def _sa(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def _cargar_vocabularios():
    import culinary_coherence as cc
    import graph_orchestrator as go
    alias = {fam: [_sa(a) for a in v] for fam, v in go._MAIN_PROTEIN_ALIASES.items()}
    verbos = [(re.compile(rx, re.IGNORECASE), m) for rx, m in cc.VERB_TO_METHOD.items()]
    return alias, verbos, go._head_dish_base_token


def rasgos(meal: dict, vocab) -> tuple:
    """(familias de proteína, técnicas, bases) sacados de los ingredientes y los pasos. Nunca del nombre."""
    alias, verbos, _head = vocab
    ings = " | ".join(_sa(x) for x in (meal.get("ingredients") or []) if isinstance(x, str))
    pasos = " ".join(str(p) for p in (meal.get("recipe") or []) if isinstance(p, str)).lower()
    prot = sorted(f for f, al in alias.items() if any(re.search(r"\b" + re.escape(a) + r"\b", ings) for a in al))
    tec = sorted({m for rx, m in verbos if rx.search(pasos)})
    bases = sorted(b for b, rs in BASES.items() if any(re.search(r"\b" + re.escape(r) + r"\b", ings) for r in rs))
    return tuple(prot), tuple(tec), tuple(bases)


def firma(meal: dict, vocab) -> tuple:
    """La plantilla de la biblioteca si la hay; si no, los rasgos. El nombre no entra: la primera versión usaba el formato
    sacado del nombre (`_head_dish_base_token`) y un renombre cambiaba la firma — justo lo que la firma existe para no ver."""
    if meal.get("_template_id"):
        return ("plantilla", str(meal["_template_id"]))
    return ("firma",) + rasgos(meal, vocab)


def comidas_por_dia(plan: dict) -> list:
    return [[m for m in (d.get("meals") or []) if isinstance(m, dict)] for d in (plan.get("days") or [])]


def medir_plan(plan: dict, vocab, tope_7d: int) -> dict:
    dias = comidas_por_dia(plan)
    n = len(dias)
    out = {"dias": n, "comidas": sum(len(d) for d in dias), "ventanas": {}, "ventanas_7d_rotas": 0, "firmas_sobre_tope": []}
    filas = [(i, m.get("meal") or "?", str(m.get("name") or ""), firma(m, vocab), rasgos(m, vocab))
             for i, d in enumerate(dias) for m in d]
    nombre_de = {}
    for f in filas:
        nombre_de.setdefault(f[3], f[2])
    for w in VENTANAS:
        if w > n:
            out["ventanas"][str(w)] = {"medida": False, "motivo": f"el plan tiene {n} días"}
            continue
        if not any(f[0] < w for f in filas):
            # un plan sin una comida no tiene «0 platos distintos»: no se midió, y se dice
            out["ventanas"][str(w)] = {"medida": False, "motivo": "sin comidas en la ventana"}
            continue
        sel = [f for f in filas if f[0] < w]
        por_franja = defaultdict(lambda: {"comidas": 0, "nombres": set(), "firmas": set(), "renombrados": 0})
        prot_tec = defaultdict(set)
        for _, franja, nombre, fi, (prot, tec, _bases) in sel:
            r = por_franja[franja]
            r["comidas"] += 1
            if nombre not in r["nombres"] and fi in r["firmas"]:
                r["renombrados"] += 1
            r["nombres"].add(nombre)
            r["firmas"].add(fi)
            for p in prot:
                prot_tec[p].add(tec)
        out["ventanas"][str(w)] = {
            "medida": True,
            "comidas": len(sel),
            "nombres_distintos": len({f[2] for f in sel}),
            "firmas_distintas": len({f[3] for f in sel}),
            "renombrados": sum(r["renombrados"] for r in por_franja.values()),
            "por_franja": {k: {"comidas": v["comidas"], "nombres": len(v["nombres"]), "firmas": len(v["firmas"]),
                               "renombrados": v["renombrados"]} for k, v in sorted(por_franja.items())},
            "proteina_con_varias_tecnicas": sorted(p for p, t in prot_tec.items() if len(t) >= 2),
            "proteina_misma_tecnica": sorted(p for p, t in prot_tec.items() if len(t) == 1),
        }
    for ini in range(0, max(0, n - 7) + 1):
        c = Counter(f[3] for f in filas if ini <= f[0] < ini + 7)
        sobre = [fi for fi, k in c.items() if k > tope_7d]
        if sobre:
            out["ventanas_7d_rotas"] += 1
            out["firmas_sobre_tope"].extend(f"{nombre_de.get(fi, '?')} {fi}"[:160] for fi in sobre)
    out["firmas_sobre_tope"] = sorted(set(out["firmas_sobre_tope"]))[:20]
    return out


def _planes_corpus() -> list:
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "culinary_corpus_*.json")))
    c = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
    return [(f"corpus:{str(p.get('plan_id'))[:8]}", p["plan_data"]) for p in c["planes"]]


def _planes_real() -> list:
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "bench_superficies_real_*.json")))
    r = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
    return [(f"real:{g.get('perfil') or g.get('plan_id')}", g["plan_data"]) for g in r.get("planes_generados") or []]


def _plan_deterministico(dias: int, country: str) -> tuple:
    import os
    os.environ.setdefault("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    os.environ.setdefault("MEALFIT_DETERMINISTIC_DAY", "1")
    # Fuera de FastAPI el pool no está abierto y `master_ingredients` sale VACÍO: sin catálogo no hay plantilla escalable
    # y los 30 días salen sin una comida (medido en la primera corrida). Igual que measure_deterministic_day_macros.py.
    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
    import db_core
    if db_core.connection_pool is not None:
        db_core.connection_pool.open()
    import deterministic_day as dd
    nut = {"target_calories": 2000, "macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}}
    skel = {"meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"], "protein_pool": []}
    fd = {"country": country, "user_id": "measure-variedad", "health_profile": {}}
    memoria, out = [], []
    for d in range(1, dias + 1):
        x = dd.build_day_for_skeleton(nut, fd, skel, d, memoria=memoria)
        out.append(x or {"meals": []})
    return (f"deterministico:{country}:{dias}d", {"days": out})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--deterministico", type=int, default=0, metavar="DIAS")
    ap.add_argument("--country", default="DO")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.deterministico:
        # Los knobs se leen al IMPORTAR `deterministic_day`, y `_cargar_vocabularios` importa el grafo, que lo importa:
        # fijarlos después dejaba el día determinista apagado y 30 días sin una comida (medido: «0 platos»).
        import os
        os.environ.setdefault("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
        os.environ.setdefault("MEALFIT_DETERMINISTIC_DAY", "1")
    import horizon
    tope = horizon.repetition_limits_for("balanced", 7)["max_exact_repeat"]
    vocab = _cargar_vocabularios()
    planes = (_planes_corpus() if a.corpus else []) + (_planes_real() if a.real else [])
    if a.deterministico:
        planes.append(_plan_deterministico(a.deterministico, a.country))
    res = {k: medir_plan(p, vocab, tope) for k, p in planes}
    art = {"schema": "2026-09-13.variedad", "fecha": date.today().isoformat(), "tope_7d_balanced": tope,
           "firma": "template_id | (familias de proteína, técnicas, bases, formato)", "ventanas": list(VENTANAS),
           "planes": res}
    for k, r in res.items():
        vs = " · ".join(f"{w}d: {v['nombres_distintos']} nombres / {v['firmas_distintas']} platos / {v['renombrados']} renombrados"
                        for w, v in r["ventanas"].items() if v.get("medida")) or "ninguna ventana cabe"
        print(f"{k:32s} {r['dias']:>2} días · {vs} · ventanas 7d rotas {r['ventanas_7d_rotas']}")
    if a.out:
        Path(a.out).write_text(json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
