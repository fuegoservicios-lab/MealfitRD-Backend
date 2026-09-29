# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-913 · 2026-09-29] ¿Es el escudo pre-INSERT un punto fijo? Lo que la SIGUIENTE pasada deshace.

    python scripts/medir_punto_fijo_escudo.py /tmp/rdv868/*.json /tmp/rdv864/*.json       # 2.ª y 3.ª pasada
    python scripts/medir_punto_fijo_escudo.py --pasadas 3 --out /tmp/punto_fijo DIR/*.json
    python scripts/medir_punto_fijo_escudo.py --json DIR/*.json
    RD_BACKEND=/opt/mealfit/backend python /tmp/medir_punto_fijo_escudo.py DIR/*.json   # el script fuera del árbol vivo

Cada fichero es una batería de `rd_battery.py` (`final_plan` = lo ENTREGADO, que ya pasó una vez por el escudo; `form` y
`user_id`). Sobre una COPIA corre `db_plans._finalize_plan_data_for_insert` otra vez (2.ª pasada) y otra (3.ª), con el mismo
contexto clínico del formulario que usa la batería, y cuenta comida a comida lo que cambia: lista, crudo, pasos, nombre,
ficha y macros. Si la última pasada cambia algo, el escudo NO es un punto fijo: un plan entregado se mueve cada vez que una
frontera lo re-finaliza.

Medido al nacer (VPS, entorno real, código vivo 890, baterías rdv801-868, 60 comidas): 2.ª pasada 34 comidas (13 con kcal
distintas), 3.ª pasada 3 — edamame 105 → 115 → 125 g, arroz integral 45 → 40 g, «10 g de avena» y «10 g de cilantro»
dropeados por `_floor_subservible_portions` (lo entregado traía líneas bajo el piso de 15 g). La cola contrato + pulido sí es
idempotente (0 en la 2.ª).

NO escribe en producción: `PYTEST_CURRENT_TEST` activa la guarda de `db_core` ANTES de importar el backend (toda escritura
cruda lanza; `system_alerts` se descarta). No llama a ningún LLM. tooltip-anchor: P1-PLAN-LOTE-913
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                              # noqa: BLE001
    pass

_BACKEND = Path(os.environ.get("RD_BACKEND") or Path(__file__).resolve().parents[1])   # en el VPS: RD_BACKEND=/opt/mealfit/backend
CAMPOS = ("ingredients", "ingredients_raw", "recipe", "name", "desc", "cals", "protein", "carbs", "fats")


def activar_guarda() -> None:
    """La guarda de escritura de `db_core` mira esta variable: se fija antes de importar nada del backend."""
    os.environ.setdefault("PYTEST_CURRENT_TEST", "medir_punto_fijo_escudo.py::medidor (call)")


def _lista(v) -> list:
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def comparar(a: dict, b: dict, plan: str = "") -> dict:
    """Comida a comida (por posición: el escudo no reordena), qué campos de `b` difieren de `a`. Puro."""
    campos: collections.Counter = collections.Counter()
    detalle: list = []
    comidas = cambiadas = 0
    dias_a, dias_b = (a or {}).get("days") or [], (b or {}).get("days") or []
    for di in range(max(len(dias_a), len(dias_b))):
        ma = (dias_a[di].get("meals") or []) if di < len(dias_a) and isinstance(dias_a[di], dict) else []
        mb = (dias_b[di].get("meals") or []) if di < len(dias_b) and isinstance(dias_b[di], dict) else []
        for mi in range(max(len(ma), len(mb))):
            comidas += 1
            if mi >= len(ma) or mi >= len(mb):
                cambiadas += 1
                campos["comida_ausente"] += 1
                presente = ma[mi] if mi < len(ma) else mb[mi]
                detalle.append({"plan": plan, "dia": di, "comida": mi, "nombre": str(presente.get("name")),
                                "campo": "comida_ausente", "antes": [], "despues": []})
                continue
            cambio = False
            for campo in CAMPOS:
                x, y = _lista(ma[mi].get(campo)), _lista(mb[mi].get(campo))
                if x == y:
                    continue
                cambio = True
                campos[campo] += 1
                fuera = [l for l in x if l not in y]
                dentro = [l for l in y if l not in x]
                detalle.append({"plan": plan, "dia": di, "comida": mi, "nombre": str(mb[mi].get("name")), "campo": campo,
                                "antes": fuera, "despues": dentro})
            cambiadas += cambio
    return {"comidas": comidas, "cambiadas": cambiadas, "campos": dict(campos), "detalle": detalle}


def medir(planes: list, escudo, pasadas: int = 2) -> dict:
    """`planes` = [(id, final_plan, form, user_id)]; `escudo(plan, form, uid) -> plan`. Trabaja sobre copias.
    `punto_fijo`: True si la ÚLTIMA pasada no cambió nada, False si cambió, None si ninguna pasada se pudo medir."""
    por_pasada = [{"pasada": n + 2, "cambiadas": 0, "campos": collections.Counter(), "detalle": []} for n in range(pasadas)]
    errores: list = []
    comidas = 0
    medidas = 0
    for pid, plan, form, uid in planes:
        previo = copy.deepcopy(plan)
        comidas += sum(len(d.get("meals") or []) for d in (previo.get("days") or []) if isinstance(d, dict))
        for n in range(pasadas):
            try:
                siguiente = escudo(copy.deepcopy(previo), form or {}, uid)
            except Exception as e:                                             # noqa: BLE001
                errores.append({"plan": pid, "pasada": n + 2, "error": f"{type(e).__name__}: {e}"})
                break
            r = comparar(previo, siguiente, plan=pid)
            por_pasada[n]["cambiadas"] += r["cambiadas"]
            por_pasada[n]["campos"].update(r["campos"])
            por_pasada[n]["detalle"].extend(r["detalle"])
            medidas += 1
            previo = siguiente
    for p in por_pasada:
        p["campos"] = dict(p["campos"])
    ultima_medida = medidas and not any(e["pasada"] == pasadas + 1 for e in errores)
    return {"planes": len(planes), "comidas": comidas, "pasadas": por_pasada, "errores": errores,
            "punto_fijo": (por_pasada[-1]["cambiadas"] == 0) if (ultima_medida and por_pasada) else None}


def _clinico_del_formulario(fd: dict) -> dict:
    """La misma forma que `db_profiles.build_clinical_form_from_profile` sobre el perfil guardado (como `rd_battery.py`)."""
    mc = fd.get("medicalConditions") or []
    if isinstance(mc, str):
        mc = [mc]
    oc = fd.get("otherConditions")
    if oc and str(oc).strip():
        mc = list(mc) + [str(oc).strip()]
    try:
        from graph_orchestrator import profile_with_free_text
        ft = profile_with_free_text(fd)
    except Exception:                                                          # noqa: BLE001
        ft = fd
    return {"gender": fd.get("gender"), "age": fd.get("age"), "medicalConditions": mc, "medications": fd.get("medications"),
            "otherConditions": oc, "otherMedications": fd.get("otherMedications"),
            "allergies": [str(a).strip() for a in (ft.get("allergies") or []) if str(a).strip()],
            "dietType": fd.get("dietType"), "dislikes": ft.get("dislikes") or [], "scheduleType": fd.get("scheduleType")}


def _escudo_real():
    """El escudo de producción con el contexto clínico del formulario. Importa el backend: la guarda va antes."""
    activar_guarda()
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    os.chdir(_BACKEND)
    try:
        from dotenv import load_dotenv
        load_dotenv(_BACKEND / ".env")
    except Exception:                                                          # noqa: BLE001
        pass
    import logging
    logging.disable(logging.WARNING)
    try:
        import db_core
        db_core.connection_pool.open()
    except Exception as e:                                                     # noqa: BLE001
        print("pool:", e)
    import db_plans

    def escudo(plan, form, uid):
        clin = _clinico_del_formulario(form)
        orig = db_plans._build_clinical_form
        db_plans._build_clinical_form = lambda _u: dict(clin)
        try:
            ins = {"user_id": uid, "plan_data": plan}
            db_plans._finalize_plan_data_for_insert(ins)
            return ins.get("plan_data")
        finally:
            db_plans._build_clinical_form = orig
    return escudo


def render(r: dict, ejemplos: int = 12) -> str:
    out = [f"planes {r['planes']} · comidas {r['comidas']}"]
    for p in r["pasadas"]:
        campos = ", ".join(f"{k} {v}" for k, v in sorted(p["campos"].items(), key=lambda kv: -kv[1])) or "nada"
        out.append(f"{p['pasada']}.ª pasada: cambian {p['cambiadas']} comidas ({campos})")
    for e in r["errores"]:
        out.append(f"ERROR {e['plan']} en la {e['pasada']}.ª pasada: {e['error']}")
    veredicto = {True: "sí", False: "NO", None: "sin medición"}[r["punto_fijo"]]
    out.append(f"PUNTO FIJO: {veredicto}")
    if r["pasadas"] and r["pasadas"][-1]["detalle"]:
        out.append("— lo que cambia en la última pasada —")
        for d in r["pasadas"][-1]["detalle"][:ejemplos]:
            out.append(f"  {d['plan']} d{d['dia']} m{d['comida']} · {d['nombre'][:60]} · {d['campo']}: "
                       f"{' | '.join(d['antes'])[:150]}  →  {' | '.join(d['despues'])[:150]}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Medidor de punto fijo del escudo pre-INSERT (sólo lectura, sin LLM)")
    ap.add_argument("ficheros", nargs="+", help="baterías de rd_battery.py (admite comodines)")
    ap.add_argument("--pasadas", type=int, default=2, help="pasadas EXTRA sobre lo entregado (2 = 2.ª y 3.ª)")
    ap.add_argument("--out", help="directorio donde dejar el detalle completo (punto_fijo.json)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    planes = []
    for f in sorted({x for g in a.ficheros for x in glob.glob(g)}):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as e:                                                 # noqa: BLE001
            print("no se lee", f, e)
            continue
        if d.get("final_plan"):
            planes.append((os.path.basename(os.path.dirname(f)) + "__" + os.path.basename(f)[:24], d["final_plan"],
                           d.get("form") or {}, d.get("user_id")))
    if not planes:
        print("ningún fichero con final_plan")
        return 2
    r = medir(planes, _escudo_real(), pasadas=max(1, a.pasadas))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        Path(a.out, "punto_fijo.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "pasadas"} | {
        "pasadas": [{k: v for k, v in p.items() if k != "detalle"} for p in r["pasadas"]]}, ensure_ascii=False)
        if a.json else render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
