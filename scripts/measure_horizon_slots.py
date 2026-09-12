# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-20 · 2026-09-12 · E6 · ARQ30-P1-03] ¿Cuántas franjas del horizonte se quedan SIN candidato viable con el
round-robin de hoy? La sombra que decide si un allocator urge.

`horizon.build_blueprint` asigna la familia de proteína por día con `pool[d % len(pool)]` y fija 3 candidatos del
registry por día × franja con los filtros del usuario (dieta, alergias, exclusiones, mercado, presupuesto, durabilidad
bajo compra única). Un allocator (CSP día × franja) sólo aporta si hoy hay franjas que se quedan sin plato **porque la
familia que le tocó al día no tiene candidato, y otra familia sí** (`rescuable_by_family`). Si las franjas vacías son
huecos de biblioteca (ninguna familia tiene plato con esos filtros), lo que falta son plantillas, no un allocator.

El blueprint ya anota esas franjas (`registry.empty_slots`, P1-PLAN-LOTE-20); este guion construye blueprints para una
matriz REPRODUCIBLE y las cuenta:

  · 6 países de mercado × los 25 perfiles clínicos del landing (`landing_benchmarks.build_landing_profiles`, fieles a
    los chips del wizard: dietas balanced/vegetarian/vegan, alergias, condiciones renal/HTA…);
  · 4 escenarios de compra: semanal · quincenal · mensual con congelador limitado · mensual SIN congelador y sin
    reposición de frescos (el cruce de E9);
  · horizonte = el ciclo de compra (7 / 15 / 30 días), 4 comidas al día.

Sólo LEE (snapshots del registry y catálogo `master_ingredients`); no escribe en la base ni llama al modelo. Fuerza en
ESTE proceso `MEALFIT_PLAN_POLICY_MODE=shadow` si está `off`, para que la Fase 2 compile la política (sin tocar el
entorno de producción).

    python scripts/measure_horizon_slots.py [--json] [--paises DO,ES] [--perfiles 5]

Veredicto: «allocator URGE» si hay franjas rescatables por otra familia; «allocator NO urge» si todas las vacías son
huecos de biblioteca (y dice cuáles); «sin huecos» si no hay franjas vacías. Medido el 2026-09-12: ver
`docs/arq30_e5_e7_diseno_canario.md` → E6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Optional

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_BACKEND)  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym (P1-PLAN-LOTE-13)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PAISES = ("DO", "ES", "US", "MX", "PR", "CO")
#: (etiqueta, groceryDuration, freezerMode, freshTopup, días del horizonte)
ESCENARIOS = (
    ("semanal", "weekly", "limited", None, 7),
    ("quincenal", "biweekly", "limited", None, 15),
    ("mensual", "monthly", "limited", None, 30),
    ("mensual_sin_congelador", "monthly", "none", "no", 30),
)


# [P2-LOGGER-EXEMPT: CLI de medición read-only; la salida ES el informe (stdout), no un log de servicio]
def _say(msg: str = "") -> None:
    print(msg)


def _pct(n: int, den: int) -> Optional[float]:
    return round(100.0 * n / den, 2) if den else None


def blueprint_para(form: dict, country: str, escenario: tuple) -> Optional[dict]:
    """El blueprint que produciría el run para este formulario y escenario (misma política ⇒ mismo blueprint)."""
    from horizon import build_blueprint, compiled_policy_for_form, _meals_per_day
    _, dur, freezer, topup, dias = escenario
    f = dict(form)
    f["country"] = country
    f["groceryDuration"] = dur
    f["freezerMode"] = freezer
    if topup is not None:
        f["freshTopup"] = topup
    comp = compiled_policy_for_form(f, country=country)
    if not comp:
        return None
    return build_blueprint(comp["effective"], total_days=dias, meals_per_day=_meals_per_day(f))


def contar(bp: dict) -> dict:
    """Las cifras de UN blueprint: franjas, candidatos por franja, vacías y por qué."""
    reg = (bp or {}).get("registry") or {}
    cands = reg.get("candidates") or {}
    dias = bp.get("days") or []
    total = sum(len(d.get("slots") or []) for d in dias)
    por_n = Counter(min(3, len(cands.get(f"{d.get('day_index')}:{s}") or [])) for d in dias for s in (d.get("slots") or []))
    vacias = list(reg.get("empty_slots") or [])
    return {
        "franjas": total, "con_3": por_n.get(3, 0), "con_2": por_n.get(2, 0), "con_1": por_n.get(1, 0),
        "vacias": len(vacias), "rescatables_por_familia": sum(1 for e in vacias if e.get("rescuable_by_family")),
        "hueco_biblioteca": sum(1 for e in vacias if not e.get("rescuable_by_family")),
        "vacias_detalle": vacias, "culture_fallbacks": len(reg.get("culture_fallbacks") or []),
        "reasignaciones": len(reg.get("family_reassignments") or []), "viable_family": bool(reg.get("viable_family")),
        "snapshot": reg.get("snapshot_hash"),
    }


def veredicto(res: dict) -> str:
    if res["blueprints"] == 0:
        return "NO CONCLUYENTE: ningún blueprint compilado"
    if res.get("reasignaciones", 0) > 0:
        return (f"ALLOCATOR MÍNIMO ENCENDIDO: {res['reasignaciones']} días con la familia movida a la que cubre MÁS franjas; "
                f"quedan {res['vacias']} franjas vacías de {res['franjas']} ({res['hueco_biblioteca']} huecos de biblioteca, "
                f"{res['rescatables_por_familia']} que ni así se cubren: fuera del pool)")
    if res["vacias"] == 0:
        return (f"SIN HUECOS: 0 franjas vacías de {res['franjas']} en {res['blueprints']} blueprints "
                f"⇒ el allocator NO urge; el round-robin no deja franjas sin plato")
    if res["rescatables_por_familia"] > 0:
        return (f"ALLOCATOR URGE: {res['rescatables_por_familia']} de {res['vacias']} franjas vacías tendrían plato con OTRA "
                f"familia de proteína ({_pct(res['rescatables_por_familia'], res['franjas'])} % de {res['franjas']} franjas); "
                f"{res['hueco_biblioteca']} son huecos de biblioteca")
    return (f"ALLOCATOR NO URGE: las {res['vacias']} franjas vacías ({_pct(res['vacias'], res['franjas'])} % de "
            f"{res['franjas']}) son huecos de BIBLIOTECA (ninguna familia tiene plato con esos filtros): faltan plantillas, "
            f"no un allocator")


def medir(paises=PAISES, escenarios=ESCENARIOS, max_perfiles: Optional[int] = None) -> dict:
    from landing_benchmarks import build_landing_profiles, strip_benchmark_meta

    tot = Counter()
    por_pais, por_escenario, por_franja, por_dieta = (defaultdict(Counter) for _ in range(4))
    peores, huecos = [], Counter()
    sin_compilar = []
    for country in paises:
        perfiles = build_landing_profiles(country)
        if max_perfiles:
            perfiles = perfiles[:max_perfiles]
        for p in perfiles:
            label = str(p.get("_label") or p.get("_id"))
            form = strip_benchmark_meta(p)
            for esc in escenarios:
                bp = blueprint_para(form, country, esc)
                if not bp:
                    sin_compilar.append(f"{country}:{label}:{esc[0]}")
                    continue
                c = contar(bp)
                tot["blueprints"] += 1
                for k in ("franjas", "con_3", "con_2", "con_1", "vacias", "rescatables_por_familia", "hueco_biblioteca",
                          "culture_fallbacks", "reasignaciones"):
                    tot[k] += c[k]
                    por_pais[country][k] += c[k]
                    por_escenario[esc[0]][k] += c[k]
                    por_dieta[str(form.get("dietType") or "balanced")][k] += c[k]
                for e in c["vacias_detalle"]:
                    por_franja[e["slot"]]["vacias"] += 1
                    por_franja[e["slot"]]["rescatables_por_familia" if e.get("rescuable_by_family") else "hueco_biblioteca"] += 1
                    huecos[f"{country} · {form.get('dietType')} · {e['slot']} · fam={e.get('family')} · día {e['day_index'] + 1}"] += 1
                if c["vacias"]:
                    peores.append({"pais": country, "perfil": label, "escenario": esc[0], "vacias": c["vacias"],
                                   "rescatables": c["rescatables_por_familia"], "franjas": c["franjas"],
                                   "ejemplos": [f"día {e['day_index'] + 1} {e['slot']} fam={e.get('family')}"
                                                f"{' (otra familia sí)' if e.get('rescuable_by_family') else ''}"
                                                for e in c["vacias_detalle"][:4]]})
    peores.sort(key=lambda x: (-x["rescatables"], -x["vacias"]))
    res = {k: int(v) for k, v in tot.items()}
    res.setdefault("blueprints", 0)
    for k in ("franjas", "con_3", "con_2", "con_1", "vacias", "rescatables_por_familia", "hueco_biblioteca", "culture_fallbacks"):
        res.setdefault(k, 0)
    res["vacias_pct"] = _pct(res["vacias"], res["franjas"])
    res["por_pais"] = {k: dict(v) for k, v in por_pais.items()}
    res["por_escenario"] = {k: dict(v) for k, v in por_escenario.items()}
    res["por_dieta"] = {k: dict(v) for k, v in por_dieta.items()}
    res["por_franja"] = {k: dict(v) for k, v in por_franja.items()}
    res["huecos_mas_frecuentes"] = [{"hueco": h, "n": n} for h, n in huecos.most_common(15)]
    res["peores_blueprints"] = peores[:12]
    res["sin_compilar"] = sin_compilar
    res["veredicto"] = veredicto(res)
    return res


def render(res: dict) -> str:
    o = [f"blueprints {res['blueprints']} · franjas {res['franjas']} · con 3 candidatos {res['con_3']} · con 2 {res['con_2']} · "
         f"con 1 {res['con_1']} · VACÍAS {res['vacias']} ({res['vacias_pct']} %) · rescatables por otra familia "
         f"{res['rescatables_por_familia']} · huecos de biblioteca {res['hueco_biblioteca']} · cocina→mercado {res['culture_fallbacks']}"
         f" · allocator mínimo {'ON' if os.environ.get('MEALFIT_HORIZON_VIABLE_FAMILY', '').lower() in ('1', 'true', 'yes', 'on') else 'off'}"
         f" · días reasignados {res.get('reasignaciones', 0)}"]
    for titulo, clave in (("por país", "por_pais"), ("por escenario", "por_escenario"), ("por dieta", "por_dieta")):
        o.append(f"  {titulo}:")
        for k, v in res[clave].items():
            o.append(f"    {k:24s} franjas {v.get('franjas', 0):5d} · vacías {v.get('vacias', 0):4d} · rescatables {v.get('rescatables_por_familia', 0):4d}"
                     f" · huecos {v.get('hueco_biblioteca', 0):4d} · con 3: {v.get('con_3', 0)}")
    if res["por_franja"]:
        o.append("  por franja (vacías): " + " · ".join(f"{k}={v.get('vacias', 0)} (rescatables {v.get('rescatables_por_familia', 0)})"
                                                        for k, v in res["por_franja"].items()))
    if res["huecos_mas_frecuentes"]:
        o.append("  huecos más frecuentes:")
        for h in res["huecos_mas_frecuentes"]:
            o.append(f"    ×{h['n']:3d} {h['hueco']}")
    if res["peores_blueprints"]:
        o.append("  peores blueprints:")
        for p in res["peores_blueprints"]:
            o.append(f"    {p['pais']} {p['perfil']} {p['escenario']}: {p['vacias']} vacías de {p['franjas']} "
                     f"(rescatables {p['rescatables']}) · {'; '.join(p['ejemplos'])}")
    if res["sin_compilar"]:
        o.append(f"  sin compilar: {len(res['sin_compilar'])} → {res['sin_compilar'][:5]}")
    o.append(res["veredicto"])
    return "\n".join(o)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--paises", default=",".join(PAISES))
    ap.add_argument("--perfiles", type=int, default=None, help="limita los perfiles del landing por país")
    ap.add_argument("--viable", action="store_true",
                    help="mide con MEALFIT_HORIZON_VIABLE_FAMILY encendido en ESTE proceso (el allocator mínimo); sin él se APAGA "
                         "explícitamente (round-robin puro), aunque el default de producción sea ON desde el lote 21")
    args = ap.parse_args(argv)
    if str(os.environ.get("MEALFIT_PLAN_POLICY_MODE", "off")).strip().lower() == "off":
        os.environ["MEALFIT_PLAN_POLICY_MODE"] = "shadow"   # sólo en este proceso: la Fase 2 tiene que compilar
    # [P1-PLAN-LOTE-21 · 2026-09-12] El knob es ON por defecto desde este lote: la medición «round-robin puro» (sin
    # --viable) tiene que APAGARLO explícitamente en este proceso, o mediría el allocator creyendo medir el diagnóstico.
    # Las dos mediciones quedan reproducibles con cualquier default. tooltip-anchor: P1-PLAN-LOTE-21-MEDIR-EXPLICITO
    os.environ["MEALFIT_HORIZON_VIABLE_FAMILY"] = "true" if args.viable else "false"
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND, ".env"))
    try:
        import db_core   # el catálogo (`_buyable`, nutrientes exigidos) sale del pool: fuera de FastAPI hay que abrirlo
        db_core.connection_pool.open()
    except Exception:
        pass
    res = medir(paises=tuple(p.strip().upper() for p in args.paises.split(",") if p.strip()), max_perfiles=args.perfiles)
    _say(json.dumps(res, ensure_ascii=False, indent=1) if args.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
