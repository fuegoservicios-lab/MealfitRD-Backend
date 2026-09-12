# -*- coding: utf-8 -*-
"""[P1-ARQ27-F3-BATERIA · 2026-09-06] Batería de entrega con las banderas de PRODUCCIÓN (ARQ27-P1-06).

El gap decía: «110 tests de siete módulos pasan en el backend directo; las sondas detectan fallos que
esos tests no cubren. Conftest mantiene algunos gates legacy apagados; los flags vivos no se
verificaron». Medido contra el `.env` del VPS el 2026-09-06, **son seis los knobs divergentes**, y el
más caro es `MEALFIT_COUNTRY_SYSTEM`: producción `true`, la suite `False`. Con él apagado los seis
países colapsan a DO, así que ningún test de la suite ha visto nunca el catálogo de ES, US, MX, PR ni
CO. Un test que no declara el flag mide otro producto.

Esta batería aplica el perfil de `prod_profile.py` y recorre una matriz de cohortes, publicando
**tasas con su denominador** por dimensión — nunca un promedio, que es donde se esconden las
regresiones de una cohorte pequeña (la vegetal, siempre).

    python scripts/delivery_battery.py                 # tabla por cohorte
    python scripts/delivery_battery.py --json          # para diffear entre versiones
    python scripts/delivery_battery.py --cohorte vegana

## Las seis dimensiones, separadas a propósito

| dimensión | qué cuenta | por qué separada |
|---|---|---|
| entrega | cohortes con ≥1 candidato en TODAS sus franjas | una cohorte sin cena no tiene plan |
| seguridad | candidatos que violan alergia o dieta | cero es el único valor aceptable |
| nutrición | candidatos sin el nutriente que la condición exige | un ausente no es un cero (I20) |
| variedad | candidatos distintos por franja frente al mínimo del modo | 1 candidato «entrega», pero repite |
| completitud | plantillas descartadas por datos incompletos | mide la deuda del catálogo, no del motor |
| sin relajación | la cohorte más restringida ⊆ la laxa | un fallback silencioso reintroduce lo excluido |

## Lo que esta batería NO hace, dicho aquí y no escondido

**No mide latencia ni coste por plan entregado.** Eso es el canary del gap y necesita generaciones
reales contra el proveedor: no se puede fabricar en una batería determinista, y fingirlo sería peor
que no tenerlo. Queda como trabajo abierto de `ARQ27-P1-06`. **[P1-PLAN-LOTE-15 · 2026-09-12] Lo mide
`scripts/canary_plan_delivery.py` sobre los planes REALES de producción (fallos, reintentos, latencia y coste por
plan, con denominador por cohorte), desde que el worker atribuye el coste al plan.**

**No ejercita el swap ni el último chunk end-to-end.** Requieren DB y LLM. Lo que sí hace es medir el
embudo con las banderas correctas, que es donde el gap encontró la ceguera. **[P1-PLAN-LOTE-15 · 2026-09-12] Lo
verifica `scripts/verify_swap_last_chunk.py` con el MISMO guard de producción sobre los planes persistidos.**

**Cero hallazgos en N cohortes no demuestra una garantía universal** — por eso la tabla imprime N
junto a cada tasa, y no solo el porcentaje.
"""
from __future__ import annotations

import argparse
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

import prod_profile  # noqa: E402

# El perfil se aplica DENTRO de `correr()`, con `perfil_aplicado()`, y el entorno vuelve al salir.
#
# La primera versión lo aplicaba aquí, en tiempo de import — y contaminó la suite entera: el test de
# la batería importa este módulo, así que 17 tests de coherencia, compras y hierbas empezaron a correr
# con los gates de producción encendidos por debajo. Una herramienta que existe para denunciar que la
# suite corre con otras banderas no puede ser quien se las cambie a los demás.
#
# Los knobs que `constants.py` congela en constantes de módulo (`COUNTRY_SYSTEM_ENABLED`) no cambian
# con esto, pero el camino que la batería mide —`_mercado()`, el catálogo, el embudo— los relee por
# llamada, y la medición sale idéntica: se comprobó comparando las dos formas.

FRANJAS = ("desayuno", "almuerzo", "cena", "merienda")

#: Matriz de cohortes. País de COCINA × dieta × alergias × condición. Deliberadamente incluye las
#: combinaciones que el embudo ya sabe duras (vegana en PR, sin lácteos ni huevo) y una imposible.
COHORTES = [
    {"id": "omnivora_do", "pais": "DO", "dieta": "omnivora", "alergias": (), "condicion": None},
    {"id": "omnivora_es", "pais": "ES", "dieta": "omnivora", "alergias": (), "condicion": None},
    {"id": "omnivora_us", "pais": "US", "dieta": "omnivora", "alergias": (), "condicion": None},
    {"id": "omnivora_mx", "pais": "MX", "dieta": "omnivora", "alergias": (), "condicion": None},
    {"id": "omnivora_pr", "pais": "PR", "dieta": "omnivora", "alergias": (), "condicion": None},
    {"id": "omnivora_co", "pais": "CO", "dieta": "omnivora", "alergias": (), "condicion": None},
    {"id": "vegetariana_do", "pais": "DO", "dieta": "vegetariana", "alergias": (), "condicion": None},
    {"id": "vegana_do", "pais": "DO", "dieta": "vegana", "alergias": (), "condicion": None},
    {"id": "vegana_pr", "pais": "PR", "dieta": "vegana", "alergias": (), "condicion": None},
    {"id": "vegana_es", "pais": "ES", "dieta": "vegana", "alergias": (), "condicion": None},
    {"id": "vegana_sin_gluten", "pais": "DO", "dieta": "vegana", "alergias": ("gluten",), "condicion": None},
    {"id": "vegana_sin_soja", "pais": "DO", "dieta": "vegana", "alergias": ("soja",), "condicion": None},
    {"id": "sin_lacteos_ni_huevo", "pais": "DO", "dieta": "omnivora",
     "alergias": ("lacteos", "huevo"), "condicion": None},
    {"id": "renal_do", "pais": "DO", "dieta": "omnivora", "alergias": (), "condicion": "renal"},
    {"id": "hta_do", "pais": "DO", "dieta": "omnivora", "alergias": (), "condicion": "hta"},
    # La cohorte MÁXIMAMENTE restringida. No se le exige cero —medido, 6 desayunos veganos sin
    # gluten, soja, frutos secos ni legumbres existen de verdad, y exigir cero habría sido inventar un
    # defecto—; se le exige que sus supervivientes sean un SUBCONJUNTO de los de la cohorte laxa. Un
    # motor que relajara en silencio devolvería algo que la laxa no tenía, y un contador no lo vería.
    {"id": "maxima_restriccion", "pais": "DO", "dieta": "vegana", "subconjunto_de": "omnivora_do",
     "alergias": ("gluten", "soja", "frutos_secos", "legumbres", "mani", "ajonjoli"), "condicion": "renal"},
]

#: Mínimo de candidatos distintos por franja para no considerar que la cohorte «repite».
VARIEDAD_MIN = 3


def _medir_cohorte(c: dict) -> dict:
    from horizon import required_nutrients
    from scripts.coverage_funnel import embudo

    # La forma es `effective["clinical"]["conditions"]`, no `{"conditions": [...]}`. Con la forma
    # equivocada `required_nutrients` es fail-open a `()` — la etapa «datos exigidos» no corría y la
    # dimensión de nutrición medía la nada mientras imprimía un porcentaje. Un fail-open silencioso
    # dentro de una MEDICIÓN es peor que un error: te da un número.
    req = required_nutrients({"clinical": {"conditions": [c["condicion"]]}} if c.get("condicion") else {})
    franjas = {}
    for slot in FRANJAS:
        r = embudo(c["pais"], slot, c["dieta"], tuple(c["alergias"]), None, "limited",
                   requiere_nutrientes=tuple(req))
        etapas = dict(r["etapas"]) if isinstance(r.get("etapas"), list) else (r.get("etapas") or {})
        franjas[slot] = {
            "elegibles": int(r.get("elegibles", 0)),
            "catalogo": int(etapas.get("catálogo", 0) or 0),
            "caidas": {k: len(v) for k, v in (r.get("caidas") or {}).items()},
            "supervivientes": list(r.get("supervivientes") or []),
        }
    elegibles = {s: v["elegibles"] for s, v in franjas.items()}
    return {
        "id": c["id"], "pais": c["pais"], "dieta": c["dieta"],
        "alergias": list(c["alergias"]), "condicion": c.get("condicion"),
        "subconjunto_de": c.get("subconjunto_de"),
        "franjas": franjas,
        "elegibles": elegibles,
        "franjas_vacias": [s for s, n in elegibles.items() if n == 0],
        "franjas_pobres": [s for s, n in elegibles.items() if 0 < n < VARIEDAD_MIN],
    }


def _plantillas(pais: str) -> dict:
    import dish_registry as dr
    return dr.templates_by_id(pais)


def _violan_alergia(f: dict) -> list:
    """Supervivientes que llevan una clase de alérgeno EXCLUIDA. Cero es el único valor aceptable."""
    ex = {a.lower() for a in f["alergias"]}
    if not ex:
        return []
    idx = _plantillas(f["pais"])
    malos = []
    for slot, v in f["franjas"].items():
        for tid in v["supervivientes"]:
            t = idx.get(str(tid)) or {}
            al = {a.lower() for a in (t.get("intrinsic_risk_attributes") or {}).get("allergens", [])}
            if ex & al:
                malos.append((slot, tid, sorted(ex & al)))
    return malos


def _violan_nutriente(f: dict) -> list:
    """Supervivientes cuyo dato del nutriente que la condición EXIGE es desconocido (I20: un ausente
    no es un cero — un perfil renal no puede recibir un plato cuyo fósforo nadie midió)."""
    if not f.get("condicion"):
        return []
    from horizon import required_nutrients
    req = set(required_nutrients({"clinical": {"conditions": [f["condicion"]]}}))
    if not req:
        return []
    idx = _plantillas(f["pais"])
    malos = []
    for slot, v in f["franjas"].items():
        for tid in v["supervivientes"]:
            t = idx.get(str(tid)) or {}
            falta = req & set(t.get("nutrition_unknown") or {})
            if falta:
                malos.append((slot, tid, sorted(falta)))
    return malos


def _completitud_por_pais(filas: list) -> dict:
    """Fracción del catálogo de cada cocina que compila entera. Mide la deuda del CATÁLOGO, no la del
    motor, y por eso su denominador son plantillas y no cohortes."""
    import dish_registry as dr
    out = {}
    for pais in sorted({f["pais"] for f in filas}):
        snap = dr.load_registry(pais) or {}
        ts = snap.get("templates") or []
        ok = sum(1 for t in ts if t.get("status") == "ok")
        out[pais] = {"n": ok, "de": len(ts), "pct": round(100.0 * ok / len(ts), 1) if ts else None}
    return out


def correr(solo: str | None = None) -> dict:
    with prod_profile.perfil_aplicado() as _cambios:
        return _correr_con_perfil(solo, _cambios)


def _correr_con_perfil(solo: str | None, _CAMBIOS: dict) -> dict:
    cohortes = [c for c in COHORTES if not solo or solo in c["id"]]
    filas = [_medir_cohorte(c) for c in cohortes]

    reales = [f for f in filas if not f["subconjunto_de"]]
    restringidas = [f for f in filas if f["subconjunto_de"]]
    por_id = {f["id"]: f for f in filas}

    def es_subconjunto(f) -> bool:
        laxa = por_id.get(f["subconjunto_de"])
        if not laxa:
            return False
        return all(set(f["franjas"][s]["supervivientes"]) <= set(laxa["franjas"][s]["supervivientes"])
                   for s in FRANJAS)

    def tasa(n, d):
        return {"n": n, "de": d, "pct": round(100.0 * n / d, 1) if d else None}

    # Seguridad y nutrición se miden sobre los SUPERVIVIENTES, no contando lo que cayó.
    #
    # La primera versión contaba «¿mordió la etapa del filtro?» y castigaba a `renal_do` y `hta_do`
    # por no tirar nada — cuando no tirar nada significa que TODA la biblioteca dominicana tiene el
    # fósforo, el potasio y el sodio medidos, que es el resultado bueno. Una métrica que llama fallo
    # al mejor caso posible habla de la métrica, no del producto.
    seguridad_ok = sum(1 for f in reales if not _violan_alergia(f))
    nutricion_ok = sum(1 for f in reales if not _violan_nutriente(f))

    return {
        "perfil": {
            "leido": prod_profile.PROFILE_READ_AT,
            "fuente": prod_profile.PROFILE_SOURCE,
            "knobs_aplicados": len(prod_profile.perfil_completo()),
            "cambios_respecto_al_entorno": _CAMBIOS,
            "excluidos_a_sabiendas": prod_profile.EXCLUIDOS_A_SABIENDAS,
        },
        "cohortes": filas,
        "completitud_catalogo": _completitud_por_pais(filas),
        "dimensiones": {
            "entrega": tasa(sum(1 for f in reales if not f["franjas_vacias"]), len(reales)),
            "seguridad": tasa(seguridad_ok, len(reales)),
            "nutricion": tasa(nutricion_ok, len(reales)),
            "variedad": tasa(sum(1 for f in reales if not f["franjas_pobres"] and not f["franjas_vacias"]),
                             len(reales)),
                        "sin_relajacion": tasa(sum(1 for f in restringidas if es_subconjunto(f)),
                                   len(restringidas)),
        },
        "no_medido": [
            "latencia y coste por plan entregado (canary: necesita generaciones reales)",
            "swap y último chunk end-to-end (necesitan DB y LLM)",
        ],
    }


def render(r: dict) -> str:
    p = r["perfil"]
    out = [f"perfil de producción leído el {p['leido']} de {p['fuente']}",
           f"knobs aplicados: {p['knobs_aplicados']}"]
    if p["cambios_respecto_al_entorno"]:
        out.append(f"  divergencias corregidas al arrancar ({len(p['cambios_respecto_al_entorno'])}):")
        for k, v in sorted(p["cambios_respecto_al_entorno"].items()):
            out.append(f"    {k:44s} {v}")
    else:
        out.append("  el entorno ya estaba alineado con producción")

    out.append("")
    out.append(f"{'cohorte':22s} {'desayuno':>9s} {'almuerzo':>9s} {'cena':>9s} {'merienda':>9s}   estado")
    out.append("-" * 78)
    for f in r["cohortes"]:
        e = f["elegibles"]
        if f["subconjunto_de"]:
            laxa = {x["id"]: x for x in r["cohortes"]}.get(f["subconjunto_de"])
            ok = laxa and all(set(f["franjas"][s]["supervivientes"])
                              <= set(laxa["franjas"][s]["supervivientes"]) for s in FRANJAS)
            estado = "subconjunto ✓" if ok else "⚠ relajó: devuelve lo que la laxa no tenía"
        elif f["franjas_vacias"]:
            estado = "⛔ sin " + ", ".join(f["franjas_vacias"])
        elif f["franjas_pobres"]:
            estado = "⚠ pobre en " + ", ".join(f["franjas_pobres"])
        else:
            estado = "ok"
        out.append(f"{f['id']:22s} {e['desayuno']:9d} {e['almuerzo']:9d} {e['cena']:9d} "
                   f"{e['merienda']:9d}   {estado}")

    out.append("")
    out.append("dimensión           tasa      denominador")
    out.append("-" * 44)
    for k, v in r["dimensiones"].items():
        pct = "—" if v["pct"] is None else f"{v['pct']:.1f} %"
        out.append(f"{k:20s}{pct:>8s}      {v['n']} de {v['de']}")

    out.append("")
    out.append("completitud del catálogo (plantillas que compilan enteras, por cocina):")
    for pais, v in r["completitud_catalogo"].items():
        pct = "—" if v["pct"] is None else f"{v['pct']:.1f} %"
        out.append(f"  {pais}  {pct:>7s}   {v['n']} de {v['de']}")

    out.append("")
    out.append("NO medido por esta batería:")
    for x in r["no_medido"]:
        out.append(f"  · {x}")
    out.append("")
    out.append("Cero hallazgos en estas cohortes no demuestra una garantía universal: el denominador")
    out.append("va impreso al lado de cada tasa justamente para que no se lea como si la demostrara.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorte", help="filtra por subcadena del id de cohorte")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from db_core import connection_pool
    if connection_pool is not None:
        try:
            connection_pool.open()   # fuera de FastAPI el pool nace cerrado y el catálogo sale vacío
        except Exception:
            pass

    r = correr(args.cohorte)
    print(json.dumps(r, ensure_ascii=False, indent=2) if args.json else render(r))
    duras = [f["id"] for f in r["cohortes"] if not f["subconjunto_de"] and f["franjas_vacias"]]
    if r["dimensiones"]["sin_relajacion"]["pct"] not in (None, 100.0):
        duras.append("relajación silenciosa")
    return 1 if duras else 0


if __name__ == "__main__":
    raise SystemExit(main())
