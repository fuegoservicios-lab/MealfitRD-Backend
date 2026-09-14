# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-45 · 2026-09-14] El orden de una receta: el pilar se ROTULA en su sitio y el desorden se MIDE.

Prueba RD del dueño (plan 40535829, 14-sep, día determinista): las 12 comidas traían recetas de la biblioteca
dominicana (la lee sólo `recipe_library`; aquí se consulta a través de él), bien ordenadas y sin los tres rótulos
«Mise en place / El Toque de Fuego / Montaje». El
reparador del contrato (`graph_orchestrator._repair_recipe_contract`) veía «falta El Toque de Fuego» y EXTRAÍA de todos
los pasos cada oración con verbo de cocción a un único paso final: «Escúrrelos bien» y «maja el ajo con el plátano»
quedaban antes de hervir el plátano, y «el pollo está listo a 74 °C» antes de cocinarlo. Los cinco `paso_incoherente`
del juez en ese plan eran eso, y `repair_stage_diff` no lo veía: sólo cuenta códigos V.

  · `rotular_fuego_en_su_sitio` / `rotular_montaje_en_su_sitio`: en una receta NARRATIVA el pilar se rotula donde ya
    está. Nunca se mueve una oración.
  · `fuera_de_orden` / `medir_plan`: cuántas oraciones de la receta de biblioteca se sirven fuera de su orden original
    (emparejadas − subsecuencia creciente más larga). Puro y determinista; no necesita catálogo.

Medido sobre las 193 recetas DO: 178 cocinan; en 138 el primer paso de cocción ya trae tiempo o temperatura, en 21 lo
trae uno posterior y en 19 ninguno; 73 terminan con un paso que empieza sirviendo.
tooltip-anchor: P1-PLAN-LOTE-45-ORDEN
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from typing import Callable, Optional

_RE_ROTULO = re.compile(r"^\s*(?:mise en place|(?:el\s+)?toque de fuego|montaje)\s*:\s*", re.IGNORECASE)
_RE_SERVIR = re.compile(r"^\s*(?:s[ií]rve\w*|emplata\w*|presenta\w*|reparte\w*|acompa[nñ]a\w*|decora\w*)\b",
                        re.IGNORECASE)
_RE_NUMERACION = re.compile(r"^\s*\d+\s*[.)-]\s*")
_RE_ORACION = re.compile(r"(?<=[.!?])\s+")
_RE_NO_PALABRA = re.compile(r"[^a-z ]+")
# Las tres etiquetas de ANOTACIÓN que el frontend no numera (`frontend/src/utils/recipeSteps.js`): no son pasos.
_NOTA_PURA = ("nota del nutricionista", "seguridad alimentaria", "ajustamos ligeramente las porciones")
UMBRAL = 0.6          # Jaccard de palabras (>2 letras) para reconocer una oración de la biblioteca ya retocada
_PAISES = ("DO", "ES", "US", "MX", "PR", "CO")


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def rotulado(paso) -> bool:
    return bool(_RE_ROTULO.match(str(paso or "")))


def nota_pura(paso) -> bool:
    t = _sa(paso)[:120]
    return any(a in t for a in _NOTA_PURA)


def es_anotacion(paso) -> bool:
    """Notas y parches deterministas (⚠ 💡 🌱 💪…): empiezan por símbolo, no por letra. No son pasos de la receta."""
    s = _RE_NUMERACION.sub("", str(paso or "").strip())
    if not s:
        return True
    return not s[0].isalpha() or nota_pura(s)


def paso_sin_rotulo(paso) -> bool:
    """Un paso de ACCIÓN de la receta que todavía no lleva ninguno de los tres rótulos."""
    return isinstance(paso, str) and not es_anotacion(paso) and not rotulado(paso)


def rotular_fuego_en_su_sitio(rec: list, es_coccion: Callable[[str], bool],
                              tiene_tiempo: Optional[Callable[[str], bool]] = None) -> Optional[int]:
    """Antepone «El Toque de Fuego: » al paso de cocción que abre el fuego, SIN mover nada. Devuelve su índice o `None`
    si la receta no tiene pasos sin rótulo que cocinen (entonces decide el llamador: extraer o sintetizar).

    Se prefiere el primer paso de cocción que ya dice tiempo o temperatura: el contrato lo exige en ese pilar, y en 21
    recetas de la biblioteca el primer paso con verbo de fuego es un «sazona… mientras preparas el sofrito»."""
    if not isinstance(rec, list):
        return None
    cands = [i for i, p in enumerate(rec) if paso_sin_rotulo(p) and es_coccion(p)]
    if not cands:
        return None
    i = next((j for j in cands if tiene_tiempo is not None and tiene_tiempo(rec[j])), cands[0])
    rec[i] = "El Toque de Fuego: " + rec[i].strip()
    return i


def rotular_montaje_en_su_sitio(rec: list) -> Optional[int]:
    """Si el ÚLTIMO paso de acción empieza sirviendo («Sirve…», «Sírvelas…», «Emplata…»), le antepone «Montaje: ». Las
    notas puras del final se saltan; un parche de acción (💪) o un paso ya rotulado detrás cierra la búsqueda."""
    if not isinstance(rec, list):
        return None
    for i in range(len(rec) - 1, -1, -1):
        p = rec[i]
        if not isinstance(p, str) or not p.strip() or nota_pura(p):
            continue
        if paso_sin_rotulo(p) and _RE_SERVIR.match(p):
            rec[i] = "Montaje: " + p.strip()
            return i
        return None
    return None


def _oraciones(paso) -> list:
    t = _RE_ROTULO.sub("", _RE_NUMERACION.sub("", str(paso or "").strip()))
    return [o for o in (x.strip() for x in _RE_ORACION.split(t)) if o]


def _clave(oracion) -> frozenset:
    return frozenset(w for w in _RE_NO_PALABRA.sub(" ", _sa(oracion)).split() if len(w) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def fuera_de_orden(recipe, pasos_biblioteca, umbral: float = UMBRAL) -> dict:
    """`{"emparejadas", "fuera"}`: oraciones de la receta servida que se reconocen en la de biblioteca, y cuántas de ellas
    están fuera de su orden original (el mínimo que habría que mover: emparejadas − subsecuencia creciente más larga)."""
    ref = [_clave(o) for p in (pasos_biblioteca or []) if isinstance(p, str) for o in _oraciones(p)]
    idx, prev = [], -1
    for p in recipe or []:
        if not isinstance(p, str) or es_anotacion(p):
            continue
        for o in _oraciones(p):
            k = _clave(o)
            if not k:
                continue
            puntos = [(_jaccard(k, r), i) for i, r in enumerate(ref)]
            mejor = max((s for s, _ in puntos), default=0.0)
            if mejor < umbral:
                continue
            empatados = [i for s, i in puntos if s == mejor]
            elegido = next((i for i in empatados if i >= prev), empatados[-1])
            idx.append(elegido)
            prev = elegido
    colas: list = []
    for x in idx:
        j = bisect.bisect_right(colas, x)
        if j == len(colas):
            colas.append(x)
        else:
            colas[j] = x
    return {"emparejadas": len(idx), "fuera": len(idx) - len(colas)}


def pasos_de_biblioteca(template_id) -> Optional[list]:
    """Los pasos congelados de una plantilla, en la biblioteca que la tenga. `None` si no hay."""
    try:
        import recipe_library as rl
    except Exception:                                                          # noqa: BLE001
        return None
    for cc in _PAISES:
        try:
            r = (rl._library(cc) or {}).get(str(template_id))
        except Exception:                                                      # noqa: BLE001
            r = None
        if isinstance(r, dict) and isinstance(r.get("pasos"), list) and r["pasos"]:
            return list(r["pasos"])
    return None


def medir_plan(plan_data, pasos_de: Optional[Callable] = None) -> dict:
    """Recorre las comidas con receta de biblioteca y cuenta las oraciones fuera de su orden. Determinista (sin tiempos):
    el informe entra en `plan_data` y los tests del seam T2 comparan planes byte a byte."""
    pasos_de = pasos_de or pasos_de_biblioteca
    medidas, fuera_total, detalle = 0, 0, []
    for d in (plan_data or {}).get("days") or []:
        if not isinstance(d, dict):
            continue
        for j, m in enumerate(d.get("meals") or []):
            if not isinstance(m, dict) or m.get("_recipe_source") != "library":
                continue
            tid = m.get("_template_id") or m.get("_recipe_template_id")
            pasos = pasos_de(tid) if tid else None
            if not pasos:
                continue
            r = fuera_de_orden(m.get("recipe") or [], pasos)
            if not r["emparejadas"]:
                continue
            medidas += 1
            if r["fuera"]:
                fuera_total += r["fuera"]
                detalle.append({"day": d.get("day"), "meal_index": j, "template_id": str(tid),
                                "fuera": r["fuera"], "emparejadas": r["emparejadas"]})
    return {"medidas": medidas, "desordenadas": len(detalle), "oraciones_fuera": fuera_total, "detalle": detalle[:20]}
