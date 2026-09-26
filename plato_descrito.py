# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-348 · 2026-09-26] «Descríbelo y lo calculo»: un plato escrito en texto libre, separado en
ingredientes EDITABLES.

El dueño: «¿Qué comiste? es poco flexible: nadie pudiera crear un plato desde cero… si no le tiró foto y no quiere
escribirle al coach». La estimación por texto que ya existía (`/consumed/estimate-macros`) devolvía UN número para
todo el plato; esto devuelve las partes («Moro de habichuelas · 230 g», «Pollo guisado · 150 g», «Aguacate · 60 g»),
cada una con sus macros, para que el usuario corrija la que no cuadre.

Casa con el catálogo SOLO los PLATOS (cocinados: `per_100g` del plato terminado). Los ALIMENTOS del catálogo no: su
`per_100g` es del alimento crudo/seco y la IA estima lo que se comió, cocido — «200 g de arroz» casado con el arroz
crudo del catálogo serían ~720 kcal en vez de ~260 (la base del número, lotes 282-286). Un plato casado toma sus
macros del catálogo (no las de la IA) y, al registrar, sus constituyentes pueden descontarse de la Nevera.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_LINEAS = 12
_MAX_GRAMOS = 2000.0


class ItemDelPlato(BaseModel):
    name: str = Field(default="", max_length=120)
    grams: float = Field(default=0.0, ge=0.0, le=5000.0)
    calories: float = Field(default=0.0, ge=0.0, le=5000.0)
    protein: float = Field(default=0.0, ge=0.0, le=500.0)
    carbs: float = Field(default=0.0, ge=0.0, le=800.0)
    healthy_fats: float = Field(default=0.0, ge=0.0, le=400.0)


class PlatoEstimado(BaseModel):
    name: str = Field(default="", max_length=120)
    items: list[ItemDelPlato] = Field(default_factory=list)


def _platos_por_nombre() -> dict:
    import food_search
    out = {}
    for slug, d in (food_search.load_dishes() or {}).items():
        label = (d or {}).get("label")
        if label:
            out[food_search._norm(label)] = (slug, label)
    return out


def prompt_del_sistema() -> str:
    """El sistema del estimador. Lleva los NOMBRES de los platos del catálogo: si el usuario comió uno, la IA lo
    nombra igual y el plato casa (con sus macros de catálogo)."""
    platos = sorted({label for _, label in _platos_por_nombre().values()})
    return (
        "Eres un nutricionista dominicano. El usuario describe en texto libre lo que comió en UNA comida. "
        "Separa lo que comió en sus partes (máximo 12) y estima cada una por separado, en porción normal de adulto "
        "salvo que el texto diga la cantidad. Pesa cada parte COMO SE COMIÓ (cocida, servida), no cruda. "
        "Si una parte es uno de estos platos, usa EXACTAMENTE ese nombre: " + "; ".join(platos) + ". "
        "Devuelve SOLO un JSON: {\"name\": \"nombre corto de la comida (≤60 caracteres)\", \"items\": [{\"name\": "
        "\"nombre de la parte\", \"grams\": gramos (número), \"calories\": kcal, \"protein\": g, \"carbs\": g, "
        "\"healthy_fats\": g}]}. Las macros de cada parte son las de ESOS gramos. Sin texto fuera del JSON."
    )


def _num(v, tope) -> float:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return 0.0
    if n != n or n in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(float(tope), n))


def lineas_del_plato(items) -> list[dict]:
    """Los ítems de la IA → líneas del componedor. Plato del catálogo → `dish:<slug>` en gramos con macros del
    catálogo; lo demás → `custom` con las macros estimadas (`estimated: True`) y sus gramos para mostrarlos."""
    import food_search
    platos = _platos_por_nombre()
    out = []
    for it in items or []:
        if hasattr(it, "model_dump"):
            it = it.model_dump()
        if not isinstance(it, dict):
            continue
        nombre = " ".join(str(it.get("name") or "").split())[:120]
        if not nombre:
            continue
        gramos = round(_num(it.get("grams"), _MAX_GRAMOS))
        casado = platos.get(food_search._norm(nombre))
        if casado and gramos >= 1:
            slug, label = casado
            r = food_search.resolve_line({"ref": f"dish:{slug}", "qty": gramos, "unit": "g"}, [])
            out.append({"ref": f"dish:{slug}", "qty": gramos, "unit": "g", "name": label,
                        "grams": gramos, "macros": r["macros"], "estimated": False})
        else:
            macros = food_search._clamp_macros({
                "kcal": _num(it.get("calories"), 5000), "protein": _num(it.get("protein"), 500),
                "carbs": _num(it.get("carbs"), 800), "fats": _num(it.get("healthy_fats"), 400),
            })
            out.append({"ref": "custom", "name": nombre, "grams": gramos or None, "macros": macros, "estimated": True})
        if len(out) == MAX_LINEAS:
            break
    return out


async def estimar_con_ia(texto: str, idioma: str | None, user_id: str) -> dict:
    """La llamada al modelo flash (el mismo del estimador de macros). Lanza si falla: el router hace el soft-fail."""
    from graph_orchestrator import ChatGLM, _plan_flash_model_name, _current_node_var, user_id_var
    from langchain_core.messages import SystemMessage, HumanMessage
    humano = f"Comida: {texto}\n" + (f"Escribe 'name' en {idioma}; los nombres de las partes, en español."
                                    if idioma else "Escribe todo en español.")
    llm = ChatGLM(model=_plan_flash_model_name(), temperature=0.1, max_retries=1, timeout=30).with_structured_output(
        PlatoEstimado, method="json_mode"
    )
    tok_node = _current_node_var.set("diary_plate_estimate")
    tok_user = user_id_var.set(user_id)
    try:
        est = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=prompt_del_sistema()), HumanMessage(content=humano)]), timeout=35,
        )
    finally:
        try:
            _current_node_var.reset(tok_node)
            user_id_var.reset(tok_user)
        except Exception:
            pass
    if not isinstance(est, PlatoEstimado):
        raise ValueError("respuesta sin forma")
    return est.model_dump()
