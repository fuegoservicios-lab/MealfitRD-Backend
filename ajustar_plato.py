# -*- coding: utf-8 -*-
"""[P1-SCAN-AI-ASSISTANT-MODAL · 2026-10-20] Ajustar plato y añadir ingredientes con IA en el modal de revisión.

Permite al usuario interactuar directamente dentro del modal de escaneo de comida:
1. Ajustar el plato completo mediante instrucciones en lenguaje natural (ej: "quítale la ensalada y ponle 2 huevos fritos").
2. Añadir un ingrediente/alimento individual por texto libre con sus macros estimadas (ej: "1 taza de café con leche").
Exentos de cuota mensual de planes (el uso va a llm_usage_events vía ChatGLM con el modelo flash).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_TOPES = {"calories": 4000.0, "protein": 350.0, "carbs": 500.0, "healthy_fats": 300.0}


class ItemComponente(BaseModel):
    name: str = Field(default="", max_length=120)
    display: str = Field(default="", max_length=120)
    qty: float = Field(default=1.0, ge=0.0, le=5000.0)
    unit: str = Field(default="unidad", max_length=40)
    calories: float = Field(default=0.0, ge=0.0, le=5000.0)
    protein: float = Field(default=0.0, ge=0.0, le=500.0)
    carbs: float = Field(default=0.0, ge=0.0, le=800.0)
    healthy_fats: float = Field(default=0.0, ge=0.0, le=400.0)


class RespuestaAjustePlato(BaseModel):
    nombre: str = Field(default="", max_length=200)
    componentes: list[ItemComponente] = Field(default_factory=list)


class RespuestaAgregarIngrediente(BaseModel):
    item: ItemComponente


class PeticionAjustarPlato(BaseModel):
    nombre: str = Field(default="", max_length=200)
    componentes: list[dict[str, Any]] = Field(default_factory=list)
    instruccion: str = Field(..., min_length=2, max_length=300)
    locale: Optional[str] = Field(default=None, max_length=16)


class PeticionAgregarIngrediente(BaseModel):
    plato: str = Field(default="", max_length=200)
    texto: str = Field(..., min_length=2, max_length=120)
    locale: Optional[str] = Field(default=None, max_length=16)


def _limpio(s: Any, n: int) -> str:
    return " ".join(str(s or "").replace('"', "").split())[:n]


def _num(v: Any, tope: float = 5000.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x != x or abs(x) == float("inf"):
        return 0.0
    return max(0.0, min(tope, x))


def normalizar_componente(c: ItemComponente | dict[str, Any]) -> dict[str, Any]:
    if hasattr(c, "model_dump"):
        data = c.model_dump()
    elif isinstance(c, dict):
        data = c
    else:
        data = {}

    name = _limpio(data.get("name") or "", 120)
    display = _limpio(data.get("display") or name, 120)
    qty = round(_num(data.get("qty") or 1.0, 5000.0), 2)
    if qty <= 0:
        qty = 1.0
    unit = _limpio(data.get("unit") or "unidad", 40) or "unidad"

    macros_raw = data.get("macros") if isinstance(data.get("macros"), dict) else data
    macros = {
        "calories": int(round(max(0.0, min(_TOPES["calories"], _num(macros_raw.get("calories")))))),
        "protein": round(max(0.0, min(_TOPES["protein"], _num(macros_raw.get("protein")))), 1),
        "carbs": round(max(0.0, min(_TOPES["carbs"], _num(macros_raw.get("carbs")))), 1),
        "healthy_fats": round(max(0.0, min(_TOPES["healthy_fats"], _num(macros_raw.get("healthy_fats")))), 1),
    }

    return {
        "name": name,
        "display": display,
        "qty": qty,
        "unit": unit,
        "macros": macros,
    }


def prompt_sistema_ajustar() -> str:
    return (
        "Eres un nutricionista dominicano y asesor experto en análisis de alimentos. Recibes un plato que el usuario "
        "está revisando, con su nombre actual y la lista de componentes detectados con sus cantidades, unidades y macros. "
        "El usuario te da una instrucción en lenguaje natural para ajustar, añadir, quitar o modificar componentes de este plato.\n\n"
        "Reglas estrictas:\n"
        "1. Quitar alimentos: Si el usuario pide quitar un alimento (ej: 'sin ensalada', 'no comí plátano'), ELIMÍNALO por completo de la lista.\n"
        "2. Reemplazar o cambiar: Si pide sustituir (ej: 'cambia arroz blanco por moro', 'en vez de carne comí pollo'), reemplázalo con el nuevo alimento y sus macros estimadas correctamente para esa porción.\n"
        "3. Añadir alimentos: Si pide agregar (ej: 'agrégale 2 huevos fritos', 'también me tomé un jugo de naranja'), agrégalos a la lista con cantidades, unidades y macros realistas para un adulto.\n"
        "4. Cambios de cantidad: Si pide 'la mitad de arroz' o 'doble ración de pollo', ajusta 'qty' y escala proporcionalmente sus macros (calorías, proteína, carbohidratos, grasas).\n"
        "5. Componentes no mencionados: DEBEN PERMANECER en la lista con sus valores y macros originales.\n"
        "6. Actualiza 'nombre' con un nombre corto y descriptivo (≤60 caracteres) que refleje la comida final.\n"
        "Devuelve SOLO un JSON con formato: {\"nombre\": \"...\", \"componentes\": [{\"name\": \"...\", \"display\": \"...\", \"qty\": número, \"unit\": \"...\", \"calories\": número, \"protein\": número, \"carbs\": número, \"healthy_fats\": número}]}. Sin texto fuera del JSON."
    )


def prompt_sistema_agregar() -> str:
    return (
        "Eres un nutricionista dominicano experto. El usuario describe en texto libre UN alimento o ingrediente que consumió y "
        "desea agregar a su comida (ej: '1 taza de café con leche', '150g de pechuga a la plancha', '2 huevos hervidos', '1 manzana mediana', '1 vaso de jugo de chinola').\n"
        "Estima el alimento con su nombre normalizado, nombre para mostrar, cantidad numérica ('qty'), unidad de medida ('unit' como 'g', 'unidad', 'taza', 'vaso', etc.) y sus macros estimadas (calories, protein, carbs, healthy_fats) para esa porción.\n"
        "Devuelve SOLO un JSON con formato: {\"item\": {\"name\": \"...\", \"display\": \"...\", \"qty\": número, \"unit\": \"...\", \"calories\": número, \"protein\": número, \"carbs\": número, \"healthy_fats\": número}}. Sin texto fuera del JSON."
    )


async def ajustar_con_ia(pet: PeticionAjustarPlato, user_id: str) -> dict[str, Any]:
    """Ajusta un plato usando el modelo flash estructurado."""
    from graph_orchestrator import ChatGLM, _plan_flash_model_name, _current_node_var, user_id_var
    from langchain_core.messages import SystemMessage, HumanMessage
    from pais_del_estimador import contexto_del_usuario

    # Construir descripción de los componentes actuales
    lineas_comps = []
    for c in pet.componentes:
        n = _limpio(c.get("name") or c.get("display") or "", 80)
        q = _num(c.get("qty") or 1.0, 5000.0)
        u = _limpio(c.get("unit") or "unidad", 30)
        m = c.get("macros") if isinstance(c.get("macros"), dict) else c
        cal = int(round(_num(m.get("calories"))))
        p = round(_num(m.get("protein")), 1)
        cb = round(_num(m.get("carbs")), 1)
        f = round(_num(m.get("healthy_fats")), 1)
        lineas_comps.append(f"- {n}: {q:g} {u} (kcal={cal}, p={p}g, c={cb}g, g={f}g)")

    texto_comps = "\n".join(lineas_comps) if lineas_comps else "(sin componentes previos)"

    humano = (
        f"Plato actual: {_limpio(pet.nombre, 200) or 'Comida'}\n"
        f"Componentes actuales:\n{texto_comps}\n\n"
        f'Instrucción del usuario: "{_limpio(pet.instruccion, 300)}"'
    )

    llm = ChatGLM(model=_plan_flash_model_name(), temperature=0.1, max_retries=1, timeout=30).with_structured_output(
        RespuestaAjustePlato, method="json_mode"
    )
    tok_node = _current_node_var.set("scan_plate_adjust")
    tok_user = user_id_var.set(user_id)
    try:
        ctx_pais = await contexto_del_usuario(user_id)
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=prompt_sistema_ajustar() + ctx_pais),
                HumanMessage(content=humano),
            ]),
            timeout=35,
        )
    finally:
        try:
            _current_node_var.reset(tok_node)
            user_id_var.reset(tok_user)
        except Exception:
            pass

    if not isinstance(resp, RespuestaAjustePlato):
        raise ValueError("respuesta de ajuste sin forma esperada")

    items_norm = [normalizar_componente(c) for c in resp.componentes if _limpio(c.name, 120)]
    return {
        "nombre": _limpio(resp.nombre or pet.nombre, 200),
        "componentes": items_norm,
    }


async def agregar_ingrediente_con_ia(pet: PeticionAgregarIngrediente, user_id: str) -> dict[str, Any]:
    """Estima un nuevo ingrediente por texto libre usando el modelo flash."""
    from graph_orchestrator import ChatGLM, _plan_flash_model_name, _current_node_var, user_id_var
    from langchain_core.messages import SystemMessage, HumanMessage
    from pais_del_estimador import contexto_del_usuario

    humano = (
        f"Plato donde se añade: {_limpio(pet.plato, 200) or 'Plato'}\n"
        f'Alimento o ingrediente a añadir: "{_limpio(pet.texto, 120)}"'
    )

    llm = ChatGLM(model=_plan_flash_model_name(), temperature=0.1, max_retries=1, timeout=25).with_structured_output(
        RespuestaAgregarIngrediente, method="json_mode"
    )
    tok_node = _current_node_var.set("scan_ingredient_add")
    tok_user = user_id_var.set(user_id)
    try:
        ctx_pais = await contexto_del_usuario(user_id)
        resp = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=prompt_sistema_agregar() + ctx_pais),
                HumanMessage(content=humano),
            ]),
            timeout=30,
        )
    finally:
        try:
            _current_node_var.reset(tok_node)
            user_id_var.reset(tok_user)
        except Exception:
            pass

    if not isinstance(resp, RespuestaAgregarIngrediente) or not resp.item:
        raise ValueError("respuesta de ingrediente sin forma esperada")

    item_norm = normalizar_componente(resp.item)
    return {
        "item": item_norm,
    }
