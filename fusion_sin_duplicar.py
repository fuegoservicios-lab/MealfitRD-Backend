# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1110 · 2026-11-19] Lo que ya está en la fila no se vuelve a sumar en la fusión.

Turno real del dueño (19-nov 00:33): dos fotos (batida de guineo con leche evaporada + plátano maduro con chuleta,
huevo y queso). El coach registró el plato (720 kcal), la batida se fusionó en él (960 kcal), lo «corrigió» varias
veces y volvió a registrar la batida: la fusión por franja (`agent._intentar_fusionar_alimento_en_comida_existente`)
la sumó OTRA VEZ (1200 kcal, 101 g de proteína) y el turno se cortó por `max_rondas` con «el registro quedó
inconsistente». La fusión no miraba si lo que llegaba ya estaba dentro.

Ahora: si la fila destino se tocó EN ESTE TURNO (su `[ID_REGISTRO_DIARIO: …]` aparece en una herramienta después del
último mensaje del usuario) y TODO lo que llega (nombre e ingredientes) ya está en ella, es el mismo alimento
registrado dos veces: no se suma. Fuera de este turno no aplica — «y también un huevo» tras un desayuno con huevos,
dicho horas después, sigue sumando como siempre.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_RE_ID = re.compile(r"\[ID_REGISTRO_DIARIO:\s*([a-zA-Z0-9_\-]+)", re.I)
_VACIAS = {"piezas", "pieza", "unidades", "unidad", "porciones", "porcion", "gramos", "medianos", "mediano",
           "maduros", "maduro", "mediana", "medianas", "chico", "chica", "grande", "grandes", "pequeno", "pequena", "cantidad", "especificada", "aprox",
           "frito", "frita", "fritos", "fritas", "hervido", "hervida", "taza", "tazas", "vaso", "vasos", "cucharada",
           "cucharadas", "lasca", "lascas", "sobre", "desde", "entre", "estilo", "tipo", "casero", "casera"}


def _plano(t: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(t or "")) if not unicodedata.combining(c)).lower()


def _palabras(*textos: Any) -> set:
    out = set()
    for t in textos:
        items = t if isinstance(t, (list, tuple)) else [t]
        for it in items:
            for w in re.findall(r"[a-z]{4,}", _plano(it)):
                w = w[:-1] if w.endswith("s") and len(w) > 4 else w      # guineos ~ guineo
                if w not in _VACIAS:
                    out.add(w)
    return out


def ids_tocados_en_el_turno(messages: list) -> set:
    """Las filas del diario que una herramienta creó o corrigió después del último mensaje del usuario."""
    ids = set()
    for m in reversed(messages or []):
        if getattr(m, "type", None) in ("human", "user") or m.__class__.__name__ == "HumanMessage":
            break
        c = getattr(m, "content", "")
        if isinstance(c, str):
            ids.update(_RE_ID.findall(c))
    return ids


def ya_esta_en_la_fila(orig: dict, tool_args: dict, messages: list) -> bool:
    """True si `tool_args` vuelve a traer algo que ya está en la fila `orig` tocada en este mismo turno."""
    try:
        if str((orig or {}).get("id")) not in ids_tocados_en_el_turno(messages):
            return False
        nuevas = _palabras(tool_args.get("meal_name"), tool_args.get("ingredients") or [])
        if not nuevas:
            return False
        ing = orig.get("ingredients") or []
        ing = ing if isinstance(ing, list) else [ing]
        return nuevas <= _palabras(orig.get("meal_name"), ing)
    except Exception:
        return False


def sin_cambios(user_id: str, orig: dict, target_slot: str) -> dict:
    """Los args de `correct_consumed_meal` que dejan la fila COMO ESTÁ («Registro confirmado sin cambios», BB-044:
    sin `ingredients` ni `nutrition`, que harían reescribir la fila)."""
    return {
        "user_id": user_id, "meal_id": str(orig.get("id")), "meal_name": str(orig.get("meal_name") or ""),
        "calories": int(orig.get("calories") or 0), "protein": int(orig.get("protein") or 0),
        "carbs": int(orig.get("carbs") or 0), "healthy_fats": int(orig.get("healthy_fats") or 0),
        "meal_type": str(orig.get("meal_type") or target_slot),
    }
