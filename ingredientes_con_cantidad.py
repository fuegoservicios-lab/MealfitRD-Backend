# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-983 · 2026-11-11] Un registro desde FOTO lleva cantidad en cada ingrediente.

Caso vivo del dueño (11-nov): la cena del mangú quedó con «salami frito (incluyendo el salami consumido)» y «mangú…
(porción estimada)»: sin gramos ni unidades, nadie puede revisar ni corregir la cifra (y el «había otro salami» se
perdió). `log_consumed_meal` rechaza UNA vez y le dice al modelo qué líneas faltan; si reintenta igual
(120 s), pasa: el registro jamás se pierde por esto. Condimentos sueltos («sal», «aceite al gusto») están exentos.
Knob `MEALFIT_PHOTO_INGREDIENT_QTY_GUARD` (True). tooltip-anchor: P1-PLAN-LOTE-983
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Iterable

_CANTIDAD = re.compile(
    r"\d|\b(?:un|una|uno|unos|unas|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|media|medio|mitad|"
    r"cuarto|pizca|chorrito|punado|cucharada|cucharadita|taza|lasca|rebanada|unidad|unidades|porcion de \w+)\b")
_CONDIMENTO = re.compile(
    r"^(?:sal|pimienta|oregano|ajo|cebolla|cilantro|perejil|limon|vinagre|especias?|sazon\w*|adobo|agua|"
    r"aceite(?: de \w+)?|mantequilla|salsa \w+|canela|comino)\b(?:.*\bal gusto\b)?")
_VENTANA_S = 120.0
_YA_AVISADO: dict = {}


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def sin_cantidad(ingredientes: Iterable) -> list:
    out = []
    for ing in ingredientes or []:
        txt = ing.get("name") if isinstance(ing, dict) else ing
        n = _norm(txt)
        if not n or _CANTIDAD.search(n) or _CONDIMENTO.match(n):
            continue
        out.append(str(txt).strip()[:80])
    return out


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PHOTO_INGREDIENT_QTY_GUARD", True)
    except Exception:                                                          # noqa: BLE001
        return True


def rechazo(user_id, meal_name, ingredientes, *, es_foto: bool, ahora: float = None):
    """Texto del rechazo, o None (= registrar). Una sola vez por usuario en 120 s."""
    if not es_foto or not activo():
        return None
    faltan = sin_cantidad(ingredientes)
    if not faltan:
        return None
    t = time.time() if ahora is None else ahora
    for k, ts in list(_YA_AVISADO.items()):
        if t - ts > _VENTANA_S:
            _YA_AVISADO.pop(k, None)
    clave = str(user_id)   # por usuario: si el modelo cambia el nombre al reintentar, igual pasa
    if clave in _YA_AVISADO:
        return None
    _YA_AVISADO[clave] = t
    return ("⚠️ NO REGISTRADO TODAVÍA: en un registro desde foto CADA ingrediente lleva su cantidad (gramos o "
            f"unidades, p. ej. «3 lascas de salami frito (≈75 g)»). Sin cantidad: {'; '.join(faltan)}. Usa lo que el "
            "usuario dijo en el chat (si corrigió cuántos había, esa cifra manda), estima el resto y vuelve a llamar "
            "a `log_consumed_meal` con las calorías y macros recalculadas. No le preguntes nada al usuario por esto "
            "ni le digas que hubo un rechazo.")
