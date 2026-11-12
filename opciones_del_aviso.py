# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-985 · 2026-11-11] El aviso de comida de alguien con restricciones nombra opciones YA FILTRADAS.

Caso vivo (cuenta de prueba, 11-nov 5:30 p. m., hipertensión): «Para merendar ahora, elige algo que ya sepas que te
cae bien y que sea bajo en sal». El lote 161 prohibió nombrar comida con alergias/dieta/condición porque el aviso es el
único texto del coach sin filtro determinista detrás — correcto, pero dejó un aviso que no dice nada.

Ahora el filtro va DELANTE: una lista corta de opciones sencillas por franja (sin embutidos ni procesados salados),
cada una pasada por el backstop clínico (`nevera_del_coach._violaciones`: alergias, dieta, embarazo, medicación,
condiciones; fail-secure) y por lo que NO LE GUSTA. El aviso solo puede nombrar esas. Si ninguna pasa, sigue como el
161 (sin nombrar). Knob `MEALFIT_NUDGE_SAFE_OPTIONS` (True). tooltip-anchor: P1-PLAN-LOTE-985
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

MAX_OPCIONES = 4

# Platos sencillos y bajos en sal, de despensa dominicana. El backstop decide cuáles le sirven a ESTA persona.
OPCIONES_POR_FRANJA = {
    "desayuno": ["Avena cocida con canela y guineo", "Huevos revueltos con casabe", "Yogur natural con lechosa",
                 "Pan integral con huevo hervido y aguacate", "Batata hervida con huevo", "Batida de guineo con leche"],
    "almuerzo": ["Pescado a la plancha con batata y ensalada", "Pollo a la plancha con arroz y vegetales",
                 "Habichuelas guisadas con arroz y ensalada", "Lentejas guisadas con arroz",
                 "Huevos con vegetales salteados y yuca", "Res magra guisada con auyama"],
    "merienda": ["Un guineo", "Una manzana", "Yogur natural", "Un puñado de nueces sin sal",
                 "Zanahoria y pepino en bastones", "Un huevo hervido", "Lechosa picada"],
    "cena": ["Pescado a la plancha con vegetales al vapor", "Pechuga de pollo a la plancha con ensalada",
             "Huevos revueltos con vegetales", "Auyama asada con queso blanco bajo en sal",
             "Habichuelas guisadas con ensalada", "Pavo a la plancha con calabacín"],
}
_FRANJA = {"desayuno": "desayuno", "breakfast": "desayuno", "almuerzo": "almuerzo", "lunch": "almuerzo",
           "merienda": "merienda", "snack": "merienda", "cena": "cena", "dinner": "cena"}


def _norm(t) -> str:
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_NUDGE_SAFE_OPTIONS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _no_le_gusta(opcion: str, no_le_gusta: list) -> bool:
    o = _norm(opcion)
    for x in no_le_gusta or []:
        for palabra in re.findall(r"[a-z]{4,}", _norm(x)):
            raiz = palabra[:-1] if palabra.endswith("s") else palabra
            if raiz and raiz in o:
                return True
    return False


def opciones_aptas(health: dict, meal: Optional[str], no_le_gusta: Optional[list] = None, violaciones=None) -> list:
    """Las opciones de la franja que pasan el backstop y sus gustos. `violaciones(nombre, perfil) -> list` inyectable."""
    franja = _FRANJA.get(_norm(meal))
    if not franja:
        return []
    import nevera_del_coach as ndc
    viol = violaciones or ndc._violaciones
    perfil = ndc._perfil(health)
    # Con alergia al maní el backstop deja pasar las nueces (otra familia), pero la reactividad cruzada es frecuente y
    # un aviso no necesita arriesgar: fuera.
    _al = perfil.get("allergies") or []
    _al = [_al] if isinstance(_al, str) else _al
    _alergias = _norm(" ".join(str(x) for x in _al if x) + " " + str(perfil.get("otherAllergies") or ""))
    _sin_frutos_secos = bool(re.search(r"\bmani\b|cacahuate|nuez|nueces|frutos secos|almendra", _alergias))
    out = []
    for op in OPCIONES_POR_FRANJA[franja]:
        if _no_le_gusta(op, no_le_gusta or []):
            continue
        if _sin_frutos_secos and re.search(r"nuez|nueces|almendra", _norm(op)):
            continue
        try:
            if viol(op, perfil):
                continue
        except Exception:                                                      # noqa: BLE001 — fail-secure
            continue
        out.append(op)
        if len(out) >= MAX_OPCIONES:
            break
    return out


def bloque(health: dict, meal: Optional[str], no_le_gusta: Optional[list] = None, violaciones=None) -> str:
    """El bloque del prompt. "" si el knob está apagado, ninguna pasa o algo falla (el aviso sigue sin nombrar)."""
    if not activo():
        return ""
    try:
        ops = opciones_aptas(health, meal, no_le_gusta, violaciones)
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-985] opciones del aviso no calculadas: {e!r}")
        return ""
    if not ops:
        return ""
    return ("\n\n✅ OPCIONES YA COMPROBADAS CONTRA SUS RESTRICCIONES [P1-PLAN-LOTE-985] (el sistema las pasó por su "
            "filtro clínico; son las ÚNICAS comidas que puedes nombrar, además de lo de SU NEVERA si aparece arriba): "
            + "; ".join(ops) + ". Propón UNA, con una porción concreta; no recites la lista. Nada de «elige algo que "
            "te caiga bien»: di qué.\n")
