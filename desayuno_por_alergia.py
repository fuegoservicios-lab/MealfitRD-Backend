# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-227 · 2026-09-25 · fuera del grafo en P1-PLAN-LOTE-240] La CATEGORÍA de desayuno se filtra por alergia.

El scrub de pools del esqueleto no la toca y el prompt del día la impone («El desayuno de este día DEBE ser de esta
categoría»). Batería real: un alérgico a gluten y huevo recibió «Avena/Cereales» → «30 g de avena» → ALÉRGENO DETECTADO
→ un intento entero quemado. Gluten ⇒ ni Avena/Cereales ni Pan/Tostadas; huevo ⇒ ni Revoltillo/Tortilla. Se reasigna a
una categoría permitida que ningún otro día use. tooltip-anchor: P1-PLAN-LOTE-227-DESAYUNO-POR-ALERGIA
"""
from __future__ import annotations

import logging

logger = logging.getLogger("graph_orchestrator")

_CATEGORIAS = ("Mangú/Tubérculos", "Batido/Bowl", "Revoltillo/Tortilla", "Avena/Cereales", "Pan/Tostadas")


def reasignar(skel_days: list, form_data) -> int:
    """Reasigna in-place la `breakfast_category` que la alergia declarada prohíbe. Devuelve cuántos días cambió."""
    cambiados = 0
    try:
        import graph_orchestrator as _go
        _bk_forb = _go._expand_allergy_declarations((form_data or {}).get("allergies") or [])
        _bk_block = set()
        if {"avena", "trigo", "pan"} & set(_bk_forb):
            _bk_block |= {"Avena/Cereales", "Pan/Tostadas"}
        if {"huevo", "huevos"} & set(_bk_forb):
            _bk_block.add("Revoltillo/Tortilla")
        if not _bk_block:
            return 0
        _bk_ok = [c for c in _CATEGORIAS if c not in _bk_block]
        _bk_usadas = [d.get("breakfast_category") for d in skel_days if isinstance(d, dict)]
        for _bkd in skel_days:
            if not isinstance(_bkd, dict) or _bkd.get("breakfast_category") not in _bk_block:
                continue
            _bk_libres = [c for c in _bk_ok if c not in _bk_usadas] or _bk_ok
            _bk_vieja = _bkd.get("breakfast_category")
            _bkd["breakfast_category"] = _bk_libres[0]
            _bk_usadas.append(_bk_libres[0])
            cambiados += 1
            logger.warning(f"🛡 [P1-PLAN-LOTE-227] Día {_bkd.get('day')}: desayuno «{_bk_vieja}» → "
                           f"«{_bk_libres[0]}» (la alergia declarada lo prohíbe)")
    except Exception as _bk_e:
        logger.debug(f"[P1-PLAN-LOTE-227] categoría de desayuno no-op: {type(_bk_e).__name__}: {_bk_e}")
    return cambiados
