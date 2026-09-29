# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-922 · 2026-09-29] El cerrador de banda no deja una porción bajo el piso cocinable que ya cumplía.

Auditoría de ia-59 (medidor de punto fijo del escudo, 913): `_rebalance_day_macros_to_target` escala TODAS las líneas
del grupo macro-dominante con un factor común (0,3-2,5) y corre DESPUÉS de `_floor_subservible_portions`. Batería rdv805,
desayuno «Bowl fresco de yogur…»: entra «15 g de avena», sale «10 g de avena»; en la pasada siguiente del escudo el piso
la DROPEA (sin headroom) y la ficha sigue diciendo «con avena y chía». Sólo el alimento que da nombre al plato estaba
protegido (lote 177). Aquí, al BAJAR, una línea que estaba en el piso o sobre él y quedaría debajo no se toca (el resto
del grupo absorbe el ajuste; la banda tolera el residuo). Las exentas del piso (aceites, especias, hierbas…) siguen
libres. Knob `MEALFIT_REBALANCE_KEEPS_FLOOR` (True). tooltip-anchor: P1-PLAN-LOTE-922
"""
from __future__ import annotations

import unicodedata


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_REBALANCE_KEEPS_FLOOR", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def bajo_el_piso(orig: str, nueva: str, db) -> bool:
    """¿Bajar `orig` a `nueva` la deja bajo el piso cocinable cuando antes lo cumplía? Nunca lanza (ante error, False)."""
    try:
        if not on():
            return False
        import graph_orchestrator as go
        piso = float(go.PORTION_SHRINK_FLOOR_G)
        g0 = db.grams_from_ingredient_string(str(orig))
        g1 = db.grams_from_ingredient_string(str(nueva))
        if not g0 or not g1 or float(g0) < piso or float(g1) >= piso:
            return False
        s = _sa(orig)
        return not any(t in s for t in go._SHRINK_FLOOR_EXEMPT_TOKENS)
    except Exception:                                                          # noqa: BLE001
        return False
