# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-936 · 2026-09-30] La compra única no manda toda la fruta a manzana.

`compra_unica.SUSTITUTOS` cambia, desde el día en que un fresco ya no aguanta, toda fruta —y el aguacate— por
«manzana». Batería real del PLATO con el 932 vivo (owner_like, días 8-11 de 30): manzana en 9 de 16 comidas y «Sardinas
con casabe, queso blanco y manzana» donde iba aguacate. El dueño (30-sep) aprobó una rueda.

Aquí, la rueda de fruta duradera (manzana 45 días, naranja 30, pera 21 en nevera: `pantry_durability`) gira por día y
comida como la rueda de proteínas del lote 214: salta lo que no aguanta hasta ese día, lo que no es seguro (alergia,
rechazo, dieta: el mismo backstop `compra_unica.es_seguro`) y deja para el final lo que el día ya lleva. El aguacate de
un plato SALADO pasa a aceitunas (despensa; hacen la misma función, grasa) con el peso de la línea, o 30 g si la línea
no trae peso; en un plato dulce o licuado sigue a la rueda de fruta, como antes.
Knob `MEALFIT_SINGLE_TRIP_FRUIT_WHEEL` (True). tooltip-anchor: P1-PLAN-LOTE-936
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

RUEDA = ("manzana", "naranja", "pera")
ACEITUNAS = "aceitunas"
_ACEITUNAS_SIN_PESO_G = 30
_DULCE = re.compile(r"\b(?:batido|smoothie|licuado|yogur|yogurt|avena|granola|postre|helado|mousse|crema dulce|"
                    r"panqueque|pancake|waffle|overnight|chocolate|cacao|miel)\b", re.IGNORECASE)


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_SINGLE_TRIP_FRUIT_WHEEL", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(s or "")).lower()
    except Exception:                                                          # noqa: BLE001
        return str(s or "").lower()


def es_dulce(plato) -> bool:
    return bool(_DULCE.search(_sa(plato)))


def elegir(hit: str, dia_abs: int, semilla: Optional[int], req: Optional[dict], evitar=(), plato: str = "",
           alergias=None, dieta=None, contexto=None) -> Optional[str]:
    """El duradero para una línea de fruta (o de aguacate) que no aguanta hasta `dia_abs`; None = el de siempre."""
    try:
        if not activo():
            return None
        import compra_unica as cu
        seguro = lambda n: cu.es_seguro(n, alergias, dieta=dieta, contexto=contexto)   # noqa: E731
        if _sa(hit) == "aguacate" and not es_dulce(plato) and seguro(ACEITUNAS):
            return ACEITUNAS
        s = int(dia_abs if semilla is None else semilla)
        rueda = [RUEDA[(s + k) % len(RUEDA)] for k in range(len(RUEDA))]
        evitar = {_sa(e) for e in (evitar or ())}
        rueda = [f for f in rueda if f not in evitar] + [f for f in rueda if f in evitar]
        for fruta in rueda:
            if cu._aguanta(f"100 g de {fruta}", int(dia_abs), req or {}) and seguro(fruta):
                return fruta
        return None
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-936] rueda de fruta no-op: {type(e).__name__}: {e}")
        return None


def aceitunas_de(texto: str, gramos: Optional[float]) -> str:
    """La línea de aceitunas con el peso de la línea de aguacate; sin peso legible, 30 g."""
    import compra_unica as cu
    prefijo = cu.cantidad_de(texto, gramos)
    if re.match(r"^\s*\d", prefijo) and re.search(r"\bg\s+de\s*$", prefijo):
        return f"{prefijo}{ACEITUNAS}"
    return f"{_ACEITUNAS_SIN_PESO_G} g de {ACEITUNAS}"


__all__ = ["RUEDA", "ACEITUNAS", "activo", "es_dulce", "elegir", "aceitunas_de"]
