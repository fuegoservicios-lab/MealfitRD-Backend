# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-803 · 2026-09-28] En el embarazo, el cerrador de proteína no elige queso blando.

Batería real (embarazo, 28-sep, con 780-801 desplegados): las dos meriendas «Lechosa fresca con maní tostado» recibieron
del cerrador «70 g / 40 g de queso cottage pasteurizado» («Acompaña con queso cottage…», «disfruta frío») y la nota del
lote 193 dice justo lo contrario: «el queso fresco o blando (blanco, de hoja, cottage, ricotta, mozzarella), aunque sea
pasteurizado, caliéntalo hasta que humee (74 °C por dentro)». Lo que el cerrador añade va aparte y frío, así que su
queso blando SIEMPRE contradice la nota. Corpus: de 70 comidas de embarazo con esa nota, 46 sirven el queso frío y 18 de
ellas por el cerrador. Con el formulario de la corrida (el ContextVar que ya fija `nevera_exigida`), en embarazo —no en
lactancia: la nota es sólo del embarazo— los candidatos que casan con `embarazo_seguro._QUESO_BLANDO` (la MISMA regex de
la nota) salen; si no queda ninguno, la lista entera. Knob `MEALFIT_PREGNANCY_CLOSER_NO_SOFT_CHEESE` (True).
tooltip-anchor: P1-PLAN-LOTE-803
"""
from __future__ import annotations


def on() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PREGNANCY_CLOSER_NO_SOFT_CHEESE", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _nombre(c) -> str:
    try:
        return str(getattr(c[2], "name", None) or c[1])
    except Exception:                                                          # noqa: BLE001
        return ""


def filtrar(cands: list) -> list:
    """Candidatos de `_safe_high_density_proteins` — `[(densidad, nombre, info)]` — sin quesos blandos en el embarazo.
    Corre ANTES del filtro de la Nevera exigida y lo consulta: si la Nevera sólo tiene queso blando, gana la Nevera (el
    revisor rechazaría otra proteína); si no, lo que se devuelve ya está dentro de ella."""
    if not cands or not on():
        return cands
    try:
        import embarazo_seguro as es
        import nevera_exigida as ne
        fd = ne._FD.get()
        if not isinstance(fd, dict) or not es._es_embarazo(fd):
            return cands
        fuera = [c for c in ne.filtrar_proteinas(cands) if not es._QUESO_BLANDO.search(_nombre(c))]
    except Exception:                                                          # noqa: BLE001
        return cands
    return fuera or cands
