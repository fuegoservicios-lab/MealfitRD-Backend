# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Las especies de pescado que el mapa de proteínas principales no veía.

`graph_orchestrator._MAIN_PROTEIN_ALIASES['pescado']` alimenta el contador de monotonía cross-día
(`cross_day_proteins`), el gate same-day del revisor, `reeleccion_dia` y el armador determinista. Su lista curada
es dominicana (tilapia, mero, chillo…) y no conocía trucha (CO), boquerones (ES), sardina, mojarra, bagre,
huachinango, lenguado ni caballa: en G24 CO la trucha de D1 y D3 no existía para ningún contador, y las sardinas
del almuerzo más la trucha de la cena del mismo día pasaban el gate. Y contaba «dorado», que en el plato es un
adjetivo: G24 CO D1 sumaba pescado por «Plátano Maduro Dorado» y un plan vivo de RD por «queso blanco fresco
dorado» — la lección de P1-DORADO-NO-ES-PEZ (`constants.PROTEIN_SYNONYMS`), que este mapa nunca heredó.

No es otra tabla de peces: las especies salen del vocabulario de pescado del escáner (`_ALLERGEN_SYNONYMS
['pescado']`, que ya incluye `vocabulario_mar.PESCADOS_EXTRA` y las altas de catálogo beta). De ahí se quita lo
que el escáner lista por SEGURIDAD y no es un pez que se sirva de principal (condimentos, fondos, adornos, una
imitación), los homónimos, y todo alias que choque con la curación de otra etiqueta (el mismo guard de
P2-PROTEIN-ALIASES-SSOT: así el atún conserva su etiqueta). Las frases inequívocas («filete de dorado/a») se
quedan. Knob `MEALFIT_BETA_FISH_SPECIES_COUNT` (False ⇒ el mapa exacto de antes). Puro; nunca lanza.
tooltip-anchor: P1-PLAN-LOTE-857-PESCADO-ESPECIES
"""
from __future__ import annotations

import logging
import unicodedata

from knobs import _env_bool

logger = logging.getLogger(__name__)

BETA_FISH_SPECIES_COUNT = _env_bool("MEALFIT_BETA_FISH_SPECIES_COUNT", True)

# Del escáner, pero no son un pez de plato: la alergia necesita verlos, la variedad no.
NO_ES_PEZ_DE_PLATO = frozenset((
    "salsa de pescado", "salsa inglesa", "worcestershire",   # condimentos (llevan anchoa)
    "cesar",                                                 # el aderezo César
    "fumet",                                                 # el fondo de pescado
    "caviar", "hueva",                                       # huevas
    "anchoa", "anchoas",                                     # el adorno de la coca o de la ensalada
    "surimi",                                                # la imitación de cangrejo
))
# Homónimos con la frontera de palabra inicial del resolvedor (`culinary_context._name_has_token`).
HOMONIMOS = frozenset((
    "dorado",   # «plátano maduro dorado», «queso dorado» — P1-DORADO-NO-ES-PEZ
    "carpa",    # «carpaccio»
))
# La frase inequívoca que sí es el pez (la del dorado ya viene de `constants.PROTEIN_SYNONYMS`).
FRASES_INEQUIVOCAS = ("filete de dorada",)


def _norm(s) -> str:
    return unicodedata.normalize("NFD", str(s or "").lower().strip()).encode("ascii", "ignore").decode("ascii")


def extender_pescado(alias_por_etiqueta: dict, vocabulario_pescado) -> tuple:
    """Muta `alias_por_etiqueta['pescado']` en su sitio: + especies del vocabulario del escáner, − homónimos.
    Devuelve `(añadidos, quitados)`. Con el knob apagado no toca nada y devuelve `([], [])`."""
    if not BETA_FISH_SPECIES_COUNT:
        return [], []
    try:
        pez = alias_por_etiqueta.get("pescado")
        if not isinstance(pez, list):
            return [], []
        quitados = [a for a in pez if _norm(a) in HOMONIMOS]
        for a in quitados:
            pez.remove(a)
        otras = {_norm(a) for lbl, als in alias_por_etiqueta.items() if lbl != "pescado" for a in (als or ())}
        tengo = {_norm(a) for a in pez}
        anadidos = []
        for cand in (*(vocabulario_pescado or ()), *FRASES_INEQUIVOCAS):
            c = _norm(cand)
            if (len(c) < 3 or c in tengo or c in NO_ES_PEZ_DE_PLATO or c in HOMONIMOS
                    or any(c.startswith(t) for t in tengo if " " not in t)          # «boquerones» ya es «boqueron»
                    or any(c in o or o in c for o in otras)):                       # guard de P2-PROTEIN-ALIASES-SSOT
                continue
            pez.append(c)
            tengo.add(c)
            anadidos.append(c)
        logger.info(f"🐟 [P1-PLAN-LOTE-857] pescado: +{len(anadidos)} especies del escáner, "
                    f"−{len(quitados)} homónimos ({', '.join(quitados) or '—'})")
        return anadidos, quitados
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-857] no se pudo extender el pescado ({type(e).__name__}: {e}) — el mapa "
                       f"sigue como estaba")
        return [], []
