# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-855 · 2026-09-29] Etiquetas de proteína pesada en un texto, por FRONTERA DE PALABRA.

El contador de monotonía cross-día (`graph_orchestrator._count_cross_day_heavy_protein_repetition`: el
`cross_day_proteins` del `variety_report`, la señal que impide el skip-when-clean de la autocrítica y la de
`reeleccion_dia`) y su espejo del armador determinista (`deterministic_day._pesadas_de`) casaban los alias
por SUBCADENA. G24 (29-sep, planes reales): Puerto Rico y Colombia salieron con `res: 3` sin un gramo de carne
de res — «res» dentro de «queso fRESco», «piña fRESca», «cilantro fRESco» — y el barrido del catálogo, las
bibliotecas y los planes encontró la otra clase viva: «pollo» dentro de «rePOLLO» (12 cadenas). Es la lección
de P1-SLOT-PROTEIN-WORDBOUNDARY y P1-DM-SPECIES-BOUNDARY, que el gate same-day ya aplica con
`culinary_context._name_has_token` y este instrumento nunca heredó.

Resolvedor: el MISMO del gate same-day (`_name_has_token`, frontera de palabra INICIAL): «pavochón» sigue
siendo pavo y «camarones» camarón, pero «repollo» deja de ser pollo. Knob
`MEALFIT_CROSS_DAY_PROTEIN_TOKEN_MATCH` (False ⇒ la subcadena de antes, byte a byte). Puro; nunca lanza.
tooltip-anchor: P1-PLAN-LOTE-855-PROTEINA-POR-TOKEN
"""
from __future__ import annotations

from culinary_context import _name_has_token
from knobs import _env_bool

CROSS_DAY_PROTEIN_TOKEN_MATCH = _env_bool("MEALFIT_CROSS_DAY_PROTEIN_TOKEN_MATCH", True)


def alias_en_texto(alias: str, texto_norm: str) -> bool:
    """¿Aparece `alias` en `texto_norm`? Ambos ya en minúscula y sin acentos. Con el knob, frontera de palabra
    inicial (SSOT del gate same-day); sin él, subcadena (la conducta previa)."""
    if not alias:
        return False
    if CROSS_DAY_PROTEIN_TOKEN_MATCH:
        return _name_has_token(alias, texto_norm)
    return alias in texto_norm


def etiquetas_en_texto(texto_norm: str, alias_por_etiqueta: dict) -> set:
    """Las etiquetas de `alias_por_etiqueta` ({etiqueta: [alias normalizados]}) presentes en `texto_norm`."""
    try:
        return {lbl for lbl, alias in (alias_por_etiqueta or {}).items()
                if any(alias_en_texto(a, texto_norm) for a in (alias or ()))}
    except Exception:                                                  # noqa: BLE001
        return set()
