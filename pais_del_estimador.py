# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-628 · 2026-09-27] El país del usuario para los prompts que ESTIMAN comida.

Los cinco estimadores —la foto (`vision_agent`), el texto libre (`/consumed/estimate-macros`, `plato_descrito`), la
duda de la foto (`ajuste_de_duda`) y el ingrediente corregido (`ingrediente_corregido`)— empiezan «Eres un
nutricionista dominicano» y ninguno recibía el país: un español que escribe «plátano» se estimaba como plátano de
cocinar, con porciones dominicanas. Este módulo da el párrafo que se AÑADE al final de su prompt de sistema.

La frontera no cambia: los nombres que el modelo devuelve siguen siendo los del catálogo, en español dominicano,
porque son el identificador con el que se descuenta la Nevera (`pantry_names_match`) y con el que el backstop de
alergias resuelve. Lo que cambia es cómo se LEE lo que el usuario escribe y lo que se ve en la foto.

Los ejemplos de nombres de otro país salen de `food_names_i18n` (`variantes_regionales`, lote 623): el mismo léxico que
amplía las alergias, no una tabla más. En RD (o sin país, o con el sistema de países apagado) la cadena es vacía y los
prompts quedan byte a byte como estaban; si el perfil no se puede leer, también.

tooltip-anchor: P1-PLAN-LOTE-628
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Donde «plátano» a secas es la banana (en RD, PR y CO es el de cocinar). Solo esta palabra: es la ambigüedad que más
# macros mueve (un plátano verde son ~220 kcal; una banana, ~105).
_PLATANO_ES_BANANA = frozenset({"ES", "MX"})
_EJEMPLOS = ("Duraznos", "Batata", "Vainitas", "Auyama", "Chinola", "Habichuelas negras")


def _ejemplos() -> str:
    try:
        from food_names_i18n import _variantes
        reg = _variantes("variantes_regionales")
    except Exception:
        return ""
    pares = [f"«{reg[c][0].lower()}» = «{c}»" for c in _EJEMPLOS if reg.get(c)]
    return ", ".join(pares)


def contexto(country) -> str:
    """El párrafo de país para el prompt de un estimador; '' en RD o si el país no se reconoce."""
    try:
        from constants import canonicalize_country, COUNTRY_PROFILES
        canon = canonicalize_country(country)
        if canon == "DO" or str(country or "").strip().upper() != canon:
            return ""
        nombre = (COUNTRY_PROFILES.get(canon) or {}).get("name_es")
    except Exception:
        return ""
    if not nombre:
        return ""
    platano = (" Allí «plátano» a secas es la banana: en el catálogo se llama «Guineo»; el de cocinar es «plátano "
               "macho» («Plátano verde» o «Plátano maduro»).") if canon in _PLATANO_ES_BANANA else ""
    ejemplos = _ejemplos()
    return (
        f"\n\nPAÍS DEL USUARIO: vive en {nombre}. Entiende lo que escribe y lo que se ve en la foto con el sentido que "
        f"tiene ALLÍ, y usa las porciones y los platos típicos de allí.{platano} Los nombres de alimentos que "
        f"devuelves siguen siendo los del catálogo de la app, en español dominicano"
        + (f" ({ejemplos})" if ejemplos else "")
        + "."
    )


def pais_de(user_id) -> Optional[str]:
    """El país del perfil, por la misma puerta que el plan (`country_for_form_data`: con el sistema apagado, 'DO')."""
    if not user_id:
        return None
    from db import get_user_profile
    from constants import country_for_form_data
    perfil = get_user_profile(user_id) or {}
    return country_for_form_data(perfil.get("health_profile") or {})


async def pais_del_usuario(user_id) -> Optional[str]:
    """`pais_de` sin bloquear el bucle; None si falla (fail-open: el prompt queda como antes)."""
    try:
        return await asyncio.to_thread(pais_de, user_id)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-628] país del usuario no leído ({type(e).__name__}): el estimador sigue sin él")
        return None


async def contexto_del_usuario(user_id) -> str:
    """`contexto(país del perfil)`; '' si no hay usuario o algo falla."""
    if not user_id:
        return ""
    return contexto(await pais_del_usuario(user_id))
