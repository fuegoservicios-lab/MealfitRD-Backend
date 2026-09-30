# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-937 · 2026-09-30] Los datos de un usuario de la Unión Europea no van al proveedor que los trata en China.

Auditoría legal para la App Store (29-sep): la transferencia a China se ampara en el consentimiento explícito
(art. 49.1.a RGPD), que el Comité Europeo interpreta de forma restrictiva; lo robusto era cláusulas tipo con ese proveedor
o enrutar la UE a otro proveedor. El dueño (30-sep) delegó la decisión: se enruta. (El nombre del proveedor vive
solo en `llm_provider`, lote 74: aquí es `PROVEEDOR_EN_CHINA`.) Hoy el único país de la UE que la
app sirve es España (`constants.COUNTRY_PROFILES`).

Cómo: `llm_provider.llm_provider_name()` —que ya leen la base, la clave y el modelo del wrapper— pregunta aquí
`proveedor_forzado()`. La región sale de dos sitios, en este orden: el país fijado por el chat con `fijar_pais` tras
fundir el formulario con el perfil (`routers/chat.py`), y el formulario de la corrida que `nevera_exigida` fija en su
ContextVar (generación inicial, bloques y regeneraciones: el grafo propaga el contexto). Sin país conocido no se
fuerza nada: el proveedor de siempre. Un knob que mandara la UE de vuelta a China no vale (cae a OpenAI).
Knobs `MEALFIT_EU_LLM_ROUTING` (True), `MEALFIT_EU_LLM_PROVIDER` (openai), `MEALFIT_EU_COUNTRIES` (ES, lista por
comas). tooltip-anchor: P1-PLAN-LOTE-937
"""
from __future__ import annotations

import contextvars
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PAIS: contextvars.ContextVar = contextvars.ContextVar("mealfit_region_ia_pais", default=None)
_UE_POR_DEFECTO = "ES"
_PROVEEDOR_UE_POR_DEFECTO = "openai"
_ALIAS = {"espana": "ES", "españa": "ES", "spain": "ES", "es-es": "ES"}


def activo() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_EU_LLM_ROUTING", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _paises_ue() -> set:
    try:
        from knobs import _env_str
        crudo = _env_str("MEALFIT_EU_COUNTRIES", _UE_POR_DEFECTO) or _UE_POR_DEFECTO
    except Exception:                                                          # noqa: BLE001
        crudo = _UE_POR_DEFECTO
    return {p.strip().upper() for p in str(crudo).split(",") if p.strip()}


def proveedor_ue() -> str:
    """El proveedor al que va la UE; nunca el que trata los datos en China (un knob que la devolviera no vale)."""
    try:
        from knobs import _env_str
        p = (_env_str("MEALFIT_EU_LLM_PROVIDER", _PROVEEDOR_UE_POR_DEFECTO) or _PROVEEDOR_UE_POR_DEFECTO).strip().lower()
    except Exception:                                                          # noqa: BLE001
        p = _PROVEEDOR_UE_POR_DEFECTO
    try:
        from llm_provider import PROVEEDOR_EN_CHINA
    except Exception:                                                          # noqa: BLE001
        PROVEEDOR_EN_CHINA = ""
    return _PROVEEDOR_UE_POR_DEFECTO if p in ("", PROVEEDOR_EN_CHINA) else p


def _canon(pais) -> Optional[str]:
    if pais is None:
        return None
    s = str(pais).strip()
    if not s:
        return None
    bajo = s.lower()
    if bajo in _ALIAS:
        return _ALIAS[bajo]
    try:
        from constants import canonicalize_country
        c = canonicalize_country(s)
        if c:
            return str(c).upper()
    except Exception:                                                          # noqa: BLE001
        pass
    return s.upper() if len(s) == 2 else None


def es_ue(pais) -> bool:
    c = _canon(pais)
    return bool(c) and c in _paises_ue()


def fijar_pais(pais):
    """Fija el país del usuario de la petición actual (el chat, tras fundir el perfil). Devuelve el token."""
    return _PAIS.set(_canon(pais))


def restaurar(token) -> None:
    try:
        _PAIS.reset(token)
    except Exception:                                                          # noqa: BLE001
        pass


def pais_actual() -> Optional[str]:
    """El país conocido de la petición: el fijado por el chat o el del formulario de la corrida."""
    p = _PAIS.get()
    if p:
        return p
    try:
        import nevera_exigida as ne
        fd = ne._FD.get()
        if isinstance(fd, dict):
            return _canon(fd.get("country"))
    except Exception:                                                          # noqa: BLE001
        pass
    return None


def proveedor_forzado() -> Optional[str]:
    """El proveedor que manda para esta petición, o None (el de siempre). Nunca lanza."""
    try:
        if not activo():
            return None
        p = pais_actual()
        if p and p in _paises_ue():
            return proveedor_ue()
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-937] no-op: {type(e).__name__}: {e}")
    return None


__all__ = ["activo", "es_ue", "fijar_pais", "restaurar", "pais_actual", "proveedor_forzado", "proveedor_ue"]
