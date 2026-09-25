"""[P1-PLAN-LOTE-225 · 2026-09-24] Textos cortos que el motor escribe en español, traducidos PARA LEER.

El escáner de comida nombra el plato y sus ingredientes en español dominicano, y así tiene que seguir: el nombre de
cada ingrediente es el identificador con el que se descuenta la Nevera (`pantry_names_match`), y el backstop del
nombre del plato (`derive_meal_name_from_description`) sólo entiende español. Lo que no puede seguir es que un tester
con la app en inglés lea «Un tazón de avena cocida con leche» en «What is it?».

Este módulo traduce, en UNA llamada al modelo flash, una lista de textos ya decididos en español, y devuelve la lista
traducida en el mismo orden. Es la última capa: nada de lo que devuelve vuelve al motor. Si el modelo tarda, falla o
devuelve algo que no cuadra, devuelve None y quien llama pinta el español, como antes: la traducción nunca puede
costar el registro.

Los nombres de idioma salen de `prompts.chat_agent._COACH_LANGUAGE_NAMES` (espejo #8 de la doc de idiomas): una tabla
más aquí sería un espejo nuevo que nadie recordaría al añadir el sexto idioma.

tooltip-anchor: P1-PLAN-LOTE-225
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LOCALE_BASE = "es-DO"
_MAX_TEXTO = 160
_MAX_TEXTOS = 40


def _nombres_de_idioma() -> dict:
    try:
        from prompts.chat_agent import _COACH_LANGUAGE_NAMES
        return dict(_COACH_LANGUAGE_NAMES)
    except Exception:
        return {}


def locale_soportado(valor) -> Optional[str]:
    """El locale si es uno de los cinco de la app; si no, None (quien llama decide el respaldo)."""
    v = str(valor or "").strip()
    if v == LOCALE_BASE or v in _nombres_de_idioma():
        return v
    return None


class _Traducciones(BaseModel):
    t: list[str] = Field(default_factory=list)


# [P1-PLAN-LOTE-225] Segundo tipo: frases cortas que la app enseña al usuario y no son nombres (lo que el coach
# recuerda de él, el momento y el motivo de un suplemento). Mismo contrato de salida.
_PROMPT_TEXTOS = (
    "Traduce al {idioma} cada texto de la lista que te paso: son frases cortas escritas en español dominicano que "
    "una app de nutrición le enseña a su usuario (lo que su coach recuerda de él, la dosis, el momento y el motivo "
    "de un suplemento). Devuelve SOLO un JSON con la forma {{\"t\": [ ... ]}}: la MISMA cantidad de textos y en el "
    "MISMO orden. Conserva las cifras y las unidades tal cual (5 g, 2 cápsulas, 1000 UI). Deja igual los nombres "
    "propios de platos típicos (mangú, sancocho, tostones). Si un texto ya está en {idioma}, devuélvelo igual. No "
    "añadas comillas, notas ni explicaciones."
)

_PROMPT = (
    "Traduce al {idioma} cada texto de la lista que te paso: son nombres de platos y de alimentos escritos en "
    "español dominicano, para que una persona los LEA en la app. "
    "Devuelve SOLO un JSON con la forma {{\"t\": [ ... ]}}: la MISMA cantidad de textos y en el MISMO orden. "
    "Traduce el nombre completo con naturalidad (\"Pechuga de pollo a la plancha\" → el nombre usual en {idioma}). "
    "Deja igual los nombres propios de platos típicos que no tienen traducción (mangú, sancocho, tostones, mofongo). "
    "No añadas cantidades, comillas, notas ni explicaciones."
)


def _preparar(textos, locale, max_chars: int = _MAX_TEXTO):
    """(idioma, textos limpios) o None si no hay nada que traducir."""
    idioma = _nombres_de_idioma().get(str(locale or ""))
    limpios = [" ".join(str(x or "").split())[:max_chars] for x in (textos or [])][:_MAX_TEXTOS]
    if not idioma or not any(limpios):
        return None
    return idioma, limpios


def _modelo(timeout_s: float):
    from graph_orchestrator import ChatGLM, _plan_flash_model_name
    llm = ChatGLM(model=_plan_flash_model_name(), temperature=0.0, max_retries=0, timeout=timeout_s)
    return llm.with_structured_output(_Traducciones, method="json_mode")


def _mensajes(idioma: str, limpios: list, tipo: str = "nombres"):
    import json
    from langchain_core.messages import SystemMessage, HumanMessage
    plantilla = _PROMPT_TEXTOS if tipo == "textos" else _PROMPT
    return [
        SystemMessage(content=plantilla.format(idioma=idioma)),
        HumanMessage(content=json.dumps(limpios, ensure_ascii=False)),
    ]


def _contexto(node: str, user_id: Optional[str]):
    """Etiqueta el gasto (`llm_usage_events`) con el nodo y el usuario; devuelve con qué deshacerlo."""
    from graph_orchestrator import _current_node_var, user_id_var
    tok_node = _current_node_var.set(node)
    tok_user = user_id_var.set(user_id) if user_id else None

    def _reset():
        try:
            _current_node_var.reset(tok_node)
            if tok_user is not None:
                user_id_var.reset(tok_user)
        except Exception:
            pass
    return _reset


def _validar(res, limpios: list, node: str, max_chars: int = _MAX_TEXTO) -> Optional[list]:
    traducidos = list(getattr(res, "t", None) or [])
    if len(traducidos) != len(limpios):
        logger.warning(f"[P1-PLAN-LOTE-225] traducción descartada ({node}): {len(traducidos)} textos para {len(limpios)}")
        return None
    out = []
    for original, trad in zip(limpios, traducidos):
        t = " ".join(str(trad or "").split())
        # Vacío, o tres veces más largo que el original: eso ya no es una traducción del nombre.
        out.append(t[:max_chars] if t and len(t) <= max(3 * len(original), 40) else original)
    return out


async def traducir_para_mostrar(textos, locale, *, user_id: Optional[str] = None,
                                node: str = "display_i18n_corto", timeout_s: float = 8.0) -> Optional[list]:
    """`textos` traducidos a `locale`, en el mismo orden; None si no aplica o si algo falla.

    Un texto vacío vuelve vacío. Un texto que el modelo devuelva vacío o desmedido vuelve en español: se mide por
    elemento, así que un fallo parcial no tira el resto."""
    prep = _preparar(textos, locale)
    if prep is None:
        return None
    idioma, limpios = prep
    try:
        llm = _modelo(timeout_s)
        mensajes = _mensajes(idioma, limpios)
        reset = _contexto(node, user_id)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-225] traducción para mostrar no disponible: {e}")
        return None
    try:
        res = await asyncio.wait_for(llm.ainvoke(mensajes), timeout=timeout_s + 1)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-225] traducción para mostrar falló ({node}, {locale}): {type(e).__name__}: {e}")
        return None
    finally:
        reset()
    return _validar(res, limpios, node)


def traducir_para_mostrar_sync(textos, locale, *, user_id: Optional[str] = None,
                               node: str = "display_i18n_corto", timeout_s: float = 6.0,
                               tipo: str = "nombres", max_chars: int = _MAX_TEXTO) -> Optional[list]:
    """La misma traducción para los endpoints síncronos (`def`, que FastAPI corre en su pool de hilos: «Cambiar
    plato», «Arreglar este día», `/api/i18n/textos`). Mismo contrato: None si no aplica o si algo falla; quien llama
    pinta el español. `tipo="textos"`: frases (recuerdos, suplementos) en vez de nombres de platos."""
    prep = _preparar(textos, locale, max_chars)
    if prep is None:
        return None
    idioma, limpios = prep
    try:
        llm = _modelo(timeout_s)
        mensajes = _mensajes(idioma, limpios, tipo)
        reset = _contexto(node, user_id)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-225] traducción para mostrar no disponible: {e}")
        return None
    try:
        res = llm.invoke(mensajes)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-225] traducción para mostrar falló ({node}, {locale}): {type(e).__name__}: {e}")
        return None
    finally:
        reset()
    return _validar(res, limpios, node, max_chars)


def nombre_de_plato_para_mostrar(nombre, locale, *, user_id: Optional[str] = None,
                                 node: str = "display_i18n_corto") -> Optional[str]:
    """El nombre de UN plato recién creado en el idioma del usuario, o None (español, locale base o fallo).

    Para el aviso y la tarjeta del plato mientras llega la traducción completa (`_display`), que el backend encola
    al persistir y tarda lo que tarda un día entero."""
    if not nombre or locale_soportado(locale) in (None, LOCALE_BASE):
        return None
    out = traducir_para_mostrar_sync([nombre], locale, user_id=user_id, node=node)
    if not out or not out[0] or out[0] == " ".join(str(nombre).split()):
        return None
    return out[0]
