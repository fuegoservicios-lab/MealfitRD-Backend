# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-995 · 2026-11-13] Solo proveedores occidentales, y Gemini 4 Hydrogen como motor del coach y del escáner.

POR QUÉ
    El dueño (13-nov) quiere quitar los avisos de transferencia a China y quedarse con proveedores occidentales:
    OpenAI, Google Gemini y Cohere. En producción ya no salía tráfico a China (14 días de `llm_usage_events`:
    gpt-6-luna, gemini-3.8-flash, gemini-decision-1 y gpt-live-1). Lo que quedaba eran CAMINOS a DeepSeek y Z.ai que
    un knob o un ID fijado podían reabrir. Este módulo los cierra: `permitir_china()` es False por defecto, y entonces
    `llm_provider` resuelve siempre a OpenAI y traduce cualquier ID `deepseek-*`/`glm-*` a su par de OpenAI.

    Gemini 4 Hydrogen (12-nov-2026, `gemini-4-hydrogen`, $1,50 in / $0,075 cacheado / $7,50 out por 1M) acepta
    texto, imagen, audio, vídeo y PDF, y según Google es unas 2,5× más rápido que 3.8 Flash. El 1-ene-2027 la 3.8
    Flash sube a ese mismo precio, así que para las fotos el cambio no encarece nada desde enero. Para el coach cuesta
    unas 15× lo que gpt-6-luna: es la decisión del dueño, y el knob `MEALFIT_CHAT_AGENT_MODEL` la revierte sin
    redeploy.

LA TRAMPA QUE ARREGLA (medida contra el API el 13-nov)
    Con herramientas, Gemini devuelve en cada `tool_call` una firma de su razonamiento
    (`extra_content.google.thought_signature`) y EXIGE que vuelva en la siguiente petición. `langchain-openai` la
    tira, y la segunda vuelta del coach (después de ejecutar la herramienta) respondía 400 «Function call is missing
    a thought_signature». `capturar_*` la guarda en `additional_kwargs["gemini_firmas"]` del AIMessage (sobrevive al
    checkpointer), y `reinyectar_firmas` la vuelve a poner en el payload. Un tool_call que no trae firma (lo escribió
    el respaldo de OpenAI) lleva el valor que Google documenta para historia ajena,
    `skip_thought_signature_validator`.

tooltip-anchor: P1-PLAN-LOTE-995
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

GEMINI_HYDROGEN = "gemini-4-hydrogen"  # [P1-PLAN-LOTE-995]

#: Los proveedores que tratan los datos en China. Ninguno se usa salvo que el knob lo permita a sabiendas.
PROVEEDORES_EN_CHINA = frozenset({"deepseek", "zai"})

CLAVE_FIRMAS = "gemini_firmas"
_FIRMA_AJENA = "skip_thought_signature_validator"

#: Los niveles de razonamiento que acepta Hydrogen (low, medium, high y xhigh). El resto del vocabulario del
#: pipeline se traduce: «sin razonar» no existe en Gemini y pasa a `low`.
_ESFUERZOS = ("low", "medium", "high", "xhigh")


def permitir_china() -> bool:
    """`MEALFIT_PERMITIR_PROVEEDORES_CHINA` (False). Ponerlo a True exige volver a mostrar el aviso de China y subir la
    versión del permiso: el texto que se acepta ya no nombra a ningún proveedor en China."""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_PERMITIR_PROVEEDORES_CHINA", False)
    except Exception:  # noqa: BLE001
        return False


def proveedor_en_china(nombre) -> bool:
    return str(nombre or "").strip().lower() in PROVEEDORES_EN_CHINA


def es_id_chino(model) -> bool:
    m = str(model or "").strip().lower()
    return m.startswith(("deepseek", "glm"))


def esfuerzo_gemini(valor) -> str:
    v = str(valor or "").strip().lower()
    if v in _ESFUERZOS:
        return v
    if v in ("max", "very_high", "ultra"):
        return "xhigh"
    return "low"


def clave_gemini() -> str:
    """La key de Gemini: la variable propia de Gemini o la de Google y, si no, `VISION_API_KEY` cuando el escáner ya va
    a Google (así vive en el VPS: misma cuenta, el mismo patrón que `coach_voz._clave` y `decision_triage`)."""
    import os
    k = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()  # [P1-VISION-GEMINI-FLASH]
    if k:
        return k
    if "googleapis.com" in os.environ.get("MEALFIT_VISION_BASE_URL", ""):
        return (os.environ.get("VISION_API_KEY") or "").strip()
    return ""


def hay_key_gemini() -> bool:
    return bool(clave_gemini())


# ───────────────────────────────────────────────────────── firmas de razonamiento (tool calls)
def _firmas_de_tool_calls(tool_calls: Iterable[dict], usar_index: bool) -> dict:
    firmas = {}
    for pos, tc in enumerate(tool_calls or []):
        extra = (tc or {}).get("extra_content")
        if not isinstance(extra, dict) or not extra:
            continue
        # En streaming Gemini manda `index: None`: entonces cuenta la posición (un tool_call por trozo, en orden).
        _idx = tc.get("index") if usar_index else None
        clave = str(_idx if _idx is not None else pos)
        firmas.setdefault(clave, extra)
    return firmas


def capturar_de_respuesta(respuesta: Any, resultado: Any) -> None:
    """Tras `_create_chat_result`: copia las firmas de cada choice a su AIMessage."""
    try:
        crudo = respuesta if isinstance(respuesta, dict) else respuesta.model_dump()
        for i, choice in enumerate(crudo.get("choices") or []):
            firmas = _firmas_de_tool_calls(((choice or {}).get("message") or {}).get("tool_calls"), usar_index=False)
            if firmas and i < len(resultado.generations):
                resultado.generations[i].message.additional_kwargs[CLAVE_FIRMAS] = firmas
    except Exception:  # noqa: BLE001 — conservar la firma nunca puede romper la respuesta
        pass


def capturar_de_trozo(trozo: dict, generacion: Any) -> None:
    """Tras `_convert_chunk_to_generation_chunk` (streaming): la firma llega en el delta del tool_call."""
    try:
        if generacion is None:
            return
        choices = trozo.get("choices") or (trozo.get("chunk") or {}).get("choices") or []
        if not choices:
            return
        firmas = _firmas_de_tool_calls(((choices[0] or {}).get("delta") or {}).get("tool_calls"), usar_index=True)
        if firmas:
            generacion.message.additional_kwargs[CLAVE_FIRMAS] = firmas
    except Exception:  # noqa: BLE001
        pass


def uso_solo_al_final(trozo: dict, generacion: Any) -> None:
    """[P1-PLAN-LOTE-996] En streaming Gemini manda el uso ACUMULADO en cada trozo (medido: 1512 → 1548 en dos
    trozos) y LangChain suma el `usage_metadata` de todos: el coach registraba 2-6× los tokens reales. Solo el
    trozo que cierra (con `finish_reason`) conserva el uso."""
    try:
        if generacion is None or getattr(generacion.message, "usage_metadata", None) is None:
            return
        choices = trozo.get("choices") or (trozo.get("chunk") or {}).get("choices") or []
        if choices and not any((c or {}).get("finish_reason") for c in choices):
            generacion.message.usage_metadata = None
    except Exception:  # noqa: BLE001
        pass


def reinyectar_firmas(payload: dict, mensajes: Iterable[Any]) -> None:
    """Antes de enviar a Gemini: cada tool_call del historial lleva su firma (o la de historia ajena)."""
    por_id = {}
    for m in mensajes or []:
        firmas = (getattr(m, "additional_kwargs", None) or {}).get(CLAVE_FIRMAS) or {}
        for pos, tc in enumerate(getattr(m, "tool_calls", None) or []):
            extra = firmas.get(str(pos))
            if extra and tc.get("id"):
                por_id[tc["id"]] = extra
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict) and not tc.get("extra_content"):
                tc["extra_content"] = por_id.get(tc.get("id")) or {"google": {"thought_signature": _FIRMA_AJENA}}


def es_gemini(model: Optional[str]) -> bool:
    return str(model or "").strip().lower().startswith("gemini")
