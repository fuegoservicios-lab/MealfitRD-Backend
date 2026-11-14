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


_CHAT_HYDROGEN_TIERS_DEFAULT = "todos"
_HYDROGEN_TIERS_ALL = frozenset({"*", "all", "todos", "todo"})
_HYDROGEN_TIERS_NONE = frozenset({"none", "off", "0", "false", "ninguno"})
_HYDROGEN_TIER_ALIASES = {"free": "gratis", "gratuito": "gratis", "gratuita": "gratis"}


def chat_hydrogen_tiers() -> str:
    """Knob `MEALFIT_CHAT_HYDROGEN_TIERS` (default `todos`): tiers que enrutan el coach a Gemini 4 Hydrogen."""
    from knobs import _env_str
    return _env_str("MEALFIT_CHAT_HYDROGEN_TIERS", _CHAT_HYDROGEN_TIERS_DEFAULT)


def _parse_hydrogen_tiers(raw: str) -> Optional[frozenset[str]]:
    tokens = [t.strip().lower() for t in str(raw or "").split(",") if t.strip()]
    if not tokens or any(t in _HYDROGEN_TIERS_ALL for t in tokens):
        return None
    if all(t in _HYDROGEN_TIERS_NONE for t in tokens):
        return frozenset()
    return frozenset(_HYDROGEN_TIER_ALIASES.get(t, t) for t in tokens if t not in _HYDROGEN_TIERS_NONE)


def usa_hydrogen_para_usuario(user_id: Optional[str] = None) -> bool:
    """[P1-PLAN-LOTE-999] True si el plan efectivo del usuario entra en `MEALFIT_CHAT_HYDROGEN_TIERS`.

    Default (`todos`): todos usan Hydrogen (comportamiento desde LOTE-995). Al restringirlo (p. ej.
    `basic,plus,ultra,admin`), `gratis` e invitados caen a `resolve_model_for_user` (`gpt-6-luna`).
    """
    allowed = _parse_hydrogen_tiers(chat_hydrogen_tiers())
    if allowed is None:
        return True
    if not allowed:
        return False
    try:
        from llm_provider import get_user_tier
        tier = str(get_user_tier(user_id) or "gratis").strip().lower()
        tier = _HYDROGEN_TIER_ALIASES.get(tier, tier)
    except Exception:  # noqa: BLE001 — fail-cheap: si falla el lookup cae a `gratis`
        tier = "gratis"
    return tier in allowed


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


_ANCHOR_TAG_OPEN = "<manual_operativo_bioboros>"
_ANCHOR_TAG_CLOSE = "</manual_operativo_bioboros>"
_ANCHOR_SYS_CORTO = (
    "Eres el coach nutricional y clínico de Bioboros. Sigue al pie de la letra "
    "el manual operativo fijado en el primer mensaje del usuario."
)
_ANCHOR_ACK = "Manual operativo de Bioboros cargado y activo."


def cache_anchor_activo() -> bool:
    """`MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR` (False desde el lote 998): ancla el manual estático como turno inicial.

    APAGADO: pasa el manual de rol sistema a rol usuario (el modelo obedece menos lo que llega como usuario y un texto
    del usuario puede contradecirlo) y el beneficio no está medido: la sonda del 13-nov (6 variantes, 17,8 k tokens)
    no devolvió tokens cacheados en NINGUNA, ni con el prompt idéntico. Encenderlo solo para un A/B contra la
    telemetría de `cached_tokens` de `llm_usage_events`."""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR", False)
    except Exception:  # noqa: BLE001
        return False


def _cache_anchor_min_chars() -> int:
    try:
        from knobs import _env_int
        return max(1000, int(_env_int("MEALFIT_GEMINI_CACHE_ANCHOR_MIN_CHARS", 7000)))
    except Exception:  # noqa: BLE001
        return 7000


def anclar_manual_estatico(payload: dict, *, minimo_chars: Optional[int] = None) -> bool:
    """Reescribe `[system(manual), user(...), ...]` a `[system(corto), user(manual), assistant(ack), user(...)]`.

    En Gemini (`generateContent` / OpenAI-compat), el prefijo `systemInstruction` largo no comparte
    cache implícito entre usuarios/sesiones distintas, mientras que el bloque estático en `contents[0]`
    sí alcanza >90% `cached_tokens` cross-user (verificado en pruebas vivas). Idempotente.
    """
    if not isinstance(payload, dict) or not cache_anchor_activo():
        return False
    msgs = payload.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    m0, m1 = msgs[0], msgs[1]
    if not isinstance(m0, dict) or m0.get("role") != "system":
        return False
    if not isinstance(m1, dict) or m1.get("role") != "user":
        return False
    sys_txt = m0.get("content")
    if not isinstance(sys_txt, str):
        return False
    umbral = _cache_anchor_min_chars() if minimo_chars is None else int(minimo_chars)
    if len(sys_txt) < umbral or _ANCHOR_TAG_OPEN in sys_txt:
        return False
    m1_content = m1.get("content")
    if isinstance(m1_content, str) and _ANCHOR_TAG_OPEN in m1_content:
        return False
    payload["messages"] = [
        {"role": "system", "content": _ANCHOR_SYS_CORTO},
        {"role": "user", "content": f"{_ANCHOR_TAG_OPEN}\n{sys_txt}\n{_ANCHOR_TAG_CLOSE}"},
        {"role": "assistant", "content": _ANCHOR_ACK},
        *msgs[1:],
    ]
    return True
