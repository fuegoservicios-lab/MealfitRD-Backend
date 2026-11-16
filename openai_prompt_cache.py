# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1003 · 2026-11-14] (G-02) Estabilidad de prefijo y prompt cache OpenAI.

Resuelve las fugas medidas por el lote 999 y G-02 que dejaban la primera llamada
de cada turno del coach con `cached_tokens = 0` (~33 000 tokens en frio, ~3,2 s):

1. Prefijo estatico cross-user (`MEALFIT_CHAT_STATIC_PREFIX_NO_UID`, default True):
   Mantiene `build_tools_instructions*_estaticas` (sin `user_id`) incluso con
   `MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT=0` (regla clinica siempre en rol `system`)
   y anexa `El user_id ...` inmediatamente DESPUES de `_bd_prefijo_chars`. De este
   modo los ~62 k caracteres del manual + instrucciones son byte-identicos entre
   todos los usuarios y entre llamadas seguidas.
2. Estabilidad del catalogo de herramientas en re-llamadas intra-turno
   (`MEALFIT_CHAT_STABLE_TOOL_SUBSET_ON_FOLLOWUP`, default True):
   Evita explotar el subconjunto de 8 herramientas a 19 al volver de un `ToolMessage`
   (mantiene la seleccion del turno + cualquier herramienta invocada desde el
   ultimo `HumanMessage`).
3. Clave y retencion extendida OpenAI (`MEALFIT_OPENAI_PROMPT_CACHE_KEY_ENABLED`,
   `MEALFIT_OPENAI_PROMPT_CACHE_RETENTION="24h"`):
   Enruta llamadas OpenAI con prefijo largo (>= 2 500 caracteres) mediante un
   `prompt_cache_key` determinista compartido entre usuarios con el mismo prefijo
   estatico y herramientas, y solicita retencion `"24h"` para sobrevivir huecos de
   varios minutos u horas entre turnos. Ante un eventual 400 del proveedor por
   parametro no soportado en un modelo concreto, desactiva fail-open en memoria ese
   campo para ese modelo y reintenta una vez.

Tooltip-anchor: P1-PLAN-LOTE-1003-PROMPT-CACHE.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import Any, Iterable, Optional

from knobs import _env_bool, _env_str

logger = logging.getLogger(__name__)

_MIN_PREFIX_CHARS_FOR_CACHE = 2500

_RETENTION_CHOICES = frozenset({"", "off", "none", "0", "false", "in_memory", "24h"})
_RETENTION_OFF_VALUES = frozenset({"", "off", "none", "0", "false"})

_CACHE_PARAM_LOCK = threading.Lock()
_UNSUPPORTED_RETENTION_MODELS: set[str] = set()
_UNSUPPORTED_CACHE_KEY_MODELS: set[str] = set()

_UID_CUT_RE = re.compile(r"(?:\n\n|^)El user_id\b")


def static_prefix_no_uid_activo() -> bool:
    """Knob `MEALFIT_CHAT_STATIC_PREFIX_NO_UID` (default True)."""
    return _env_bool("MEALFIT_CHAT_STATIC_PREFIX_NO_UID", True)


def stable_tool_subset_activo() -> bool:
    """Knob `MEALFIT_CHAT_STABLE_TOOL_SUBSET_ON_FOLLOWUP` (default True)."""
    return _env_bool("MEALFIT_CHAT_STABLE_TOOL_SUBSET_ON_FOLLOWUP", True)


def openai_prompt_cache_key_activo() -> bool:
    """Knob `MEALFIT_OPENAI_PROMPT_CACHE_KEY_ENABLED` (default True)."""
    return _env_bool("MEALFIT_OPENAI_PROMPT_CACHE_KEY_ENABLED", True)


def openai_prompt_cache_retention_value() -> str:
    """Knob `MEALFIT_OPENAI_PROMPT_CACHE_RETENTION` (default '24h'; '' si apagado)."""
    val = _env_str("MEALFIT_OPENAI_PROMPT_CACHE_RETENTION", "24h", choices=_RETENTION_CHOICES)
    return "" if val in _RETENTION_OFF_VALUES else val


def tools_usadas_en_turno(messages: Optional[Iterable[Any]]) -> set[str]:
    """Nombres de herramientas invocadas desde el ultimo mensaje humano."""
    if not messages:
        return set()
    used: set[str] = set()
    for msg in reversed(list(messages)):
        role_or_type = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role_or_type in ("human", "user"):
            break
        if role_or_type == "tool":
            t_name = getattr(msg, "name", None)
            if t_name:
                used.add(str(t_name))
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc_name = tc.get("name") or (tc.get("function") or {}).get("name")
            else:
                tc_name = getattr(tc, "name", None)
            if tc_name:
                used.add(str(tc_name))
    return used


def _volatile_cut_patterns() -> tuple[re.Pattern[str], ...]:
    pats: list[re.Pattern[str]] = [_UID_CUT_RE]
    try:
        from prompt_desglose import _VOLATILE_BLOCK_SPECS
        pats.extend(pat for _, pat in _VOLATILE_BLOCK_SPECS)
    except Exception:  # noqa: BLE001
        pass
    return tuple(pats)


def extract_static_sys_prefix(sys_text: str) -> str:
    """Recorta al primer marcador dinamico/de usuario para alinear `prompt_cache_key`."""
    if not sys_text:
        return ""
    cut = len(sys_text)
    for pat in _volatile_cut_patterns():
        m = pat.search(sys_text)
        if m is not None and m.start() < cut:
            cut = m.start()
    return sys_text[:cut]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text"):   # input_text: Responses
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _extract_system_and_tool_names(payload: dict) -> tuple[str, list[str]]:
    sys_parts: list[str] = []
    # [P1-PLAN-LOTE-1043 · A-07 paso 3 / A-20] La API de Responses lleva la conversación en `input`, no en `messages`:
    # sin esto, ningún turno por Responses recibía la clave de caché.
    if payload.get("instructions"):
        sys_parts.append(_message_text(payload.get("instructions")))
    for msg in payload.get("messages") or payload.get("input") or ():
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in ("system", "developer"):
            txt = _message_text(msg.get("content"))
            if txt:
                sys_parts.append(txt)
        else:
            break
    tool_names: list[str] = []
    for tool in payload.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        name = fn.get("name") if isinstance(fn, dict) else tool.get("name")
        if name:
            tool_names.append(str(name))
    return "\n".join(sys_parts), tool_names


def _payload_alcanza_umbral_cache(full_sys_text: str, tool_names: list[str]) -> bool:
    estimated_chars = len(full_sys_text) + len(tool_names) * 600
    return estimated_chars >= _MIN_PREFIX_CHARS_FOR_CACHE


def _build_cache_key_digest(model_key: str, tool_names: list[str], full_sys_text: str) -> str:
    static_head = extract_static_sys_prefix(full_sys_text) or full_sys_text
    digest_input = f"{model_key}\n{','.join(tool_names)}\n{static_head}".encode("utf-8", errors="replace")
    return f"mf1:{hashlib.sha256(digest_input).hexdigest()[:24]}"


def compute_prompt_cache_key(payload: dict, model_name: Optional[str] = None) -> Optional[str]:
    """Genera una clave determinista <= 64 chars compartida entre usuarios del mismo prefijo."""
    if not openai_prompt_cache_key_activo():
        return None
    full_sys_text, tool_names = _extract_system_and_tool_names(payload)
    if not _payload_alcanza_umbral_cache(full_sys_text, tool_names):
        return None
    return _build_cache_key_digest(_model_key(model_name, payload), tool_names, full_sys_text)


def _model_key(model_name: Optional[str], payload: Optional[dict] = None) -> str:
    return str(model_name or (payload or {}).get("model") or "").strip().lower()


def inyectar_openai_prompt_cache(payload: dict, *, model_name: Optional[str] = None) -> None:
    """Inyecta `prompt_cache_key` y `prompt_cache_retention` en un payload OpenAI elegible."""
    if not isinstance(payload, dict):
        return
    m_key = _model_key(model_name, payload)
    full_sys_text, tool_names = _extract_system_and_tool_names(payload)
    if not _payload_alcanza_umbral_cache(full_sys_text, tool_names):
        return

    with _CACHE_PARAM_LOCK:
        skip_key = m_key in _UNSUPPORTED_CACHE_KEY_MODELS
        skip_ret = m_key in _UNSUPPORTED_RETENTION_MODELS

    if not skip_key and "prompt_cache_key" not in payload and openai_prompt_cache_key_activo():
        payload["prompt_cache_key"] = _build_cache_key_digest(m_key, tool_names, full_sys_text)

    retention = openai_prompt_cache_retention_value()
    if retention and not skip_ret and "prompt_cache_retention" not in payload:
        payload["prompt_cache_retention"] = retention

    _registrar_huella_del_prefijo(payload, full_sys_text)


def _registrar_huella_del_prefijo(payload: dict, full_sys_text: str) -> None:
    """[REVISION-LOTES-1001-1011] Huella del prefijo cacheable, para aislar por qué la PRIMERA llamada de cada turno
    sigue con `cached_tokens = 0` en producción (medido el 14-nov: 0 de 12 turnos, incluidos 3 seguidos del mismo
    usuario en 1 minuto con las mismas 8 tools y el mismo largo de prefijo). OpenAI cachea tools + mensajes por
    prefijo: comparar estas huellas entre dos turnos dice qué parte cambia, sin gastar IA. Solo log."""
    try:
        tools_json = json.dumps(payload.get("tools") or [], sort_keys=True, ensure_ascii=False)
        estatico = extract_static_sys_prefix(full_sys_text) or ""
        logger.info(
            "[G-02 HUELLA] key=%s tools=%s/%d sys_head=%s/%d sys_2k=%s msgs=%d",
            payload.get("prompt_cache_key"),
            hashlib.sha256(tools_json.encode("utf-8", "replace")).hexdigest()[:10], len(tools_json),
            hashlib.sha256(estatico.encode("utf-8", "replace")).hexdigest()[:10], len(estatico),
            hashlib.sha256(full_sys_text[:2000].encode("utf-8", "replace")).hexdigest()[:10],
            len(payload.get("messages") or ()),
        )
    except Exception:  # noqa: BLE001
        pass


def registrar_error_param_cache(exc: Exception, model_name: Optional[str]) -> bool:
    """Detecta un rechazo 400 por `prompt_cache_retention`/`prompt_cache_key` y desactiva en memoria.

    Devuelve True SOLO la primera vez que se registra un campo rechazado para ese
    modelo, permitiendo un reintento unico inmediato sin bucles.
    """
    m_key = _model_key(model_name)
    if not m_key:
        return False
    status = getattr(exc, "status_code", None)
    msg = str(exc or "").lower()
    is_param_400 = (status == 400) or (
        "400" in msg and any(tok in msg for tok in ("invalid", "unknown", "unsupported", "unrecognized"))
    )
    if not is_param_400:
        return False

    registered = False
    with _CACHE_PARAM_LOCK:
        if "prompt_cache_retention" in msg and m_key not in _UNSUPPORTED_RETENTION_MODELS:
            _UNSUPPORTED_RETENTION_MODELS.add(m_key)
            logger.warning(
                "⚠️ [P1-PLAN-LOTE-1003] %r rechazo `prompt_cache_retention`; se desactiva fail-open para este modelo.",
                m_key,
            )
            registered = True
        if "prompt_cache_key" in msg and m_key not in _UNSUPPORTED_CACHE_KEY_MODELS:
            _UNSUPPORTED_CACHE_KEY_MODELS.add(m_key)
            logger.warning(
                "⚠️ [P1-PLAN-LOTE-1003] %r rechazo `prompt_cache_key`; se desactiva fail-open para este modelo.",
                m_key,
            )
            registered = True
    return registered


def reset_unsupported_cache_models_for_tests() -> None:
    """Helper solo para tests unitarios."""
    with _CACHE_PARAM_LOCK:
        _UNSUPPORTED_RETENTION_MODELS.clear()
        _UNSUPPORTED_CACHE_KEY_MODELS.clear()


# Pre-registrar los 4 knobs en `knobs._KNOBS_REGISTRY` al importar el modulo.
_ = (
    static_prefix_no_uid_activo(),
    stable_tool_subset_activo(),
    openai_prompt_cache_key_activo(),
    openai_prompt_cache_retention_value(),
)
