# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1007 · G-13 · 2026-11-14] Escalado determinista de turnos difíciles con tope duro USD.

Reglas (D3 aprobada por el dueño):
- Modelo de escalado: `gpt-6.1-terra` (`GPT61_TERRA`, US$ 1,00/M in · US$ 0,05/M cached · US$ 5,00/M out).
- Tope duro por turno: US$ 0,03 (`MEALFIT_CHAT_HARD_TURN_MAX_COST_USD=0.03`).
- Sólo escala cuando:
  1) `MEALFIT_CHAT_HARD_TURN_ESCALATION_ENABLED` está activo (True por defecto),
  2) NO hay un override manual distinto en `MEALFIT_CHAT_AGENT_MODEL` o `MEALFIT_CHAT_AGENT_MODEL` ya apunta al destino,
  3) El turno presenta señales clínicas/complejidad objetivas (múltiples condiciones/alergias + consulta de riesgo,
     o reintento post-nudge con perfil clínico / multi-foto, o multi-foto con perfil clínico activo),
  4) La estimación conservadora (caché frío) del costo acumulado + llamada en `gpt-6.1-terra` con `max_tokens` acotado
     es <= US$ 0,03; si excede el tope, NO escala y se queda en el modelo principal (`gpt-6-luna`).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from knobs import _env_bool, _env_float, _env_int, _env_str

logger = logging.getLogger(__name__)

_RE_HARD_INTENT = re.compile(
    r"\b(alergi\w*|celia\w*|diabet\w*|renal\w*|embaraz\w*|lactanc\w*|hipertens\w*|bariatr\w*|"
    r"tiroid\w*|medicament\w*|interacc\w*|sintom\w*|reacci[oó]n\w*|sustitu\w*|reemplaz\w*|"
    r"puedo\s+comer|me\s+hace\s+da[nñ]o|peligr\w*|prohib\w*|plan\w*)\b",
    re.IGNORECASE,
)

_EMPTY_CLINICAL = frozenset({"", "ninguna", "ninguno", "none", "n/a", "na", "no", "sin alergias", "ningun"})


def hard_turn_escalation_enabled() -> bool:
    return _env_bool("MEALFIT_CHAT_HARD_TURN_ESCALATION_ENABLED", True)


def hard_turn_model_name() -> Optional[str]:
    from llm_provider import GPT61_TERRA, is_openai_model

    raw = (_env_str("MEALFIT_CHAT_HARD_TURN_MODEL", GPT61_TERRA) or "").strip()
    if not raw or raw.lower() in ("off", "none", "0", "false"):
        return None
    if is_openai_model(raw) and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        return None
    return raw


def hard_turn_max_cost_usd() -> float:
    return _env_float(
        "MEALFIT_CHAT_HARD_TURN_MAX_COST_USD",
        0.03,
        validator=lambda v: 0.001 <= v <= 0.25,
    )


def hard_turn_max_output_tokens() -> int:
    val = _env_int("MEALFIT_CHAT_HARD_TURN_MAX_OUTPUT_TOKENS", 1000)
    return max(256, min(val, 4096))


def hard_turn_reasoning_effort() -> str:
    raw = (_env_str("MEALFIT_CHAT_HARD_TURN_EFFORT", "low") or "low").strip().lower()
    return raw if raw in ("none", "low", "medium", "high") else "low"


def _latest_human_text(messages: list[Any]) -> str:
    for m in reversed(messages or []):
        mtype = getattr(m, "type", None) or getattr(m, "role", None)
        if mtype in ("human", "user") or m.__class__.__name__ == "HumanMessage":
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "") if isinstance(p, dict) else p) for p in content)
            return str(content or "")
    return ""


def _clinical_flags_count(form_data: dict[str, Any]) -> int:
    if not isinstance(form_data, dict):
        return 0
    count = 0
    for key in ("allergies", "medicalConditions"):
        raw = form_data.get(key)
        if isinstance(raw, str):
            items = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
        else:
            items = []
        valid = [x for x in items if x.lower() not in _EMPTY_CLINICAL]
        count += len(valid)
    diet = str(form_data.get("dietType") or "").strip().lower()
    if diet and diet not in ("balanced", "equilibrada", "normal", "ninguna", "none"):
        count += 1
    return count


def detect_hard_turn_signals(state: dict[str, Any]) -> list[str]:
    """Devuelve las señales objetivas que clasifican un turno como difícil."""
    if not isinstance(state, dict):
        return []
    signals: list[str] = []
    form_data = state.get("form_data") or {}
    clinical_n = _clinical_flags_count(form_data)
    retried = bool(
        state.get("diary_claim_retried")
        or state.get("plate_photo_retried")
        or state.get("photo_log_retried")
    )
    multi_photo = len(state.get("turn_plate_photos") or []) >= 2
    human_txt = _latest_human_text(state.get("messages") or [])
    hard_intent = bool(_RE_HARD_INTENT.search(human_txt))

    if clinical_n >= 2 and hard_intent:
        signals.append("multi_clinical_intent")
    if retried and (clinical_n >= 1 or multi_photo):
        signals.append("post_nudge_retry")
    if multi_photo and clinical_n >= 1:
        signals.append("multiphoto_clinical")
    return signals


def estimate_turn_spent_micros(tracker: Any) -> int:
    """Suma conservadora de USD micros ya gastados en llamadas previas del mismo turno."""
    if tracker is None:
        return 0
    calls = getattr(tracker, "llm_calls", None) or []
    if not calls:
        return 0
    from db_profiles import compute_llm_cost_micros

    total = 0
    for c in list(calls):
        if not isinstance(c, dict):
            continue
        m = c.get("model")
        in_t = c.get("input_tokens")
        out_t = c.get("output_tokens")
        ca_t = c.get("cached_tokens") or 0
        if m and in_t is not None and out_t is not None:
            cm = compute_llm_cost_micros(m, in_t, out_t, ca_t)
            if cm is not None:
                total += cm
    return total


def estimate_call_max_micros(
    model: str,
    llm_messages: list[Any],
    active_tools: list[Any],
    max_output_tokens: int,
) -> Optional[int]:
    """Estimación conservadora (0% caché, ~3 chars/token) para respetar el tope por turno."""
    from db_profiles import compute_llm_cost_micros

    chars = 0
    for m in llm_messages or []:
        c = getattr(m, "content", "")
        if isinstance(c, list):
            chars += sum(len(str(p)) for p in c)
        else:
            chars += len(str(c or ""))
    # +180 toks aproximados por herramienta ligada + overhead de mensajes
    in_est = max(500, (chars // 3) + (len(active_tools or []) * 180) + 500)
    return compute_llm_cost_micros(model, in_est, max_output_tokens, cached_tokens=0)


def evaluar_turno_dificil(
    state: dict[str, Any],
    llm_messages: list[Any],
    primary_model: str,
    active_tools: list[Any],
    *,
    tracker: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Decide si escalar un turno difícil a `gpt-6.1-terra` respetando el tope US$ 0,03/turno.

    Retorna `(modelo_efectivo, {"escalated": bool, ...})`.
    """
    if not hard_turn_escalation_enabled():
        return primary_model, {"escalated": False, "reason": "disabled"}

    from llm_provider import GPT6_LUNA

    # Respetar override explícito cuando apunta a un modelo distinto del baseline `gpt-6-luna`
    # (en el VPS `.env`, `MEALFIT_CHAT_AGENT_MODEL=gpt-6-luna` es precisamente la base elegible).
    _override = (_env_str("MEALFIT_CHAT_AGENT_MODEL", "") or "").strip()
    target = hard_turn_model_name()
    if not target or target.lower() == str(primary_model or "").strip().lower():
        return primary_model, {"escalated": False, "reason": "no_target_or_same"}
    if _override and _override.lower() != GPT6_LUNA.lower():
        return primary_model, {"escalated": False, "reason": "explicit_model_override"}

    signals = detect_hard_turn_signals(state)
    if not signals:
        return primary_model, {"escalated": False, "reason": "normal_turn"}

    max_cost_usd = hard_turn_max_cost_usd()
    max_budget_micros = int(round(max_cost_usd * 1_000_000))
    max_out = hard_turn_max_output_tokens()
    spent_micros = estimate_turn_spent_micros(tracker)
    est_micros = estimate_call_max_micros(target, llm_messages, active_tools, max_out)

    if est_micros is None or (spent_micros + est_micros) > max_budget_micros:
        logger.info(
            "💸 [G-13 HARD-TURN] Escalado omitido por tope USD ($%.4f est + $%.4f previo > $%.2f tope); "
            "se mantiene %r (signals=%s)",
            (est_micros or 0) / 1_000_000.0,
            spent_micros / 1_000_000.0,
            max_cost_usd,
            primary_model,
            signals,
        )
        return primary_model, {
            "escalated": False,
            "reason": "budget_cap_exceeded",
            "signals": signals,
            "est_usd": round((spent_micros + (est_micros or 0)) / 1_000_000.0, 5),
            "cap_usd": max_cost_usd,
        }

    total_est_usd = round((spent_micros + est_micros) / 1_000_000.0, 5)
    logger.info(
        "🚀 [G-13 HARD-TURN] Escalando turno difícil %r → %r (est_max=$%.4f <= tope=$%.2f; signals=%s)",
        primary_model,
        target,
        total_est_usd,
        max_cost_usd,
        signals,
    )
    return target, {
        "escalated": True,
        "from_model": primary_model,
        "to_model": target,
        "signals": signals,
        "max_tokens": max_out,
        "reasoning_effort": hard_turn_reasoning_effort(),
        "est_usd": total_est_usd,
        "cap_usd": max_cost_usd,
    }


def construir_llm_turno_dificil(
    model: str,
    hard_info: dict[str, Any],
    *,
    timeout_s: float,
    retry_kwargs: Optional[dict[str, Any]] = None,
    voice_kwargs: Optional[dict[str, Any]] = None,
):
    """Construye el cliente LLM acotado para un turno escalado sin duplicar callsites en `agent.py`."""
    import llm_provider

    rkw = dict(retry_kwargs or {})
    if llm_provider.is_openai_model(model):
        return llm_provider.build_chat_llm(
            model,
            timeout=timeout_s,
            reasoning_effort=hard_info.get("reasoning_effort", "low"),
            max_tokens=hard_info.get("max_tokens", hard_turn_max_output_tokens()),
            **rkw,
        )
    vkw = dict(voice_kwargs or {})
    return llm_provider.ChatGLM(
        model=model,
        temperature=0.7,
        timeout=timeout_s,
        max_tokens=hard_info.get("max_tokens", hard_turn_max_output_tokens()),
        **rkw,
        **vkw,
    )
