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
    # [P1-PLAN-LOTE-1038 · A-15] era `plan\w*`: en el replay de 461 mensajes reales casó 23 veces y NINGUNA era un turno
    # difícil («a la plancha», «plántano», «planta»…). Ahora solo la palabra «plan» / «planes».
    r"puedo\s+comer|me\s+hace\s+da[nñ]o|peligr\w*|prohib\w*|plan(?:es)?)\b",
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


def chat_user_daily_budget_usd() -> float:
    """[P1-PLAN-LOTE-1010 · G-19] Presupuesto diario por usuario USD para chat (0 = desactivado)."""
    return _env_float(
        "MEALFIT_CHAT_USER_DAILY_BUDGET_USD",
        0.25,
        validator=lambda v: 0.0 <= v <= 50.0,
    )


def user_exempt_from_chat_budget(user_id: Optional[str]) -> bool:
    """[P1-PLAN-LOTE-1016 · A-05] Las cuentas de administración (`MEALFIT_ADMIN_USER_IDS`) y `MEALFIT_CHAT_BUDGET_EXEMPT_USER_IDS` no
    se bloquean por el tope diario: el dueño probando el coach llegaría al límite el mismo día (13-nov: 3 cuentas pasaron de US$ 0,25)."""
    if not user_id:
        return False
    uid = str(user_id).strip().lower()
    try:
        from admin_acceso import admin_ids
        if uid in admin_ids():
            return True
    except Exception as exc:  # noqa: BLE001
        from fail_open import fail_open
        fail_open("budget_exempt_admin_ids", exc)
    extra = _env_str("MEALFIT_CHAT_BUDGET_EXEMPT_USER_IDS", "")
    if uid in {x.strip().lower() for x in extra.split(",") if x.strip()}:
        return True
    from coach_tope_diario import es_cuenta_de_prueba   # [P1-PLAN-LOTE-1045 · D5] las cuentas de prueba, exentas
    return es_cuenta_de_prueba(uid)


def check_user_daily_chat_budget(
    user_id: Optional[str],
    *,
    execute_sql_query: Any = None,
) -> dict[str, Any]:
    """[P1-PLAN-LOTE-1010 · G-19] Evalúa gasto USD 24h del usuario sin coste de IA (fail-open).

    - `< 80%`: `status='ok'`, permite turno y escalado normal.
    - `>= 80% y < 100%`: `status='soft_80'`, bloquea escalado a `gpt-6.1-terra`.
    - `>= 100%`: `status='hard_100'`, pausa suavemente el turno sin llamar al LLM.
    """
    limit = chat_user_daily_budget_usd()
    if limit <= 0.0 or not user_id or user_id == "guest" or user_exempt_from_chat_budget(user_id):
        return {
            "status": "ok",
            "allow_escalation": True,
            "block_turn": False,
            "spent_usd": 0.0,
            "limit_usd": limit,
        }
    try:
        if execute_sql_query is None:
            from db_core import execute_sql_query as _ssot_q
            execute_sql_query = _ssot_q
        row = execute_sql_query(
            """
            SELECT COALESCE(SUM(cost_usd_micros), 0)::bigint AS spent_micros
            FROM public.llm_usage_events
            WHERE user_id = %s
              AND (node LIKE 'chat%%' OR node = 'vision_scan')
              AND created_at >= NOW() - INTERVAL '24 hours'
            """,
            (str(user_id),),
            fetch_one=True,
        ) or {}
        if isinstance(row, dict):
            spent_micros = int(row.get("spent_micros") or 0)
        elif isinstance(row, (list, tuple)) and row:
            spent_micros = int(row[0] or 0)
        else:
            spent_micros = 0
        spent_usd = round(spent_micros / 1_000_000.0, 6)
        if spent_usd >= limit:
            from coach_tope_diario import libre_desde   # [P1-PLAN-LOTE-1045 · D5] el aviso dice la hora real
            return {
                "status": "hard_100",
                "allow_escalation": False,
                "block_turn": True,
                "spent_usd": spent_usd,
                "limit_usd": limit,
                "libre_desde": libre_desde(str(user_id), limit, execute_sql_query=execute_sql_query),
            }
        if spent_usd >= (0.80 * limit):
            return {
                "status": "soft_80",
                "allow_escalation": False,
                "block_turn": False,
                "spent_usd": spent_usd,
                "limit_usd": limit,
            }
        return {
            "status": "ok",
            "allow_escalation": True,
            "block_turn": False,
            "spent_usd": spent_usd,
            "limit_usd": limit,
        }
    except Exception as exc:
        from fail_open import fail_open   # [P1-PLAN-LOTE-1014 · A-04] el tope que falla en silencio no protege nada
        fail_open("chat_budget_check", exc, critico=True)
        return {
            "status": "ok",
            "allow_escalation": True,
            "block_turn": False,
            "spent_usd": 0.0,
            "limit_usd": limit,
        }


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
    if isinstance(state, dict) and state.get("chat_budget_soft_degraded"):
        return primary_model, {"escalated": False, "reason": "user_daily_budget_soft_cap"}
    # [P1-PLAN-LOTE-1016 · A-07] terra nunca había atendido un turno real: por canario (por defecto solo `MEALFIT_CHAT_CANARY_USER_IDS`)
    # hasta que un turno suyo quede registrado con su coste. `MEALFIT_CHAT_CANARY_TERRA=on` lo abre a todos; `off` lo cierra.
    from coach_canario import registrar_brazo as _registrar_brazo
    if not _registrar_brazo(tracker, (state or {}).get("user_id") if isinstance(state, dict) else None, "terra"):
        return primary_model, {"escalated": False, "reason": "canary_off"}

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
        _m_low = str(model or "").strip().lower()
        if "terra" in _m_low and _m_low.startswith("gpt-6") and "use_responses_api" not in rkw:
            rkw["use_responses_api"] = True
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
