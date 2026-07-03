"""[P1-NEXT-LEVEL-BATCH · 2026-07-02] Modelo de gustos APRENDIDO del uso (no declarado).

El motor trataba igual al usuario en su plan 1 y en su plan 50: toda la personalización
venía de lo DECLARADO en el formulario (dislikes/alergias). Este módulo aprende de lo que
el usuario HACE:
  - swap-away: reemplazó un plato Y cambió la proteína principal → señal negativa débil
    (peso 1.0) sobre la proteína vieja. Si mantuvo la proteína, el swap no era sobre ella
    → cero señal (anti-ruido).
  - chat-replace: pidió cambiar un plato por chat → negativa débil sobre la proteína vieja
    si cambió; FUERTE (peso 2.0) si el texto contiene negación explícita del token
    ("no me gusta", "odio", "sin pollo", "quita el pollo").

Consumo: `build_taste_context(user_id)` produce un bloque de prompt que se APPENDEA al
`taste_profile` existente del context builder → fluye al planner (esqueleto/pools) y al
day-generator sin tocar sus prompts. Umbral de score ≥ 2.0 (dos swaps o un chat fuerte)
y ventana de 90 días → un evento aislado no re-dirige el motor. Señal SUAVE (prompt):
las alergias/dieta siguen siendo los únicos vetos duros.

Fail-open TOTAL: sin tabla / sin DB / guest → cero señales, cero contexto, cero excepción
hacia el caller. Guests (user_id None/'guest'/session) nunca escriben ni leen.

Knob: MEALFIT_TASTE_MODEL (default ON — prompt-soft, riesgo bajo).
Migración: migrations/p1_user_taste_events_2026_07_02.sql (ambos dirs, SSOT).
tooltip-anchor: P1-NEXT-LEVEL-TASTE. Test: test_p1_next_level_batch.py.
"""
from __future__ import annotations

import logging
import re

from knobs import _env_bool, _env_float, _env_int

logger = logging.getLogger(__name__)

TASTE_MODEL_ENABLED = _env_bool("MEALFIT_TASTE_MODEL", True)
TASTE_MIN_SCORE = _env_float("MEALFIT_TASTE_MIN_SCORE", 2.0, validator=lambda v: 0.5 <= v <= 10.0)
TASTE_WINDOW_DAYS = _env_int("MEALFIT_TASTE_WINDOW_DAYS", 90, validator=lambda v: 7 <= v <= 365)
TASTE_MAX_TOKENS = _env_int("MEALFIT_TASTE_MAX_TOKENS", 6, validator=lambda v: 1 <= v <= 12)
# [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-G) TASTE v2 — señales POSITIVAS: el modelo solo aprendía
# rechazos; lo que el usuario ELIGE activamente (proteína del plato al que swapea / la que pide
# por chat) no dejaba rastro. Las positivas se guardan con peso NEGATIVO en el mismo score →
# compensan automáticamente los avoid (2 swap-away + 3 swap-to = neto bajo umbral) y, con señal
# fuerte acumulada, generan la línea PREFIERE del contexto. Rollback: MEALFIT_TASTE_POSITIVE=false.
TASTE_POSITIVE_ENABLED = _env_bool("MEALFIT_TASTE_POSITIVE", True)
TASTE_POSITIVE_MIN_SCORE = _env_float("MEALFIT_TASTE_POSITIVE_MIN_SCORE", 2.0, validator=lambda v: 0.5 <= v <= 10.0)

# Negación explícita: la señal es FUERTE solo si el texto contiene una frase de rechazo
# Y menciona el token de la proteína (evita atribuir "no me gusta este plato" a la proteína).
_STRONG_NEG_PHRASES = ("no me gusta", "no me agrada", "odio", "detesto", "no quiero", "quita", "sin ")


def _is_real_user(user_id) -> bool:
    u = str(user_id or "").strip().lower()
    return bool(u) and u != "guest" and not u.startswith("guest") and len(u) >= 30  # UUID-ish


def _protein_syns() -> dict:
    """SSOT de sinónimos de proteína — lazy para evitar ciclos de import (vive en tools)."""
    try:
        from tools import _SD_PROT_SYNS_CM
        return _SD_PROT_SYNS_CM
    except Exception:
        return {}


def protein_token_of(meal_name: str) -> str | None:
    """Proteína canónica del nombre del plato (word-boundary + accent-strip)."""
    try:
        from constants import strip_accents
        name = strip_accents(str(meal_name or "").lower())
        for canon, syns in _protein_syns().items():
            if any(re.search(r"\b" + s + r"\b", name) for s in syns):
                return canon
    except Exception:
        pass
    return None


def _record(user_id, token: str, signal: str, weight: float, source: str) -> bool:
    try:
        from db import execute_sql_write
        execute_sql_write(
            "INSERT INTO user_taste_events (user_id, token, signal, weight, source) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(user_id), str(token)[:60], str(signal)[:30], float(weight), str(source)[:30]),
        )
        return True
    except Exception as _e:
        logger.debug(f"[P1-NEXT-LEVEL-TASTE] record no-op: {type(_e).__name__}: {_e}")
        return False


def record_swap_away(user_id, old_meal_name: str, new_meal_name: str, source: str = "swap") -> bool:
    """Señal negativa DÉBIL sobre la proteína del plato reemplazado — SOLO si la proteína
    cambió (si el usuario mantuvo la proteína, el swap no era sobre ella)."""
    if not TASTE_MODEL_ENABLED or not _is_real_user(user_id):
        return False
    old_p = protein_token_of(old_meal_name)
    new_p = protein_token_of(new_meal_name)
    if not old_p or old_p == new_p:
        return False
    return _record(user_id, old_p, "swap_away", 1.0, source)


def record_chat_replace(user_id, old_meal_name: str, new_meal_name: str, changes: str) -> bool:
    """Señal negativa por reemplazo vía chat: fuerte (2.0) con negación explícita del token
    en `changes`; débil (1.0) si solo cambió la proteína."""
    if not TASTE_MODEL_ENABLED or not _is_real_user(user_id):
        return False
    old_p = protein_token_of(old_meal_name)
    if not old_p:
        return False
    try:
        from constants import strip_accents
        ch = strip_accents(str(changes or "").lower())
    except Exception:
        ch = str(changes or "").lower()
    syns = _protein_syns().get(old_p, (old_p,))
    _token_mentioned = any(re.search(r"\b" + s + r"\b", ch) for s in syns)
    strong = _token_mentioned and any(p in ch for p in _STRONG_NEG_PHRASES)
    if strong:
        return _record(user_id, old_p, "chat_negative", 2.0, "chat")
    new_p = protein_token_of(new_meal_name)
    if old_p != new_p:
        return _record(user_id, old_p, "chat_replace", 1.0, "chat")
    return False


def record_swap_to(user_id, old_meal_name: str, new_meal_name: str, source: str = "swap") -> bool:
    """[P2-AUDIT-V6-BATCH · 2026-07-03] (P2-G) Señal POSITIVA débil sobre la proteína del plato
    ELEGIDO al persistir un swap — SOLO si la proteína cambió (elegirla activamente ≠ mantenerla
    por inercia). Peso -1.0 (offset del score avoid)."""
    if not TASTE_MODEL_ENABLED or not TASTE_POSITIVE_ENABLED or not _is_real_user(user_id):
        return False
    old_p = protein_token_of(old_meal_name)
    new_p = protein_token_of(new_meal_name)
    if not new_p or old_p == new_p:
        return False
    return _record(user_id, new_p, "swap_to", -1.0, source)


def record_chat_request_positive(user_id, new_meal_name: str, changes: str) -> bool:
    """[P2-AUDIT-V6-BATCH · 2026-07-03] (P2-G) Señal POSITIVA fuerte cuando el usuario PIDIÓ la
    proteína por chat ("ponme camarones") y el plato entregado la trae. Guard anti-falso-positivo:
    la mención no debe venir precedida de una frase de rechazo ("sin pollo", "no quiero res")."""
    if not TASTE_MODEL_ENABLED or not TASTE_POSITIVE_ENABLED or not _is_real_user(user_id):
        return False
    new_p = protein_token_of(new_meal_name)
    if not new_p:
        return False
    try:
        from constants import strip_accents
        ch = strip_accents(str(changes or "").lower())
    except Exception:
        ch = str(changes or "").lower()
    syns = _protein_syns().get(new_p, (new_p,))
    for s in syns:
        for m in re.finditer(r"\b" + s + r"\b", ch):
            prefix = ch[max(0, m.start() - 20):m.start()]
            if any(p in prefix for p in _STRONG_NEG_PHRASES):
                continue
            return _record(user_id, new_p, "chat_positive", -2.0, "chat")
    return False


def positive_tokens_for_user(user_id) -> list[str]:
    """[P2-AUDIT-V6-BATCH · 2026-07-03] (P2-G) Tokens con score NETO fuertemente positivo
    (SUM(weight) ≤ -umbral) en la ventana — lo que el usuario elige consistentemente."""
    if not TASTE_MODEL_ENABLED or not TASTE_POSITIVE_ENABLED or not _is_real_user(user_id):
        return []
    try:
        from db import execute_sql_query
        rows = execute_sql_query(
            "SELECT token, SUM(weight) AS score FROM user_taste_events "
            "WHERE user_id = %s AND created_at > NOW() - make_interval(days => %s) "
            "GROUP BY token HAVING SUM(weight) <= %s ORDER BY score ASC LIMIT %s",
            (str(user_id), int(TASTE_WINDOW_DAYS), -float(TASTE_POSITIVE_MIN_SCORE), int(TASTE_MAX_TOKENS)),
            fetch_all=True,
        ) or []
        return [str(r.get("token")) for r in rows if r.get("token")]
    except Exception as _e:
        logger.debug(f"[P2-AUDIT-V6-BATCH] (P2-G) read positivas no-op: {type(_e).__name__}: {_e}")
        return []


def negative_tokens_for_user(user_id) -> list[str]:
    """Tokens con score negativo acumulado ≥ umbral en la ventana (más señal primero)."""
    if not TASTE_MODEL_ENABLED or not _is_real_user(user_id):
        return []
    try:
        from db import execute_sql_query
        rows = execute_sql_query(
            "SELECT token, SUM(weight) AS score FROM user_taste_events "
            "WHERE user_id = %s AND created_at > NOW() - make_interval(days => %s) "
            "GROUP BY token HAVING SUM(weight) >= %s ORDER BY score DESC LIMIT %s",
            (str(user_id), int(TASTE_WINDOW_DAYS), float(TASTE_MIN_SCORE), int(TASTE_MAX_TOKENS)),
            fetch_all=True,
        ) or []
        return [str(r.get("token")) for r in rows if r.get("token")]
    except Exception as _e:
        logger.debug(f"[P1-NEXT-LEVEL-TASTE] read no-op: {type(_e).__name__}: {_e}")
        return []


def build_taste_context(user_id) -> str:
    """Bloque de prompt con lo aprendido del uso. '' si no hay señal (byte-equivalente
    → preserva el prompt-cache para usuarios sin señales).
    [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-G) además de los rechazos, la línea PREFIERE con las
    proteínas que el usuario ELIGE consistentemente (swap-to / pedido por chat)."""
    tokens = negative_tokens_for_user(user_id)
    positives = positive_tokens_for_user(user_id)
    if not tokens and not positives:
        return ""
    parts = ["\n🧠 APRENDIDO DEL USO (señal de comportamiento, NO declarada en el formulario):"]
    if tokens:
        parts.append(
            f" el usuario reemplaza consistentemente los platos cuya proteína principal es: {', '.join(tokens)}. "
            "REDUCE su presencia en los pools y platos (no las elimines del todo si son necesarias "
            "nutricionalmente — es preferencia aprendida, NO alergia)."
        )
    if positives:
        parts.append(
            f" El usuario ELIGE consistentemente: {', '.join(positives)} — cuando encajen con el "
            "objetivo y la variedad del día, PREFIÉRELAS sobre equivalentes (sin repetirlas en exceso)."
        )
    return "".join(parts)
