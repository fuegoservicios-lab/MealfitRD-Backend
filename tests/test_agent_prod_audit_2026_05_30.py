"""[AGENT-PROD-AUDIT · 2026-05-30] Regresión consolidada de los gaps cerrados
en la auditoría profunda del subsistema del agente (workflow adversario de 14
dimensiones → verificación manual → implementación).

Cubre (parser-based salvo el funcional de cache_empty):
  - P1-PROACTIVE-TZ            proactive_agent pasa tz_offset a get_consumed_meals_today
  - P2-PROACTIVE-NUDGE-BUDGET-TZ  nudge count contra el día AST, no UTC
  - P1-CHAT-BILL-VERIFIED-UID  billing por verified_user_id (cierra bypass user_id==session_id)
  - P2-FACT-SAVE-FAIL-LOUD     fact pipeline chequea el return de save_user_fact
  - P2-EMBED-NO-CACHE-EMPTY    cache_empty=False en embedding callsites (+ funcional)
  - P3-WATER-ATOMIC-DELTA      log_water_glass usa upsert atómico ON CONFLICT
  - P3-SUMMARY-ARCHIVE-GUARD   delete de summaries gateado en archive OK
  - P3-VISION-FAIL-ERROR-LOG   vision failure a logger.error (Sentry)

Parser-based para correr en el venv DB-less del repo. El IDOR + async offload
del diary viven en test_p1_diary_upload_guest_idor.py.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _code_only(src: str) -> str:
    """Quita el contenido de comentarios `#...` línea-por-línea para que las
    aserciones NEGATIVAS no matcheen el código viejo citado en los comentarios
    explicativos del fix. (Lección de auditorías previas: anclar al statement
    real, no a substrings del comentario.)"""
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


# ---------------------------------------------------------------------------
# P1-PROACTIVE-TZ + P2-PROACTIVE-NUDGE-BUDGET-TZ
# ---------------------------------------------------------------------------
def test_proactive_passes_tz_offset_to_consumed():
    src = _read("proactive_agent.py")
    assert "P1-PROACTIVE-TZ" in src
    assert "_proactive_tz_offset_min" in src, "Falta el helper del knob de offset"
    # La llamada a get_consumed_meals_today debe pasar tz_offset_mins.
    assert re.search(
        r'get_consumed_meals_today\(\s*\n?\s*user_id\s*,\s*\n?\s*date_str=now_ast\.strftime\("%Y-%m-%d"\)\s*,\s*\n?\s*tz_offset_mins=_proactive_tz_offset_min\(\)\s*,?\s*\n?\s*\)',
        src,
    ), "get_consumed_meals_today debe recibir tz_offset_mins=_proactive_tz_offset_min()"


def test_nudge_budget_counts_ast_day_not_utc():
    src = _read("proactive_agent.py")
    assert "P2-PROACTIVE-NUDGE-BUDGET-TZ" in src
    # Ya no debe usar el conteo UTC crudo `DATE(sent_at) = CURRENT_DATE` en el
    # CÓDIGO (el comentario del fix sí lo cita como pre-fix → strip comentarios).
    assert "DATE(sent_at) = CURRENT_DATE" not in _code_only(src), (
        "El conteo de nudges UTC (DATE(sent_at)=CURRENT_DATE) debe haberse "
        "reemplazado por la conversión a día AST"
    )
    assert "America/Santo_Domingo" in src, (
        "El conteo de nudges debe convertir a la zona AST 'America/Santo_Domingo'"
    )


# ---------------------------------------------------------------------------
# P1-CHAT-BILL-VERIFIED-UID
# ---------------------------------------------------------------------------
def test_chat_bills_on_verified_user_id():
    src = _read("routers/chat.py")
    assert "P1-CHAT-BILL-VERIFIED-UID" in src
    # Ambos sitios facturan por verified_user_id.
    assert 'log_api_usage(verified_user_id, "gemini_chat")' in src, (
        "Debe facturar log_api_usage sobre verified_user_id (token-verified)"
    )
    # El gate explotable user_id != session_id ya NO debe gobernar el billing:
    # no debe quedar un log_api_usage(user_id, ...) ACTIVO (el comentario del fix
    # lo cita como pre-fix → strip comentarios antes de la aserción negativa).
    assert 'log_api_usage(user_id, "gemini_chat")' not in _code_only(src), (
        "El billing NO debe usar el user_id del body (bypass user_id==session_id)"
    )


# ---------------------------------------------------------------------------
# P2-FACT-SAVE-FAIL-LOUD
# ---------------------------------------------------------------------------
def test_fact_pipeline_checks_save_return():
    src = _read("fact_extractor.py")
    assert "P2-FACT-SAVE-FAIL-LOUD" in src
    # El save de nuevos hechos captura el return y solo cuenta en éxito.
    assert re.search(
        r'_saved\s*=\s*save_user_fact\(', src
    ), "Debe capturar el return de save_user_fact"
    assert re.search(
        r'if\s+_saved\s*:', src
    ), "Debe contar saved_count/merge_count solo si _saved es truthy"
    # Debe emitir un error-level visible en el path de fallo.
    assert "save_user_fact NO persistió" in src


# ---------------------------------------------------------------------------
# P2-EMBED-NO-CACHE-EMPTY (parser + funcional)
# ---------------------------------------------------------------------------
def test_embedding_callsites_opt_out_of_caching_empty():
    fx = _read("fact_extractor.py")
    va = _read("vision_agent.py")
    assert "cache_empty=False" in fx, "fact_extractor.get_embedding debe usar cache_empty=False"
    assert "cache_empty=False" in va, "vision _cached_multimodal_embedding debe usar cache_empty=False"


def test_centralized_cache_respects_cache_empty_flag():
    """Funcional: con cache_empty=False un resultado falsy NO se cachea
    (se re-ejecuta); con el default (True) sí se cachea (backward-compat)."""
    import cache_manager as cm

    calls = {"n": 0}

    @cm.centralized_cache(ttl_seconds=999999, maxsize=10, cache_empty=False)
    def emb(_t):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [0.1]

    assert emb("k") == []          # primer intento falla -> []
    assert emb("k") == [0.1]       # NO cacheó el [] -> re-ejecuta y obtiene el bueno
    assert calls["n"] == 2

    calls2 = {"n": 0}

    @cm.centralized_cache(ttl_seconds=999999, maxsize=10)  # default cache_empty=True
    def emb2(_t):
        calls2["n"] += 1
        return [] if calls2["n"] == 1 else [9]

    assert emb2("k") == []
    assert emb2("k") == []          # cacheó el [] -> NO re-ejecuta (comportamiento histórico)
    assert calls2["n"] == 1


# ---------------------------------------------------------------------------
# P3-WATER-ATOMIC-DELTA
# ---------------------------------------------------------------------------
def test_water_glass_uses_atomic_upsert():
    src = _read("tools.py")
    assert "P3-WATER-ATOMIC-DELTA" in src
    # Incremento atómico en SQL: glasses + delta dentro del ON CONFLICT.
    assert "ON CONFLICT (user_id, log_date)" in src
    assert "GREATEST(0, LEAST(50, water_intake_log.glasses + %s))" in src, (
        "El upsert debe sumar el delta en SQL (atómico), no en Python"
    )
    assert "RETURNING glasses" in src


# ---------------------------------------------------------------------------
# P3-SUMMARY-ARCHIVE-GUARD
# ---------------------------------------------------------------------------
def test_master_summary_delete_gated_on_archive():
    src = _read("memory_manager.py")
    assert "P3-SUMMARY-ARCHIVE-GUARD" in src
    assert re.search(r'_archived\s*=\s*archive_summaries\(', src), (
        "Debe capturar el return de archive_summaries"
    )
    assert re.search(r'if\s+_archived\s*:', src), (
        "delete_summaries debe ejecutarse solo si _archived es truthy"
    )


# ---------------------------------------------------------------------------
# P3-VISION-FAIL-ERROR-LOG
# ---------------------------------------------------------------------------
def test_vision_failure_logs_error_level():
    src = _read("vision_agent.py")
    assert "P3-VISION-FAIL-ERROR-LOG" in src
    # El except del invoke de vision debe loguear a error (no warning).
    assert "process_image_with_vision falló" in src
    assert re.search(r'logger\.error\(.*process_image_with_vision falló', src), (
        "El fallo de vision debe ir a logger.error para que Sentry lo capture"
    )
