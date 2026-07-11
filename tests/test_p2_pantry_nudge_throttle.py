"""[P2-PANTRY-NUDGE-THROTTLE · 2026-07-11] Canal ÚNICO para las notificaciones de
"nevera vacía" con cooldown por usuario.

Feedback vivo del owner: la misma idea le llegaba desde 6+ emisores distintos (pausa
per-chunk, recordatorios per-chunk, validación final, recovery, freeze) — "me sale
mucho y a veces triplicada". Ahora: no importa cuántos chunks/eventos lo pidan, UNA
notificación por ventana de 6h (knob MEALFIT_PANTRY_NUDGE_COOLDOWN_HOURS), con copy
unificado y claro ("Tu Nevera está vacía — agrega tus alimentos para que tu plan
empiece a funcionar").

tooltip-anchor: P2-PANTRY-NUDGE-THROTTLE
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Funcional: cooldown suprime, expirado envía, KV se actualiza
# ---------------------------------------------------------------------------

def _run_nudge(last_sent_iso):
    import cron_tasks as ct
    sent = []
    writes = []

    def _fake_q(sql, params=(), fetch_one=False, **kw):
        if "app_kv_store" in sql:
            return {"value": {"sent_at": last_sent_iso}} if last_sent_iso else None
        return None

    with patch.object(ct, "execute_sql_query", _fake_q), \
         patch.object(ct, "execute_sql_write", lambda *a, **k: writes.append(a)), \
         patch.object(ct, "_dispatch_push_notification", lambda **kw: sent.append(kw)):
        ok = ct._dispatch_pantry_nudge("user-123")
    return ok, sent, writes


def test_cooldown_suppresses_within_window():
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    ok, sent, writes = _run_nudge(recent)
    assert ok is False and sent == [] and writes == [], (
        "2h < cooldown 6h → suprimido (era la causa del triplicado)"
    )


def test_sends_after_cooldown_and_records_kv():
    old = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    ok, sent, writes = _run_nudge(old)
    assert ok is True and len(sent) == 1
    assert "Nevera está vacía" in sent[0]["title"], "copy unificado y claro"
    assert "empiece a funcionar" in sent[0]["body"], "el copy pedido por el owner"
    assert writes, "el envío registra sent_at en KV (base del cooldown)"


def test_first_ever_sends():
    ok, sent, _ = _run_nudge(None)
    assert ok is True and len(sent) == 1


# ---------------------------------------------------------------------------
# 2. Todos los emisores de la clase pasan por el canal
# ---------------------------------------------------------------------------

def test_all_pantry_emitters_rewired():
    n = _CRON.count("_dispatch_pantry_nudge(")
    # 1 def + 7 callsites (el push de snapshot-stale se REVIRTIÓ a directo a propósito:
    # tiene su propio cooldown per-user en health_profile y copy semántico propio).
    assert n >= 8, f"esperaba >=8 (1 def + 7 callsites), hay {n} — ¿algún emisor volvió al push directo?"
    # los títulos viejos de la clase ya no se emiten directo
    # como TÍTULO de push directo (la frase puede vivir en bodies de otros avisos,
    # p.ej. el push de cambio de zona horaria — esos no son de esta clase).
    for legacy in ("Actualiza tu nevera para continuar",
                   "Refresca tu nevera para continuar tu plan",
                   "No pudimos confirmar tu nevera",
                   "Tu plan necesita revisión de nevera"):
        assert ('title="' + legacy + '"') not in _CRON, f"emisor legacy sin throttle: {legacy!r}"


def test_knob_and_failopen():
    i = _CRON.find("def _dispatch_pantry_nudge")
    blk = _CRON[i: i + 3200]
    assert 'MEALFIT_PANTRY_NUDGE_COOLDOWN_HOURS", 6' in blk, "cooldown 6h (pedido del owner)"
    assert "pantry_nudge_last:" in blk, "KV por usuario"
    assert "Fail-open" in blk or "fail-open" in blk.lower(), (
        "error de KV → envía igual (mejor un aviso de más que un usuario congelado sin saberlo)"
    )


def test_marker_anchored_in_source():
    assert _CRON.count("P2-PANTRY-NUDGE-THROTTLE") >= 4
