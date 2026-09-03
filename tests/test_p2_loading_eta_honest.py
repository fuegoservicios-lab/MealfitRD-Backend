"""[P2-LOADING-ETA-HONEST · 2026-09-03] La pantalla de carga deja de prometer «3-6 minutos».

Medido sobre los 34 bloques 1 completados por la cola desde el flip (2026-09-02): mediana 583 s
(~10 min), p90 925 s (~15 min), mín 255 s, máx 1856 s. El copy fijo mentía en casi todos los
planes. `GET /api/plans/generation-eta` sirve el p50/p90 REAL de los últimos 14 días (cache 10 min,
cero LLM ⇒ `get_verified_user_id` + limitador, no `verify_api_quota`); el frontend lo consume y
cae a un rango prudente si no llega.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
SRC = (BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _rp():
    return pytest.importorskip("routers.plans")


def test_eta_shape_from_aggregate_row():
    rp = _rp()
    assert rp._generation_eta_from_row({"n": 34, "p50": 583.4, "p90": 925.0}) == {
        "p50_s": 583, "p90_s": 925, "n": 34, "window_days": rp._GENERATION_ETA_WINDOW_DAYS}
    # p90 nunca por debajo de p50 (percentiles sobre pocas filas pueden empatar hacia abajo)
    assert rp._generation_eta_from_row({"n": 9, "p50": 600, "p90": 590})["p90_s"] == 600
    # pocas muestras ⇒ nulls (el frontend cae al rango prudente), nunca una cifra inventada
    assert rp._generation_eta_from_row({"n": 3, "p50": 100, "p90": 200})["p50_s"] is None
    assert rp._generation_eta_from_row({})["n"] == 0 and rp._generation_eta_from_row(None)["p90_s"] is None
    assert rp._generation_eta_from_row({"n": 10, "p50": 0, "p90": 0})["p50_s"] is None


def test_endpoint_is_quota_exempt_with_limiter_and_cached():
    m = re.search(r'@router\.get\("/generation-eta"\)\s*\nasync def api_generation_eta\((.*?)\):', SRC, re.DOTALL)
    assert m, "endpoint /generation-eta ausente"
    sig = m.group(1)
    assert "Depends(get_verified_user_id)" in sig and "Depends(_ETA_LIMITER)" in sig
    assert "verify_api_quota" not in sig
    assert "_ETA_LIMITER = RateLimiter(max_calls=30, period_seconds=60)" in SRC
    assert "_GENERATION_ETA_TTL_S = 600" in SRC and "_GENERATION_ETA_MIN_SAMPLES = 5" in SRC
    # la cifra sale del tiempo de pared del bloque 1 completado, acotado a [60 s, 2 h]
    assert "WHERE week_number = 1" in SRC and "AND status = 'completed'" in SRC
    assert "EXTRACT(EPOCH FROM (updated_at - created_at)) BETWEEN 60 AND 7200" in SRC


def test_claude_md_documents_the_exemption():
    md = (BACKEND / "CLAUDE.md").read_text(encoding="utf-8")
    assert "`/generation-eta`" in md and "P2-LOADING-ETA-HONEST" in md


def test_marker_present():
    assert "P2-LOADING-ETA-HONEST" in SRC
