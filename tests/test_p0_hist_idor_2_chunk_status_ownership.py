"""[P0-HIST-IDOR-2 · 2026-05-10] `GET /{plan_id}/chunk-status` DEBE
chequear ownership del plan antes de serializar el payload.

Bug original (audit Historial 2026-05-10):
    El handler `api_chunk_status` (plans.py:3330) leía `user_id` de
    `meal_plans` pero NUNCA lo comparaba contra `verified_user_id`.
    Cualquier user authenticated podía leer info de planes ajenos:
      - `last_learning_hint` con `quality_history[-1].score` del
        `user_profiles.health_profile` del DUEÑO ajeno.
      - `failed_chunks` con `attempts` (telemetría operacional).
      - `paused_chunks` con `reason_code` derivado de
        `pipeline_snapshot._pause_reason`/`_pantry_pause_reason`/
        `dead_letter_reason` (estado de pantry/tz/learning del dueño).
      - `tier_breakdown` (calidad de generación por tier).
      - `_user_action_required` (CTA preformateado),
        `_recovery_exhausted_chunks`, `_pantry_degraded_summary`.

    Polling-friendly: el frontend lo llama cada 2-5s durante la
    generación → IDOR explotable a alta frecuencia. Identificado
    como follow-up en P1-HIST-AUDIT-NEW-1.

Estrategia del test (parser estático, mismo patrón que
`test_p0_hist_idor_1_retry_chunk_ownership.py`):
    1. Localizar `api_chunk_status` y extraer body hasta el
       siguiente top-level `def`/`@router`.
    2. Verificar comparación `user_id != verified_user_id` (con
       casts a str para defender contra UUID vs str).
    3. Verificar que el mismatch lanza 404 (no 403 — no leak existencia).
    4. Verificar guard `if not verified_user_id: 401`.

Drift detection bidireccional:
    - Si alguien revierte el ownership check → falla
      `test_chunk_status_has_ownership_compare`.
    - Si alguien lo cambia a 403 → falla `test_chunk_status_returns_404_on_ownership_mismatch`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — el endpoint "
            f"fue renombrado/eliminado. Si el rename es intencional, "
            f"actualizar este test con el nuevo nombre."
        )
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def chunk_status_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_chunk_status")


def test_chunk_status_has_ownership_compare(chunk_status_body: str):
    """Comparación explícita `user_id != verified_user_id` (con cast
    a str para defender contra UUID vs str). Sin esto, el endpoint
    serializa info de planes ajenos al primer atacante authenticated
    que conozca un plan_id.
    """
    # Aceptamos `str(user_id) != str(verified_user_id)` o
    # `str(plan["user_id"]) != str(verified_user_id)` o cualquier
    # variante con strip/cast. Patrón flexible.
    pattern = re.compile(
        r"str\s*\(\s*\w+(?:\[[^\]]+\])?\s*\)\s*!=\s*str\s*\(\s*verified_user_id\s*\)",
        re.IGNORECASE,
    )
    assert pattern.search(chunk_status_body), (
        "P0-HIST-IDOR-2 regresión: `api_chunk_status` NO compara "
        "el `user_id` leído de meal_plans contra `verified_user_id`. "
        "Sin este check, todo el payload (last_learning_hint con "
        "quality_score del dueño, paused_chunks reason_codes, "
        "tier_breakdown, _user_action_required) se filtra a "
        "cualquier user authenticated que pase un plan_id ajeno. "
        "Patrón requerido: `if str(user_id) != str(verified_user_id): "
        "raise HTTPException(404, ...)` — mismo contrato que "
        "/blocked_reasons (plans.py:3637-3645) y DELETE /{plan_id}."
    )


def test_chunk_status_returns_404_on_ownership_mismatch(chunk_status_body: str):
    """El mismatch del ownership debe lanzar 404 (no 403). Devolver
    403 leak la existencia del plan ajeno — útil para enumeración.
    """
    # Buscar bloque "if str(...) != str(verified_user_id): raise HTTPException(status_code=404"
    # multi-linea con whitespace tolerante. Regex narra: comparison
    # str()!=str(verified_user_id) seguida de raise 404 dentro de la
    # misma rama.
    pattern = re.compile(
        r"str\s*\([^)]+\)\s*!=\s*str\s*\(\s*verified_user_id\s*\)\s*:\s*"
        r"[\r\n]+\s*(?:#[^\r\n]*[\r\n]+\s*)*"
        r"raise\s+HTTPException\s*\(\s*status_code\s*=\s*404",
        re.IGNORECASE | re.DOTALL,
    )
    assert pattern.search(chunk_status_body), (
        "P0-HIST-IDOR-2 regresión: el mismatch del ownership check "
        "no lanza HTTPException(status_code=404). Si lanza 403, "
        "filtramos la existencia del plan ajeno (atacante puede "
        "enumerar plan_ids válidos por la diferencia 403/404). "
        "Si no lanza nada, el bug original sigue vivo."
    )


def test_chunk_status_authenticated_required(chunk_status_body: str):
    """`if not verified_user_id: raise 401` previo al SELECT. Sin
    este guard, un request sin auth (verified_user_id = None) llega
    al SELECT `WHERE id=%s` sin filtro de user, y el ownership
    compare `None != row.user_id` siempre dispara 404 — ofuscando el
    error real (debería ser 401).
    """
    pattern = re.compile(
        r"if\s+not\s+verified_user_id\s*:\s*[\r\n]+\s*"
        r"raise\s+HTTPException\s*\(\s*status_code\s*=\s*401",
        re.IGNORECASE,
    )
    assert pattern.search(chunk_status_body), (
        "P0-HIST-IDOR-2 regresión: falta `if not verified_user_id: "
        "raise HTTPException(401, ...)` al inicio del handler. "
        "Sin este guard temprano, requests sin auth caen al 404 "
        "del ownership compare, lo que es semánticamente incorrecto."
    )


def test_marker_anchor_present():
    """El nombre de este archivo contiene el slug del marker
    `P0-HIST-IDOR-2` para que `test_p2_hist_audit_14_marker_test_link`
    lo correlacione con `_LAST_KNOWN_PFIX` cuando bumpee.
    """
    expected_slug = "p0_hist_idor_2_chunk_status_ownership"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo de test debe contener el slug del "
        "P-fix para que `test_p2_hist_audit_14_marker_test_link` lo "
        "matchee con el marker `_LAST_KNOWN_PFIX` del app.py."
    )
