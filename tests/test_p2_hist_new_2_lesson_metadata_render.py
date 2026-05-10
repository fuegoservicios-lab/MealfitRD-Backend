"""[P2-HIST-NEW-2 · 2026-05-09] Tests del campo `metadata` en el
endpoint `/{plan_id}/lessons` y su render en el tab Lecciones.

Bug original (audit profundo Historial 2026-05-09):
    `chunk_lesson_telemetry` tiene un campo `metadata` jsonb donde
    los crons escriben contexto arbitrario por lección
    (`{score: 85, threshold: 50}`, `{retries: 3, error: "..."}`,
    etc.). El endpoint /lessons ya devolvía el campo pero el frontend
    lo descartaba — diagnóstico potencial perdido en el modal del
    Historial.

Fix:
    - Backend: coerción defensiva (no-dict → None) para garantizar
      que el frontend reciba un shape estable.
    - Frontend: render de hasta 3 chips inline `key: value` con
      sanitización por tipo (number/boolean/string/object) +
      truncate. Si hay >3 keys, chip "+N más" con JSON completo en
      el title= tooltip.

Cobertura backend:
    1. Anchor del marker en endpoint Y en History.jsx.
    2. SQL del SELECT incluye `metadata`.
    3. Response shape: `metadata: dict|null`.
    4. Coerción: non-dict (string, list, etc.) → None.
    5. dict vacío → preservar como dict vacío (frontend skipea por
       Object.entries length=0).
    6. Pass-through de dict válido sin mutación.
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


_USER_A = "11111111-1111-1111-1111-111111111111"
_PLAN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _build_test_client():
    from routers.plans import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _base_lesson_row(**overrides):
    base = {
        "id": "1",
        "event": "rejected_meal_synthesized",
        "week_number": 1,
        "synthesized_count": 1,
        "queue_count": 0,
        "metadata": None,
        "created_at": datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Anchors
# ---------------------------------------------------------------------------
def test_marker_present_in_endpoint():
    from routers.plans import api_plan_lessons_detail
    src = inspect.getsource(api_plan_lessons_detail)
    assert "P2-HIST-NEW-2" in src, (
        "Endpoint /lessons debe citar `P2-HIST-NEW-2` para que un grep "
        "+ git blame lleve directo al fix de la coerción defensiva."
    )


def test_marker_present_in_history_jsx():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    history_jsx = repo_root / "frontend" / "src" / "pages" / "History.jsx"
    assert history_jsx.exists()
    text = history_jsx.read_text(encoding="utf-8")
    assert "[P2-HIST-NEW-2" in text


# ---------------------------------------------------------------------------
# 2. SQL contract
# ---------------------------------------------------------------------------
def test_sql_select_includes_metadata():
    """SELECT debe pedir `metadata` para que el row arrive al handler.
    Sin esto, el response devuelve `metadata: None` siempre y el render
    nunca se dispara."""
    from routers.plans import api_plan_lessons_detail
    src = inspect.getsource(api_plan_lessons_detail)
    norm = re.sub(r"\s+", " ", src)
    assert re.search(
        r"SELECT[\s\S]*?metadata[\s\S]*?FROM\s+chunk_lesson_telemetry",
        norm,
        re.IGNORECASE,
    ), "SQL SELECT debe incluir el campo `metadata`."


# ---------------------------------------------------------------------------
# 3. Response shape: pass-through de dict válido
# ---------------------------------------------------------------------------
def test_response_passes_through_dict_metadata():
    """Pass-through: dict no-vacío llega tal cual al response."""
    fake_meta = {"score": 85, "threshold": 50, "tier": "high"}
    fake_row = _base_lesson_row(metadata=fake_meta)

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/lessons")
    assert r.status_code == 200, r.text
    body = r.json()
    lessons = body["lessons"]
    assert len(lessons) == 1
    assert lessons[0]["metadata"] == fake_meta


def test_response_handles_none_metadata():
    """Plan con lecciones sin metadata → metadata=None viaja como
    null. Frontend skipea el render."""
    fake_row = _base_lesson_row(metadata=None)

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/lessons")
    assert r.status_code == 200
    assert r.json()["lessons"][0]["metadata"] is None


# ---------------------------------------------------------------------------
# 4. Coerción defensiva: tipos no-dict → None
# ---------------------------------------------------------------------------
def test_response_coerces_string_metadata_to_none():
    """Caso edge: row con metadata=string (jsonb que serializó como
    text). El handler debe coercer a None — el frontend asume dict
    y rompería con un string."""
    fake_row = _base_lesson_row(metadata="raw_string_legacy_corrupted")

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/lessons")
    assert r.status_code == 200
    assert r.json()["lessons"][0]["metadata"] is None


def test_response_coerces_list_metadata_to_none():
    """Caso edge: list jsonb (no esperado pero posible). Coerción a
    None."""
    fake_row = _base_lesson_row(metadata=["a", "b"])

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/lessons")
    assert r.status_code == 200
    assert r.json()["lessons"][0]["metadata"] is None


def test_response_preserves_empty_dict():
    """dict vacío {} se preserva (no coerce a None). Es válido
    semánticamente — significa "metadata escrito pero sin keys".
    Frontend skipea el render por Object.entries.length === 0."""
    fake_row = _base_lesson_row(metadata={})

    def _fake(query, params=None, **kwargs):
        if "FROM meal_plans WHERE id" in query:
            return {"id": _PLAN_A}
        return [fake_row]

    client = _build_test_client()
    from auth import verify_api_quota
    client.app.dependency_overrides[verify_api_quota] = lambda: _USER_A

    with patch("db_core.execute_sql_query", side_effect=_fake):
        r = client.get(f"/api/plans/{_PLAN_A}/lessons")
    assert r.status_code == 200
    # Empty dict preserved (handler check is `not isinstance(_meta, dict)`,
    # which is False for {}).
    assert r.json()["lessons"][0]["metadata"] == {}
