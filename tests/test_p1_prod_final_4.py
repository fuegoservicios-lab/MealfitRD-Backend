"""[P1-PROD-FINAL-4 · 2026-05-24] Test umbrella + cross-link del bundle.

Bundle 2-en-1 que cierra los 2 P1 backend residuales del audit prod-readiness
2026-05-24 (post P2-EMBED-TELEMETRY, P0 limpio):

  GAP-1 P1-ASYNC-SYNC-DB-BLOCKING:
    Handlers `async def` en routers/preferences.py, routers/billing.py,
    routers/plans.py ejecutaban funciones DB **síncronas** sin envolver
    en `asyncio.to_thread`, bloqueando el event loop ~10-200ms por
    roundtrip Supabase. Mismo modo de fallo que P2-AUTH-ASYNC-SLEEP
    cerró para `auth.py`. Bajo carga (≥50 req/s) → throttling severo
    de todos los demás handlers async (chat stream, webhook PayPal,
    diary upload).

  GAP-2 P1-CHAT-TTS-TIMEOUT-HARDCODED:
    `routers/chat.py:493` exhibía `httpx.AsyncClient(timeout=15.0)`
    literal. El test blanket `test_p1_new_httpx_timeout.py` solo cubría
    `billing.py` y no detectó este gap. Sin rollback sin redeploy si
    ElevenLabs degrada latencia.

Este test ancla el bundle (anchor + marker) y delega las verificaciones
detalladas a:
  - `test_p1_new_httpx_timeout.py` (extendido a chat.py en este bundle).
  - Verificaciones inline acá del `asyncio.to_thread` wrapping en los 3
    routers tocados.

Cross-link guard P2-HIST-AUDIT-14: el slug `p1_prod_final_4` matchea
el marker `P1-PROD-FINAL-4`.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_APP_PY = _BACKEND / "app.py"
_PREFERENCES = _BACKEND / "routers" / "preferences.py"
_BILLING = _BACKEND / "routers" / "billing.py"
_PLANS = _BACKEND / "routers" / "plans.py"
_CHAT = _BACKEND / "routers" / "chat.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — Anchors presentes en los 3 routers tocados + chat.py
# ---------------------------------------------------------------------------
def test_anchor_async_sync_in_preferences():
    src = _read(_PREFERENCES)
    assert "P1-ASYNC-SYNC-DB-BLOCKING" in src, (
        "Falta anchor `P1-ASYNC-SYNC-DB-BLOCKING` en backend/routers/preferences.py. "
        "Sin anchor, un futuro reader que vea `await asyncio.to_thread(...)` no "
        "sabrá el modo de fallo que cierra (event loop bloqueado por DB sync)."
    )


def test_anchor_async_sync_in_billing():
    src = _read(_BILLING)
    assert "P1-ASYNC-SYNC-DB-BLOCKING" in src, (
        "Falta anchor `P1-ASYNC-SYNC-DB-BLOCKING` en backend/routers/billing.py."
    )


def test_anchor_async_sync_in_plans():
    src = _read(_PLANS)
    assert "P1-ASYNC-SYNC-DB-BLOCKING" in src, (
        "Falta anchor `P1-ASYNC-SYNC-DB-BLOCKING` en backend/routers/plans.py."
    )


def test_anchor_tts_timeout_in_chat():
    src = _read(_CHAT)
    assert "P1-CHAT-TTS-TIMEOUT-HARDCODED" in src, (
        "Falta anchor `P1-CHAT-TTS-TIMEOUT-HARDCODED` en backend/routers/chat.py."
    )


# ---------------------------------------------------------------------------
# Sección 2 — `asyncio` importado + helper/uso en los 3 routers
# ---------------------------------------------------------------------------
def test_asyncio_imported_in_preferences():
    src = _read(_PREFERENCES)
    assert re.search(r"^import asyncio\b", src, re.MULTILINE), (
        "preferences.py debe importar `asyncio` para usar `to_thread`."
    )


def test_asyncio_imported_in_billing():
    src = _read(_BILLING)
    assert re.search(r"^import asyncio\b", src, re.MULTILINE), (
        "billing.py debe importar `asyncio` para usar `to_thread`."
    )


def test_to_thread_used_in_preferences():
    """Los 4 handlers async (memory get/set + water tracker get/set) deben
    usar `asyncio.to_thread` para cada call DB sync."""
    src = _read(_PREFERENCES)
    matches = re.findall(r"await asyncio\.to_thread\(", src)
    assert len(matches) >= 4, (
        f"preferences.py debe tener ≥4 callsites `await asyncio.to_thread(...)` "
        f"(memory get/set + water tracker get/set). Encontrados: {len(matches)}."
    )


def test_run_sync_db_in_thread_helper_in_billing():
    """billing.py declara `_run_sync_db_in_thread` helper para wrappear los thunks
    DB sync desde handlers async.

    [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: los thunks ya no son
    `supabase.table(...).execute()` (PostgREST) sino lambdas sobre
    `execute_sql_query`/`execute_sql_write` (SQL directo). El helper se
    conserva con el mismo nombre (tooltip-anchor) y la misma propiedad:
    ningún roundtrip DB inline en el event loop."""
    src = _read(_BILLING)
    assert re.search(
        r"async def _run_sync_db_in_thread\(.*?\):\s*\n\s*return await asyncio\.to_thread",
        src,
        re.DOTALL,
    ), (
        "billing.py debe declarar `async def _run_sync_db_in_thread(thunk):` que "
        "delegue a `asyncio.to_thread`. Sin este helper los callsites quedan "
        "verbose y un nuevo callsite olvidaría el wrap fácilmente."
    )
    # Callsites DB despachados via helper desde handlers async. Mínimo
    # esperado: 7 (mismo floor que el audit original — 6 callsites + 1
    # boy-scout /discount/validate). El helper sync `_persist_billing_alert`
    # escribe directo (no async handler), igual que pre-migración.
    wrapped = re.findall(r"await _run_sync_db_in_thread\(", src)
    assert len(wrapped) >= 7, (
        f"billing.py debe tener ≥7 callsites `await _run_sync_db_in_thread(...)`. "
        f"Encontrados: {len(wrapped)}."
    )
    # El transporte PostgREST quedó eliminado fail-loud del módulo: cero
    # callsites `supabase.table(` residuales.
    assert "supabase.table(" not in src, (
        "billing.py no debe conservar callsites PostgREST `supabase.table(` "
        "tras P1-NEON-DB-MIGRATION (los fallbacks REST fueron eliminados "
        "fail-loud)."
    )


def test_to_thread_used_in_plans_pending_pipeline_and_depleted():
    """plans.py debe tener ≥5 callsites async.to_thread (pending-status,
    ack, /depleted-items GET, POST, DELETE) post-bundle."""
    src = _read(_PLANS)
    # Heurística: cada uno de los 5 handlers fue tocado. Verificamos
    # que el patrón `await asyncio.to_thread(` aparezca cerca de los
    # nombres de las funciones DB target.
    db_fns = [
        "get_pending_pipeline",
        "clear_pending_pipeline",
        "list_depleted_items",
        "bulk_upsert_depleted_items",
        "delete_depleted_item",
    ]
    for fn in db_fns:
        # Match `await asyncio.to_thread(<fn>` con espacios opcionales
        pat = re.compile(
            rf"await\s+asyncio\.to_thread\(\s*{re.escape(fn)}\b"
        )
        assert pat.search(src), (
            f"plans.py debe envolver `{fn}` con `await asyncio.to_thread(...)`. "
            f"Sin esto el handler async bloquea event loop por ~10-200ms."
        )


# ---------------------------------------------------------------------------
# Sección 3 — Marker bumped
# ---------------------------------------------------------------------------
def test_marker_bumped_to_p1_prod_final_4():
    """[Relajado por P1-FRONTEND-FINAL-1 · 2026-05-24] Sibling date-floor:
    el marker debe tener fecha >= 2026-05-24 (día del cierre del bundle
    P1-PROD-FINAL-4). Exact-match removido tras supersede del Bundle #2
    frontend del mismo día — patrón emergente desde P1-PROD-FINAL-1."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor 2026-05-24 "
        f"(día del cierre P1-PROD-FINAL-4)."
    )


def test_marker_date_meets_p1_prod_final_4_floor():
    """Date-floor sibling para futuros supersedes."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor}."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Cross-link guard P2-HIST-AUDIT-14
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    """El slug del marker `p1_prod_final_4` matchea el archivo. El cross-link
    enforcer `test_p2_hist_audit_14_marker_test_link.py` verifica que el slug
    del marker DEBE matchear al menos un archivo `tests/test_<slug>*.py`."""
    src = _read(Path(__file__))
    assert "P1-PROD-FINAL-4" in src
