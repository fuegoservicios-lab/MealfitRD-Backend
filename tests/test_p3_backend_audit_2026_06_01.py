"""[P3-BACKEND-AUDIT · 2026-06-01] Tests de regresión parser-based para el
bundle de auditoría de backend (velocidad + seguridad).

4 gaps P3 confirmados por la auditoría (find→verify adversarial + análisis
determinista de `pg_stat_statements`/logs de prod sobre la DB viva):

  1. P3-LESSON-TELEMETRY-EXISTS — los 2 INSERT de `chunk_lesson_telemetry`
     en cron_tasks.py degradan a `INSERT ... SELECT ... WHERE EXISTS` en
     lugar de `VALUES`. Cierra el ERROR recurrente de Postgres en prod
     (`violates foreign key constraint chunk_lesson_telemetry_meal_plan_id_fkey`
     observado decenas de veces/24h) cuando el parent `meal_plans` fue borrado
     mid-flight: ahora el INSERT afecta 0 filas en silencio, sin línea ERROR
     a nivel DB ni re-buffer.

  2. P3-HEALTH-LEAK-STRIP — el endpoint PÚBLICO `GET /api/system/health/plan-graph`
     ya no filtra `type(e).__name__: {e}` al cliente en su rama except (info-disclosure
     de símbolos/módulos vía ImportError). El logger ya preserva el detalle full
     server-side. Consistente con safe_error_detail / P2-HEALTH-UID-STRIP / P3-READY-REASON.

  3. P3-NOTIF-EVENTLOOP — `subscribe_push` es `def` plano y `unsubscribe_push`
     offloadea su psycopg síncrono vía `asyncio.to_thread`, evitando bloquear el
     event loop (anti-patrón `async def` + I/O bloqueante directo).

  4. P3-RESTOCK-BULK-DELETE — `/restock` borra los depleted items repuestos con un
     único bulk DELETE (`bulk_delete_depleted_items`, `= ANY(%s)`) en lugar del loop
     N+1 de `delete_depleted_item` por nombre.

Parser-based (regex sobre source) — no importa los módulos para evitar disparar
schedulers/DB init y para correr sin el entorno completo (langgraph/apscheduler).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"
_SYSTEM = _BACKEND / "routers" / "system.py"
_NOTIF = _BACKEND / "routers" / "notifications.py"
_DB_INV = _BACKEND / "db_inventory.py"
_PLANS = _BACKEND / "routers" / "plans.py"
_APP = _BACKEND / "app.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. P3-LESSON-TELEMETRY-EXISTS
# ---------------------------------------------------------------------------
def test_telemetry_inserts_use_where_exists_guard():
    """Ambos INSERT de chunk_lesson_telemetry usan el guard WHERE EXISTS."""
    src = _read(_CRON)
    # El guard exacto contra el parent meal_plans, presente en los 2 sitios
    # (flush buffer + _record helper).
    n = len(re.findall(
        r"WHERE EXISTS \(SELECT 1 FROM meal_plans WHERE id = %s\)", src
    ))
    assert n >= 2, (
        f"Esperaba ≥2 guards `WHERE EXISTS (SELECT 1 FROM meal_plans WHERE id = %s)` "
        f"en cron_tasks.py (flush + _record); encontré {n}."
    )
    assert "P3-LESSON-TELEMETRY-EXISTS" in src, "anchor P3-LESSON-TELEMETRY-EXISTS ausente"


def test_telemetry_inserts_no_longer_use_bare_values():
    """Anti-regresión: ningún INSERT a chunk_lesson_telemetry vuelve al patrón
    `VALUES (...)` bare (que disparaba el FK ERROR cuando el plan fue borrado)."""
    src = _read(_CRON)
    # El patrón viejo: INSERT INTO chunk_lesson_telemetry ... VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb
    bare = re.search(
        r"INSERT INTO chunk_lesson_telemetry[\s\S]{0,260}?VALUES \(%s, %s, %s, %s, %s, %s, %s::jsonb",
        src,
    )
    assert bare is None, (
        "Se detectó un INSERT INTO chunk_lesson_telemetry con `VALUES (...)` bare. "
        "Debe usar `SELECT ... WHERE EXISTS (SELECT 1 FROM meal_plans WHERE id = %s)` "
        "para no violar el FK cuando el parent fue borrado mid-flight."
    )


# ---------------------------------------------------------------------------
# 2. P3-HEALTH-LEAK-STRIP
# ---------------------------------------------------------------------------
def _plan_graph_health_body(src: str) -> str:
    m = re.search(
        r"def get_plan_graph_health\(\):([\s\S]+?)(?=\n\nclass |\n@router\.)",
        src,
    )
    assert m is not None, "no se pudo aislar get_plan_graph_health()"
    return m.group(1)


def test_public_health_endpoint_does_not_leak_exception_type():
    """La rama except del endpoint público no interpola `type(e)` en el detail
    que viaja al cliente."""
    body = _plan_graph_health_body(_read(_SYSTEM))
    assert "type(e).__name__" not in body, (
        "get_plan_graph_health() (endpoint PÚBLICO) filtra `type(e).__name__` al "
        "cliente. Mantener el detalle solo en logger.error server-side."
    )
    assert "P3-HEALTH-LEAK-STRIP" in body, "anchor P3-HEALTH-LEAK-STRIP ausente en la función"
    assert "detalle en logs del servidor" in body, (
        "el mensaje genérico esperado no está presente"
    )


def test_public_health_endpoint_still_logs_full_detail_server_side():
    """Defensa-en-profundidad: el detalle full se sigue logueando server-side."""
    body = _plan_graph_health_body(_read(_SYSTEM))
    assert re.search(r"logger\.error\(f\"\[P1-9\][^\n]*\{e\}", body), (
        "el logger.error que preserva el detalle full server-side desapareció"
    )


# ---------------------------------------------------------------------------
# 3. P3-NOTIF-EVENTLOOP
# ---------------------------------------------------------------------------
def test_subscribe_push_is_sync_def():
    src = _read(_NOTIF)
    assert "async def subscribe_push(" not in src, (
        "subscribe_push debe ser `def` plano (FastAPI lo threadpool-ea); como "
        "`async def` con psycopg síncrono bloquearía el event loop."
    )
    assert "def subscribe_push(" in src, "subscribe_push no encontrado"


def test_unsubscribe_push_offloads_db_to_thread():
    src = _read(_NOTIF)
    assert "import asyncio" in src, "falta `import asyncio`"
    assert "def _delete_push_subscription_sync(" in src, (
        "helper síncrono _delete_push_subscription_sync ausente"
    )
    assert "asyncio.to_thread(_delete_push_subscription_sync" in src, (
        "unsubscribe_push debe offloadear el DELETE vía asyncio.to_thread"
    )
    assert "P3-NOTIF-EVENTLOOP" in src, "anchor P3-NOTIF-EVENTLOOP ausente"


# ---------------------------------------------------------------------------
# 4. P3-RESTOCK-BULK-DELETE
# ---------------------------------------------------------------------------
def test_bulk_delete_depleted_items_helper_exists():
    src = _read(_DB_INV)
    m = re.search(
        r"def bulk_delete_depleted_items\([\s\S]+?(?=\ndef )",
        src,
    )
    assert m is not None, "bulk_delete_depleted_items no encontrado en db_inventory.py"
    body = m.group(0)
    assert "= ANY(%s)" in body, "el bulk delete debe usar `= ANY(%s)` (un solo round-trip)"
    assert "lower(trim(ingredient_name))" in body, (
        "debe preservar el match case-insensitive vía lower(trim(...))"
    )
    assert "WHERE user_id = %s" in body, "filtro user_id (I2) ausente en el bulk delete"
    assert "P3-RESTOCK-BULK-DELETE" in body, "anchor P3-RESTOCK-BULK-DELETE ausente"


def test_restock_uses_bulk_delete_not_loop():
    src = _read(_PLANS)
    assert "from db_inventory import bulk_delete_depleted_items" in src, (
        "/restock debe importar bulk_delete_depleted_items"
    )
    assert "bulk_delete_depleted_items(user_id, _names_to_clear)" in src, (
        "/restock debe invocar el bulk delete con _names_to_clear"
    )
    # Anti-regresión: el loop N+1 en el bloque de cleanup de restock no debe volver.
    assert "for _nm in _names_to_clear:" not in src, (
        "El loop N+1 `for _nm in _names_to_clear` reapareció en /restock; usar bulk delete."
    )


# ---------------------------------------------------------------------------
# 5. Marker cross-link
# ---------------------------------------------------------------------------
def test_marker_bumped_to_this_bundle():
    """El marker _LAST_KNOWN_PFIX existe y es posterior (o igual) a este bundle.

    `_LAST_KNOWN_PFIX` es el marker del ÚLTIMO P-fix mergeado en HEAD (ver
    CLAUDE.md → 'Convenciones del repo'); por diseño lo sobreescribe cada
    P-fix posterior. Por eso este test NO puede congelar el marker en
    `P3-BACKEND-AUDIT · ` — cuando este bundle se mergeó ESE era el último,
    pero P-fixes posteriores (actual: el bundle de triage de junio) ya lo
    bumpearon legítimamente. La aserción durable es: el marker está presente,
    bien formado (`Pn-... · YYYY-MM-DD`), y su fecha >= la de este bundle
    (2026-06-01) — lo que prueba que NO fue revertido a un marker más viejo.
    El formato/floor canónico lo enforza `test_p3_1_last_known_pfix_freshness.py`."""
    from datetime import datetime, date

    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _read(_APP)
    )
    assert m is not None, "_LAST_KNOWN_PFIX no encontrado"
    marker = m.group(1)
    fmt = re.match(r"^P\d+(?:-[A-Z0-9]+)+\s+·\s+(\d{4}-\d{2}-\d{2})$", marker)
    assert fmt is not None, (
        f"marker actual = {marker!r}; esperaba formato 'Pn-X · YYYY-MM-DD'."
    )
    marker_date = datetime.strptime(fmt.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 6, 1), (
        f"marker actual = {marker!r} con fecha {marker_date} < 2026-06-01 "
        f"(fecha de este bundle P3-BACKEND-AUDIT). El marker fue revertido a "
        f"un P-fix más viejo que este bundle."
    )
