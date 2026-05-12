"""[P1-OBS-1 · 2026-05-12] Tests parser-based para `GET /api/system/admin/health-snapshot`.

Anchor: P1-OBS-1-HEALTH-SNAPSHOT.

El endpoint es contrato con SRE / dashboards (Grafana, /admin scripts). Estos
tests anclan tres garantías load-bearing:

1. **Auth**: el endpoint invoca `_verify_admin_token` antes de cualquier query.
   Sin esta defensa, cualquier visitante podría enumerar circuit breakers y
   alerts (leak de telemetría operacional + posible IDOR-ish).
2. **Cobertura de keys**: las 8 keys del contrato siempre se serializan
   (incluso si las sub-queries fallan, valor `None`/`[]`/`{}`). Si alguien
   borra una key, dashboards se rompen silenciosamente.
3. **Nodos observados**: `_HEALTH_SNAPSHOT_WATCHDOG_NODES` lista los 3 ticks
   que prod monitorea (`_hardfloor_autoheal_tick`, `_hot_table_bloat_tick`,
   `_pipeline_metrics_silence_check_tick`). Si se borra uno sin actualizar
   el test, deploy-lag invisible.
"""
from pathlib import Path
import re


_ROUTER_PATH = Path(__file__).resolve().parents[1] / "routers" / "system.py"


def _read_router_source() -> str:
    return _ROUTER_PATH.read_text(encoding="utf-8")


def test_health_snapshot_anchor_present():
    """El marker P1-OBS-1-HEALTH-SNAPSHOT debe estar en el código para
    que P2-HIST-AUDIT-14 (cross-link marker-test) no genere falso negativo
    y para que un futuro grep `P1-OBS-1` aterrice acá."""
    src = _read_router_source()
    assert "P1-OBS-1" in src, "Anchor P1-OBS-1 ausente en routers/system.py"
    assert "P1-OBS-1-HEALTH-SNAPSHOT" in src, (
        "Anchor extendido P1-OBS-1-HEALTH-SNAPSHOT ausente — necesario "
        "para que SRE encuentre el endpoint vía grep."
    )


def test_health_snapshot_endpoint_decorator_present():
    """Verificar que el endpoint está registrado en el router como
    `GET /admin/health-snapshot` (prefix `/api/system` añadido por el
    APIRouter del módulo)."""
    src = _read_router_source()
    assert re.search(
        r'@router\.get\(\s*["\']/admin/health-snapshot["\']\s*\)',
        src,
    ), (
        "Decorator @router.get('/admin/health-snapshot') ausente — el endpoint "
        "P1-OBS-1 fue renombrado o eliminado. Si lo moviste a otra ruta, "
        "actualizá este test + dashboards externos."
    )


def test_health_snapshot_requires_admin_token():
    """El handler DEBE invocar `_verify_admin_token` antes de cualquier
    SELECT — si no, el endpoint expone telemetría operacional sin auth.

    Escaneamos solo el cuerpo de `admin_health_snapshot`, no todo el archivo
    (que tiene 3+ endpoints admin más). Buscamos `_verify_admin_token` en
    las primeras ~20 líneas del handler.
    """
    src = _read_router_source()
    handler_match = re.search(
        r"def admin_health_snapshot\(request: Request\):(.*?)(?=^def |^@router\.|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert handler_match, "Función admin_health_snapshot no encontrada"
    body = handler_match.group(1)
    # Las primeras líneas son docstring; el verify debe estar antes de
    # cualquier execute_sql_query (best-effort blocks abajo).
    verify_idx = body.find("_verify_admin_token(")
    sql_idx = body.find("execute_sql_query(")
    assert verify_idx != -1, (
        "admin_health_snapshot NO invoca _verify_admin_token — endpoint sin "
        "auth filtraría operacional info sin Bearer CRON_SECRET."
    )
    assert sql_idx == -1 or verify_idx < sql_idx, (
        "_verify_admin_token debe aparecer ANTES del primer execute_sql_query "
        "para que el 401/403 corte antes de tocar la DB."
    )


def test_health_snapshot_returns_all_contract_keys():
    """Las 8 keys del contrato (más `success`) deben construirse y
    serializarse en el return final. Test es regex sobre el dict literal,
    no runtime — atrapa borrado accidental al refactorizar.
    """
    src = _read_router_source()
    handler_match = re.search(
        r"def admin_health_snapshot\(request: Request\):(.*?)(?=^def |^@router\.|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert handler_match
    body = handler_match.group(1)
    return_match = re.search(r"return\s*\{(.*?)\}\s*$", body, re.DOTALL | re.MULTILINE)
    assert return_match, "No se encontró return dict en admin_health_snapshot"
    ret = return_match.group(1)
    required_keys = (
        '"success"',
        '"live_marker"',
        '"expected_marker"',
        '"drift"',
        '"open_alerts"',
        '"metrics_15min"',
        '"watchdog_ticks"',
        '"stuck_chunks"',
        '"dead_lettered_chunks"',
        '"open_circuit_breakers"',
    )
    missing = [k for k in required_keys if k not in ret]
    assert not missing, (
        f"Keys faltantes en return del health-snapshot: {missing}. "
        f"Contrato con Grafana/SRE roto."
    )


def test_health_snapshot_watchdog_nodes_constant():
    """`_HEALTH_SNAPSHOT_WATCHDOG_NODES` lista los 3 ticks observables.
    Si alguien borra uno, deploy-lag invisible queda fuera del snapshot.

    Los 3 nodos canónicos al cierre P1-OBS-1:
      - `_hardfloor_autoheal_tick` (tick principal — su silencio = deploy lag)
      - `_hot_table_bloat_tick` (P2-AUDIT-2)
      - `_pipeline_metrics_silence_check_tick` (P2-OBSERVABILITY-1)
    """
    src = _read_router_source()
    assert "_HEALTH_SNAPSHOT_WATCHDOG_NODES" in src, (
        "Constante _HEALTH_SNAPSHOT_WATCHDOG_NODES ausente — el endpoint "
        "P1-OBS-1 necesita la lista canónica."
    )
    required_nodes = (
        '"_hardfloor_autoheal_tick"',
        '"_hot_table_bloat_tick"',
        '"_pipeline_metrics_silence_check_tick"',
    )
    tuple_match = re.search(
        r"_HEALTH_SNAPSHOT_WATCHDOG_NODES\s*=\s*\((.*?)\)",
        src,
        re.DOTALL,
    )
    assert tuple_match, "_HEALTH_SNAPSHOT_WATCHDOG_NODES no definido como tuple"
    tuple_body = tuple_match.group(1)
    missing = [n for n in required_nodes if n not in tuple_body]
    assert not missing, (
        f"Nodos faltantes en _HEALTH_SNAPSHOT_WATCHDOG_NODES: {missing}. "
        f"Si quitaste alguno deliberadamente (e.g., watchdog renombrado), "
        f"actualizá este test + la docstring del endpoint en el mismo PR."
    )


def test_health_snapshot_best_effort_per_block():
    """Cada bloque de queries debe estar en try/except — un blip transitorio
    en una tabla (lock conflict, tipo inesperado) NO debe tumbar el endpoint
    entero. Si alguien refactoriza y consolida en un solo try, fallamos.
    """
    src = _read_router_source()
    handler_match = re.search(
        r"def admin_health_snapshot\(request: Request\):(.*?)(?=^def |^@router\.|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert handler_match
    body = handler_match.group(1)
    try_count = len(re.findall(r"^\s+try:\s*$", body, re.MULTILINE))
    assert try_count >= 5, (
        f"Esperamos ≥5 bloques try independientes en admin_health_snapshot "
        f"(uno por sub-query: live_marker, expected_marker, open_alerts, "
        f"metrics_15min, watchdog_ticks, plan_chunk_queue, circuit_breakers); "
        f"encontrados: {try_count}. Si consolidaste, romperás la resiliencia "
        f"best-effort del endpoint."
    )
