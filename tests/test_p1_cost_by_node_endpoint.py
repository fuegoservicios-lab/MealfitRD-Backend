"""[P1-COST-BY-NODE-ENDPOINT · 2026-05-16] Admin endpoint
`GET /api/system/admin/cost-by-node` que agrega `llm_usage_events` para
darle visibilidad consumible a la columna `node` populada por
P1-COST-INSTRUMENTATION-PHASE2.

Tests anclan:
  1. Endpoint registrado en `routers/system.py` con método GET.
  2. Auth gateado por `_verify_admin_token` (CRON_SECRET) + rate limit.
  3. Query SQL usa `make_interval(hours => %s)` (parametrizado correcto,
     no f-string que abriría injection) + `COALESCE(node, '(unattributed)')`
     para visibilizar rows sin atribución.
  4. Response shape: total_calls, total_usd, total_cached_pct, unattributed,
     by_node (ordenado por costo desc).
  5. Clamp del query param `hours` a [1, 720] (defensa contra full-scan).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SYSTEM_PY = _BACKEND_ROOT / "routers" / "system.py"


def test_endpoint_registered_get_admin_cost_by_node():
    """Endpoint registrado con método GET en routers/system.py."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    assert '@router.get("/admin/cost-by-node")' in src, (
        "Endpoint GET /admin/cost-by-node no registrado."
    )
    assert "def admin_cost_by_node(request: Request" in src, (
        "Función handler `admin_cost_by_node` no encontrada."
    )


def test_endpoint_uses_admin_auth_and_rate_limit():
    """Gateado por `_verify_admin_token` + `_check_admin_rate_limit`,
    consistente con el resto de admin endpoints."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def admin_cost_by_node\(.*?(?=\n@router\.|\Z)",
        src,
        re.DOTALL,
    )
    assert fn, "Cuerpo de `admin_cost_by_node` no aislable."
    body = fn.group(0)
    assert "_verify_admin_token(request.headers.get(\"authorization\"))" in body, (
        "Falta verificación de Bearer CRON_SECRET — endpoint admin sin auth."
    )
    assert "_check_admin_rate_limit(request)" in body, (
        "Falta rate limit admin — atacante con token podría spammear queries pesadas."
    )


def test_query_uses_parametrized_interval():
    """`make_interval(hours => %s)` es parametrizado seguro. Cualquier
    f-string con la ventana embebida sería SQL injection."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def admin_cost_by_node\(.*?(?=\n@router\.|\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    assert "make_interval(hours => %s)" in body, (
        "Query no usa `make_interval(hours => %s)` parametrizado. "
        "Verifica que no haya f-string embebiendo `hours` (SQL injection)."
    )
    # Negative: el parámetro NO debe estar embebido como f-string.
    assert re.search(r'NOW\(\)\s*-\s*INTERVAL\s*f"', body) is None, (
        "Detectado `INTERVAL f\"...\"` — abre SQL injection. Usar "
        "`make_interval(hours => %s)` con tuple param."
    )


def test_query_uses_coalesce_node():
    """`COALESCE(node, '(unattributed)')` visibiliza calls fuera del pipeline
    (agent tools, scripts, pre-phase 2) sin omitirlas del agregado."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    assert "COALESCE(node, '(unattributed)')" in src, (
        "Query no usa COALESCE para `node` — rows con NULL desaparecerían "
        "del leaderboard, perdiendo información de calls sin atribución."
    )


def test_hours_param_clamp():
    """`hours` debe clampearse a [1, 720] para evitar full-scan accidental.
    720h = 30 días, máximo razonable para no escanear toda la tabla."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def admin_cost_by_node\(.*?(?=\n@router\.|\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    assert "max(1, min(720, int(hours)" in body, (
        "Falta clamp del query param `hours` a [1, 720]. Sin clamp, un "
        "atacante puede forzar `hours=999999` y disparar full-scan."
    )


def test_response_includes_aggregate_keys():
    """Response debe incluir TODAS las keys agregadas que el endpoint promete
    en su docstring — sin esto el caller no puede confiar en el shape."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def admin_cost_by_node\(.*?(?=\n@router\.|\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    expected_keys = [
        '"success"',
        '"window_hours"',
        '"total_calls"',
        '"total_usd"',
        '"total_cached_pct"',
        '"unattributed"',
        '"by_node"',
    ]
    for key in expected_keys:
        assert key in body, (
            f"Response no incluye {key} — contrato roto vs docstring."
        )


def test_response_orders_by_total_usd_desc():
    """SQL ORDER BY `cost_micros_sum DESC` para que el primer elemento del
    `by_node` sea siempre el nodo más caro. SRE espera ese orden."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def admin_cost_by_node\(.*?(?=\n@router\.|\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    assert "ORDER BY cost_micros_sum DESC" in body, (
        "Query no ordena por costo desc — el endpoint no sería un "
        "leaderboard si el SRE tiene que ordenar manualmente."
    )


def test_cost_usd_micros_converted_to_dollars_in_response():
    """Costo se guarda en μUSD (int) pero el response debe exponer USD (float)
    para que el SRE no tenga que dividir por 1M mentalmente."""
    src = _SYSTEM_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def admin_cost_by_node\(.*?(?=\n@router\.|\Z)",
        src,
        re.DOTALL,
    )
    assert fn
    body = fn.group(0)
    # División por 1M para convertir μUSD → USD
    assert "/ 1_000_000.0" in body, (
        "Response no convierte μUSD a USD. SRE vería '4831027' en lugar "
        "de '$4.83' — costo cognitivo innecesario."
    )
