"""[P1-CHANGE-OUTCOME-TELEMETRY · 2026-08-05] Cada cambio que pide el usuario deja
una fila con su desenlace.

POR QUÉ EXISTE. Inventario de `pipeline_metrics` el 2026-08-05: 18.644 filas de
`coherence_guard_validation`, 2.727 de `review_plan`… y **CERO** de swap o de
regeneración de día. El desenlace de cada cambio vivía solo en el journal del VPS,
que rota — así que la pregunta que un landing querría responder («¿los cambios de
plato salen al primer intento, y cuándo no, por qué?») no tenía datos, y cada
sesión de uso se evaporaba.

Guarda el DESENLACE, no el intento: una operación = una fila. `error_code` dice
por qué falló (despensa / macros / retries agotados / proveedor caído), que es lo
que convierte un porcentaje en algo accionable.

tooltip-anchor: P1-CHANGE-OUTCOME-TELEMETRY
"""
import io
import json
import re
from pathlib import Path

_PLANS = Path(__file__).resolve().parents[1] / "routers" / "plans.py"


def _src() -> str:
    return io.open(_PLANS, encoding="utf-8").read()


def _sin_comentarios(bloque: str) -> str:
    return "\n".join(l for l in bloque.splitlines() if not l.lstrip().startswith("#"))


def test_existe_el_emisor():
    assert "def _emit_change_outcome_metric(" in _src()


def test_escribe_a_pipeline_metrics_con_node_propio():
    src = _src()
    i = src.index("def _emit_change_outcome_metric(")
    bloque = _sin_comentarios(src[i:i + 2200])
    assert "INSERT INTO pipeline_metrics" in bloque
    assert 'f"change_{kind}"' in bloque, (
        "El node debe identificar el tipo de cambio para poder agrupar sin parsear texto."
    )


def test_es_best_effort():
    """Telemetría que puede tumbar un cambio de plato es peor que no tenerla."""
    src = _src()
    i = src.index("def _emit_change_outcome_metric(")
    assert "except Exception" in src[i:i + 2200]


def test_cubre_exito_y_fallo_del_swap():
    src = _sin_comentarios(_src())
    assert re.search(r'_emit_change_outcome_metric\(\s*\n?\s*"swap",\s*"ok"', src), "falta el éxito del swap"
    assert re.search(r'_emit_change_outcome_metric\(\s*\n?\s*"swap",\s*"failed"', src), "falta el fallo del swap"


def test_cubre_todas_las_ramas_de_fallo_del_dia():
    """Un fallo es un fallo: las CUATRO ramas emiten.

    Son 4 y no 2 — dos de despensa, una de retries agotados y una de proveedor
    caído. Cubrir solo las dos «obvias» dejaría el denominador mal y el
    porcentaje del landing sería optimista sin que nadie lo notara.
    """
    src = _sin_comentarios(_src())
    fallos = re.findall(r'"regen_day", "failed", user_id=user_id, error_code="([a-z_]+)"', src)
    assert len(fallos) == 4, f"esperadas 4 ramas de fallo del día, encontradas {len(fallos)}: {fallos}"
    assert set(fallos) == {
        "pantry_insufficient_for_goal", "ai_unavailable", "ai_exhausted_retries",
    }, fallos
    # Y el éxito.
    assert re.search(r'_emit_change_outcome_metric\(\s*\n?\s*"regen_day",\s*"ok"', src)


def test_el_exito_del_dia_distingue_el_parcial():
    """`kept` > 0 es un éxito PARCIAL: sin ese campo, un día con 3 de 4 platos
    cambiados contaría igual que uno completo."""
    src = _sin_comentarios(_src())
    i = src.index('"regen_day", "ok"')
    bloque = src[i:i + 400]
    assert "kept=" in bloque and "regenerated=" in bloque
    assert "band_score=" in bloque, "sin band_score no se puede cruzar con la precisión de macros"


def test_emite_una_fila_por_operacion():
    """Verificación por ejecución, no por forma."""
    import routers.plans as rp
    import db

    capturado = []
    orig = db.execute_sql_write
    try:
        db.execute_sql_write = lambda sql, params=None, *a, **kw: capturado.append((sql, params))
        rp._emit_change_outcome_metric(
            "swap", "failed", user_id="u1", error_code="swap_llm_retries_exhausted")
        rp._emit_change_outcome_metric("regen_day", "ok", user_id="u1", kept=1, band_score=0.75)
    finally:
        db.execute_sql_write = orig

    assert len(capturado) == 2
    nodes = [p[2] for _, p in capturado]
    assert nodes == ["change_swap", "change_regen_day"]
    meta0 = json.loads(capturado[0][1][5])
    assert meta0["outcome"] == "failed"
    assert meta0["error_code"] == "swap_llm_retries_exhausted"


def test_un_fallo_de_db_no_rompe_el_cambio():
    """Si la telemetría revienta, el usuario igual recibe su plato."""
    import routers.plans as rp
    import db

    orig = db.execute_sql_write
    try:
        def _boom(*a, **kw):
            raise RuntimeError("db caida")
        db.execute_sql_write = _boom
        rp._emit_change_outcome_metric("swap", "ok", user_id="u1")  # no debe levantar
    finally:
        db.execute_sql_write = orig
