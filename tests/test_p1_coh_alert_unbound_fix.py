"""[P1-COH-ALERT-UNBOUND-FIX · 2026-06-25] El cron diario `_shopping_coherence_alert_job` fallaba en
CADA corrida con `UnboundLocalError: cannot access local variable 'execute_sql_query'`.

Causa: `execute_sql_query` se importa a nivel de módulo (cron_tasks.py ~L22), pero dentro de la función
había un `from db_core import execute_sql_query` LOCAL (en el bloque de tracking de fallos). En Python,
una asignación a un nombre en CUALQUIER punto del cuerpo lo vuelve local en TODO el scope → el
`plans = execute_sql_query(...)` PREVIO (fetch de planes) referenciaba el local antes de su asignación →
UnboundLocalError → la alerta diaria de coherencia quedaba rota.

Fix: eliminar el import local redundante (el de módulo lo provee). Este test ancla que NO vuelva.
NOTA: el chequeo es consciente de COMENTARIOS (el fix dejó una explicación que MENCIONA el import) →
solo cuentan líneas de código reales (no `#`).
"""
import os
import re

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slice_top_level_func(src: str, name: str) -> str:
    start = src.find(f"\ndef {name}(")
    if start < 0 and src.startswith(f"def {name}("):
        start = 0
    assert start >= 0, f"no se encontró def {name}("
    nxt = re.search(r"\ndef ", src[start + 1:])
    end = (start + 1 + nxt.start()) if nxt else len(src)
    return src[start:end]


def _code_lines(block: str):
    """Líneas de CÓDIGO (descarta vacías y comentarios), preservando la indentación original."""
    return [l for l in block.splitlines() if l.strip() and not l.strip().startswith("#")]


def test_no_local_execute_sql_query_import_in_coh_alert_job():
    with open(os.path.join(_BACKEND, "cron_tasks.py"), encoding="utf-8") as f:
        src = f.read()
    body = _slice_top_level_func(src, "_shopping_coherence_alert_job")
    offenders = [l for l in _code_lines(body)
                 if l.strip().startswith("from db_core import execute_sql_query")]
    assert not offenders, (
        "Re-añadido `from db_core import execute_sql_query` LOCAL en _shopping_coherence_alert_job → "
        f"reintroduce el UnboundLocalError. Offenders: {offenders}"
    )
    # Y la función debe SEGUIR usando execute_sql_query (resuelto del import de módulo).
    assert "execute_sql_query(" in body


def test_module_level_execute_sql_query_import_present():
    # El import de módulo (la fuente correcta del nombre) debe existir SIN indentar (top-level).
    with open(os.path.join(_BACKEND, "cron_tasks.py"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert any(l.startswith("from db_core import execute_sql_query") for l in lines), (
        "Falta el import de módulo `from db_core import execute_sql_query` (top-level) en cron_tasks.py"
    )
