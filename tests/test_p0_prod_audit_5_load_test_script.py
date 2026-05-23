"""[P0-PROD-AUDIT-1 · 2026-05-23] Invariantes del script load_test_db_pool.py.

Gap original (audit 2026-05-23 — B-P0-5):
    3 pools DB (`connection_pool`, `async_connection_pool`,
    `chat_checkpoint_pool`) con knobs `MEALFIT_DB_POOL_*`. Ningún test
    ejecutado validó saturación bajo carga real. Riesgo: primer pico
    de tráfico tumba el backend.

Fix:
    `scripts/load_test_db_pool.py` ejecuta load test async (httpx) contra
    endpoints representativos del backend. Mide p50/p95/p99 + error rate +
    throughput. Documentado en `docs/runbooks/db_pool_load_test.md`.

Por qué un test del script (no solo el script en sí):
    El script es una herramienta operacional — si alguien lo borra "porque
    nunca lo usé", el gap reabre. Este test ancla existencia + importabilidad
    + declaración de scenarios canónicos.

NOTA: este test NO ejecuta el load test real (que requiere backend corriendo).
Es un sanity check estructural — el load test se ejecuta manualmente desde
runbook o desde job de CI específico de staging.

Tooltip-anchor: P0-PROD-AUDIT-1-LOAD-TEST | audit 2026-05-23.
"""
from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _BACKEND_ROOT / "scripts" / "load_test_db_pool.py"
_RUNBOOK_PATH = _BACKEND_ROOT / "docs" / "runbooks" / "db_pool_load_test.md"

_MODULE_KEY = "_p0_prod_audit_5_load_test_module"


def _import_script_module():
    """Importa el script como módulo. Asegura `sys.modules[name]` ANTES de
    `exec_module` — sin esto, `@dataclass` falla con AttributeError
    porque el ProcessClass machinery de dataclasses busca el módulo en
    sys.modules para resolver type hints (Python 3.11+ behavior).
    """
    spec = importlib.util.spec_from_file_location(_MODULE_KEY, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, (
        f"importlib NO puede resolver {_SCRIPT_PATH} — problema de path o permisos."
    )
    module = importlib.util.module_from_spec(spec)
    # Insertar ANTES del exec_module para que dataclass.__module__ resuelva.
    sys.modules[_MODULE_KEY] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(_MODULE_KEY, None)
        pytest.fail(
            f"Script {_SCRIPT_PATH} no es importable: {type(e).__name__}: {e}. "
            f"Revisar typos / imports."
        )
    return module


def test_script_exists() -> None:
    assert _SCRIPT_PATH.exists(), (
        f"Load test script ausente en {_SCRIPT_PATH}. Cierre del gap "
        f"B-P0-5 perdido."
    )


def test_script_is_executable() -> None:
    """El script debe tener bit ejecutable + shebang `#!/usr/bin/env python3`.
    Si alguien borra el shebang, `./scripts/load_test_db_pool.py` falla con
    `Permission denied` o `bad interpreter` cuando el operador lo invoca
    bajo presión de incidente.
    """
    # En Windows, los bits ejecutables no aplican — skip ese check.
    if os.name != "nt":
        mode = _SCRIPT_PATH.stat().st_mode
        assert mode & stat.S_IXUSR, (
            f"Script {_SCRIPT_PATH} no tiene bit ejecutable. "
            f"Restaurar con: chmod +x {_SCRIPT_PATH}"
        )
    # Shebang.
    first_line = _SCRIPT_PATH.read_text(encoding="utf-8").split("\n")[0]
    assert first_line.startswith("#!") and "python" in first_line, (
        f"Script {_SCRIPT_PATH} no tiene shebang `#!/usr/bin/env python3`. "
        f"Invocación directa rompe."
    )


def test_script_is_importable() -> None:
    """El script debe parsear sin errores de sintaxis. Test de sanity:
    si alguien introduce un typo Python, este test falla loud antes de
    que un operador trate de ejecutarlo bajo presión.

    NO ejecuta el `main()` — solo importa el módulo.
    """
    _import_script_module()


def test_script_declares_canonical_scenarios() -> None:
    """El script debe declarar al menos los scenarios canónicos: health,
    ready, version. Sin ellos, el smoke baseline (sin auth) NO se puede
    ejecutar.
    """
    module = _import_script_module()
    scenarios = getattr(module, "SCENARIOS", None)
    assert isinstance(scenarios, dict), (
        "Script no exporta `SCENARIOS: dict`. Restaurar la declaración top-level."
    )
    required = {"health", "ready", "version"}
    missing = required - set(scenarios.keys())
    assert not missing, (
        f"Scenarios canónicos ausentes en SCENARIOS: {missing}. "
        f"Sin ellos, smoke baseline (no-auth) imposible."
    )


def test_script_documented_in_runbook() -> None:
    """El runbook db_pool_load_test.md debe existir + referenciar el script
    + documentar los SOPs de quick start, stress test, y diagnóstico.
    """
    assert _RUNBOOK_PATH.exists(), (
        f"Runbook ausente en {_RUNBOOK_PATH}. Sin documentación, el script "
        f"es inutilizable bajo presión de incidente (el operador no sabe "
        f"qué thresholds significan PASS/WARN/FAIL)."
    )
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    required_sections = [
        "Quick start",
        "Verdict",
        "Diagnóstico",
    ]
    missing = [s for s in required_sections if s not in text]
    assert not missing, (
        f"Runbook db_pool_load_test.md incompleto — secciones ausentes: {missing}."
    )
    # El script debe ser referenciado por path.
    assert "load_test_db_pool.py" in text, (
        "Runbook no referencia el script `load_test_db_pool.py`. Cross-link "
        "perdido."
    )


def test_script_evaluate_verdict_handles_empty_report() -> None:
    """Sanity: `evaluate_verdict` NO debe crashear con un reporte vacío
    (e.g. si el target estuvo down y 0 requests completaron).

    El bug clásico: división por cero en `count` o `len(latencies_ms)`.
    """
    module = _import_script_module()
    evaluate_verdict = getattr(module, "evaluate_verdict", None)
    assert evaluate_verdict is not None, "Script no exporta `evaluate_verdict`."

    empty_report = {"by_scenario": {}}
    thresholds = {
        "error_rate_warn_pct": 1.0,
        "error_rate_fail_pct": 5.0,
        "health_p95_warn_ms": 200,
        "health_p95_fail_ms": 500,
        "api_p95_warn_ms": 1500,
        "api_p95_fail_ms": 3000,
    }
    verdict, findings = evaluate_verdict(empty_report, thresholds)
    # Sin scenarios, el verdict default es PASS (no hay hallazgos).
    assert verdict == "PASS", (
        f"evaluate_verdict con report vacío debe ser PASS, fue {verdict}. "
        f"findings={findings}"
    )
