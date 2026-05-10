"""[P1-A · 2026-05-08] Tests del cierre del audit del registry global de knobs.

Bug original (audit 2026-05-08, post-P3-B):
  Diez sites en `cron_tasks.py`, `routers/plans.py` y `shopping_calculator.py`
  consumían `MEALFIT_*` vía `os.environ.get(...)` raw — bypass del contrato
  P3-NEW-D que requiere que cada knob se auto-registre en `_KNOBS_REGISTRY`
  vía `_env_int/_env_float/_env_bool/_env_str`. Sin registro:
    - `/health/version` no incluye el knob en su snapshot.
    - `_log_active_knobs()` startup no muestra overrides aplicados.
    - SRE que cambia el knob en prod sin redeploy no puede confirmar
      que tomó efecto.

Fix:
  1. `cron_tasks.py` importa `_env_int, _env_float` desde graph_orchestrator
     y los usa en los 8 sites runtime-evaluados.
  2. `routers/plans.py` añade `_env_int` al import bloque y migra 2 sites.
  3. `shopping_calculator.py` usa el patrón lazy con fallback a
     `os.environ.get` (espejo P1-3) en 2 sites.

Cobertura:
  - test_coh_alert_knobs_registered: 3 knobs MEALFIT_COH_ALERT_*.
  - test_coherence_metrics_knobs_registered: 3 knobs.
  - test_perishable_cycle_knobs_registered: 2 knobs.
  - test_reactivate_lookback_knob_registered: 1 knob.
  - test_perishable_knobs_via_shopping_calculator_path.
  - test_perishable_knobs_via_routers_plans_path.
  - test_invalid_float_falls_back_to_default: validator de cap_ratio.
  - test_no_raw_os_environ_get_in_migrated_sites: regresión estática.
"""
import importlib
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def go_module():
    """Import limpio de graph_orchestrator. Reload garantiza registry hidratado
    desde el env actual del fixture."""
    import graph_orchestrator
    return graph_orchestrator


# ---------------------------------------------------------------------------
# Cron-side knobs (cron_tasks.py)
# ---------------------------------------------------------------------------
def test_coh_alert_knobs_registered(go_module, monkeypatch):
    """Los 3 knobs `MEALFIT_COH_ALERT_*` consumidos por
    `_shopping_coherence_alert_job` se registran al ser leídos."""
    monkeypatch.setenv("MEALFIT_COH_ALERT_MIN_PLANS", "12")
    monkeypatch.setenv("MEALFIT_COH_ALERT_CAP_RATIO", "0.07")
    monkeypatch.setenv("MEALFIT_COH_ALERT_PLAN_FRACTION", "0.15")

    # Forzar lectura via los wrappers — `_env_int`/`_env_float` se exponen en go.
    assert go_module._env_int("MEALFIT_COH_ALERT_MIN_PLANS", 5) == 12
    assert go_module._env_float(
        "MEALFIT_COH_ALERT_CAP_RATIO", 0.05, validator=lambda v: 0.0 < v < 1.0
    ) == 0.07
    assert go_module._env_float(
        "MEALFIT_COH_ALERT_PLAN_FRACTION", 0.10, validator=lambda v: 0.0 < v < 1.0
    ) == 0.15

    snap = go_module.get_knobs_registry_snapshot()
    for name, expected in (
        ("MEALFIT_COH_ALERT_MIN_PLANS", 12),
        ("MEALFIT_COH_ALERT_CAP_RATIO", 0.07),
        ("MEALFIT_COH_ALERT_PLAN_FRACTION", 0.15),
    ):
        assert name in snap, f"{name} ausente del registry tras lectura"
        assert snap[name]["value"] == expected
        assert snap[name]["is_override"] is True


def test_coherence_metrics_knobs_registered(go_module, monkeypatch):
    """Los 3 knobs `MEALFIT_COHERENCE_METRICS_*` (lookback/max_plans/interval)
    se registran al ser leídos."""
    monkeypatch.setenv("MEALFIT_COHERENCE_METRICS_LOOKBACK_H", "2.5")
    monkeypatch.setenv("MEALFIT_COHERENCE_METRICS_MAX_PLANS", "500")
    monkeypatch.setenv("MEALFIT_COHERENCE_METRICS_INTERVAL_MIN", "30")

    import math
    go_module._env_float(
        "MEALFIT_COHERENCE_METRICS_LOOKBACK_H",
        1.0,
        validator=lambda v: v > 0 and math.isfinite(v),
    )
    go_module._env_int("MEALFIT_COHERENCE_METRICS_MAX_PLANS", 1000)
    go_module._env_int("MEALFIT_COHERENCE_METRICS_INTERVAL_MIN", 60)

    snap = go_module.get_knobs_registry_snapshot()
    assert snap["MEALFIT_COHERENCE_METRICS_LOOKBACK_H"]["value"] == 2.5
    assert snap["MEALFIT_COHERENCE_METRICS_MAX_PLANS"]["value"] == 500
    assert snap["MEALFIT_COHERENCE_METRICS_INTERVAL_MIN"]["value"] == 30


def test_perishable_cycle_knobs_registered(go_module, monkeypatch):
    """`MEALFIT_PERISHABLE_CYCLE_DAYS_MAX` y `MEALFIT_PERISHABLE_CYCLE_DAYS`
    se registran al ser leídos vía `_env_int`."""
    monkeypatch.setenv("MEALFIT_PERISHABLE_CYCLE_DAYS_MAX", "45")
    monkeypatch.setenv("MEALFIT_PERISHABLE_CYCLE_DAYS", "10")

    go_module._env_int("MEALFIT_PERISHABLE_CYCLE_DAYS_MAX", 30)
    go_module._env_int("MEALFIT_PERISHABLE_CYCLE_DAYS", 7)

    snap = go_module.get_knobs_registry_snapshot()
    assert snap["MEALFIT_PERISHABLE_CYCLE_DAYS_MAX"]["value"] == 45
    assert snap["MEALFIT_PERISHABLE_CYCLE_DAYS"]["value"] == 10
    # `is_override=True` para ambos (env presente, parse OK).
    assert snap["MEALFIT_PERISHABLE_CYCLE_DAYS_MAX"]["is_override"] is True
    assert snap["MEALFIT_PERISHABLE_CYCLE_DAYS"]["is_override"] is True


def test_reactivate_knobs_registered(go_module, monkeypatch):
    """`MEALFIT_REACTIVATE_SHOPPING_LIST_INTERVAL_MIN` y
    `MEALFIT_REACTIVATE_LOOKBACK_DAYS` se registran."""
    monkeypatch.setenv("MEALFIT_REACTIVATE_SHOPPING_LIST_INTERVAL_MIN", "120")
    monkeypatch.setenv("MEALFIT_REACTIVATE_LOOKBACK_DAYS", "180")

    go_module._env_int("MEALFIT_REACTIVATE_SHOPPING_LIST_INTERVAL_MIN", 60)
    go_module._env_int("MEALFIT_REACTIVATE_LOOKBACK_DAYS", 365)

    snap = go_module.get_knobs_registry_snapshot()
    assert snap["MEALFIT_REACTIVATE_SHOPPING_LIST_INTERVAL_MIN"]["value"] == 120
    assert snap["MEALFIT_REACTIVATE_LOOKBACK_DAYS"]["value"] == 180


# ---------------------------------------------------------------------------
# Validator de rango: cap_ratio fuera de (0,1) cae a default
# ---------------------------------------------------------------------------
def test_invalid_float_falls_back_to_default(go_module, monkeypatch):
    """Si el operador setea `MEALFIT_COH_ALERT_CAP_RATIO=1.5` (fuera de rango),
    el validator captura, loguea WARNING y cae al default 0.05.
    `parse_failed=True` debe quedar marcado para que `_log_active_knobs`
    lo destaque en la línea WARNING."""
    monkeypatch.setenv("MEALFIT_COH_ALERT_CAP_RATIO", "1.5")
    val = go_module._env_float(
        "MEALFIT_COH_ALERT_CAP_RATIO", 0.05, validator=lambda v: 0.0 < v < 1.0
    )
    assert val == 0.05
    snap = go_module.get_knobs_registry_snapshot()
    info = snap["MEALFIT_COH_ALERT_CAP_RATIO"]
    assert info["parse_failed"] is True
    assert info["value"] == 0.05
    assert info["is_override"] is False, (
        "valor fuera de rango efectivamente no es un override válido — el valor "
        "que está en uso es el default."
    )


# ---------------------------------------------------------------------------
# Regresión estática: nada de `os.environ.get("MEALFIT_*")` en los sites
# migrados (excepto los fallbacks defensivos dentro de try/except).
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _read(rel_path: str) -> str:
    full = _BACKEND_DIR / rel_path
    return full.read_text(encoding="utf-8")


def test_cron_tasks_no_raw_perishable_or_coh_reads():
    """Regresión: las migraciones P1-A no se deben revertir. Ningún
    `os.environ.get("MEALFIT_PERISHABLE_*"|"MEALFIT_COH_*"|"MEALFIT_REACTIVATE_*"|"MEALFIT_COHERENCE_METRICS_*")`
    debe quedar en cron_tasks.py."""
    src = _read("cron_tasks.py")
    forbidden = [
        r'os\.environ\.get\(\s*["\']MEALFIT_PERISHABLE_CYCLE_DAYS',
        r'os\.environ\.get\(\s*["\']MEALFIT_COH_ALERT_',
        r'os\.environ\.get\(\s*["\']MEALFIT_COHERENCE_METRICS_',
        r'os\.environ\.get\(\s*["\']MEALFIT_REACTIVATE_',
    ]
    for pattern in forbidden:
        matches = re.findall(pattern, src)
        assert not matches, (
            f"Regresión P1-A: encontrado patrón raw `{pattern}` en cron_tasks.py — "
            f"debe usar `_env_int`/`_env_float` para auto-registro en _KNOBS_REGISTRY. "
            f"Matches: {matches[:3]}"
        )


def test_routers_plans_no_raw_perishable_reads():
    """Regresión: routers/plans.py no debe re-introducir lecturas raw de
    MEALFIT_PERISHABLE_*."""
    src = _read("routers/plans.py")
    matches = re.findall(
        r'os\.environ\.get\(\s*["\']MEALFIT_PERISHABLE_', src
    )
    assert not matches, (
        f"Regresión P1-A: routers/plans.py reintrodujo {matches} via os.environ.get raw."
    )


def test_shopping_calculator_uses_knobs_helpers():
    """[P2-1 · 2026-05-08] Tras extraer los helpers a `backend/knobs.py`,
    `shopping_calculator.py` ya no necesita el lazy-import + try/except: importa
    directamente `_knob_env_int` (y compañeros) a top-level. Este test asegura
    que los knobs PERISHABLE pasan por `_knob_env_int` y NO regresan a
    `os.environ.get` raw.

    Histórico: P1-A 2026-05-08-late introdujo el alias `_env_int_ssot` con
    fallback try/except como apaño contra imports circulares. P2-1 cerró el
    ciclo movido los helpers a `knobs.py` (cero deps), eliminando el fallback.
    """
    src = _read("shopping_calculator.py")
    # Top-level import desde el SSOT (acepta single-line o multi-line en `(...)`).
    assert re.search(
        r"from\s+knobs\s+import[\s\S]*?_env_int\s+as\s+_knob_env_int", src
    ), (
        "shopping_calculator.py debe importar `_env_int as _knob_env_int` desde "
        "`knobs` a top-level. Sin esto los knobs SEMANTIC_INIT/EMBED_INIT/PERISHABLE "
        "vuelven a bypassear _KNOBS_REGISTRY."
    )
    # Knobs PERISHABLE pasan por el helper SSOT.
    assert re.search(
        r"_knob_env_int\(\s*['\"]MEALFIT_PERISHABLE_CYCLE_DAYS_MAX['\"]", src
    ), "Falta la lectura SSOT `_knob_env_int('MEALFIT_PERISHABLE_CYCLE_DAYS_MAX', ...)`."
    assert re.search(
        r"_knob_env_int\(\s*['\"]MEALFIT_PERISHABLE_CYCLE_DAYS['\"]", src
    ), "Falta la lectura SSOT `_knob_env_int('MEALFIT_PERISHABLE_CYCLE_DAYS', ...)`."
    # Los `_env_int_local`/`_env_float_local` legacy quedaron eliminados — sin
    # `def` y sin call sites. Permitimos menciones en comentarios históricos
    # (que documentan el migration path) buscando solo patrones ejecutables.
    assert not re.search(r"def\s+_env_int_local\b", src), (
        "Función `_env_int_local` debe haber sido eliminada (P2-1) — bypassa `_KNOBS_REGISTRY`."
    )
    assert not re.search(r"def\s+_env_float_local\b", src), (
        "Función `_env_float_local` debe haber sido eliminada (P2-1) — bypassa `_KNOBS_REGISTRY`."
    )
    assert not re.search(r"_env_int_local\s*\(", src), (
        "Call site `_env_int_local(...)` activo encontrado — debe usar `_knob_env_int` (P2-1)."
    )
    assert not re.search(r"_env_float_local\s*\(", src), (
        "Call site `_env_float_local(...)` activo encontrado — debe usar `_knob_env_float` (P2-1)."
    )
