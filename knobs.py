"""[P2-1 · 2026-05-08] SSOT del registry de knobs `MEALFIT_*` (extraído de
`graph_orchestrator.py`).

Este módulo aísla los helpers `_env_int`/`_env_float`/`_env_bool`/`_env_str`
y la dict `_KNOBS_REGISTRY` para que módulos importados POR
`graph_orchestrator.py` (notablemente `constants.py`) puedan registrar sus
knobs sin crear una dependencia circular en module-init.

Histórico:
  - P3-NEW-D 2026-05-08 introdujo `_env_int`/`_env_float`/`_env_bool` con
    auto-registro en `_KNOBS_REGISTRY` dentro de `graph_orchestrator.py`.
  - P1-2 2026-05-08 añadió `_env_str(name, default, choices)`.
  - P1-A 2026-05-08-late migró 10 knobs de cron_tasks/routers/plans, pero
    quedaron ~14 fuera del registry porque viven en módulos importados POR
    graph_orchestrator (constants.py) o en sitios module-init (app.py:79,
    shopping_calculator.py:120/134/135) que no pueden hacer
    `from graph_orchestrator import ...` sin riesgo de circular.
  - P2-1 2026-05-08 extrae los helpers a este módulo aislado para cerrar
    el bypass: cualquier knob `MEALFIT_*` puede ahora registrarse desde
    cualquier módulo del backend sin acoplarlo a graph_orchestrator.

Diseño:
  - Cero dependencias internas (sólo `os`/`logging`). Cualquier módulo del
    backend puede importarlo a top-level sin generar ciclos.
  - `graph_orchestrator.py` re-exporta `_env_int`/`_env_float`/`_env_bool`/
    `_env_str`/`_KNOBS_REGISTRY`/`get_knobs_registry_snapshot`/`_register_knob`
    para preservar todos los call sites que ya importan desde ahí.
  - El registry es una dict mutable global; `from knobs import _KNOBS_REGISTRY`
    desde múltiples módulos comparte el mismo objeto, así que las
    inserciones de constants.py son visibles en `_log_active_knobs` (que
    itera el mismo dict desde graph_orchestrator).

Observabilidad pura: cero cambio de comportamiento en los helpers.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

_KNOBS_REGISTRY: dict = {}


def _register_knob(
    name: str,
    type_label: str,
    default: Any,
    raw: Optional[str],
    value: Any,
    *,
    parse_failed: bool = False,
) -> None:
    """Anota el knob en el registry. Idempotente — el último registro gana
    si por alguna razón se llama dos veces (no debería pasar con consts
    de módulo, pero defensivo contra reload patterns en tests)."""
    _KNOBS_REGISTRY[name] = {
        "type": type_label,
        "default": default,
        "raw": raw,
        "value": value,
        "is_override": raw is not None and raw != "" and not parse_failed,
        "parse_failed": parse_failed,
    }


def _env_int(
    name: str,
    default: int,
    validator: Optional[Callable[[int], bool]] = None,
) -> int:
    """[P2-KNOBS-ENV-INT-NO-VALIDATOR · 2026-05-24] `validator` opcional:
    callable `(int) -> bool`. Si retorna False, loguea WARNING + cae al
    default + marca `parse_failed`. Útil para rangos (ej.
    `lambda v: 1 <= v <= 100_000` para sizes/thresholds).

    Pre-fix `_env_int` no aceptaba `validator=` (a diferencia de
    `_env_float`). Resultado: knobs int críticos requerían `clamp` manual
    post-lectura — `_EMBEDDING_CACHE_MAXSIZE` lo hacía con `max/min`,
    pero `MEALFIT_CB_FAILURE_THRESHOLD=0` (o negativo) pasaba silencioso
    y abría el breaker eternamente. Simetría con `_env_float` cierra el
    gap. Tooltip-anchor: P2-KNOBS-ENV-INT-NO-VALIDATOR.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        _register_knob(name, "int", default, raw, default)
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            f"[KNOBS] env {name}={raw!r} no es int. Usando default={default}."
        )
        _register_knob(name, "int", default, raw, default, parse_failed=True)
        return default
    if validator is not None:
        try:
            ok = bool(validator(value))
        except Exception as _e:
            logging.getLogger(__name__).warning(
                f"[KNOBS] env {name}={value} validator falló ({_e}). Usando default={default}."
            )
            _register_knob(name, "int", default, raw, default, parse_failed=True)
            return default
        if not ok:
            logging.getLogger(__name__).warning(
                f"[KNOBS] env {name}={value} fuera de rango permitido. Usando default={default}."
            )
            _register_knob(name, "int", default, raw, default, parse_failed=True)
            return default
    _register_knob(name, "int", default, raw, value)
    return value


def _env_float(
    name: str,
    default: float,
    validator: Optional[Callable[[float], bool]] = None,
) -> float:
    """[P1-3 · 2026-05-08] `validator` opcional: callable `(float) -> bool`.
    Si retorna False, loguea WARNING + cae al default + marca `parse_failed`.
    Útil para rangos (ej. `lambda v: 0.0 < v < 1.0` para fracciones)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        _register_knob(name, "float", default, raw, default)
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            f"[KNOBS] env {name}={raw!r} no es float. Usando default={default}."
        )
        _register_knob(name, "float", default, raw, default, parse_failed=True)
        return default
    if validator is not None:
        try:
            ok = bool(validator(value))
        except Exception as _e:
            logging.getLogger(__name__).warning(
                f"[KNOBS] env {name}={value} validator falló ({_e}). Usando default={default}."
            )
            _register_knob(name, "float", default, raw, default, parse_failed=True)
            return default
        if not ok:
            logging.getLogger(__name__).warning(
                f"[KNOBS] env {name}={value} fuera de rango permitido. Usando default={default}."
            )
            _register_knob(name, "float", default, raw, default, parse_failed=True)
            return default
    _register_knob(name, "float", default, raw, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    """P1-NEW-1: parser laxo de booleanos. Acepta `1/true/yes/on` (case-insensitive)
    como verdadero; cualquier otro valor no vacío como falso. Default si vacío.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        _register_knob(name, "bool", default, raw, default)
        return default
    value = raw.strip().lower() in ("1", "true", "yes", "on")
    _register_knob(name, "bool", default, raw, value)
    return value


def thinking_budget_kwargs(model_name: str, env_var: str, default: int) -> dict:
    """[P2-COST-THINKING-CAP-EXT · 2026-06-01] Devuelve `{"thinking_budget": N}`
    para CAPAR el razonamiento (reasoning/thinking) de modelos thinking-capable
    (gemini-3.5-flash / *-pro) en nodos de relleno-de-schema FUERA del pipeline
    de planes — chat swap_llm, modify-meal tool, fact-extractor PRO. Espejo
    parametrizado por knob del helper `_thinking_budget_kwargs` de
    `graph_orchestrator.py` (P1-COST-THINKING-CAP), que solo cubre day-gen +
    correctores.

    Por qué importa para COSTO: los reasoning tokens facturan como OUTPUT
    (~$9/M en flash, 6x el input). Sin cap, un nodo thinking-capable puede
    emitir miles de tokens de razonamiento (day-gen sin cap llegó a 19,162 tok
    de output, ~80% del costo del plan) en tareas que solo rellenan un schema.
    Un techo GENEROSO (default 2048, el valor A/B-validado de day-gen que hace
    una tarea MÁS dura — un día completo) recorta el runaway patológico SIN
    tocar el razonamiento normal de un solo plato/extracción → ahorro sin
    pérdida de calidad.

    Semántica del knob (clamp [-1, 32768]):
      - N >= 0  → cap el reasoning en N tokens.
      - N < 0   → sentinela: sin cap (reasoning libre, comportamiento legacy /
                  rollback sin redeploy).
    flash-lite NO soporta thinking_config → dict vacío (evita pasar un kwarg
    no soportado que rompería la construcción del LLM). Tooltip-anchor:
    P2-COST-THINKING-CAP-EXT.
    """
    budget = _env_int(env_var, default, validator=lambda v: -1 <= v <= 32768)
    if budget < 0:
        return {}
    if not model_name or "lite" in model_name.lower():
        return {}
    return {"thinking_budget": budget}


def is_production() -> bool:
    """[P2-PROD-AUDIT-3 · 2026-05-30] SSOT del check de entorno productivo.

    ANTES 5 gates de seguridad (billing.py is_sandbox ×3, app.py _IS_PRODUCTION +
    webhook fail-secure) usaban `os.environ.get("ENVIRONMENT") == "production"`
    exact-match case/whitespace-sensitive, mientras plans.py YA normalizaba con
    `.strip().lower()`. Un typo `Production`/`PRODUCTION`/` production ` flippeaba
    `is_sandbox=True` y volvía bypasseables el sandbox de PayPal + (con un knob
    `MEALFIT_ALLOW_WEBHOOK_UNSIGNED` residual) la verificación de firma del webhook.
    Este helper normaliza lower+strip; todos los gates lo consumen.
    """
    return os.environ.get("ENVIRONMENT", "").strip().lower() == "production"


def _env_str(name: str, default: str, choices: Optional[set[str]] = None) -> str:
    """[P1-2 · 2026-05-08] Knob de tipo string normalizado (lower+strip), con
    validación opcional de `choices`. Auto-registra en `_KNOBS_REGISTRY`.

    Si `choices` es no-None y el valor no coincide, loguea WARNING y cae al
    default (registrando `parse_failed=True` para que el inventario lo destaque).
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        _register_knob(name, "str", default, raw, default)
        return default
    value = raw.strip().lower()
    if choices is not None and value not in choices:
        logging.getLogger(__name__).warning(
            f"[KNOBS] env {name}={raw!r} no es valor permitido "
            f"(esperado: {sorted(choices)}). Usando default={default!r}."
        )
        _register_knob(name, "str", default, raw, default, parse_failed=True)
        return default
    _register_knob(name, "str", default, raw, value)
    return value


def get_knobs_registry_snapshot() -> dict:
    """[P3-NEW-D] Snapshot read-only del registry de knobs. Útil para
    endpoints de diagnóstico (`/api/system/knobs`) o para tests que
    verifican que un knob fue resuelto al valor esperado.

    Devuelve copia para que mutaciones del caller no afecten el registry."""
    return {name: dict(info) for name, info in _KNOBS_REGISTRY.items()}
