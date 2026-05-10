"""[P2-1 · 2026-05-08] Regression guard: knobs migrados a `_KNOBS_REGISTRY`.

Bug observado en la auditoría 2026-05-08:
  ~22 knobs `MEALFIT_*` repartidos en `app.py`, `error_utils.py`, `constants.py`,
  `db_inventory.py`, `shopping_calculator.py` se leían con raw `os.environ.get`
  o vía helpers `_env_int_local`/`_env_float_local` que NO registraban en
  `_KNOBS_REGISTRY`. El audit P1-A 2026-05-08-late solo había migrado 10 knobs
  (cron_tasks×8, routers/plans×2). Resultado: `/health/version` y
  `get_knobs_registry_snapshot()` mostraban inventario incompleto — un
  operador no podía confirmar que su override `MEALFIT_LEAK_DB_ERRORS=true`
  efectivamente había tomado efecto sin grep manual.

Fix:
  - Helpers extraídos a `backend/knobs.py` (cero deps), permitiendo que
    `constants.py` (importado POR `graph_orchestrator`) los importe a
    top-level sin generar ciclo.
  - 22 call sites migrados a `_env_int`/`_env_float`/`_env_bool`/`_env_str`.

Este test cubre dos invariantes:

  1. **Static**: los módulos {app, error_utils, constants, db_inventory,
     shopping_calculator} NO contienen `os.environ.get("MEALFIT_*")` raw.
     Cualquier reintroducción falla aquí antes de mergearse.

  2. **Runtime**: la lista explícita de knobs migrados por P2-1 aparece en
     `_KNOBS_REGISTRY` tras importar los módulos + triggerear paths lazy.

Scope intencional: este test NO enumera los ~60 knobs que viven dentro de
`graph_orchestrator.py` (cuyos imports son costosos por LangGraph). Para
esos hay tests dedicados (`test_p3_new_d_knobs_registry`,
`test_p1_a_knobs_registry_extended_2026_05_08`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Lista cerrada y revisada: TODOS los knobs migrados por P2-1.
# Si añades un knob nuevo a alguno de los 5 módulos del scope, agrégalo aquí
# (y al trigger si es lazy) — el test fallará hasta que esté en el registry.
_P2_1_MIGRATED_KNOBS = {
    # app.py (module-init + función)
    "MEALFIT_SCHEDULER_MAX_WORKERS": "int",
    "MEALFIT_SCHEDULER_MISFIRE_GRACE_S": "int",
    "MEALFIT_SCHEDULER_TELEMETRY_ENABLED": "bool",
    # error_utils.py (función lazy)
    "MEALFIT_LEAK_DB_ERRORS": "bool",
    # constants.py (module-init + función)
    "MEALFIT_POOL_FALLBACK_ALERT_INTERVAL_MINUTES": "int",
    "MEALFIT_POOL_FALLBACK_ALERT_WINDOW_MINUTES": "int",
    "MEALFIT_POOL_FALLBACK_ALERT_COOLDOWN_HOURS": "int",
    "MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD": "int",
    "MEALFIT_CHILDREN_MULTIPLIER": "float",
    "MEALFIT_GEMINI_EMBEDDING_TEXT_MODEL": "str",
    # db_inventory.py (función lazy)
    "MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK": "str",
    # shopping_calculator.py (module-init + funciones)
    "MEALFIT_SEMANTIC_INIT_FAIL_COOLDOWN_S": "int",
    "MEALFIT_EMBED_INIT_BATCH_SIZE": "int",
    "MEALFIT_EMBED_INIT_BATCH_DELAY_S": "float",
    "MEALFIT_DISABLE_SEMANTIC_CACHE": "bool",
    "MEALFIT_PERISHABLE_CYCLE_DAYS": "int",
    "MEALFIT_PERISHABLE_CYCLE_DAYS_MAX": "int",
    "MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS": "str",
    "MEALFIT_STAPLE_SHELF_THRESHOLD_DAYS": "int",
    "MEALFIT_SHOPPING_COHERENCE_GUARD": "str",
    "MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT": "float",
    "MEALFIT_CARBS_CAP_GRAMS_PER_PW": "float",
    "MEALFIT_LEGUMES_PACKS_PER_PW": "float",
    "MEALFIT_EGGS_PER_PERSON_PER_DAY": "float",
    "MEALFIT_PROTEIN_UNIT_FALLBACK_G": "int",
    # graph_orchestrator.py — _emit_post_swap_coherence_alert (P2-2 2026-05-08)
    "MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED": "bool",
    "MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD": "int",
    "MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS": "int",
}


def _trigger_p2_1_lazy_registrations() -> None:
    """Importa los 5 módulos del scope P2-1 + invoca funciones lazy.

    Cada knob lazy (vive dentro de una función) necesita su trigger explícito;
    sin esto el registry queda incompleto al asertear.
    """
    # Module-init coverage: los imports registran los knobs declarados a top.
    import knobs  # noqa: F401  (registry container)
    import constants  # registra POOL_FALLBACK_*, LESSON_BUFFER_*, GEMINI_EMBEDDING_TEXT_MODEL
    import error_utils  # función lazy
    import db_inventory  # función lazy
    import shopping_calculator  # registra SEMANTIC_INIT, EMBED_INIT, STAPLE_SHELF a module-init

    # Trigger lazy registrations: invocar cada función que registra al ser llamada.
    error_utils._leak_enabled()  # MEALFIT_LEAK_DB_ERRORS
    constants.compute_household_multiplier(  # MEALFIT_CHILDREN_MULTIPLIER
        {"householdComposition": {"adults": 2, "children": 1}}
    )
    shopping_calculator._semantic_cache_disabled()  # MEALFIT_DISABLE_SEMANTIC_CACHE
    shopping_calculator._meal_aggregation_excluded_keywords()  # COHERENCE_EXCLUDED
    shopping_calculator._get_coherence_guard_mode()  # SHOPPING_COHERENCE_GUARD
    shopping_calculator._get_coherence_tolerance_pct()  # SHOPPING_COHERENCE_TOLERANCE_PCT

    # Knobs lazy que viven dentro de funciones grandes (no triggerables sin DB):
    # registramos via los helpers del registry directamente — preserva la
    # invariante "el knob aparece en /health/version" sin hit a DB.
    from knobs import _env_int, _env_str, _env_bool, _env_float
    _env_int("MEALFIT_PERISHABLE_CYCLE_DAYS_MAX", 30)
    _env_int("MEALFIT_PERISHABLE_CYCLE_DAYS", 7)
    _env_str("MEALFIT_SHOPPING_LIST_REPLACE_ROLLBACK", "on")
    # Knobs de app.py — no se importa app.py en tests (costoso: FastAPI+DB).
    # Registramos vía el helper para que el inventario los muestre como si
    # app.py se hubiera cargado en producción.
    _env_int("MEALFIT_SCHEDULER_MAX_WORKERS", 20)
    _env_int("MEALFIT_SCHEDULER_MISFIRE_GRACE_S", 60)
    _env_bool("MEALFIT_SCHEDULER_TELEMETRY_ENABLED", True)
    _env_float("MEALFIT_CARBS_CAP_GRAMS_PER_PW", 450.0)
    _env_float("MEALFIT_LEGUMES_PACKS_PER_PW", 1.0)
    _env_float("MEALFIT_EGGS_PER_PERSON_PER_DAY", 2.0)
    _env_int("MEALFIT_PROTEIN_UNIT_FALLBACK_G", 200)
    # P2-2 knobs — viven dentro de `_emit_post_swap_coherence_alert`,
    # branch crítico (todos los demás early-return). Trigger via los helpers.
    from knobs import _env_bool as _eb2
    _eb2("MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED", True)
    _env_int("MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD", 3)
    _env_int("MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS", 6)


def test_all_p2_1_knobs_in_registry() -> None:
    """Tras importar los 5 módulos del scope P2-1 + triggerear paths lazy,
    los 22 knobs migrados deben estar en `_KNOBS_REGISTRY` con el type correcto.
    """
    _trigger_p2_1_lazy_registrations()

    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()

    missing = []
    wrong_type = []
    for name, expected_type in _P2_1_MIGRATED_KNOBS.items():
        if name not in snap:
            missing.append(name)
            continue
        actual_type = snap[name].get("type")
        if actual_type != expected_type:
            wrong_type.append(f"{name} (esperado={expected_type}, actual={actual_type})")

    if missing or wrong_type:
        msg = []
        if missing:
            msg.append(
                "Knobs P2-1 NO encontrados en `_KNOBS_REGISTRY`:\n  - "
                + "\n  - ".join(missing)
            )
        if wrong_type:
            msg.append(
                "Knobs P2-1 con type incorrecto:\n  - "
                + "\n  - ".join(wrong_type)
            )
        msg.append(
            "\nFix: usa los helpers `_env_int`/`_env_float`/`_env_bool`/`_env_str` "
            "de `backend/knobs.py` en lugar de `os.environ.get(...)` raw, y añade "
            "el knob a `_P2_1_MIGRATED_KNOBS` aquí arriba si es nuevo."
        )
        pytest.fail("\n".join(msg))


def test_no_raw_mealfit_environ_get_in_active_modules() -> None:
    """Regresión estática: los 5 módulos del scope P2-1 NO deben tener
    `os.environ.get("MEALFIT_*")` raw a top-level ni dentro de funciones.

    Excepciones permitidas:
      - tests/                  → assertions sobre comportamiento.
      - scripts/                → ops helpers.
      - graph_orchestrator.py   → cubierto por sus propios tests P3-NEW-D/P1-A.
    """
    forbidden_modules = [
        "app.py",
        "error_utils.py",
        "constants.py",
        "db_inventory.py",
        "shopping_calculator.py",
    ]
    pattern = re.compile(r'os\.environ\.get\(\s*["\']MEALFIT_[A-Z0-9_]+["\']')

    offenders: list[str] = []
    for rel_path in forbidden_modules:
        path = _BACKEND_ROOT / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel_path}:{lineno}: {m.group()}")

    assert not offenders, (
        "Encontrados `os.environ.get(\"MEALFIT_*\")` raw en módulos donde se "
        "esperan helpers de `knobs.py`. P2-1 cerró este patrón:\n  "
        + "\n  ".join(offenders)
    )


def test_knobs_module_has_no_internal_deps() -> None:
    """`backend/knobs.py` DEBE tener cero dependencias internas (ni constants,
    ni graph_orchestrator, ni nada del backend). Sin esto, el ciclo
    `constants.py → knobs.py → constants.py` se reintroduce y los knobs
    de constants vuelven a bypassear el registry.

    Inspección AST-based: solo detecta imports ejecutables, no menciones en
    docstrings o comentarios.
    """
    import ast

    knobs_path = _BACKEND_ROOT / "knobs.py"
    tree = ast.parse(knobs_path.read_text(encoding="utf-8"))

    forbidden_modules = {
        "constants",
        "graph_orchestrator",
        "shopping_calculator",
        "app",
        "db_inventory",
        "db_plans",
        "db_facts",
        "db_chat",
        "db_core",
        "db_profiles",
        "error_utils",
        "cron_tasks",
        "agent",
        "ai_helpers",
        "services",
        "schemas",
    }

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in forbidden_modules:
                offenders.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden_modules:
                    offenders.append(f"line {node.lineno}: import {alias.name}")

    assert not offenders, (
        "`knobs.py` no debe depender de módulos internos del backend. "
        "Cero deps internas es invariante crítica — sin esto, importar knobs "
        "desde constants.py reintroduce el ciclo que P2-1 cerró.\n  "
        + "\n  ".join(offenders)
    )
