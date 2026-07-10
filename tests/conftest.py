"""
Shared fixtures for backend E2E tests.

Provides `seeded_user_profile`: inserts a synthetic user into user_profiles
and user_inventory so that E2E chunk tests never skip due to missing DB data.
[P1-E2E-FIXTURE-NEON · 2026-07-10] Neon no tiene schema `auth` — user_profiles es la raíz.
"""
# [P1-VERIFIED-ONLY-DEFAULT-ON · 2026-07-02] El default de CÓDIGO del knob verified-only pasó a
# True (cierra la regresión silenciosa ".env reseteado ⇒ enforcement apagado"). El baseline
# HISTÓRICO de esta suite se escribió con el enforcement OFF (los tests de coherencia construyen
# planes con alimentos sintéticos off-catálogo a propósito) → lo fijamos explícito aquí. Los tests
# del knob (test_p3_verified_ingredients_only / test_p1_objective_v4_batch) lo activan con
# monkeypatch cuando prueban el path ON. setdefault: una env var real del operador SIEMPRE gana.
import os as _os_conftest
_os_conftest.environ.setdefault("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "false")
# [P2-AUDIT-V5-BATCH · 2026-07-02] (GAP-14) Mismo patrón para strict-all-reasons: el default de
# CÓDIGO pasó a True en agent.py (cierra ".env reseteado ⇒ cravings/weekend vuelven a permitir
# ingredientes externos"); los tests legacy de cravings/weekend asumen el baseline OFF.
_os_conftest.environ.setdefault("MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS", "false")
# [P1-GATES-FLIP-ON · 2026-07-03] (audit v6 · P1-4) Los 3 gates OFF-de-nacimiento pasaron a ON
# en código con la serie del gym baseline (20 perfiles: contract 0/20 retry, ceiling 4/20,
# per-day floor 9/20). El baseline HISTÓRICO de la suite se escribió con los gates OFF (los
# fixtures construyen planes sintéticos que dispararían el sodio/contract gate a propósito) →
# se fijan OFF aquí. Los tests del flip (test_p1_gates_flip_on) verifican el default de CÓDIGO
# por source-parse y activan el path ON con monkeypatch. setdefault: env real del operador gana.
_os_conftest.environ.setdefault("MEALFIT_SODIUM_EXCESS_GATE", "false")
_os_conftest.environ.setdefault("MEALFIT_RECIPE_CONTRACT_GATE", "false")
_os_conftest.environ.setdefault("MEALFIT_MICRO_CLOSER_PERDAY", "false")

# [P0-5] Eagerly resolve real `langgraph` BEFORE any test module loads. Several
# test files do `sys.modules.setdefault('langgraph', MagicMock())` to support
# environments without the package, but `setdefault` only checks if the key is
# already in `sys.modules` — it cannot tell whether the existing entry is the
# real package or a previously-installed MagicMock. When the alphabetically-first
# test (e.g. test_chunk_learning_appears_in_prompt.py) ran its `setdefault`
# without `langgraph` yet in `sys.modules`, it installed a MagicMock, and every
# subsequent `from langgraph.checkpoint.memory import MemorySaver` (transitively
# pulled in by `cron_tasks` → `agent`) raised
# `ModuleNotFoundError: 'langgraph' is not a package`. Importing it here primes
# `sys.modules` with the real package so all later `setdefault`s become no-ops.
# Only stub if the real submodule path is genuinely unimportable (CI without
# the dependency installed).
try:
    import langgraph  # noqa: F401
    import langgraph.graph  # noqa: F401
    import langgraph.graph.message  # noqa: F401
    import langgraph.checkpoint.memory  # noqa: F401
except Exception:
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault("langgraph", MagicMock())
    sys.modules.setdefault("langgraph.graph", MagicMock())
    sys.modules.setdefault("langgraph.graph.message", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint.memory", MagicMock())
    sys.modules.setdefault("langgraph.checkpoint.postgres", MagicMock())

# [P0-5 · P0-DEEPSEEK-MIGRATION 2026-06-12] Same eager-import for
# `langchain_openai` (base client of the DeepSeek provider, see
# `llm_provider.py`). If a test file installs a partial stub first, a later
# import of the real surface (e.g. via cron_tasks → ai_helpers →
# llm_provider) raises ImportError. Importing the real package here primes
# sys.modules with the full surface, and subsequent stub `setdefault` /
# `_install_stub` calls become no-ops because the key is already populated.
try:
    import langchain_openai  # noqa: F401
    from langchain_openai import (  # noqa: F401
        ChatOpenAI,
        OpenAIEmbeddings,
    )
except Exception:
    import sys
    from unittest.mock import MagicMock
    if "langchain_openai" not in sys.modules:
        _stub = MagicMock()
        # ChatDeepSeek(ChatOpenAI) llama super().__init__(**kwargs); un stub
        # `= object` peta (object.__init__ no acepta kwargs) y rompe la colección
        # en entornos sin langchain_openai. Un stub-class que traga **kwargs sí
        # permite instanciar las subclases.
        class _StubLLM:
            def __init__(self, *args, **kwargs):
                pass
        _stub.ChatOpenAI = _StubLLM
        _stub.OpenAIEmbeddings = _StubLLM
        sys.modules["langchain_openai"] = _stub

import uuid
import json
import pytest
from datetime import datetime, timezone

from db_core import execute_sql_write, execute_sql_query, connection_pool


# ---------------------------------------------------------------------------
# Ensure the connection pool is open for test sessions
# ---------------------------------------------------------------------------
if connection_pool and not getattr(connection_pool, '_opened', False):
    connection_pool.open()


# ---------------------------------------------------------------------------
# Marker registration (also declared in pytest.ini at backend root)
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: End-to-end tests requiring a live database")


# ---------------------------------------------------------------------------
# Core fixture: synthetic user + plan_id, with full teardown
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_user_profile():
    """Create a throwaway user in user_profiles → user_inventory (Neon: sin auth.users).

    Yields (user_id, plan_id).  Teardown removes all traces in FK-safe order.
    """
    user_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    email = f"e2e-test-{user_id[:8]}@test.local"

    # --- Setup -----------------------------------------------------------
    # Pre-clean any leftover data from a previously interrupted test with
    # the same UUID (astronomically unlikely, but handles partial teardowns).
    for tbl in ("plan_chunk_queue", "meal_plans", "user_inventory"):
        execute_sql_write(f"DELETE FROM {tbl} WHERE user_id = %s", (user_id,))
    execute_sql_write("DELETE FROM user_profiles WHERE id = %s", (user_id,))

    # [P1-E2E-FIXTURE-NEON · 2026-07-10] Neon NO tiene schema `auth` (P1-NEON-DB-MIGRATION):
    # `user_profiles` es la tabla raíz (cero FKs). El INSERT a auth.users mataba en SETUP
    # los 8 E2E de chunks 7/15/30d + 23 tests más desde la migración (2026-06-12) —
    # relation "auth.users" does not exist. `email` va directo en user_profiles.
    # 1. user_profiles (raíz)
    health_profile = {
        "age": 30,
        "weight": 75,
        "height": 170,
        "gender": "M",
        "goal": "maintain",
        "activityLevel": "moderate",
        "dietType": "Omnívora",
        "allergies": [],
        "budget": "medium",
        "householdSize": 1,
        "tz_offset_minutes": -240,
    }
    execute_sql_write(
        "INSERT INTO user_profiles (id, email, health_profile) VALUES (%s, %s, %s::jsonb) "
        "ON CONFLICT (id) DO UPDATE SET health_profile = EXCLUDED.health_profile",
        (user_id, email, json.dumps(health_profile, ensure_ascii=False)),
    )

    # 3. user_inventory  (enough staples so pantry checks pass)
    pantry_items = [
        ("Pechuga de Pollo", 1000, "g"),
        ("Arroz", 2000, "g"),
        ("Habichuelas", 500, "g"),
        ("Res", 800, "g"),
        ("Pescado", 600, "g"),
        ("Huevos", 12, "unidad"),
        ("Aceite de Oliva", 500, "ml"),
        ("Cebolla", 500, "g"),
        ("Ajo", 100, "g"),
        ("Tomate", 400, "g"),
    ]
    for name, qty, unit in pantry_items:
        execute_sql_write(
            "INSERT INTO user_inventory (user_id, ingredient_name, quantity, unit) "
            "VALUES (%s, %s, %s, %s)",
            (user_id, name, qty, unit),
        )

    yield user_id, plan_id

    # --- Teardown (FK-safe order) ----------------------------------------
    execute_sql_write(
        "DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,)
    )
    execute_sql_write(
        "DELETE FROM meal_plans WHERE id = %s", (plan_id,)
    )
    # Also clean any plans created with this user_id outside the fixture plan_id
    execute_sql_write(
        "DELETE FROM plan_chunk_queue WHERE user_id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM meal_plans WHERE user_id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM user_inventory WHERE user_id = %s", (user_id,)
    )
    execute_sql_write(
        "DELETE FROM user_profiles WHERE id = %s", (user_id,)
    )
