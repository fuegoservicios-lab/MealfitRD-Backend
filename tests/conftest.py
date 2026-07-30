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

# [P1-CONFTEST-EAGER-GO · 2026-07-26] MISMO patrón, aplicado al módulo del repo que más se
# stubea: `graph_orchestrator`.
#
# Varios archivos de test instalan stubs con la forma `if "X" not in sys.modules: <stub>` EN
# TIEMPO DE IMPORT (p.ej. test_p0_a_zombie_partial_finalize, test_p2_crons_health_aggregate).
# Ese guard no distingue "no está cargado" de "está cargado el real": si el archivo que lo
# ejecuta se colecta ANTES que nadie importe el módulo de verdad, deja un stub parcial
# instalado para TODA la sesión. Los tests posteriores que hagan `import graph_orchestrator`
# reciben ese stub y revientan con
#
#     AttributeError: module 'graph_orchestrator' has no attribute '<lo que sea>'
#
# Es exactamente el fallo que este conftest ya documenta y cura para `langgraph` (P0-5) — la
# misma trampa del `setdefault`, otro módulo. Medido 2026-07-26: el glob
# polish/display/finalize daba 105 rojos + 24 errores de colección, y los mismos archivos
# pasaban en verde corriendo solos.
#
# Importarlo aquí, antes de que se colecte ningún test, hace que todos esos guards sean no-op.
# Fail-open: si el import real falla (entorno sin deps), se deja pasar y cada test se apaña
# como hasta ahora — este conftest no debe ser quien tumbe la suite.
try:
    import graph_orchestrator  # noqa: F401
except Exception as _e_go_eager:  # pragma: no cover - depende del entorno
    import sys as _sys_go
    print(f"[P1-CONFTEST-EAGER-GO] no se pudo pre-importar graph_orchestrator: "
          f"{type(_e_go_eager).__name__}: {_e_go_eager}", file=_sys_go.stderr)

import sys
import uuid
import json
import pytest
from datetime import datetime, timezone

import db_core
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
# [P0-TEST-DB-ISOLATION · 2026-07-29] Guarda de escrituras reales a Neon PRODUCCIÓN.
#
# `db_core._guard_test_write_to_prod` (el único cuello de botella de escritura,
# `execute_sql_write`) bloquea con RuntimeError cualquier INSERT/UPDATE/DELETE que
# ocurra bajo pytest, A MENOS que el test en curso esté marcado `@pytest.mark.e2e`
# (o `pytestmark = pytest.mark.e2e` a nivel de módulo). Esta fixture autouse es la
# que le dice a `db_core` cuál es "el test en curso": lee el marker del nodo ANTES
# de que corra cualquier fixture de setup (incl. `seeded_user_profile` / `fresh_plan`
# de los archivos E2E), y restaura el valor previo al salir — necesario porque los
# tests corren en el MISMO proceso/módulo y `_CURRENT_TEST_IS_E2E` es un flag global.
#
# Censo P0-TEST-DB-ISOLATION: de los 12 archivos que escriben de verdad hoy, 8 ya
# llevaban `@pytest.mark.e2e`; los 4 restantes (los del incidente) lo ganan en el
# mismo commit que corrige su fixture `fresh_plan`. Con eso, ESTA guarda no rompe
# ningún test existente — solo cierra la puerta a que un test NUEVO, no marcado,
# escriba prod por accidente.
@pytest.fixture(autouse=True)
def _guard_test_writes_to_prod(request):
    is_e2e = request.node.get_closest_marker("e2e") is not None
    _previo = db_core._CURRENT_TEST_IS_E2E
    db_core._CURRENT_TEST_IS_E2E = is_e2e
    try:
        yield
    finally:
        db_core._CURRENT_TEST_IS_E2E = _previo


# ---------------------------------------------------------------------------
# [P1-MODULE-IDENTITY-RESTORE · 2026-07-26] Restaurar la IDENTIDAD de los módulos
# sensibles después de cada test.
#
# Varios tests necesitan re-importar un módulo para que un knob lazy relea el env:
#
#     if "graph_orchestrator" in sys.modules:
#         del sys.modules["graph_orchestrator"]
#     import graph_orchestrator
#
# El re-import es legítimo. Lo que NO lo es: dejar en `sys.modules` un objeto de
# identidad DISTINTA para el resto de la sesión. Los tests ya cargados retienen
# una referencia al módulo v1; los que corran después reciben el v2. Un
# `patch.object(modulo, "X")` parchea uno y el código bajo prueba lee el otro.
#
# Medido por bisect 2026-07-26: **14 de 37 rojos de `tests/test_p1_*.py` eran esto**
# — archivos que en aislamiento pasan y en suite fallan. Dos culpables distintos
# (`test_p1_1_convert_amount_density_fallback`, `test_p1_2_env_str_coherence_block_action`)
# y seis archivos más con el mismo patrón sin curar, cada uno con su forma.
#
# El repo ya conocía el fallo: `test_fatigue_decay_constants_consistency` lo documenta
# como P0-5 y se cura A MANO ("SALVAR el módulo original y RESTAURARLO en el finally").
# La cura per-archivo no escala — el séptimo que lo escriba mal vuelve a envenenar la
# sesión. Esto lo cierra en UN sitio, para los que existen y los que vengan.
#
# El test SÍ recibe su módulo fresco (la fixture restaura al SALIR), así que ninguno
# pierde lo que necesita: solo deja de contaminar a los siguientes.
# Lista DELIBERADAMENTE estrecha: solo los dos módulos cuya contaminación se rastreó por bisect.
#
# La primera versión vigilaba además `shopping_calculator`, `constants`, `cron_tasks` y `app`.
# Medido sobre la suite COMPLETA: 225 → 211 fallos, pero con **2 nuevos**
# (`test_chunked_generation::test_edge_case_one_or_two_days`,
# `test_p1_bigfruit_gram_hint::test_hint_yields_serving_macros_not_whole_fruit`) — dos tests que
# pasaban gracias a la contaminación de otro y que, al restaurarles la identidad original, se
# quedaron sin ella. Ninguno falla en aislamiento.
#
# Restaurar de más también rompe. Se limita al par medido; si aparece otra contaminación
# rastreada, se añade con su bisect, no por si acaso.
# [P1-SUITE-SWEEP · 2026-07-27] +shopping_calculator: `test_p1_3_shopping_coherence_knobs_registered`
# lo borra de sys.modules sin restaurar (su `_fresh_modules()` borra graph_orchestrator Y
# shopping_calculator; este fixture solo devolvía el primero). Bisect del par: ese archivo +
# las víctimas reproduce 3 de los 4 rojos de orden (caps_last_word ×2, recipe_step failsafe).
_MODULOS_VIGILADOS = ("graph_orchestrator", "db_inventory", "shopping_calculator")


@pytest.fixture(autouse=True)
def _restaurar_identidad_de_modulos():
    _antes = {m: sys.modules.get(m) for m in _MODULOS_VIGILADOS}
    yield
    for _m, _obj in _antes.items():
        if _obj is not None and sys.modules.get(_m) is not _obj:
            sys.modules[_m] = _obj


# [P1-CATALOG-INDEX-NO-STICKY · 2026-07-29] Hermana de la de arriba, para el OTRO vector: la
# identidad del módulo puede estar intacta y su CACHÉ envenenada.
#
# Cadena medida (3 eslabones, el culpable ~800 ficheros antes en orden alfabético):
#   1. `test_chunked_learning_propagation.py::_run_process` parchea `db_core.connection_pool` con un
#      MagicMock. `shopping_calculator` importó `connection_pool` por valor (sigue siendo el real y
#      pasa el `if`), pero `execute_sql_query` lee `db_core.connection_pool` en tiempo de llamada →
#      `get_master_ingredients()` cachea el MagicMock como catálogo. El `patch` se revierte al
#      salir; `_master_cache` NO, porque es estado de módulo y nadie lo restauraba.
#   2. El primer `_phantom_catalog_index()` posterior itera el MagicMock → TypeError → fail-open →
#      cachea `{}`… y el guard de entrada es `is not None`. (Eso era un bug de PRODUCCIÓN y se
#      arregló ahí; este fixture es la otra mitad.)
#   3. Con el índice vacío el resolvedor devuelve `(None, None)` y lo memoiza, así que
#      `_raw_display_parallel_by_food` da True para listas PERMUTADAS y `_sync_one_raw_line` no
#      escala nada — 8 rojos en 6 ficheros, todos verdes en aislamiento.
#
# Punto ÚNICO a propósito: hay ~25 ficheros que parchean `db_core.connection_pool`, y cualquiera
# puede volver a envenenar el catálogo. Se mantiene la disciplina de la fixture de arriba (limitar
# a lo medido): estos 4 caches son los que el bisect atravesó, no una lista preventiva.
_CACHES_CONTAMINABLES = (
    ("shopping_calculator", "_master_cache", None),
    ("graph_orchestrator", "_PHANTOM_CATALOG_INDEX_CACHE", None),
    ("graph_orchestrator", "_CATALOG_DENSITY_INDEX_CACHE", None),
)


@pytest.fixture(autouse=True)
def _limpiar_caches_de_catalogo():
    yield
    for _mod_name, _attr, _val in _CACHES_CONTAMINABLES:
        _mod = sys.modules.get(_mod_name)
        if _mod is not None and hasattr(_mod, _attr):
            try:
                setattr(_mod, _attr, _val)
            except Exception:
                pass
    _go = sys.modules.get("graph_orchestrator")
    _memo = getattr(_go, "_LINE_FOOD_GRAMS_CACHE", None) if _go is not None else None
    if isinstance(_memo, dict):
        _memo.clear()      # memoiza (None, None) por línea: sobrevive al reset de los índices


# ---------------------------------------------------------------------------
# Core fixture: synthetic user + plan_id, with full teardown
# ---------------------------------------------------------------------------
def _safe_write(query: str, params: tuple, label: str) -> None:
    """[P0-TEST-DB-ISOLATION · 2026-07-29] DELETE de teardown aislado: un fallo en
    UNA sentencia (lock, blip de red) ya no aborta las siguientes — cada tabla se
    limpia independientemente. No es protección contra SIGKILL (nada lo es; por
    eso el marker `_test_fixture` en `plan_data` — reapable por
    `_sweep_synthetic_test_plans` — es la defensa que sí sobrevive un proceso
    muerto a mitad de corrida), pero sí cierra el caso más común y más barato de
    curar: una excepción individual (no una interrupción del proceso) que dejaba
    el resto del teardown sin ejecutar.
    """
    try:
        execute_sql_write(query, params)
    except Exception as _err:
        import sys as _sys_safe_write
        print(
            f"[P0-TEST-DB-ISOLATION] teardown falló (no bloqueante) — {label}: {_err}",
            file=_sys_safe_write.stderr,
        )


@pytest.fixture
def seeded_user_profile():
    """Create a throwaway user in user_profiles → user_inventory (Neon: sin auth.users).

    Yields (user_id, plan_id).  Teardown removes all traces in FK-safe order.
    """
    user_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    email = f"e2e-test-{user_id[:8]}@test.local"

    # --- Setup -------------------------------------------------------------
    # [P0-TEST-DB-ISOLATION · 2026-07-29] Envuelto en try/except: si el INSERT
    # de user_profiles o alguno de los de user_inventory revienta a mitad
    # (DB blip, columna renombrada), las filas YA insertadas antes del fallo se
    # limpian aquí mismo en vez de quedar huérfanas — el `yield` nunca se
    # alcanza en ese caso, así que el teardown de abajo (post-yield) NO
    # correría por sí solo. Se relanza la excepción original: el test sigue
    # fallando de forma visible, solo que sin dejar basura en prod.
    try:
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
    except Exception:
        _safe_write("DELETE FROM user_inventory WHERE user_id = %s", (user_id,), "setup-fail cleanup user_inventory")
        _safe_write("DELETE FROM meal_plans WHERE user_id = %s", (user_id,), "setup-fail cleanup meal_plans")
        _safe_write("DELETE FROM plan_chunk_queue WHERE user_id = %s", (user_id,), "setup-fail cleanup plan_chunk_queue")
        _safe_write("DELETE FROM user_profiles WHERE id = %s", (user_id,), "setup-fail cleanup user_profiles")
        raise

    try:
        yield user_id, plan_id
    finally:
        # --- Teardown (FK-safe order) — cada DELETE aislado, ver _safe_write. ---
        _safe_write("DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,), "plan_chunk_queue(plan_id)")
        _safe_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,), "meal_plans(plan_id)")
        # Also clean any plans created with this user_id outside the fixture plan_id
        _safe_write("DELETE FROM plan_chunk_queue WHERE user_id = %s", (user_id,), "plan_chunk_queue(user_id)")
        _safe_write("DELETE FROM meal_plans WHERE user_id = %s", (user_id,), "meal_plans(user_id)")
        _safe_write("DELETE FROM user_inventory WHERE user_id = %s", (user_id,), "user_inventory")
        _safe_write("DELETE FROM user_profiles WHERE id = %s", (user_id,), "user_profiles")


# ---------------------------------------------------------------------------
# [P1-TEST-RESIDUE-DETECTOR · 2026-07-27] Avisar si la corrida deja filas de test
# vivas en la base de PRODUCCIÓN.
#
# Contexto medido hoy: esta suite escribe en Neon PROD (no hay base de test; el
# `seeded_user_profile` de arriba crea un usuario real y lo borra en teardown).
# Eso NO es basura acumulada — se midió: 0 residuos de corridas anteriores, el
# teardown es fiable — pero sí dos riesgos:
#
#   1. Contaminación del monitoreo. Un vigilante que avise de "planes nuevos" ve
#      los fixtures como si fueran de usuarios reales. Hoy casi tomo un plan de
#      test (27 días, lista de 4 items) como la evidencia para girar
#      `MEALFIT_SHOPPING_COHERENCE_GUARD` a `block`.
#   2. Riesgo de cola: si una corrida muere a mitad (Ctrl-C, timeout, OOM), el
#      teardown no corre y el usuario sintético se queda en prod.
#
# Montar una base de test aparte NO está justificado hoy: habría que mantener
# sincronizado el catálogo de 204 alimentos, y estos tests valen precisamente
# porque validan contra el catálogo REAL. Lo que sí compensa es esto: convertir
# un riesgo silencioso en uno visible.
#
# NO falla la suite a propósito — un residuo no invalida los resultados, solo hay
# que limpiarlo. Se imprime al final, donde el operador lo ve.
def pytest_sessionfinish(session, exitstatus):
    try:
        filas = execute_sql_query(
            "SELECT id, email FROM user_profiles "
            "WHERE email LIKE 'e2e-test-%' OR email LIKE '%@test.local'"
        ) or []
    except Exception as _e:  # sin DB / red caída: el detector nunca estorba
        return
    if not filas:
        return
    import sys as _sys_res
    print(
        f"\n[P1-TEST-RESIDUE-DETECTOR] AVISO: {len(filas)} usuario(s) de test VIVOS en la base "
        f"de producción tras la corrida — el teardown no completó:",
        file=_sys_res.stderr,
    )
    for _f in filas[:10]:
        print(f"    {str(_f.get('id'))[:8]}  {_f.get('email')}", file=_sys_res.stderr)
    print(
        "    Limpieza: DELETE de user_inventory / meal_plans / plan_chunk_queue / "
        "user_profiles por ese user_id (mismo orden FK-safe que el fixture).",
        file=_sys_res.stderr,
    )
