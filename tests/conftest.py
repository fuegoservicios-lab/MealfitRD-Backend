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
import os
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

# [P0-5 · P0-LLM-PROVIDER-MIGRATION 2026-06-12] Same eager-import for
# `langchain_openai` (base client of the GLM provider, see
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
        # ChatGLM(ChatOpenAI) llama super().__init__(**kwargs); un stub
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

# [P0-CI-VERDICT · 2026-09-04] Mismo patrón para los otros módulos que los ficheros de test
# stubean con `if "X" not in sys.modules` en tiempo de import (`memory_manager`, `services`,
# `agent`, `db`, `db_inventory`). Medido 2026-09-04: `test_p0_3_legacy_learning_atomicity`
# instalaba un `memory_manager` de juguete sin `summarize_and_prune` y, cuando se colectaba
# antes que nadie importara el real, `routers.plans` moría con
# `ImportError: cannot import name 'summarize_and_prune' from 'memory_manager' (unknown
# location)` en un fichero que nada tenía que ver. Fail-open, igual que arriba.
for _eager_mod in ("memory_manager", "services", "agent", "db", "db_inventory"):
    try:
        __import__(_eager_mod)
    except Exception as _e_eager:  # pragma: no cover - depende del entorno
        import sys as _sys_eager
        print(f"[P0-CI-VERDICT] no se pudo pre-importar {_eager_mod}: "
              f"{type(_e_eager).__name__}: {_e_eager}", file=_sys_eager.stderr)

import ast
import sys
import uuid
import json
import pytest
import re
from datetime import datetime, timezone
from pathlib import Path

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
    config.addinivalue_line("markers", "needs_local_data: needs the live catalog/database or backend/.env (skipped in CI)")
    config.addinivalue_line(
        "markers",
        "frontend_cross_repo: tests whose subject includes the sibling frontend repo",
    )


# [P2-DEPLOY-FRONTEND-SALTA-LA-PARIDAD · 2026-08-23] Clasificación por
# propiedad: cada TEST que construya o consuma una ruta al repo frontend recibe
# el marker. Una lista manual de 246 tests quedaría obsoleta el primer día que
# nazca una paridad nueva; marcar el fichero entero ejecutaría miles de vecinos
# backend-only de los grandes batch files.
_FRONTEND_CROSS_REPO_CACHE: dict[Path, frozenset[str]] = {}


def _frontend_literal_or_alias(node: ast.AST, known_names: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            identifier = child.id.lower()
            if child.id in known_names or identifier == "frontend" or identifier.startswith(
                ("frontend_", "_frontend", "_front")
            ):
                return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            literal = child.value.lower().replace("\\", "/").strip()
            normalized_path = "/" + literal.strip("/") + "/"
            if (
                literal == "frontend"
                or literal.startswith("../frontend")
                or "/frontend/" in normalized_path
                or "frontend/src" in literal
            ):
                return True
    return False


def _assigned_names(node: ast.AST) -> set[str]:
    targets = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    return {
        child.id
        for target in targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name)
    }


def _frontend_cross_repo_test_names(path: Path) -> frozenset[str]:
    path = Path(path)
    if path in _FRONTEND_CROSS_REPO_CACHE:
        return _FRONTEND_CROSS_REPO_CACHE[path]
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        result = frozenset()
        _FRONTEND_CROSS_REPO_CACHE[path] = result
        return result

    frontend_values = {"frontend_repo_path"}
    changed = True
    while changed:
        changed = False
        for statement in tree.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None or not _frontend_literal_or_alias(value, frontend_values):
                continue
            before = len(frontend_values)
            frontend_values.update(_assigned_names(statement))
            changed = changed or len(frontend_values) != before

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    dependent_functions: set[str] = set()
    changed = True
    while changed:
        changed = False
        for function in functions:
            body = function.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            arguments = {
                arg.arg
                for arg in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            called_functions = {
                node.func.id
                for statement in body
                for node in ast.walk(statement)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            depends = (
                bool(arguments.intersection(dependent_functions))
                or bool(called_functions.intersection(dependent_functions))
                or any(
                    _frontend_literal_or_alias(statement, frontend_values)
                    for statement in body
                )
            )
            if not depends:
                continue
            if function.name not in dependent_functions:
                dependent_functions.add(function.name)
                changed = True
            for global_node in (
                node for node in ast.walk(function) if isinstance(node, ast.Global)
            ):
                before = len(frontend_values)
                frontend_values.update(global_node.names)
                changed = changed or len(frontend_values) != before

    result = frozenset(name for name in dependent_functions if name.startswith("test_"))
    _FRONTEND_CROSS_REPO_CACHE[path] = result
    return result


def _is_frontend_cross_repo_test_file(path: Path) -> bool:
    return bool(_frontend_cross_repo_test_names(path))


# ---------------------------------------------------------------------------
# [P0-CI-VERDICT · 2026-09-04] El CI del backend llevaba semanas sin veredicto.
#
# Medido en el run 1451 de `main` (2026-09-03): 1.412 failed + 464 errors. El marker
# `frontend_cross_repo` se ponía… y nadie lo leía: ningún hook saltaba los tests marcados, así
# que en un checkout sin el hermano privado (el secret `SIBLING_REPO_TOKEN` no está definido)
# 1.122 tests reventaban con FileNotFoundError sobre `frontend/`. Otros 177 buscaban
# `migrations/` y 74 `CLAUDE.md` en la RAÍZ del workspace (el runner sólo tiene `backend/`);
# el resto pedía artefactos que NO están versionados en ningún repo (`deploy-mealfit.ps1`,
# los runbooks de `~/.claude/…/memory`, `scratch/README.md`) o el catálogo VIVO de
# `master_ingredients` (sin DB en el runner).
#
# Un rojo permanente entrena a ignorar el CI entero. La degradación tiene que ser LEGIBLE:
#   1. Sin hermano frontend ⇒ los tests que lo leen se SALTAN (marker surgical + detección
#      gruesa por literal de ruta en el módulo, porque el AST no alcanza a los 414 ficheros).
#   2. `pytest_runtest_makereport` convierte en SKIP —con la razón— los fallos cuyo único
#      motivo es un artefacto fuera del repo backend que no existe en este checkout, y los
#      que dependen del catálogo vivo cuando no hay pool (la línea de log lo delata).
#   3. `ci.yml` emula la raíz del workspace para lo que SÍ versiona el backend
#      (`CLAUDE.md`, `migrations/`): esos ~250 tests vuelven a EJECUTARSE de verdad.
# Un skip cuenta y se ve (`-rs`); un rojo por FileNotFoundError no dice nada de producción.
# Anclado por tests/test_p0_ci_verdict.py.
# ---------------------------------------------------------------------------
import re as _re_ci

_CI_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_CI_WORKSPACE_ROOT = _CI_BACKEND_ROOT.parent
_CI_FRONTEND_ROOT = _CI_WORKSPACE_ROOT / "frontend"
_CI_FRONTEND_PRESENT = (_CI_FRONTEND_ROOT / "src").is_dir()
# Literal de ruta al hermano: `/ "frontend"`, `joinpath("frontend"`, `"frontend/src/…"`,
# `"../frontend"`. NO una mención en prosa (la palabra sin comillas no cuenta).
_CI_FRONTEND_PATH_LITERAL_RE = _re_ci.compile(
    r"""/\s*["']frontend["']|joinpath\(\s*["']frontend["']|["'](?:\.\./)?frontend["'/]"""
)
_CI_MODULE_MENTIONS_FRONTEND_CACHE: dict[Path, bool] = {}
# Artefactos del workspace-root que NINGÚN repo versiona. Citados por nombre en los asserts
# (`assert X.exists(), "falta scratch/README.md"`): si el nombre aparece en el mensaje y el
# fichero no existe en la raíz del workspace, el test no puede evaluarse en este checkout.
_CI_WORKSPACE_ROOT_ARTIFACTS = (
    "deploy-mealfit.ps1",
    "scratch/README.md",
    ".gitignore",
)
_CI_CATALOG_UNAVAILABLE_LOG = "No connection_pool available to fetch master_ingredients"
_CI_ABS_PATH_RE = _re_ci.compile(r"(?<![\w\-])((?:/[^\s'\"`:,;)\]]+)+)")


def _ci_module_mentions_frontend(path: Path) -> bool:
    path = Path(path)
    cached = _CI_MODULE_MENTIONS_FRONTEND_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        source = ""
    result = bool(_CI_FRONTEND_PATH_LITERAL_RE.search(source))
    _CI_MODULE_MENTIONS_FRONTEND_CACHE[path] = result
    return result

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] CLAUDE.md vive en el workspace raíz (repo PRIVADO): en el
# CI del backend sin `SIBLING_REPO_TOKEN` no existe, y 121 tests lo leen como literal de ruta
# (`_REPO_ROOT / "CLAUDE.md"`). Antes reventaban por construcción; ahora se SALTAN con motivo, y
# el conteo de `skipped` de pytest deja visible cuánto no se verificó. El frontend (público) y
# `migrations/` (enlace a la copia SSOT) sí están, así que sus tests corren.
_WORKSPACE = Path(__file__).resolve().parents[2]
_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _db_available() -> bool:
    # El checkout del dueño tiene `backend/.env` (con la URL de Neon); el CI no. Se decide por el
    # .env o por una señal EXPLÍCITA, no por `NEON_DATABASE_URL` suelta: el entorno del job resultó
    # tenerla definida y los módulos `needs_local_data` corrieron (y fallaron) sin base real.
    return (_BACKEND_DIR / ".env").exists() or os.environ.get("MEALFIT_TESTS_HAVE_DB") == "1"


# (regex sobre el FUENTE del módulo de test, requisito presente?, motivo del skip). Se evalúa
# una vez por archivo. En el checkout completo del dueño todo está y nada se salta; en el CI del
# backend cada familia se salta con su motivo y el conteo de `skipped` lo deja visible.
_LOCAL_ONLY = (
    (re.compile(r"""["']CLAUDE\.md["']"""), lambda: (_WORKSPACE / "CLAUDE.md").exists(),
     "workspace raíz ausente (repo privado): este test lee CLAUDE.md"),
    (re.compile(r"run_ci\.ps1|deploy-mealfit\.ps1|scripts/README|docs/superpowers"),
     lambda: (_WORKSPACE / "scripts" / "run_ci.ps1").exists() or (_WORKSPACE / "deploy-mealfit.ps1").exists(),
     "workspace raíz ausente (repo privado): este test lee scripts/ o docs/ del workspace"),
    (re.compile(r"\.claude[/\\\\]projects|runbook_"), lambda: (Path.home() / ".claude" / "projects").exists(),
     "memoria local del dueño ausente (~/.claude/projects): este test lee un runbook"),
    (re.compile(r"""["']\.env["']"""), lambda: (_BACKEND_DIR / ".env").exists(),
     "backend/.env ausente (secretos locales): este test lee el .env"),
    (re.compile(r"psycopg\.connect\(|connection_pool\.open\(|load_dotenv\("), _db_available,
     "sin base de datos (NEON_DATABASE_URL ni backend/.env): este test la necesita"),
)
_local_only_cache: dict = {}


def _local_only_reason(path: Path) -> str | None:
    key = str(path)
    if key not in _local_only_cache:
        reason = None
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            src = ""
        for rx, present, why in _LOCAL_ONLY:
            if rx.search(src) and not present():
                reason = why
                break
        _local_only_cache[key] = reason
    return _local_only_cache[key]


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.frontend_cross_repo
    skip_frontend = pytest.mark.skip(
        reason=f"[P0-CI-VERDICT] repo hermano frontend ausente: {_CI_FRONTEND_ROOT}"
    )
    for item in items:
        item_path = Path(str(getattr(item, "path", item.fspath)))
        item_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        surgical = item_name in _frontend_cross_repo_test_names(item_path)
        if surgical:
            item.add_marker(marker)
        if not _CI_FRONTEND_PRESENT and (surgical or _ci_module_mentions_frontend(item_path)):
            item.add_marker(skip_frontend)
        reason = _local_only_reason(item_path)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
        elif item.get_closest_marker("needs_local_data") and not _db_available():
            item.add_marker(pytest.mark.skip(
                reason="sin base de datos ni backend/.env: este módulo se declara needs_local_data"))


def _ci_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ci_path_outside_backend(raw: str):
    """Ruta absoluta fuera del repo backend que NO existe → razón de skip; si no, None.

    Sólo cuentan rutas bajo la raíz del workspace (`parents[2]`: el hermano frontend, el
    deploy script, `scratch/`…) o bajo el HOME del usuario (los runbooks de `~/.claude`). Un
    fragmento como `/frontend` suelto dentro de un mensaje de aserción NO es un artefacto
    ausente — sin esta acotación un assert que citara una ruta se convertía en skip.
    """
    try:
        candidate = Path(raw)
    except (TypeError, ValueError):
        return None
    if not candidate.is_absolute():
        return None
    if _ci_under(candidate, _CI_BACKEND_ROOT):
        return None  # dentro del backend: un fichero que falta AQUÍ sí es un fallo real
    if not (_ci_under(candidate, _CI_WORKSPACE_ROOT) or _ci_under(candidate, Path.home())):
        return None
    if candidate.exists():
        return None
    return f"artefacto fuera del repo backend ausente en este checkout: {candidate}"


def _ci_fuera_de_repo_skip_reason(excinfo, report):
    exc = excinfo.value
    msg = str(exc)
    # 1) FileNotFoundError / pytest.fail / AssertionError que citan una ruta absoluta fuera
    #    del backend y que no existe (hermano frontend, runbooks de memoria, deploy script…).
    candidates = []
    if isinstance(exc, FileNotFoundError) and getattr(exc, "filename", None):
        candidates.append(str(exc.filename))
    candidates.extend(_CI_ABS_PATH_RE.findall(msg))
    for raw in candidates:
        reason = _ci_path_outside_backend(raw.rstrip(".'\"`"))
        if reason:
            return reason
    # 2) Artefactos del workspace-root no versionados, citados por nombre.
    if isinstance(exc, (AssertionError, pytest.fail.Exception)):
        for rel in _CI_WORKSPACE_ROOT_ARTIFACTS:
            if rel in msg and not (_CI_WORKSPACE_ROOT / rel).exists():
                return f"artefacto del workspace-root no versionado ausente: {rel}"
    # 3) Catálogo vivo: sin pool, `get_master_ingredients()` devuelve vacío y lo dice en el
    #    log. Un test que lo necesitaba no puede evaluarse aquí; con DB corre entero.
    if getattr(db_core, "connection_pool", None) is None:
        for _name, content in getattr(report, "sections", ()):
            if _CI_CATALOG_UNAVAILABLE_LOG in content:
                return "catálogo vivo (master_ingredients) requerido y sin DB en este entorno"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Mismo contrato en COLECCIÓN: un módulo que lee `frontend/…` o un artefacto del workspace
    a nivel de import muere antes de tener items (pytest lo reporta como «ERROR collecting» y
    ningún hook por-test lo alcanza). Si el único motivo es un artefacto fuera del repo, el
    módulo entero se salta con la razón en vez de tumbar la sesión."""
    outcome = yield
    report = outcome.get_result()
    if not report.failed or report.longrepr is None:
        return
    text = str(report.longrepr)
    reason = None
    for raw in _CI_ABS_PATH_RE.findall(text):
        reason = _ci_path_outside_backend(raw.rstrip(".'\"`"))
        if reason:
            break
    if reason is None and not _CI_FRONTEND_PRESENT and "FileNotFoundError" in text and "frontend" in text:
        reason = f"repo hermano frontend ausente: {_CI_FRONTEND_ROOT}"
    if reason is None:
        return
    report.outcome = "skipped"
    report.longrepr = (str(getattr(collector, "path", collector.fspath)), 0, f"[P0-CI-VERDICT] {reason}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when not in ("setup", "call") or not report.failed or call.excinfo is None:
        return
    reason = _ci_fuera_de_repo_skip_reason(call.excinfo, report)
    if reason is None:
        return
    report.outcome = "skipped"
    report.longrepr = (str(item.path), item.location[1] or 0, f"[P0-CI-VERDICT] {reason}")


# ---------------------------------------------------------------------------
# [P2-CI-BACKEND-CERO-TESTS · 2026-08-23] Repos hermanos opcionales
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def frontend_repo_path() -> Path:
    """Resuelve ``../frontend`` tarde y salta solo tests cross-repo si no está.

    El workflow del repo backend hace checkout exclusivamente de backend. Las
    pruebas de paridad pueden usar el frontend cuando existe en el workspace de
    desarrollo, pero su ausencia nunca debe abortar la colección completa.
    """

    sibling = Path(__file__).resolve().parents[2] / "frontend"
    if not (sibling / "src").is_dir():
        pytest.skip(f"repo hermano ausente: {sibling}")
    return sibling


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
#
# [P1-COUNTRY-SYSTEM-F2 · Task 10 · 2026-08-18] 5º cache, MISMA enfermedad, cadena causal
# DISTINTA (no requiere `db_core.connection_pool` mockeado): `_VERIFIED_SHOPPING_NAMES`
# (`shopping_calculator.py`) deriva de `get_master_ingredients()` con su propio TTL de 300s.
# ~65 ficheros monkeypatchean `get_master_ingredients` directamente; si alguno dispara una
# derivación real (`_is_verified_for_shopping`/`_get_verified_shopping_name_set`) mientras el
# mock está activo, `monkeypatch` restaura la FUNCIÓN al salir pero NO el set derivado — sobrevive
# envenenado hasta 300s para el siguiente test que lo lea, sea cual sea su fichero. Encontrado por
# el reviewer de Task 9 corriendo la suite completa (orden-dependiente: verde en aislamiento,
# rojo en suite — `test_mereyes_es_verificado_para_compras_tras_el_alias` /
# `test_ciruela_ya_existe_en_catalogo_con_precio_cero_cambio_de_catalogo`,
# `test_p1_country_system_f2.py`); reproducido con el mismo mecanismo antes de curarlo (evidencia
# RED/GREEN completa en `.superpowers/sdd/2026-08-17-paises-fase-2/task-10-report.md`).
_CACHES_CONTAMINABLES = (
    ("shopping_calculator", "_master_cache", None),
    ("graph_orchestrator", "_PHANTOM_CATALOG_INDEX_CACHE", None),
    ("graph_orchestrator", "_CATALOG_DENSITY_INDEX_CACHE", None),
    ("shopping_calculator", "_VERIFIED_SHOPPING_NAMES", None),
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
_TABLAS_USER_ID_CACHE: list | None = None

# Las que ya limpian los DELETE explícitos del teardown, en su orden de claves foráneas.
# Se excluyen del barrido para no repetir trabajo — no por corrección: repetir un DELETE
# ya hecho es inofensivo.
_TABLAS_YA_EXPLICITAS = frozenset({"plan_chunk_queue", "meal_plans", "user_inventory", "user_profiles"})


def _tablas_con_user_id() -> list:
    """[P1-TEARDOWN-SWEEP · 2026-08-12] Tablas de `public` con columna `user_id`.

    Se pregunta al CATÁLOGO en vez de mantener una lista: una lista escrita el día que
    había tres tablas se queda corta en cuanto alguien añade la cuarta, y nadie se entera
    hasta que aparecen miles de filas huérfanas. Esta pregunta no envejece.

    Se consulta UNA vez por sesión (el esquema no cambia a mitad de corrida) y se cachea:
    el teardown corre por cada test.

    Fail-soft a la lista conocida: si el catálogo no se puede leer, es preferible limpiar
    lo de siempre que reventar el teardown entero y dejarlo TODO sucio.
    """
    global _TABLAS_USER_ID_CACHE
    if _TABLAS_USER_ID_CACHE is not None:
        return _TABLAS_USER_ID_CACHE

    try:
        filas = execute_sql_query(
            """
            SELECT c.table_name
              FROM information_schema.columns c
              JOIN information_schema.tables t
                ON t.table_schema = c.table_schema AND t.table_name = c.table_name
             WHERE c.table_schema = 'public'
               AND c.column_name = 'user_id'
               AND t.table_type = 'BASE TABLE'
             ORDER BY c.table_name
            """,
            fetch_all=True,
        ) or []
        nombres = [
            r["table_name"] if isinstance(r, dict) else r[0]
            for r in filas
        ]
        _TABLAS_USER_ID_CACHE = [n for n in nombres if n not in _TABLAS_YA_EXPLICITAS]
    except Exception:
        _TABLAS_USER_ID_CACHE = []

    return _TABLAS_USER_ID_CACHE


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
            # [P3-FIXTURE-TZ-SIGN · 2026-08-22] +240, no -240. Convención `getTimezoneOffset()`:
            # POSITIVO al OESTE, así que República Dominicana es +240. Con el signo invertido este
            # fixture situaba a cada usuario de e2e en UTC+4 —Bakú, ocho horas de diferencia—
            # mientras decía modelar RD, y un caso de frontera de día montado así es
            # autoconsistente: pasa igual contra el código correcto que contra el del signo al
            # revés, porque el error se cancela consigo mismo.
            # Verificado contra las cinco cuentas reales de producción, que llevan +240 todas.
            "tz_offset_minutes": 240,
        }
        execute_sql_write(
            "INSERT INTO user_profiles (id, email, health_profile) VALUES (%s, %s, %s::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET health_profile = EXCLUDED.health_profile",
            (user_id, email, json.dumps(health_profile, ensure_ascii=False)),
        )

        # 3. user_inventory  (enough staples so pantry checks pass)
        #
        # [P2-E2E-PANTRY-STOCK · 2026-08-15] El pollo se dimensiona por lo que COMEN
        # los mocks, no "a ojo". `test_chunked_30days_e2e` encola 9 chunks × 3 días y
        # su mock pide "100g pollo" TODOS los días: 2.700 g. Con 1.000 g las reservas
        # —que se acumulan tras cada merge y NO se liberan al completar (los
        # `release_chunk_reservations` son todos rutas de error)— agotaban la fila en
        # el chunk 5, y en el 6 la fila DESAPARECÍA de la nevera viva
        # (`db_inventory.py`: `available = max(qty - reserved, 0)` y luego
        # `if qty <= 0: continue`). El guard duro post-merge lo reportaba como
        # "Ingredientes COMPLETAMENTE INEXISTENTES: 100g pollo" — un mensaje que
        # acusa de ausencia lo que en realidad estaba RESERVADO por los chunks
        # anteriores del mismo plan.
        #
        # POR QUÉ APARECIÓ AHORA y no antes: hasta `P1-PANTRY-NAME-RESOLUTION`
        # (2026-08-07) la reserva era un no-op silencioso — buscaba por igualdad
        # exacta y 'Pechuga de pollo' ≠ 'Pechuga de Pollo', así que la nevera del
        # fixture nunca se vaciaba. Aquel P-fix arregló el descuento y destapó que
        # este fixture estaba 2,7× corto. El test no se rompió: se volvió honesto.
        #
        # ⚠️ NO se arregla stubeando `reserve_plan_ingredients` como hacen los tests
        # hermanos. Este es el E2E: la reserva real ES cobertura, y precisamente del
        # camino que cambió hace ocho días. Se arregla dándole de comer.
        #
        # 5.000 g = 2.700 g necesarios + margen. El margen no es adorno: si algún día
        # `pantry_names_match` resuelve "150g res" contra la fila `Res` (hoy NO lo
        # hace, por tokens distintos), ese ingrediente empezará a reservar también.
        pantry_items = [
            ("Pechuga de Pollo", 5000, "g"),
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

        # [P1-TEARDOWN-SWEEP · 2026-08-12] Barrido de TODA tabla con `user_id`, antes de
        # borrar el perfil.
        #
        # POR QUÉ EXISTE. Este teardown limpiaba tres tablas a mano. Las suites escriben
        # en muchas más —telemetría, coste, métricas de chunk, frecuencias de
        # ingrediente—, así que cada corrida dejaba filas cuyo dueño desaparecía un
        # instante después. Medido en producción el 2026-08-12: **7.540 filas huérfanas
        # de 600 dueños fantasma**, ninguno con un solo plan, comida o mensaje. Y el
        # 42% del libro de coste por usuario era de ellos, o sea que cualquier análisis
        # de gasto por persona estaba contaminado.
        #
        # POR QUÉ SE DERIVA DEL ESQUEMA Y NO ES UNA LISTA. Añadir los seis nombres que
        # faltaban hoy repetiría el error: la lista se escribió cuando había tres tablas,
        # y se quedó corta sin que nadie la tocara. Lo que no envejece es la pregunta
        # «¿qué tablas tienen un `user_id`?», y esa la contesta el catálogo. Una tabla
        # nueva queda cubierta el día que se crea.
        #
        # Va DESPUÉS de los DELETE explícitos de arriba (que son los que respetan el
        # orden de claves foráneas entre plan y cola) y ANTES de `user_profiles`, que es
        # la raíz. Cada uno aislado: una tabla que falle no puede dejar las demás sucias.
        for _tabla in _tablas_con_user_id():
            _safe_write(f"DELETE FROM {_tabla} WHERE user_id::text = %s", (user_id,), f"sweep {_tabla}")

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
def _describir_destino() -> tuple[str, str]:
    """[P1-TEST-RESIDUE-TARGET · 2026-07-31] ¿Contra qué base acabamos de correr?

    Devuelve `(nombre, gravedad)` para que los detectores digan la verdad en vez de
    asumir. Los dos avisos afirmaban "contra la base de PRODUCCIÓN" sin mirar — y en
    la primera corrida contra un branch de Neon eso ya era falso. Un detector que
    exagera se aprende a ignorar, y entonces no sirve el día que acierta.

    Reusa `_db_target_is_nonprod` (P0-TEST-DB-DUAL-URL), que exige que las DOS URLs
    sean no-producción: así el aviso no puede decir "branch de test" en una
    configuración a medias, que es justo cuando más importaría no relajarse.
    """
    try:
        es_nonprod, _ = db_core._db_target_is_nonprod()
    except Exception:
        return "la base configurada", "AVISO"
    if es_nonprod:
        return "el branch de test (no producción)", "NOTA"
    return "la base de PRODUCCIÓN", "AVISO"


def pytest_sessionfinish(session, exitstatus):
    # [P1-TEST-RESIDUE-TELEMETRY · 2026-07-31] PRIMERO el chequeo de telemetría fantasma:
    # el de usuarios de abajo tiene `return`s tempranos, así que colgarlo al final lo
    # dejaría sin ejecutar justo cuando NO hay usuarios residuales — que es el caso
    # normal y precisamente cuando la telemetría sí puede haber quedado sucia.
    _reportar_telemetria_fantasma()

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
    # [P1-TEST-RESIDUE-TARGET · 2026-07-31] Decir contra QUÉ base, no asumir producción.
    _donde_u, _grav_u = _describir_destino()
    print(
        f"\n[P1-TEST-RESIDUE-DETECTOR] {_grav_u}: {len(filas)} usuario(s) de test VIVOS en "
        f"{_donde_u} tras la corrida — el teardown no completó:",
        file=_sys_res.stderr,
    )
    for _f in filas[:10]:
        print(f"    {str(_f.get('id'))[:8]}  {_f.get('email')}", file=_sys_res.stderr)
    print(
        "    Limpieza: DELETE de user_inventory / meal_plans / plan_chunk_queue / "
        "user_profiles por ese user_id (mismo orden FK-safe que el fixture).",
        file=_sys_res.stderr,
    )


# [P1-TEST-RESIDUE-TELEMETRY · 2026-07-31] El detector de arriba mira `user_profiles`
# y ahí se queda corto, porque el residuo que MÁS daño hizo no tenía perfil que mirar.
#
# Incidente del 31 jul: `chunk_lesson_telemetry` tenía 2.237 filas de 447 `user_id` que
# NUNCA existieron como perfil — el 94% de la tabla. Consecuencias medidas:
#
#   1. El cron de flota `_alert_high_synthesized_lesson_ratio` llevaba semanas midiendo
#      la suite de tests en vez del producto, y disparó una alerta por ello.
#   2. Falseó una cifra que se reportó como diagnóstico ("843 eventos en la semana 2"
#      cuando los reales eran 26). El ruido de tests no solo escribe en producción:
#      contamina las mediciones que haces SOBRE producción, incluidas las que usas para
#      decidir si hay un bug.
#
# Por qué aquí y no montando una base aparte: esa decisión está tomada y documentada
# arriba (el catálogo de 204 alimentos debe ser el REAL). Esto no la revisa — hace lo
# que esa misma decisión promete, "convertir un riesgo silencioso en uno visible",
# para el residuo que el detector original no podía ver.
#
# Barato: UNA query agregada, sin recorrer tablas. Best-effort como su hermano.
def _reportar_telemetria_fantasma():
    """Invocado desde `pytest_sessionfinish`, NO es un hook.

    pytest solo llama hooks por nombre exacto: si esto se llamara
    `pytest_sessionfinish_algo` no lo ejecutaría nadie y sería un detector inerte —
    verde para siempre, vigilando nada.
    """
    # [P1-TEARDOWN-SWEEP · 2026-08-12] Antes miraba UNA tabla: `chunk_lesson_telemetry`.
    # Por eso no vio nada cuando había 7.540 filas huérfanas repartidas en SEIS tablas
    # (llm_usage_events 2.576, pipeline_metrics 1.868, plan_chunk_metrics 1.225,
    # chunk_deferrals 886, ingredient_frequencies 690 y, sí, 295 en la que sí miraba).
    #
    # Un detector que vigila una tabla de seis no es que avise poco: es que su silencio
    # se lee como «todo limpio». Ahora pregunta al catálogo igual que el barrido del
    # teardown, así que una tabla nueva entra en la vigilancia el día que se crea y no el
    # día que alguien se acuerda.
    _tablas = ["meal_plans", "user_inventory", "plan_chunk_queue"] + list(_tablas_con_user_id())
    if not _tablas:
        return
    _union = "\n UNION ALL\n".join(
        f"""SELECT '{t}' AS tabla, count(*)::int AS n,
                   count(DISTINCT x.user_id::text)::int AS usuarios
              FROM {t} x
             WHERE x.user_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM user_profiles p
                                WHERE p.id::text = x.user_id::text)
            HAVING count(*) > 0"""
        for t in _tablas
    )
    try:
        filas = execute_sql_query(_union) or []
    except Exception:  # sin DB / red caída: el detector nunca estorba
        return
    if not filas:
        return
    import sys as _sys_tel
    _donde, _gravedad = _describir_destino()
    for _f in filas:
        print(
            f"\n[P1-TEST-RESIDUE-TELEMETRY] {_gravedad}: {_f.get('n')} fila(s) en "
            f"{_f.get('tabla')} de {_f.get('usuarios')} user_id SIN perfil — telemetría "
            f"fantasma escrita por la suite contra {_donde}.",
            file=_sys_tel.stderr,
        )
    if _gravedad == "AVISO":
        print(
            "    Por qué importa: varios crons agregan estas tablas SIN filtrar por "
            "usuario, así que estas filas mueven métricas de producto — y el libro de "
            "coste por usuario deja de poder responder cuánto gasta la gente de verdad.\n"
            "    Limpieza (por cada tabla de arriba):\n"
            "      DELETE FROM <tabla> x WHERE x.user_id IS NOT NULL AND NOT EXISTS\n"
            "        (SELECT 1 FROM user_profiles p WHERE p.id::text = x.user_id::text);",
            file=_sys_tel.stderr,
        )
    else:
        print(
            "    Inocuo aquí: en un branch el residuo no toca ninguna métrica de "
            "producto. Se sigue reportando porque significa que un teardown no "
            "completó, y eso conviene saberlo antes de que la corrida sea contra prod.",
            file=_sys_tel.stderr,
        )
