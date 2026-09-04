"""[P0-CI-VERDICT · 2026-09-04] El CI del backend vuelve a dar un veredicto.

Medido en el run 1451 de `main` (2026-09-03): **1.412 failed + 464 errors** sobre ~22.000
tests, y no era código roto — era la disposición del runner:

  · 1.122 tests leían `frontend/` (el hermano privado no se clona sin `SIBLING_REPO_TOKEN`).
    El marker `frontend_cross_repo` SE PONÍA… y nadie lo leía: ningún hook saltaba lo marcado.
  · 177 buscaban `migrations/` y 74 `CLAUDE.md` en la raíz del workspace (`parents[2]`), donde
    viven como copia SSOT de los ficheros del backend; el runner sólo tiene `backend/`.
  · El resto pedía artefactos que NINGÚN repo versiona (`deploy-mealfit.ps1`, runbooks de
    `~/.claude/…/memory`, `scratch/README.md`, el `.env` local) o el catálogo VIVO de
    `master_ingredients` (sin DB en el runner).

Un rojo permanente entrena a ignorar el CI entero, y dentro vivían el P0 clínico de alérgenos y
los guards de i18n. La degradación tiene que ser LEGIBLE: un skip con razón cuenta y se ve; un
FileNotFoundError no dice nada de producción.

Tres piezas, ancladas aquí:
  1. `tests/conftest.py`: los tests marcados `frontend_cross_repo` (y los módulos con un literal
     de ruta al hermano) se SALTAN cuando el frontend no está; `pytest_runtest_makereport`
     convierte en SKIP los fallos cuyo único motivo es un artefacto fuera del repo o el catálogo
     vivo ausente.
  2. `.github/workflows/ci.yml`: emula la raíz del workspace para lo que el backend SÍ versiona
     (`CLAUDE.md`, `migrations/`) y lista los skips por razón (`-rs`).
  3. `graph_orchestrator.py`: `logger` definido junto a su import (P0-ORCH-LOGGER-BOOT) — el
     aviso de import de P2-ORCH-8 lo usaba 1.700 líneas antes de que existiera.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_CONFTEST = (_BACKEND / "tests" / "conftest.py").read_text(encoding="utf-8")
_CI = (_BACKEND / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ── 1. conftest: los marcados se saltan de verdad ─────────────────────────────

def test_los_items_frontend_cross_repo_se_saltan_sin_hermano():
    i = _CONFTEST.index("def pytest_collection_modifyitems(")
    cuerpo = _CONFTEST[i:i + 1500]
    assert "pytest.mark.skip(" in cuerpo and "_CI_FRONTEND_PRESENT" in cuerpo, (
        "El marker `frontend_cross_repo` se ponía y nadie lo leía: sin un `skip` condicionado a "
        "la presencia del hermano, 1.122 tests reventaban con FileNotFoundError en CI."
    )
    assert "_ci_module_mentions_frontend(item_path)" in cuerpo, (
        "Falta la detección gruesa por literal de ruta: el AST surgical no alcanza a los 414 "
        "módulos que construyen rutas a `frontend/` de otras formas."
    )


def test_el_detector_grueso_reconoce_los_literales_reales_y_no_la_prosa():
    m = re.search(r"_CI_FRONTEND_PATH_LITERAL_RE = _re_ci\.compile\(\n\s*r\"\"\"(.+?)\"\"\"\n\)", _CONFTEST, re.S)
    assert m, "no encuentro _CI_FRONTEND_PATH_LITERAL_RE en el conftest"
    rx = re.compile(m.group(1))
    for literal in ('_ROOT / "frontend" / "src"', "REPO_ROOT.joinpath('frontend', 'src')",
                    'os.path.join(ROOT, "frontend/src/pages")', "Path('../frontend')"):
        assert rx.search(literal), f"el detector debe reconocer {literal!r}"
    for prosa in ('"""El frontend hermano…"""', "# el frontend no importa aquí", "'front end'"):
        assert not rx.search(prosa), f"el detector NO debe disparar con prosa: {prosa!r}"


def test_makereport_convierte_en_skip_solo_lo_que_no_se_puede_evaluar_aqui():
    assert "def pytest_runtest_makereport(item, call):" in _CONFTEST
    assert "def _ci_fuera_de_repo_skip_reason(excinfo, report):" in _CONFTEST
    i = _CONFTEST.index("def _ci_fuera_de_repo_skip_reason(")
    cuerpo = _CONFTEST[i:i + 3000]
    # Un fichero que falta DENTRO del backend es un fallo real, nunca un skip.
    j = _CONFTEST.index("def _ci_path_outside_backend(")
    fuera = _CONFTEST[j:j + 900]
    assert "_ci_under(candidate, _CI_BACKEND_ROOT)" in fuera and "return None  # dentro del backend" in fuera
    # …y sólo cuentan rutas bajo el workspace o el HOME: un `/frontend` suelto en un mensaje de
    # aserción no es un artefacto ausente (falso positivo medido el 2026-09-04 en este mismo test).
    assert "_ci_under(candidate, _CI_WORKSPACE_ROOT) or _ci_under(candidate, Path.home())" in fuera
    # El catálogo vivo se detecta por la línea de log que emite shopping_calculator sin pool.
    assert "_CI_CATALOG_UNAVAILABLE_LOG" in cuerpo and 'connection_pool", None) is None' in cuerpo
    sc = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "No connection_pool available to fetch master_ingredients" in sc, (
        "La línea de log que delata el catálogo ausente cambió en shopping_calculator.py: "
        "actualiza _CI_CATALOG_UNAVAILABLE_LOG en el conftest o el skip deja de reconocerla."
    )


def test_los_modulos_stubeados_en_import_se_precargan():
    """`test_p0_3_legacy_learning_atomicity` instala un `memory_manager` de juguete si nadie
    importó el real antes; precargarlo hace el guard no-op (mismo patrón que P1-CONFTEST-EAGER-GO)."""
    for mod in ("memory_manager", "services", "agent"):
        assert f'"{mod}"' in _CONFTEST.split("for _eager_mod in (")[1].split(")")[0], (
            f"{mod} debe estar en la lista de pre-imports del conftest"
        )


# ── 2. ci.yml: la raíz del workspace emulada y los skips visibles ─────────────

def test_ci_emula_la_raiz_para_lo_que_el_backend_versiona():
    assert "ln -s backend/CLAUDE.md CLAUDE.md" in _CI and "ln -s backend/migrations migrations" in _CI, (
        "Sin los enlaces, 177 tests de migraciones y 74 de CLAUDE.md mueren con FileNotFoundError "
        "en el runner (la raíz del workspace no existe allí)."
    )
    assert "ln -s backend/.github" not in _CI, (
        "NO enlazar `.github` en la raíz: test_p1_live_2_ci_gate (G31) exige que la raíz NO tenga "
        "un `workflows/ci.yml` (el monorepo muerto)."
    )


def test_ci_lista_los_skips_por_razon():
    m = re.search(r"pytest tests/ -v --tb=short -m \"not e2e\"([^\n]*)", _CI)
    assert m and "-rs" in m.group(1), (
        "El paso de pytest debe llevar `-rs`: un verde con cientos de skips se lee sólo si el "
        "resumen dice POR QUÉ se saltaron (hermano ausente / catálogo vivo / artefacto)."
    )


# ── 3. graph_orchestrator: el logger existe antes de usarse ───────────────────

def test_logger_definido_antes_del_primer_uso_en_import():
    tree = ast.parse(_GO)
    def_line = next(
        n.lineno for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "logger" for t in n.targets)
    )
    first_use = next(
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "logger"
    )
    assert def_line < first_use, (
        f"`logger` se usa en la línea {first_use} y se define en la {def_line}: con "
        "MEALFIT_LLM_MAX_PER_USER < PLAN_CHUNK_SIZE el módulo moría con NameError al importarse "
        "(P0-ORCH-LOGGER-BOOT)."
    )
    assert "P0-ORCH-LOGGER-BOOT" in _GO
