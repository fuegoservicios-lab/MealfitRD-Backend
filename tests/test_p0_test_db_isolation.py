"""
P0-TEST-DB-ISOLATION · 2026-07-29

Un fixture de test (`fresh_plan` en test_chunk_corrupted_plan_data_pauses.py, y 3
copias hermanas) elegía su usuario "víctima" con `SELECT id FROM user_profiles
LIMIT 1` — sin ORDER BY, un accidente de orden físico de almacenamiento, no
intención — y ese SELECT devolvía determinísticamente al OWNER real. El INSERT
en `meal_plans` que seguía escribió un plan corrupto (sin `plan_data.name`) en la
cuenta REAL y rompió su dashboard; sobrevivió porque el proceso de test murió
antes del `yield`'s teardown.

Este archivo ancla las 4 defensas del fix (censo completo: CLAUDE.md sección
`P0-TEST-DB-ISOLATION` / memoria del proyecto):

  1. Guarda en tiempo de ejecución (`db_core._guard_test_write_to_prod`, cableada
     en el ÚNICO cuello de botella `execute_sql_write`): bloquea escrituras reales
     bajo pytest a menos que el test esté marcado `@pytest.mark.e2e`. No-op fuera
     de pytest — nunca toca runtime normal.
  2. Guarda estática (parser-based) sobre TODO `backend/tests/*.py`: ningún
     fixture puede volver a elegir una víctima con `SELECT ... FROM <tabla real>
     LIMIT 1` sin WHERE/ORDER BY.
  3. Los 4 archivos corregidos estampan `_test_fixture: true` (+ el `name` legado
     `Plan Sintético... — Test...`) en cada `meal_plans` que siembran, para que
     `cron_tasks._sweep_synthetic_test_plans` los reclame si el teardown no corre.
  4. El predicado del sweep amplió su OR a ese flag SIN volverse un catch-all que
     pudiera atrapar contenido real (el censo encontró 22 planes reales del
     owner sin `name` — un predicado `name IS NULL` habría sido catastrófico).

Para cada aserción: si saboteas la implementación en memoria, esta aserción
(y solo esta) debe fallar. Ver el reporte de la sesión para el detalle de qué
sabotaje disparó qué aserción.
"""
import glob
import os
import re

import pytest

import db_core


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)

# Mismo patrón que causó el incidente: FROM <tabla real> directo a LIMIT 1, sin
# WHERE ni ORDER BY de por medio. Deliberadamente amplio (no solo user_profiles)
# — el mismo accidente es igual de peligroso sobre cualquier tabla user-scoped.
#
# El lookahead `(?=["'])` exige que el match termine justo donde cierra el
# string literal de Python (`"...LIMIT 1"` / `'...LIMIT 1'`) — es lo que
# distingue una query REAL ejecutándose de la prosa de ESTE MISMO archivo (y de
# los 4 corregidos) que describe el bug entre backticks para documentarlo. Sin
# el lookahead, este propio test se auto-detecta como violación.
UNORDERED_LIMIT1_RE = re.compile(
    r"FROM\s+(user_profiles|meal_plans|user_inventory|plan_chunk_queue|"
    r"consumed_meals|user_facts)\s+LIMIT\s+1(?=[\"'])",
    re.IGNORECASE,
)

FIXED_FIXTURE_FILES = [
    "test_chunk_corrupted_plan_data_pauses.py",
    "test_p0_3_guard_pause_hard_fail.py",
    "test_p1_2_chunk_deferrals_telemetry.py",
    "test_p1_4_logging_preference.py",
]


def _all_test_files():
    return sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py")))


# ---------------------------------------------------------------------------
# 1. La guarda dispara al escribir contra una URL de DB con pinta de producción.
# ---------------------------------------------------------------------------
class TestGuardFiresAgainstProdLookingUrl:
    def setup_method(self):
        self._orig_flag = db_core._CURRENT_TEST_IS_E2E

    def teardown_method(self):
        db_core._CURRENT_TEST_IS_E2E = self._orig_flag

    def test_fires_when_unmarked_and_prod_url(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_module::fake_test")
        monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", raising=False)
        monkeypatch.setenv(
            "NEON_DATABASE_URL",
            "postgresql://user:pw@ep-cool-lake-123456.us-east-2.aws.neon.tech/mealfit",
        )
        db_core._CURRENT_TEST_IS_E2E = False
        with pytest.raises(RuntimeError, match="P0-TEST-DB-ISOLATION"):
            db_core._guard_test_write_to_prod(
                "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)"
            )

    def test_fires_via_the_real_execute_sql_write_chokepoint(self, monkeypatch):
        """No solo la función standalone: el cableado real en execute_sql_write."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_module::fake_test")
        monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", raising=False)
        monkeypatch.setenv(
            "NEON_DATABASE_URL",
            "postgresql://user:pw@ep-cool-lake-123456.us-east-2.aws.neon.tech/mealfit",
        )
        db_core._CURRENT_TEST_IS_E2E = False
        with pytest.raises(RuntimeError, match="P0-TEST-DB-ISOLATION"):
            # Si la guarda NO estuviera cableada aquí, esto intentaría abrir una
            # conexión real (y fallaría de otra forma, o peor, escribiría).
            db_core.execute_sql_write(
                "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
                ("x", "y", "{}"),
            )

    def test_escape_hatch_env_var_bypasses_guard(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_module::fake_test")
        monkeypatch.setenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", "1")
        monkeypatch.setenv(
            "NEON_DATABASE_URL",
            "postgresql://user:pw@ep-cool-lake-123456.us-east-2.aws.neon.tech/mealfit",
        )
        db_core._CURRENT_TEST_IS_E2E = False
        # No debe lanzar — es el escape hatch documentado en el mensaje de error.
        db_core._guard_test_write_to_prod("INSERT INTO meal_plans (id) VALUES (%s)")

    def test_does_not_fire_when_db_url_already_looks_like_test(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_module::fake_test")
        monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", raising=False)
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://u:p@localhost:5432/mealfit_test")
        db_core._CURRENT_TEST_IS_E2E = False
        db_core._guard_test_write_to_prod("INSERT INTO meal_plans (id) VALUES (%s)")


# ---------------------------------------------------------------------------
# 2. La guarda NO dispara para escrituras runtime normales (fuera de pytest) ni
#    para tests marcados e2e.
# ---------------------------------------------------------------------------
class TestGuardDoesNotFireLegitimately:
    def setup_method(self):
        self._orig_flag = db_core._CURRENT_TEST_IS_E2E

    def teardown_method(self):
        db_core._CURRENT_TEST_IS_E2E = self._orig_flag

    def test_no_fire_outside_pytest(self, monkeypatch):
        """Runtime real (app en producción, cron, dev local sin pytest)."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        db_core._CURRENT_TEST_IS_E2E = False
        # No debe lanzar pese a NO estar marcado e2e — PYTEST_CURRENT_TEST ausente
        # es la señal de "esto no es un test corriendo".
        db_core._guard_test_write_to_prod("INSERT INTO meal_plans (id) VALUES (%s)")

    def test_no_fire_when_marked_e2e(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_module::fake_test")
        monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", raising=False)
        monkeypatch.setenv(
            "NEON_DATABASE_URL",
            "postgresql://user:pw@ep-cool-lake-123456.us-east-2.aws.neon.tech/mealfit",
        )
        db_core._CURRENT_TEST_IS_E2E = True
        db_core._guard_test_write_to_prod("INSERT INTO meal_plans (id) VALUES (%s)")

    @pytest.mark.e2e
    def test_conftest_autouse_fixture_sets_flag_true_when_test_is_marked_e2e(self):
        """Prueba el cableado REAL de conftest.py (no un monkeypatch del flag):
        este test method está marcado @pytest.mark.e2e — al llegar aquí,
        `_guard_test_writes_to_prod` (autouse) ya debió fijar el flag en True."""
        assert db_core._CURRENT_TEST_IS_E2E is True

    def test_conftest_autouse_fixture_sets_flag_false_when_test_is_unmarked(self):
        """Compañero sin marker: mismo mecanismo, valor opuesto."""
        assert db_core._CURRENT_TEST_IS_E2E is False


# ---------------------------------------------------------------------------
# 3. Ningún fixture en backend/tests/ elige una víctima con LIMIT 1 sin
#    WHERE/ORDER BY — parser-based sobre TODO el directorio, no solo los 4
#    archivos conocidos, para atrapar una reintroducción futura.
# ---------------------------------------------------------------------------
def test_no_unordered_victim_selection_limit1_in_any_test_file():
    offenders = []
    for path in _all_test_files():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in UNORDERED_LIMIT1_RE.finditer(text):
            offenders.append(f"{os.path.basename(path)}: {m.group(0)!r}")
    assert not offenders, (
        "Patrón de selección de víctima sin ORDER BY/WHERE encontrado — adopta un "
        "usuario REAL determinísticamente. Usa la fixture `seeded_user_profile` "
        "de conftest.py (o tu propio `str(uuid.uuid4())`) en su lugar:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 4. Los 4 archivos corregidos estampan el marker que el sweep cron matchea, Y
#    ya no llevan el patrón roto, Y quedan marcados @pytest.mark.e2e (para que
#    la guarda #1 no los bloquee).
# ---------------------------------------------------------------------------
def test_fixed_fixture_files_stamp_the_sweep_marker():
    for fname in FIXED_FIXTURE_FILES:
        path = os.path.join(TESTS_DIR, fname)
        assert os.path.isfile(path), f"{fname} no existe — ¿se movió/renombró?"
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert "_test_fixture" in text and re.search(r'"_test_fixture"\s*:\s*True', text), (
            f"{fname}: falta el flag boolean '_test_fixture': True en su plan_data "
            f"sembrado — el sweep no lo reclamaría si el teardown no corre."
        )
        assert re.search(r"Plan Sintético.*—\s*Test", text), (
            f"{fname}: falta el `name` legado 'Plan Sintético ... — Test ...' "
            f"que matchea el predicado ILIKE original del sweep."
        )
        assert re.search(r"pytest\.mark\.e2e|pytestmark\s*=\s*pytest\.mark\.e2e", text), (
            f"{fname}: sin @pytest.mark.e2e / pytestmark, la guarda #1 bloquearía "
            f"sus escrituras reales."
        )
        assert not UNORDERED_LIMIT1_RE.search(text), (
            f"{fname}: todavía contiene el patrón roto original."
        )


# ---------------------------------------------------------------------------
# 5. El predicado del sweep NO puede matchear un plan de un usuario real con
#    contenido real: solo el `name` legado exacto O el flag explícito
#    `_test_fixture = 'true'` — nunca un catch-all como "name IS NULL" (el
#    censo P0-TEST-DB-ISOLATION midió 22 planes REALES del owner sin `name`;
#    ese catch-all los habría arrastrado al sweep).
# ---------------------------------------------------------------------------
def test_sweep_predicate_cannot_match_real_user_content():
    path = os.path.join(BACKEND_DIR, "cron_tasks.py")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    start = text.index("def _sweep_synthetic_test_plans")
    next_def = re.search(r"\ndef [A-Za-z_]", text[start + 10:])
    assert next_def is not None, "No se encontró el cierre de _sweep_synthetic_test_plans."
    body = text[start: start + 10 + next_def.start()]

    # Aislar el TEXTO SQL real (dentro de las f-strings triple-quoted), no el
    # docstring/comentarios que rodean la función — esos mencionan
    # '_test_fixture'/'Plan Sintético' en prosa varias veces y un simple
    # `body.count(...) >= 2` los contaría como si fueran las dos queries reales,
    # dejando pasar un sabotaje que solo rompe UNA de las dos SQL de verdad.
    select_match = re.search(r'SELECT id, user_id.*?"""', body, re.DOTALL)
    update_match = re.search(r'UPDATE meal_plans.*?"""', body, re.DOTALL)
    assert select_match is not None, "No se encontró el bloque SQL del SELECT de candidatos."
    assert update_match is not None, "No se encontró el bloque SQL del UPDATE abandoned."

    # El flag explícito y el name legado deben aparecer en AMBAS queries reales:
    # una en el SELECT de candidatos, otra en el UPDATE que marca `abandoned` —
    # deben estar de acuerdo (un drift entre los dos sería un bug de
    # reapabilidad silenciosa: el SELECT encuentra el plan pero el UPDATE ya no
    # lo alcanza, dejándolo "encontrado pero nunca marcado").
    for _label, _sql in (("SELECT", select_match.group(0)), ("UPDATE", update_match.group(0))):
        assert "_test_fixture" in _sql, (
            f"{_label}: falta el flag '_test_fixture' en el SQL real (no solo en "
            f"comentarios/docstring)."
        )
        assert "Plan Sintético" in _sql, (
            f"{_label}: falta el name legado 'Plan Sintético...— Test...' en el "
            f"SQL real (no solo en comentarios/docstring)."
        )

    # Guardrail negativo: nunca un predicado que trate la AUSENCIA de `name` (o
    # de la presencia/ausencia de cualquier otra key) como señal de
    # test-fixture — eso SÍ atraparía contenido real (censo: 22 planes REALES
    # del owner sin `name`). Nótese que esto es MÁS estrecho que "IS NULL no
    # aparece en ningún lado del cuerpo" — la función tiene un `dead_lettered_at
    # IS NULL` legítimo y no relacionado en el UPDATE de cancelación de chunks;
    # lo que se prohíbe es específicamente ese IS NULL aplicado a `name` (o
    # ausencia de key vía `?`).
    _dangerous_catchalls = (
        "'name' IS NULL",
        "'name') IS NULL",
        "NOT (plan_data ? 'name')",
        "NOT plan_data ? 'name'",
    )
    for _pat in _dangerous_catchalls:
        assert _pat not in body, (
            f"Predicado peligroso encontrado: {_pat!r} — trataría la AUSENCIA de "
            f"'name' como señal de test-fixture y atraparía planes reales sin "
            f"nombre (censo: 22 planes del owner, 2026-07-29)."
        )

