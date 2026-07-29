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

[P0-TEST-DB-ISOLATION-2 · 2026-07-29 · code review] Tres huecos cerrados tras
adversarial review del fix original:

  5. La guarda #1 (`_guard_test_write_to_prod`) ahora se cablea TAMBIÉN a nivel
     de conexión (`db_core._install_write_guard_on_connection`, invocado desde
     `configure_sync_conn`/`configure_async_conn`) — no solo dentro de
     `execute_sql_write`. Censo del review: 27+ call sites (`update_plan_data_atomic`,
     el endpoint `/name`, cron_tasks.py, db_profiles.py, dreaming.py,
     routers/notifications.py) escriben abriendo `connection_pool.connection()`
     crudo y bypaseaban la guarda #1 por completo.
  6. `UNORDERED_LIMIT1_RE` dejó de operar sobre texto crudo del archivo (el
     lookahead `(?=["'])` era evadible por SQL multi-line triple-quoted,
     literales adyacentes partidos en su propia línea, concatenación `+`, y
     punto y coma final — los 4 estilos idiomáticos de este mismo repo). Ahora
     opera sobre el `ast.Constant.value` ya evaluado (`_iter_test_file_sql_literal_texts`),
     excluyendo docstrings por posición estructural (primer statement de
     Module/ClassDef/FunctionDef) en vez de por adyacencia de comillas.
  7. `test_sweep_predicate_cannot_match_real_user_content` ya no depende solo
     de un blacklist de 4 substrings — `_evaluate_sql_predicate_against_plan_data`
     evalúa el predicado REAL extraído del source contra payloads `plan_data`
     realistas (incluyendo un plan de owner sin `name`, el caso censado) y
     falla LOUD (`_UnrecognizedPredicateClause`) ante cualquier sintaxis que no
     reconozca — un catch-all reformulado ya no puede pasar en silencio.
"""
import ast
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
# [P0-TEST-DB-ISOLATION-2 · 2026-07-29] Ya NO lleva el lookahead `(?=["'])` de
# cierre — ese lookahead exigía que el "1" de "LIMIT 1" terminara pegado a la
# comilla de cierre del MISMO literal Python, y era la única defensa contra
# auto-detectar la prosa de ESTE archivo (y de los 4 corregidos) que documenta
# el bug entre backticks. El problema (hallado en review): ese mismo requisito
# hacía evadible la detección por CUALQUIERA de los 4 estilos idiomáticos de
# este repo — SQL triple-quoted con `LIMIT 1` en su propia línea antes del
# `\"\"\"` de cierre (cron_tasks.py), literales adyacentes partidos en su
# propia línea (`"...FROM tabla " "LIMIT 1"`, db_plans.py), concatenación
# `+`, o punto y coma final — en todos esos casos el carácter tras el "1" en
# el TEXTO CRUDO es un salto de línea o la comilla de APERTURA del siguiente
# literal, no la de cierre del mismo.
#
# El regex ahora se aplica sobre el valor YA EVALUADO de cada string-literal
# (`_iter_test_file_sql_literal_texts`, más abajo) — el mismo string que
# psycopg recibiría en runtime, sin comillas ni saltos de línea de por medio —
# así que el `\b` final basta; la exclusión de prosa/docstrings se resuelve
# estructuralmente (posición del nodo AST), no por adyacencia de comillas.
UNORDERED_LIMIT1_RE = re.compile(
    r"FROM\s+(user_profiles|meal_plans|user_inventory|plan_chunk_queue|"
    r"consumed_meals|user_facts)\s+LIMIT\s+1\b",
    re.IGNORECASE,
)

FIXED_FIXTURE_FILES = [
    "test_chunk_corrupted_plan_data_pauses.py",
    "test_p0_3_guard_pause_hard_fail.py",
    "test_p1_2_chunk_deferrals_telemetry.py",
    "test_p1_4_logging_preference.py",
]


def _all_test_files():
    """Todos los `test_*.py` de este directorio, EXCEPTO este propio archivo.

    [P0-TEST-DB-ISOLATION-2 · 2026-07-29] Este archivo (`test_p0_test_db_isolation.py`)
    es un META-test: contiene, A PROPÓSITO, texto de ejemplo con el patrón peligroso
    `FROM <tabla> LIMIT 1` como FIXTURES de datos (no como código que la suite
    ejecuta contra una DB real) para probar que `_find_unordered_victim_selections`
    SÍ lo detecta bajo los 4 estilos de formateo evasivos — ver
    `TestUnorderedLimit1DetectorSurvivesSourceFormattingStyles`. Ese texto vive
    dentro de string-literals de este mismo archivo (no dentro de un `ast.parse`
    anidado — el detector no re-parsea recursivamente Python-dentro-de-un-string),
    así que un self-scan RAW de este archivo se auto-detectaría — un falso positivo
    estructural, no el "prosa evade el lookahead" que el detector SÍ debe seguir
    atrapando en cualquier OTRO archivo de fixtures. Este archivo no siembra
    `meal_plans`/`user_profiles` (no tiene código de fixture real que escriba a
    Neon), así que excluirlo del sweep no reduce cobertura de seguridad real.
    """
    self_name = os.path.basename(__file__)
    return sorted(
        p for p in glob.glob(os.path.join(TESTS_DIR, "test_*.py"))
        if os.path.basename(p) != self_name
    )


def _iter_test_file_sql_literal_texts(source: str):
    """Extrae el contenido YA EVALUADO de los string-literals de un archivo
    Python — incluyendo concatenación implícita adyacente (`"a" "b"` → el
    parser de `ast` ya las fusiona en un solo `Constant` antes de que las
    veamos), concatenación con `+`, y las partes literales de f-strings —
    pero EXCLUYE los docstrings (módulo/clase/función: primer statement del
    body, criterio estructural, no textual) para no auto-detectar la prosa
    que documenta el bug entre backticks en este mismo archivo.

    [P0-TEST-DB-ISOLATION-2 · 2026-07-29] Reemplaza el regex-sobre-texto-crudo
    original. Operar sobre el AST hace irrelevante dónde caigan los saltos de
    línea o las comillas de cada literal — es el mismo string que psycopg
    recibiría en runtime, así que ningún estilo de formateo del SQL (multi-line,
    literales partidos, concatenación `+`) puede evadir la detección solo por
    reformatear DÓNDE están las comillas en el archivo fuente.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    # Docstrings: primer statement de Module/ClassDef/FunctionDef, si es una
    # expresión de solo-string. Identificados por `id()` del nodo Constant
    # para excluirlos abajo sin tener que re-caminar el árbol por posición.
    docstring_ids = set()
    containers = [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for c in containers:
        body = getattr(c, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_ids.add(id(body[0].value))

    def _eval_str_concat(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return None if id(node) in docstring_ids else node.value
        if isinstance(node, ast.JoinedStr):  # f-string: partes literales, placeholder por {expr}
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    parts.append(" ")
            return "".join(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _eval_str_concat(node.left)
            right = _eval_str_concat(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            val = _eval_str_concat(node)
            if val:
                yield val


def _find_unordered_victim_selections(source: str):
    """Devuelve la lista de matches de `UNORDERED_LIMIT1_RE` sobre los
    string-literals REALES (no-docstring) de `source` — SSOT usado por ambos
    tests #3 y #4 de este archivo."""
    offenders = []
    for literal_text in _iter_test_file_sql_literal_texts(source):
        offenders.extend(m.group(0) for m in UNORDERED_LIMIT1_RE.finditer(literal_text))
    return offenders


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
        for match in _find_unordered_victim_selections(text):
            offenders.append(f"{os.path.basename(path)}: {match!r}")
    assert not offenders, (
        "Patrón de selección de víctima sin ORDER BY/WHERE encontrado — adopta un "
        "usuario REAL determinísticamente. Usa la fixture `seeded_user_profile` "
        "de conftest.py (o tu propio `str(uuid.uuid4())`) en su lugar:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 3b. [P0-TEST-DB-ISOLATION-2 · code review] El detector #3 sobrevive a los 4
#     estilos idiomáticos de este repo que evadían el regex-sobre-texto-crudo
#     original (lookahead de comilla adyacente) — y sigue sin auto-detectar la
#     prosa/docstring que documenta el bug.
# ---------------------------------------------------------------------------
class TestUnorderedLimit1DetectorSurvivesSourceFormattingStyles:
    def test_catches_triple_quoted_multiline_limit_on_own_line(self):
        """Estilo de cron_tasks.py: SQL triple-quoted con `LIMIT 1` en su
        propia línea antes del `\"\"\"` de cierre. El regex original con
        lookahead de comilla NO lo atrapaba (el carácter tras "1" es un
        salto de línea, no una comilla)."""
        source = '''
def fresh_plan_evasion_1(cursor):
    cursor.execute(
        """
        SELECT id
        FROM user_profiles
        LIMIT 1
        """
    )
'''
        assert _find_unordered_victim_selections(source)

    def test_catches_adjacent_split_literal_own_line(self):
        """Estilo de db_plans.py: literales de string adyacentes (concat
        implícita de Python), cada uno en su propia línea. El texto CRUDO
        tiene una comilla de cierre + salto de línea + comilla de apertura
        entre 'user_profiles' y 'LIMIT' — rompía el `\\s+` del regex
        original."""
        source = '''
def fresh_plan_evasion_2(cursor):
    cursor.execute(
        "SELECT id FROM user_profiles "
        "LIMIT 1"
    )
'''
        assert _find_unordered_victim_selections(source)

    def test_catches_plus_concatenated_variant(self):
        source = '''
def fresh_plan_evasion_3(cursor):
    query = "SELECT id FROM user_profiles " + "LIMIT 1"
    cursor.execute(query)
'''
        assert _find_unordered_victim_selections(source)

    def test_catches_trailing_semicolon_variant(self):
        source = '''
def fresh_plan_evasion_4(cursor):
    cursor.execute("SELECT id FROM user_profiles LIMIT 1;")
'''
        assert _find_unordered_victim_selections(source)

    def test_does_not_flag_ordered_filtered_query(self):
        """Negativo: WHERE + ORDER BY entre FROM y LIMIT es exactamente lo
        que el fix pide usar — nunca debe dispararse contra esto."""
        source = '''
def legit_query(cursor, user_id):
    cursor.execute(
        """
        SELECT id FROM user_profiles
        WHERE id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,),
    )
'''
        assert not _find_unordered_victim_selections(source)

    def test_does_not_self_detect_module_docstring_prose(self):
        """Regresión: el propio archivo (y cualquier otro) documenta el bug
        citando `SELECT id FROM user_profiles LIMIT 1` entre backticks
        dentro de un docstring — eso NUNCA debe contar como violación real,
        ni con el nuevo extractor AST."""
        source = '''
"""
Este módulo documenta un incidente causado por
`SELECT id FROM user_profiles LIMIT 1` sin ORDER BY.
"""
import os
'''
        assert not _find_unordered_victim_selections(source)

    def test_does_not_self_detect_function_docstring_prose(self):
        source = '''
def some_test():
    """Regresión conocida: `SELECT id FROM meal_plans LIMIT 1` adoptaba la
    fila de un usuario real."""
    pass
'''
        assert not _find_unordered_victim_selections(source)

    # Nótese que NO hay un test "este archivo, leído crudo del disco, no se
    # auto-detecta" — a diferencia de la prosa/docstrings (excluidos
    # estructuralmente, ver `test_does_not_self_detect_*` arriba), los 4 tests
    # `test_catches_*` de esta clase embeben el patrón peligroso como DATO
    # (string fixture) dentro de código Python real de ESTE archivo, y el
    # detector no re-parsea recursivamente Python-dentro-de-un-string — un
    # self-scan raw SIEMPRE los encontraría. `_all_test_files()` excluye este
    # archivo del sweep de producción por esa razón exacta (ver su docstring).


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
        assert not _find_unordered_victim_selections(text), (
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


# ---------------------------------------------------------------------------
# 5b. [P0-TEST-DB-ISOLATION-2 · 2026-07-29 · code review] Cierre BEHAVIORAL de
#     #5: el blacklist de 4 substrings de arriba solo prueba que el texto NO
#     contiene esas 4 frases exactas — un catch-all reformulado semánticamente
#     idéntico (p.ej. `OR COALESCE(plan_data->>'name', '') = ''`) lo evade en
#     silencio porque la substring no está en la lista. Estos tests EJECUTAN
#     el predicado real extraído del source contra payloads `plan_data`
#     realistas, y el evaluador falla LOUD ante sintaxis que no reconoce —
#     así CUALQUIER reescritura no auditada del predicado (no solo las 4
#     frases del blacklist) rompe la suite.
# ---------------------------------------------------------------------------
class _UnrecognizedPredicateClause(Exception):
    """El evaluador no reconoce esta cláusula — se propaga (test failure) en
    vez de dejarla pasar en silencio como `True`/`False` arbitrario."""


_ATOM_HASKEY_NOT_RE = re.compile(r"NOT\s*\(\s*plan_data\s*\?\s*'(?P<key>\w+)'\s*\)", re.IGNORECASE)
_ATOM_HASKEY_RE = re.compile(r"plan_data\s*\?\s*'(?P<key>\w+)'", re.IGNORECASE)
_ATOM_ILIKE_RE = re.compile(r"plan_data->>'(?P<key>\w+)'\s+ILIKE\s+'(?P<pattern>[^']*)'", re.IGNORECASE)
_ATOM_EQ_RE = re.compile(r"plan_data->>'(?P<key>\w+)'\s*=\s*'(?P<value>[^']*)'")
_ATOM_ISNULL_RE = re.compile(r"plan_data->>'(?P<key>\w+)'\s+IS\s+NULL", re.IGNORECASE)


def _ilike_match(value, pattern: str) -> bool:
    """Emula Postgres ILIKE (case-insensitive, `%`=comodín) via fnmatch."""
    if value is None:
        return False
    import fnmatch
    fn_pattern = pattern.replace("%", "*").replace("_", "?")
    return fnmatch.fnmatch(str(value).lower(), fn_pattern.lower())


def _evaluate_sql_predicate_against_plan_data(sql_fragment: str, plan_data: dict) -> bool:
    """Evalúa (best-effort, gramática limitada al vocabulario que
    `_sweep_synthetic_test_plans` usa hoy: `->>'key' ILIKE`, `->>'key' =`,
    `->>'key' IS NULL`, `? 'key'` / `NOT (... ? 'key')`, `AND`/`OR`) el
    fragmento WHERE extraído del SOURCE de producción contra un dict
    `plan_data` realista. Sustituye cada cláusula atómica RECONOCIDA por su
    valor booleano Python y evalúa el resto como expresión pura de
    True/False/and/or/not/paréntesis.

    Cualquier cláusula NO reconocida (p.ej. `COALESCE(...)`, `LENGTH(...)`,
    cualquier función SQL fuera del vocabulario de arriba) deja texto
    residual no-booleano → `_UnrecognizedPredicateClause`. Esto es
    intencional: un catch-all reformulado que esta gramática no entiende
    DEBE romper la suite (forzar auditoría humana), no pasar en silencio."""
    s = sql_fragment.replace("%%", "%")  # des-escapar el %% de psycopg (ver ILIKE patterns)

    # Orden importa: NOT(plan_data ? 'k') antes que el positivo plan_data ? 'k'.
    s = _ATOM_HASKEY_NOT_RE.sub(lambda m: "False" if m.group("key") in plan_data else "True", s)
    s = _ATOM_HASKEY_RE.sub(lambda m: "True" if m.group("key") in plan_data else "False", s)
    s = _ATOM_ILIKE_RE.sub(
        lambda m: "True" if _ilike_match(plan_data.get(m.group("key")), m.group("pattern")) else "False",
        s,
    )
    s = _ATOM_EQ_RE.sub(
        lambda m: "True" if (
            plan_data.get(m.group("key")) is not None
            and str(plan_data.get(m.group("key"))) == m.group("value")
        ) else "False",
        s,
    )
    s = _ATOM_ISNULL_RE.sub(lambda m: "True" if plan_data.get(m.group("key")) is None else "False", s)

    s = re.sub(r"\bAND\b", "and", s, flags=re.IGNORECASE)
    s = re.sub(r"\bOR\b", "or", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNOT\b", "not", s, flags=re.IGNORECASE)
    s = s.strip()

    if not re.fullmatch(r"[\sA-Za-z()]+", s) or not re.search(r"\b(True|False)\b", s):
        raise _UnrecognizedPredicateClause(
            f"No pude traducir a booleano puro tras sustituir cláusulas conocidas: "
            f"residual={s!r} (original={sql_fragment!r})"
        )
    # eval() seguro aquí: `s` YA fue validado por el fullmatch de arriba a contener
    # EXCLUSIVAMENTE los tokens `True`/`False`/`and`/`or`/`not`/espacios/paréntesis — todo
    # lo demás (los literales `'...'` de los operandos SQL originales) fue reemplazado por
    # esas sustituciones ANTES de llegar aquí. No hay input de usuario ni texto SQL crudo
    # en `s` en este punto; `__builtins__` vacío además bloquea cualquier llamada aunque el
    # fullmatch tuviera un hueco. No usar `ast.literal_eval` porque el resultado no es un
    # literal Python (es una expresión booleana con `and`/`or`/`not`), y no usar
    # `eval(s)` sin namespace: el `{"__builtins__": {}}` explícito es la defensa real.
    return bool(eval(compile(s, "<sweep_predicate>", "eval"), {"__builtins__": {}}, {}))


def _extract_sweep_predicate_clauses(cron_tasks_source: str):
    """Devuelve (select_clause, update_clause): el bloque `(... OR ...)` de
    `plan_data->>'name' ILIKE ... OR plan_data->>'_test_fixture' = ...` de
    CADA una de las dos queries reales de `_sweep_synthetic_test_plans`
    (SELECT candidatos, UPDATE abandoned) — mismo aislamiento de bloques que
    usa `test_sweep_predicate_cannot_match_real_user_content` arriba."""
    start = cron_tasks_source.index("def _sweep_synthetic_test_plans")
    next_def = re.search(r"\ndef [A-Za-z_]", cron_tasks_source[start + 10:])
    body = cron_tasks_source[start: start + 10 + next_def.start()]

    select_match = re.search(r'SELECT id, user_id.*?"""', body, re.DOTALL)
    update_match = re.search(r'UPDATE meal_plans.*?"""', body, re.DOTALL)

    clause_re = re.compile(r"\(\s*plan_data->>'name'.*?\)", re.DOTALL)
    select_clause = clause_re.search(select_match.group(0)) if select_match else None
    update_clause = clause_re.search(update_match.group(0)) if update_match else None
    return (
        select_clause.group(0) if select_clause else None,
        update_clause.group(0) if update_clause else None,
    )


class TestSweepPredicateBehaviorAgainstRealisticPayloads:
    _REAL_OWNER_PLAN_NO_NAME = {"generation_status": "complete", "days": [{"day_name": "Lunes"}]}
    _REAL_OWNER_PLAN_CUSTOM_NAME = {"name": "Mi plan de la abuela"}
    _REAL_FIXTURE_LEGACY_NAME = {"name": "Plan Sintético 7 días — Test Chunks"}
    _REAL_FIXTURE_FLAG = {"name": "Mi plan", "_test_fixture": "true"}

    def _clauses(self):
        path = os.path.join(BACKEND_DIR, "cron_tasks.py")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        select_clause, update_clause = _extract_sweep_predicate_clauses(text)
        assert select_clause and update_clause, (
            "No se pudo extraer el bloque OR de una de las dos queries reales — "
            "¿cambió la forma del SQL en _sweep_synthetic_test_plans?"
        )
        return select_clause, update_clause

    def test_real_predicate_does_not_match_owner_plan_without_name(self):
        """El caso censado: 22 planes REALES del owner sin `name`. El
        predicado REAL (ejecutado, no grepeado) no debe marcarlos candidatos."""
        for clause in self._clauses():
            assert _evaluate_sql_predicate_against_plan_data(clause, self._REAL_OWNER_PLAN_NO_NAME) is False

    def test_real_predicate_does_not_match_owner_plan_with_custom_name(self):
        for clause in self._clauses():
            assert _evaluate_sql_predicate_against_plan_data(clause, self._REAL_OWNER_PLAN_CUSTOM_NAME) is False

    def test_real_predicate_matches_known_legacy_fixture_name(self):
        """Control positivo — si esto falla, el sweep dejó de reconocer lo
        que SÍ debe limpiar."""
        for clause in self._clauses():
            assert _evaluate_sql_predicate_against_plan_data(clause, self._REAL_FIXTURE_LEGACY_NAME) is True

    def test_real_predicate_matches_explicit_test_fixture_flag(self):
        for clause in self._clauses():
            assert _evaluate_sql_predicate_against_plan_data(clause, self._REAL_FIXTURE_FLAG) is True


class TestPredicateEvaluatorFailsLoudOnUnrecognizedSyntax:
    """Ancla el comportamiento fail-loud: si un futuro cambio a
    `_sweep_synthetic_test_plans` introduce sintaxis fuera del vocabulario
    reconocido (ILIKE / `= 'literal'` / IS NULL / `?` de existencia de key),
    el evaluador debe REVENTAR en vez de dejarlo pasar en silencio — así
    `TestSweepPredicateBehaviorAgainstRealisticPayloads` (que llama al mismo
    evaluador contra el source REAL) falla ruidosamente ante cualquier
    reescritura no auditada, no solo ante las 4 frases hardcoded del
    blacklist de la sección #5."""

    def test_unrecognized_clause_raises_instead_of_silently_passing(self):
        with pytest.raises(_UnrecognizedPredicateClause):
            _evaluate_sql_predicate_against_plan_data(
                "(COALESCE(plan_data->>'name', '') = '')",
                {"generation_status": "complete"},
            )

    def test_reviewers_exact_catchall_example_would_have_swept_a_real_plan(self):
        """Documenta — con una traducción DIRECTA a Python del catch-all que
        el reviewer propuso como escenario de fallo (COALESCE está fuera de
        la gramática del mini-evaluador, ver test anterior) — que ese
        catch-all SÍ habría marcado un plan REAL del owner sin `name` como
        candidato al sweep. Es la razón concreta por la que ese patrón está
        prohibido (sección #5, blacklist) Y por la que el evaluador debe
        fallar loud en vez de dejarlo pasar (test anterior)."""
        owner_plan_no_name = {"generation_status": "complete", "days": []}
        coalesce_name_empty_catchall = str(owner_plan_no_name.get("name") or "") == ""
        assert coalesce_name_empty_catchall is True


# ---------------------------------------------------------------------------
# 6. [P0-TEST-DB-ISOLATION-2 · 2026-07-29 · code review] La guarda #1 cubre
#    TAMBIÉN escrituras crudas que abren `connection_pool.connection()`
#    directo (bypaseando `execute_sql_write`) — p.ej. `update_plan_data_atomic`
#    (db_plans.py, el patrón PREFERIDO de la invariante I7), el endpoint
#    `/name`, cron_tasks.py, db_profiles.py, dreaming.py,
#    routers/notifications.py. Se instala envolviendo `conn.cursor` desde
#    `configure_sync_conn`/`configure_async_conn` — corre UNA VEZ por
#    conexión física, cubre TODOS los checkouts futuros de esa conexión.
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append(query)
        return "ok"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self):
        self._cursor = _FakeCursor()

    def cursor(self, *args, **kwargs):
        return self._cursor


class TestConnectionLevelWriteGuardCoversRawCallsites:
    def setup_method(self):
        self._orig_flag = db_core._CURRENT_TEST_IS_E2E

    def teardown_method(self):
        db_core._CURRENT_TEST_IS_E2E = self._orig_flag

    def _prod_env(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake_module::fake_test")
        monkeypatch.delenv("MEALFIT_ALLOW_TEST_WRITES_TO_PROD", raising=False)
        monkeypatch.setenv(
            "NEON_DATABASE_URL",
            "postgresql://user:pw@ep-cool-lake-123456.us-east-2.aws.neon.tech/mealfit",
        )

    def test_install_write_guard_blocks_raw_cursor_write(self, monkeypatch):
        """El helper standalone: un `cursor.execute("UPDATE ...")` crudo —
        exactamente lo que hace `update_plan_data_atomic` — se bloquea aunque
        nunca pase por `execute_sql_write`."""
        self._prod_env(monkeypatch)
        db_core._CURRENT_TEST_IS_E2E = False
        conn = _FakeConn()
        db_core._install_write_guard_on_connection(conn)
        cursor = conn.cursor()
        with pytest.raises(RuntimeError, match="P0-TEST-DB-ISOLATION"):
            cursor.execute("UPDATE meal_plans SET plan_data = %s WHERE id = %s", ("{}", "x"))

    def test_install_write_guard_lets_selects_through(self, monkeypatch):
        """Solo escrituras se guardan — un SELECT crudo (p.ej. el `SELECT
        plan_data FROM meal_plans WHERE id = %s FOR UPDATE` de
        `update_plan_data_atomic`) no debe bloquearse: la guarda original
        nunca protegió lecturas."""
        self._prod_env(monkeypatch)
        db_core._CURRENT_TEST_IS_E2E = False
        conn = _FakeConn()
        db_core._install_write_guard_on_connection(conn)
        cursor = conn.cursor()
        result = cursor.execute("SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE", ("x",))
        assert result == "ok"
        assert cursor.executed == ["SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE"]

    def test_install_write_guard_bypassed_when_e2e(self, monkeypatch):
        self._prod_env(monkeypatch)
        db_core._CURRENT_TEST_IS_E2E = True
        conn = _FakeConn()
        db_core._install_write_guard_on_connection(conn)
        cursor = conn.cursor()
        result = cursor.execute("UPDATE meal_plans SET plan_data = %s WHERE id = %s", ("{}", "x"))
        assert result == "ok"

    def test_real_configure_sync_conn_wires_the_guard(self, monkeypatch):
        """No solo el helper standalone: el cableado REAL en
        `configure_sync_conn` — el callback que psycopg_pool invoca al crear
        cada conexión física del pool sync. Sin esto cableado ahí,
        `update_plan_data_atomic` y el resto de los 27+ call sites que abren
        `connection_pool.connection()` crudo seguirían sin protección aunque
        el helper standalone funcione perfecto."""
        self._prod_env(monkeypatch)
        db_core._CURRENT_TEST_IS_E2E = False
        conn = _FakeConn()
        conn.autocommit = False
        conn.prepare_threshold = 5
        db_core.configure_sync_conn(conn)
        cursor = conn.cursor()
        with pytest.raises(RuntimeError, match="P0-TEST-DB-ISOLATION"):
            cursor.execute("DELETE FROM meal_plans WHERE id = %s", ("x",))

    def test_real_configure_async_conn_wires_the_guard(self, monkeypatch):
        """Variante async del test anterior — mismo cableado real, pool
        async (`configure_async_conn`)."""
        import asyncio

        self._prod_env(monkeypatch)
        db_core._CURRENT_TEST_IS_E2E = False

        class _FakeAsyncCursor:
            async def execute(self, query, params=None):
                return "ok"

        class _FakeAsyncConn:
            def __init__(self):
                self._cursor = _FakeAsyncCursor()

            def cursor(self, *args, **kwargs):
                return self._cursor

            async def set_autocommit(self, v):
                pass

        conn = _FakeAsyncConn()
        conn.prepare_threshold = 5

        asyncio.run(db_core.configure_async_conn(conn))
        cursor = conn.cursor()

        async def _run():
            await cursor.execute("UPDATE meal_plans SET plan_data = %s WHERE id = %s", ("{}", "x"))

        with pytest.raises(RuntimeError, match="P0-TEST-DB-ISOLATION"):
            asyncio.run(_run())

    def test_real_configure_checkpoint_conn_wires_the_guard(self, monkeypatch):
        """El pool del LangGraph checkpointer también queda cubierto —
        `checkpoints`/`checkpoint_writes` son tablas user-scoped igual que
        `meal_plans`."""
        self._prod_env(monkeypatch)
        db_core._CURRENT_TEST_IS_E2E = False
        conn = _FakeConn()
        conn.autocommit = False
        conn.prepare_threshold = 5
        db_core.configure_checkpoint_conn(conn)
        cursor = conn.cursor()
        with pytest.raises(RuntimeError, match="P0-TEST-DB-ISOLATION"):
            cursor.execute("INSERT INTO checkpoints (id) VALUES (%s)", ("x",))

