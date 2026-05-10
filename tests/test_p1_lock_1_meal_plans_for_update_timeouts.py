"""[P1-LOCK-1 · 2026-05-10] Regression guard: cada `SELECT … FOR UPDATE`
sobre `meal_plans` DEBE estar precedido por `set_meal_plan_for_update_timeouts(cursor)`.

Bug observado (auditoría 2026-05-10):
    Logs Postgres prod mostraron una transacción esperando 92.5s un
    `AccessExclusiveLock` sobre tuple de `meal_plans` antes de ser
    cancelada por `statement_timeout`. Sin `lock_timeout` local explícito,
    el default de Postgres (lock_timeout=0 = infinito) deja al caller
    esperando indefinidamente. Cuatro call sites tomaban
    `SELECT … FOR UPDATE` sobre `meal_plans` sin bound de espera:
        - `routers/plans.py:/shift-plan`
        - `cron_tasks.py:T1 worker merge`
        - `cron_tasks.py:T2 worker merge`
        - `cron_tasks.py:_background_rolling_refill`

Fix:
    Helper `db_plans.set_meal_plan_for_update_timeouts(cursor)` que setea
    `SET LOCAL lock_timeout` + `SET LOCAL statement_timeout` desde dos
    knobs (`MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS` /
    `MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS`). Aplicado en los 4 sites.

Cobertura:
    1. Helper existe, lee ambos knobs y emite los 2 SET LOCAL.
    2. Knobs se auto-registran en `_KNOBS_REGISTRY` al ser leídos.
    3. Cada `meal_plans … FOR UPDATE` en los 2 archivos relevantes tiene
       `set_meal_plan_for_update_timeouts` antes en el mismo bloque
       (defensivo contra futuros sites nuevos que olviden el helper).

Excluido del scope:
    `db_plans.update_plan_data_atomic` ([db_plans.py:263](backend/db_plans.py#L263))
    tiene su propio `SET LOCAL lock_timeout` con knob
    `CHUNK_LEARNING_LOCK_TIMEOUT_MS` desde antes (P0-2). NO debe migrar
    al nuevo helper — su patrón es válido y intencional.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_HELPER_NAME = "set_meal_plan_for_update_timeouts"

# Archivos con call sites que DEBEN usar el helper.
# `db_plans.py` queda fuera intencionalmente (ver docstring del módulo).
_FILES_TO_AUDIT = [
    _BACKEND_ROOT / "routers" / "plans.py",
    _BACKEND_ROOT / "cron_tasks.py",
]

# Detector: cada `FOR UPDATE` cuya query asociada (delimitada por el
# `cursor.execute(` / `execute_sql_query(` / `query =` previo más cercano)
# mencione `meal_plans`. Esto evita falsos positivos del estilo "una
# query sin FOR UPDATE seguida cerca por otra con FOR UPDATE no
# relacionada" (bug atrapado durante implementación de P1-LOCK-1).
_FOR_UPDATE_RE = re.compile(r"\bFOR\s+UPDATE\b", re.IGNORECASE)
# Patrón de "inicio de query": un cursor.execute(, execute_sql_query(,
# o asignación `query = "..."`. Definido ampliamente porque el repo
# usa los 3 estilos. Cualquier ocurrencia de uno de estos delimita
# el comienzo de la query asociada al `FOR UPDATE` siguiente.
_QUERY_START_RE = re.compile(
    r"(?:cursor\.execute\s*\(|execute_sql_query\s*\(|execute_sql_write\s*\(|query\s*=)",
    re.IGNORECASE,
)


def _is_in_python_comment(text: str, offset: int) -> bool:
    """Heurística simple: la posición está en un Python comment si en
    la misma línea, antes del offset, hay un `#` (asume que `#` no
    aparece dentro de un string SQL — verdad en el repo actual).
    """
    line_start = text.rfind("\n", 0, offset) + 1
    line_prefix = text[line_start:offset]
    return "#" in line_prefix


def _is_in_python_docstring(text: str, offset: int) -> bool:
    """Heurística: estamos dentro de un docstring si el conteo de
    `\"\"\"` (o `'''`) antes del offset es impar — significa que un
    docstring está abierto y aún no se cierra.

    Limitación: si el código contiene `\"\"\"` literal en otra posición
    (raro), confunde el conteo. No vista en el repo actual.
    """
    triple_double = text.count('"""', 0, offset)
    triple_single = text.count("'''", 0, offset)
    return (triple_double % 2 == 1) or (triple_single % 2 == 1)


def _find_for_update_sites(text: str) -> list[int]:
    """Devuelve offsets (caracter) de cada `FOR UPDATE` que cumple:
        (a) NO está en un Python comment (filtra menciones en docs/notes).
        (b) Su query asociada — del último `cursor.execute(`/
            `execute_sql_query(`/`query =` previo hasta el `FOR UPDATE` —
            mencione `meal_plans`. Excluye FOR UPDATE de otras tablas.
    """
    sites: list[int] = []
    for m in _FOR_UPDATE_RE.finditer(text):
        for_update_offset = m.start()
        # (a) Skip menciones dentro de comentarios Python.
        if _is_in_python_comment(text, for_update_offset):
            continue
        # (a') Skip menciones dentro de docstrings (función/módulo).
        if _is_in_python_docstring(text, for_update_offset):
            continue
        # (b) Buscar el inicio de la query asociada.
        query_starts = list(_QUERY_START_RE.finditer(text, 0, for_update_offset))
        if not query_starts:
            continue
        query_start = query_starts[-1].start()
        query_window = text[query_start:for_update_offset]
        if re.search(r"\bmeal_plans\b", query_window, re.IGNORECASE):
            sites.append(for_update_offset)
    return sites


def _line_of_offset(text: str, offset: int) -> int:
    """1-indexed line number del offset dado."""
    return text.count("\n", 0, offset) + 1


def _helper_called_before(text: str, for_update_offset: int, max_lookback_chars: int = 4000) -> bool:
    """¿Aparece la llamada al helper en los caracteres previos al
    `FOR UPDATE`? Lookback amplio (4000 chars ≈ 80 líneas) cubre
    bloques `with conn ... with cursor ... helper(); execute(...)`
    sin dar falsos positivos por contextos lejanos.
    """
    start = max(0, for_update_offset - max_lookback_chars)
    window = text[start:for_update_offset]
    return f"{_HELPER_NAME}(" in window


# ---------------------------------------------------------------------------
# 1. Helper en db_plans.py existe y lee los 2 knobs
# ---------------------------------------------------------------------------
class TestHelper:
    def test_helper_function_defined_in_db_plans(self):
        """El helper existe como función pública de `db_plans`."""
        from db_plans import set_meal_plan_for_update_timeouts
        assert callable(set_meal_plan_for_update_timeouts)

    def test_helper_emits_both_set_local_statements(self, monkeypatch):
        """El helper invoca `SET LOCAL lock_timeout` Y `SET LOCAL
        statement_timeout` (en ese orden o cualquiera, pero ambos)."""
        captured: list[str] = []

        class _StubCursor:
            def execute(self, sql, params=None):
                captured.append(sql)

        from db_plans import set_meal_plan_for_update_timeouts
        set_meal_plan_for_update_timeouts(_StubCursor())

        joined = "\n".join(captured).lower()
        assert "set local lock_timeout" in joined, (
            "Helper no emitió `SET LOCAL lock_timeout`. Sin esto, el caller "
            "espera indefinidamente por el row lock — bug original P1-LOCK-1."
        )
        assert "set local statement_timeout" in joined, (
            "Helper no emitió `SET LOCAL statement_timeout`. Sin esto, "
            "una tx que mantenga el lock con I/O lento dentro queda sin "
            "gate superior — gate redundante con el de la sesión pero "
            "explícito por defensa-en-profundidad."
        )

    def test_helper_reads_both_knobs(self, monkeypatch):
        """Al ejecutar el helper, ambos knobs se auto-registran en
        `_KNOBS_REGISTRY` con valores int parseados."""
        monkeypatch.setenv("MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS", "7777")
        monkeypatch.setenv("MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS", "44444")

        # Reset del registry para captura limpia.
        from knobs import _KNOBS_REGISTRY
        _KNOBS_REGISTRY.pop("MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS", None)
        _KNOBS_REGISTRY.pop("MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS", None)

        class _StubCursor:
            def execute(self, sql, params=None):
                pass

        from db_plans import set_meal_plan_for_update_timeouts
        set_meal_plan_for_update_timeouts(_StubCursor())

        assert "MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS" in _KNOBS_REGISTRY
        assert _KNOBS_REGISTRY["MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS"]["value"] == 7777
        assert "MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS" in _KNOBS_REGISTRY
        assert _KNOBS_REGISTRY["MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS"]["value"] == 44444

    def test_helper_uses_knob_values_in_sql(self, monkeypatch):
        """El SQL emitido debe contener los valores de los knobs (no
        valores hardcoded). Defensa contra refactor que olvide propagar."""
        monkeypatch.setenv("MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS", "1234")
        monkeypatch.setenv("MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS", "5678")

        captured: list[str] = []

        class _StubCursor:
            def execute(self, sql, params=None):
                captured.append(sql)

        from db_plans import set_meal_plan_for_update_timeouts
        set_meal_plan_for_update_timeouts(_StubCursor())

        joined = "\n".join(captured)
        assert "1234ms" in joined, (
            "Valor del knob lock_timeout no aparece en el SQL emitido. "
            "¿Helper hardcoded? Revisar db_plans.set_meal_plan_for_update_timeouts."
        )
        assert "5678ms" in joined, (
            "Valor del knob statement_timeout no aparece en el SQL emitido."
        )

    def test_helper_swallows_set_local_exception(self, monkeypatch):
        """Si `SET LOCAL` falla (Postgres viejo, permisos), el helper
        debe loggear debug pero NO propagar — el comportamiento previo
        (sin timeout) es estrictamente menos seguro pero la tx debe seguir.
        """
        class _FailingCursor:
            def execute(self, sql, params=None):
                raise RuntimeError("permission denied for SET LOCAL")

        from db_plans import set_meal_plan_for_update_timeouts
        # No debe lanzar.
        set_meal_plan_for_update_timeouts(_FailingCursor())


# ---------------------------------------------------------------------------
# 2. Cada `meal_plans … FOR UPDATE` en los archivos auditados tiene
#    el helper antes en el mismo bloque
# ---------------------------------------------------------------------------
class TestCallSitesUseHelper:
    @pytest.mark.parametrize(
        "file_path",
        _FILES_TO_AUDIT,
        ids=[str(p.relative_to(_BACKEND_ROOT)) for p in _FILES_TO_AUDIT],
    )
    def test_each_for_update_preceded_by_helper(self, file_path: Path):
        """Por cada `FROM meal_plans … FOR UPDATE` en el archivo, el helper
        `set_meal_plan_for_update_timeouts(cursor)` debe aparecer en los
        ~80 líneas previas. Sin esto, el bug P1-LOCK-1 puede regresar
        silenciosamente (un nuevo call site copy-paste sin el helper)."""
        text = file_path.read_text(encoding="utf-8")
        sites = _find_for_update_sites(text)

        assert sites, (
            f"No se encontró ningún `FROM meal_plans ... FOR UPDATE` en "
            f"{file_path}. Si el archivo cambió y ya no contiene esta "
            f"query, removerlo de `_FILES_TO_AUDIT` en este test."
        )

        unprotected = []
        for offset in sites:
            if not _helper_called_before(text, offset):
                line_no = _line_of_offset(text, offset)
                unprotected.append(f"{file_path.name}:{line_no}")

        assert not unprotected, (
            f"Sites con `FROM meal_plans ... FOR UPDATE` SIN llamada previa "
            f"a `{_HELPER_NAME}(cursor)`: {unprotected}\n"
            f"Esto reintroduce el bug P1-LOCK-1: el caller puede esperar "
            f"indefinidamente por el row lock. Añadir antes del execute:\n"
            f"    from db_plans import {_HELPER_NAME}\n"
            f"    {_HELPER_NAME}(cursor)\n"
        )

    def test_at_least_one_site_per_audited_file(self):
        """Sanity: ambos archivos auditados deben tener al menos un site
        (si quedaron sin sites tras un refactor, este test pide que el
        operador remueva el archivo de la lista o investigue dónde se
        movió la query)."""
        for file_path in _FILES_TO_AUDIT:
            text = file_path.read_text(encoding="utf-8")
            sites = _find_for_update_sites(text)
            assert sites, (
                f"`{file_path.relative_to(_BACKEND_ROOT)}` está en "
                f"`_FILES_TO_AUDIT` pero ya no contiene queries "
                f"`meal_plans ... FOR UPDATE`. Si el refactor las movió, "
                f"actualizar la lista; si las eliminó, también."
            )

    def test_total_audited_sites_count(self):
        """Defensa contra falsa cobertura: confirma que estamos
        auditando los 4 sites reales conocidos al cierre de P1-LOCK-1.
        Si añades un 5º site, este número sube y el operador lo nota."""
        total = 0
        for file_path in _FILES_TO_AUDIT:
            text = file_path.read_text(encoding="utf-8")
            total += len(_find_for_update_sites(text))
        # 1 en routers/plans.py (/shift-plan) + 3 en cron_tasks.py (T1, T2, bg-refill).
        assert total >= 4, (
            f"Esperaba >= 4 sites (1 routers/plans + 3 cron_tasks); "
            f"encontré {total}. ¿Se eliminó algún call site sin actualizar "
            f"este test?"
        )


# ---------------------------------------------------------------------------
# 3. db_plans.update_plan_data_atomic NO migra al nuevo helper
#    (intencional — tiene su propio knob CHUNK_LEARNING_LOCK_TIMEOUT_MS)
# ---------------------------------------------------------------------------
def test_update_plan_data_atomic_keeps_its_own_lock_timeout():
    """Documentación viva: `update_plan_data_atomic` ([db_plans.py])
    mantiene su propio `SET LOCAL lock_timeout` con knob distinto
    (`CHUNK_LEARNING_LOCK_TIMEOUT_MS`). NO debe migrar al nuevo helper.
    Si alguien lo migra por accidente y este test falla, revisar si la
    migración es intencional (entonces actualizar este test) o no
    (revertir).
    """
    db_plans_path = _BACKEND_ROOT / "db_plans.py"
    text = db_plans_path.read_text(encoding="utf-8")
    # En `update_plan_data_atomic` debe seguir el `SET LOCAL lock_timeout`
    # original con la cadena `CHUNK_LEARNING_LOCK_TIMEOUT_MS` cerca.
    func_match = re.search(
        r"def update_plan_data_atomic\([\s\S]+?(?=\ndef |\Z)",
        text,
    )
    assert func_match, "`update_plan_data_atomic` no encontrada en db_plans.py"
    body = func_match.group(0)
    assert "CHUNK_LEARNING_LOCK_TIMEOUT_MS" in body, (
        "`update_plan_data_atomic` perdió la referencia a su knob original "
        "`CHUNK_LEARNING_LOCK_TIMEOUT_MS`. Si migró al nuevo helper "
        "`set_meal_plan_for_update_timeouts`, actualizar este test."
    )
    assert "SET LOCAL lock_timeout" in body, (
        "`update_plan_data_atomic` perdió su `SET LOCAL lock_timeout`. "
        "Si migró al nuevo helper, actualizar este test."
    )
