"""[P1-SWAP-PERSIST-ATOMIC · 2026-05-22] Test del refactor del handler
`/swap-meal/persist` de UPDATE plano con `jsonb_set` chained a
`update_plan_data_atomic` (FOR UPDATE row lock + mutator callback).

Pre-fix (GAP-2 audit production-readiness 2026-05-22):
    El handler `api_swap_meal_persist` usaba `execute_sql_write` con
    `jsonb_set` chained, sin row lock. La protección contra lost-update
    se sostenía en dos hechos parciales:

      1. `jsonb_set` server-side serializa per-row a nivel Postgres
         (los dos UPDATE se ejecutan secuencialmente sobre la misma
         tupla — el segundo opera sobre el resultado del primero).
      2. El worker típicamente persiste `days[7+]` mientras el swap
         opera sobre `days[0-5]` → no se solapan en path.

    Pero la garantía NO era estructural:
      - Si un futuro refactor del mutator cambia a operaciones overlap
        (e.g., también muta `_plan_modified_at` que el worker bumpea),
        un read-modify-write podría perderse.
      - Otros endpoints (`/recalculate-shopping-list`, `/recipe/expand`)
        YA usan `update_plan_data_atomic` con FOR UPDATE (P0-2 +
        P1-AUDIT-1). El swap-meal/persist era el outlier que violaba
        I7 en espíritu.

    Audit production-readiness 2026-05-22 marcó esto como GAP-2 (🟡
    "race condition en /swap-meal/persist sin row lock").

Cierre P1-SWAP-PERSIST-ATOMIC:
    Handler migrado a `update_plan_data_atomic(plan_id, _swap_mutator,
    user_id=verified_user_id)`. El mutator:
      1. Resuelve `days[day_index].meals[meal_index]` y muta el slot.
      2. Pop de las 4 keys `aggregated_shopping_list*` (recalc downstream).
      3. Sello `_plan_modified_at` con timestamp ISO UTC.
      4. (opcional) Flag `is_restocked = False` si el caller lo señala.

    El SELECT inicial de ownership (`AND user_id = %s`) se preserva
    como defensa-en-depth + para 404-on-not-found antes del network
    round-trip al helper atómico.

Tooltip-anchor: P1-SWAP-PERSIST-ATOMIC | audit 2026-05-22
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    assert m, f"`def {fn_name}(` no encontrado en plans.py"
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def swap_persist_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_swap_meal_persist")


# ---------------------------------------------------------------------------
# Section A — Migración a update_plan_data_atomic
# ---------------------------------------------------------------------------
class TestAtomicMigration:

    def test_handler_uses_update_plan_data_atomic(
        self, swap_persist_body: str
    ):
        """El handler DEBE invocar `update_plan_data_atomic` con `plan_id`
        y un mutator callback. Sin esto, regresó al UPDATE plano y se
        reabre el riesgo lost-update con _chunk_worker."""
        assert "update_plan_data_atomic" in swap_persist_body, (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler ya no invoca "
            "`update_plan_data_atomic`. Si volvió a `execute_sql_write` "
            "directo, perdimos el FOR UPDATE row lock que serializa "
            "contra `_chunk_worker` concurrente."
        )

    def test_atomic_helper_receives_user_id_kwarg(
        self, swap_persist_body: str
    ):
        """`update_plan_data_atomic` DEBE recibir `user_id=verified_user_id`
        kwarg explícito. Sin esto, el helper loguea `[I2-MISS]` warning
        y el UPDATE no filtra `AND user_id = %s` → degrada la defensa-en-depth."""
        pattern = re.compile(
            r"update_plan_data_atomic[^)]*user_id\s*=\s*verified_user_id",
            re.DOTALL,
        )
        assert pattern.search(swap_persist_body), (
            "P1-SWAP-PERSIST-ATOMIC regresión: el caller a "
            "`update_plan_data_atomic` no pasa `user_id=verified_user_id`. "
            "Esto produce `[I2-MISS]` en logs y degrada defensa-en-depth "
            "I2 (UPDATE sin `AND user_id = %s`)."
        )

    def test_mutator_pops_aggregated_shopping_list_keys(
        self, swap_persist_body: str
    ):
        """El mutator DEBE eliminar las 4 keys `aggregated_shopping_list*`
        para forzar recalc downstream — mismo contrato que el SQL legacy
        con `#- '{aggregated_shopping_list*}'`."""
        # Cualquier forma de pop/del de las 4 keys es aceptable.
        # Verificamos al menos `aggregated_shopping_list` literal.
        assert "aggregated_shopping_list_weekly" in swap_persist_body, (
            "P1-SWAP-PERSIST-ATOMIC regresión: el mutator no maneja "
            "`aggregated_shopping_list_weekly`. Sin strip de las 4 listas, "
            "el frontend vería una lista de compras stale tras swap-ear "
            "un meal con ingredientes nuevos."
        )
        assert "aggregated_shopping_list_biweekly" in swap_persist_body
        assert "aggregated_shopping_list_monthly" in swap_persist_body

    def test_mutator_writes_plan_modified_at(self, swap_persist_body: str):
        """El sello CAS `_plan_modified_at` DEBE bumpearse. El sort del
        Historial usa esta key (P0-3) para mostrar el plan recientemente
        modificado primero. Sin el bump, swap-ear un meal antiguo no lo
        promueve en el sort."""
        assert "_plan_modified_at" in swap_persist_body, (
            "P1-SWAP-PERSIST-ATOMIC regresión: el mutator no bumpea "
            "`_plan_modified_at`. El Historial perdería el sort por "
            "recencia tras un swap."
        )

    def test_mutator_handles_clear_is_restocked(
        self, swap_persist_body: str
    ):
        """El flag opcional `clear_is_restocked` DEBE seguir funcionando —
        el frontend lo envía cuando detecta uncovered ingredients post-swap
        (lógica P0-1 conservada client-side)."""
        assert "clear_is_restocked" in swap_persist_body, (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler ya no maneja "
            "`clear_is_restocked`. La lógica de uncovered ingredients "
            "post-swap (P0-1) quedaría sin defensa."
        )
        assert "is_restocked" in swap_persist_body, (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler ya no escribe "
            "`is_restocked` en plan_data."
        )


# ---------------------------------------------------------------------------
# Section B — 404 y race contra DELETE
# ---------------------------------------------------------------------------
class TestRaceConditions:

    def test_handles_empty_result_from_atomic_as_404(
        self, swap_persist_body: str
    ):
        """`update_plan_data_atomic` retorna `{}` si el row no existe
        (race contra `DELETE /{plan_id}` entre el SELECT inicial y el
        FOR UPDATE). El handler DEBE mapear este caso a 404 explícito,
        NO a 500 ni a 200 silent."""
        # Buscamos un patrón `if not result:` (o `if result == {}:` etc.)
        # seguido de `raise HTTPException(status_code=404`.
        # Usamos `.` permisivo (DOTALL match newlines) para no acoplar al
        # nombre exacto de la variable (`result`/`updated`/etc.) ni a
        # caracteres `{}` que aparecen en comentarios (e.g., `{plan_id}`).
        pattern = re.compile(
            r"if\s+not\s+\w+:.{0,600}?HTTPException\s*\(\s*status_code\s*=\s*404",
            re.DOTALL,
        )
        assert pattern.search(swap_persist_body), (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler no maneja el "
            "caso `update_plan_data_atomic` retorna `{}` (row desapareció "
            "en el ínterin). Sin esto, un swap durante DELETE concurrente "
            "retornaría 200 silent en lugar de 404 honesto."
        )


# ---------------------------------------------------------------------------
# Section C — Error mapping del mutator
# ---------------------------------------------------------------------------
class TestErrorMapping:

    def test_index_error_mapped_to_400(self, swap_persist_body: str):
        """Si `day_index` o `meal_index` están fuera de rango del plan
        REAL (no del bound check inicial — eso son 400 más arriba), el
        mutator levanta IndexError. El handler DEBE mapear a 400, NO 500."""
        # Buscamos un except que capture IndexError (y/o ValueError) y
        # devuelva HTTPException(400).
        pattern = re.compile(
            r"except\s*\(?\s*(?:IndexError|ValueError)[^)]*\)?[^}]{0,600}HTTPException\s*\(\s*status_code\s*=\s*400",
            re.DOTALL,
        )
        assert pattern.search(swap_persist_body), (
            "P1-SWAP-PERSIST-ATOMIC regresión: el handler no mapea "
            "IndexError/ValueError del mutator a HTTPException(400). "
            "Sin esto, un day_index fuera de rango devolvería 500 "
            "(operacional alarm) en vez de 400 (client error legítimo)."
        )


# ---------------------------------------------------------------------------
# Section D — SELECT inicial de ownership preservado
# ---------------------------------------------------------------------------
class TestOwnershipSelect:

    def test_initial_select_filters_user_id(self, swap_persist_body: str):
        """Aunque el helper atómico ya filtra `user_id` internamente, el
        SELECT inicial DEBE preservarse para devolver 404 honesto antes
        del network round-trip al helper. Defensa-en-depth."""
        where_pattern = re.compile(
            r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
            re.IGNORECASE | re.DOTALL,
        )
        assert where_pattern.search(swap_persist_body), (
            "P1-SWAP-PERSIST-ATOMIC regresión: el SELECT inicial de "
            "ownership ya no filtra `AND user_id = %s`. Eso elimina la "
            "primera capa de defensa contra IDOR y degrada la observabilidad "
            "(404 honesto pre-atomic vs `{}` post-atomic)."
        )


# ---------------------------------------------------------------------------
# Section E — Marker
# ---------------------------------------------------------------------------
def test_marker_anchor_filename():
    expected_slug = "p1_swap_persist_atomic"
    assert expected_slug in __file__.replace("\\", "/").lower()


def test_marker_in_source(swap_persist_body: str):
    """Tooltip-anchor: el marker `P1-SWAP-PERSIST-ATOMIC` DEBE aparecer
    en el cuerpo del handler para que un futuro grep cross-reference el
    test al fix."""
    assert "P1-SWAP-PERSIST-ATOMIC" in swap_persist_body, (
        "P1-SWAP-PERSIST-ATOMIC regresión: el marker tooltip-anchor "
        "fue removido del comment block del handler. Sin el marker, "
        "un grep cross-reference futuro pierde el cross-link al test."
    )
