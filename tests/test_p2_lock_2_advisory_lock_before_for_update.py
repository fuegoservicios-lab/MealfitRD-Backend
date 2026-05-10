"""[P2-LOCK-2 · 2026-05-10] Regression guard: cada `SELECT … FOR UPDATE`
sobre `meal_plans` DEBE estar precedido por `acquire_meal_plan_advisory_lock`
en el mismo bloque transaccional. Cierra la asimetría de orden detectada
durante implementación de P1-LOCK-1.

Bug original (deadlock potencial bounded por P1-LOCK-1 timeout):
    Pre-P2-LOCK-2, dos sites tomaban locks en orden invertido vs los workers:
        - `/shift-plan` y `_background_rolling_refill`:
              FOR UPDATE → advisory lock (orden A)
        - T1 worker, T2 worker:
              advisory lock → FOR UPDATE (orden B)

    Si /shift-plan y worker corrían simultáneos sobre el mismo plan,
    cada uno sostenía un recurso esperando el otro:
        Worker:    advisory(plan_X) ✓ → FOR UPDATE(plan_X) BLOCKED
        /shift:    FOR UPDATE(plan_X) ✓ → advisory(plan_X)  BLOCKED
    Postgres detecta dentro de `deadlock_timeout` (1s default) y aborta
    una de las txs con `deadlock_detected`. P1-LOCK-1 bound el wait con
    statement_timeout (otra red de seguridad), pero la causa raíz era
    order-asymmetry.

Fix:
    /shift-plan y bg-refill ahora siguen el orden de los workers:
        1. set_meal_plan_for_update_timeouts(cursor)  ← bound waits
        2. SELECT id FROM meal_plans … LIMIT 1        ← resolve sin lock
        3. acquire_meal_plan_advisory_lock(plan_id)   ← advisory primero
        4. SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE
                                                       ← row lock segundo

Cobertura:
    Para cada `meal_plans … FOR UPDATE` en `routers/plans.py` y
    `cron_tasks.py` (excluyendo comentarios/docstrings y excluyendo
    `db_plans.update_plan_data_atomic` que es helper de bajo nivel),
    verificar que `acquire_meal_plan_advisory_lock` aparece en los
    ~80 lines previos en el mismo bloque.

    Si alguien añade un nuevo call site sin el advisory lock antes
    (o lo invierte de vuelta a "FOR UPDATE primero"), este test falla
    loud con la línea ofensora.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
# Los call sites usan aliases por ergonomía (`_p04_acquire_lock`,
# `_p02_acquire_lock`, etc.) para mantener tags P-fix legibles inline.
# Este regex cubre tanto las invocaciones aliased como el nombre raw.
# Si alguien introduce una nueva alias siguiendo el patrón `_pXX_acquire_lock`
# o `_pXXX_acquire_lock`, el regex la captura. Una alias totalmente nueva
# (e.g. `_meal_lock`) NO matchea y el test falla — forzando que el operador
# actualice este regex y conscientemente añada el sitio.
_ADVISORY_CALL_RE = re.compile(
    r"\b(?:_p\d+_acquire_lock|acquire_meal_plan_advisory_lock)\s*\(",
)

# Mismos archivos que P1-LOCK-1 (sites de meal_plans FOR UPDATE).
_FILES_TO_AUDIT = [
    _BACKEND_ROOT / "routers" / "plans.py",
    _BACKEND_ROOT / "cron_tasks.py",
]


# ---------------------------------------------------------------------------
# Detector reutilizado de test_p1_lock_1_*.py — copiado en lugar de
# importado para que cada test sea self-contained (importar entre tests
# añade fragilidad a discovery).
# ---------------------------------------------------------------------------
_FOR_UPDATE_RE = re.compile(r"\bFOR\s+UPDATE\b", re.IGNORECASE)
_QUERY_START_RE = re.compile(
    r"(?:cursor\.execute\s*\(|execute_sql_query\s*\(|execute_sql_write\s*\(|query\s*=)",
    re.IGNORECASE,
)


def _is_in_python_comment(text: str, offset: int) -> bool:
    line_start = text.rfind("\n", 0, offset) + 1
    line_prefix = text[line_start:offset]
    return "#" in line_prefix


def _is_in_python_docstring(text: str, offset: int) -> bool:
    triple_double = text.count('"""', 0, offset)
    triple_single = text.count("'''", 0, offset)
    return (triple_double % 2 == 1) or (triple_single % 2 == 1)


def _find_for_update_sites(text: str) -> list[int]:
    sites: list[int] = []
    for m in _FOR_UPDATE_RE.finditer(text):
        offset = m.start()
        if _is_in_python_comment(text, offset):
            continue
        if _is_in_python_docstring(text, offset):
            continue
        query_starts = list(_QUERY_START_RE.finditer(text, 0, offset))
        if not query_starts:
            continue
        query_start = query_starts[-1].start()
        query_window = text[query_start:offset]
        if re.search(r"\bmeal_plans\b", query_window, re.IGNORECASE):
            sites.append(offset)
    return sites


def _line_of_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _advisory_called_before(
    text: str, for_update_offset: int, max_lookback_chars: int = 4000
) -> bool:
    """¿Aparece una llamada al advisory lock en los caracteres previos
    al `FOR UPDATE`? Cubre alias `_pXX_acquire_lock(` y nombre raw
    `acquire_meal_plan_advisory_lock(`. Lookback ≈ 80 líneas."""
    start = max(0, for_update_offset - max_lookback_chars)
    window = text[start:for_update_offset]
    return bool(_ADVISORY_CALL_RE.search(window))


# ---------------------------------------------------------------------------
# 1. Cada FOR UPDATE precedido por advisory lock
# ---------------------------------------------------------------------------
class TestAdvisoryBeforeForUpdate:
    @pytest.mark.parametrize(
        "file_path",
        _FILES_TO_AUDIT,
        ids=[str(p.relative_to(_BACKEND_ROOT)) for p in _FILES_TO_AUDIT],
    )
    def test_each_for_update_preceded_by_advisory(self, file_path: Path):
        """Por cada `FROM meal_plans ... FOR UPDATE` en el archivo, el
        helper `acquire_meal_plan_advisory_lock(cursor, ...)` debe
        aparecer en los ~80 líneas previas. Sin esto, el bug P2-LOCK-2
        regresa: orden de adquisición (FOR UPDATE primero, advisory
        después) introduce deadlock potencial vs workers.
        """
        text = file_path.read_text(encoding="utf-8")
        sites = _find_for_update_sites(text)

        assert sites, (
            f"No se encontró ningún `FROM meal_plans ... FOR UPDATE` en "
            f"{file_path}. Si el archivo cambió, removerlo de "
            f"`_FILES_TO_AUDIT`."
        )

        out_of_order = []
        for offset in sites:
            if not _advisory_called_before(text, offset):
                line_no = _line_of_offset(text, offset)
                out_of_order.append(f"{file_path.name}:{line_no}")

        assert not out_of_order, (
            f"Sites con `FROM meal_plans ... FOR UPDATE` SIN llamada previa "
            f"al advisory lock (alias `_pXX_acquire_lock` o raw "
            f"`acquire_meal_plan_advisory_lock`): {out_of_order}\n"
            f"Esto reintroduce el bug P2-LOCK-2: orden de adquisición "
            f"asimétrico vs workers (advisory → FOR UPDATE) puede producir "
            f"deadlock. Refactor sugerido:\n"
            f"    1. SELECT id FROM meal_plans WHERE ... LIMIT 1  (sin FOR UPDATE)\n"
            f"    2. acquire_meal_plan_advisory_lock(cursor, plan_id, purpose='general')\n"
            f"    3. SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE\n"
        )


# ---------------------------------------------------------------------------
# 2. Sanity: confirmar que todos los sites tienen advisory antes
# ---------------------------------------------------------------------------
def test_total_sites_all_advisory_protected():
    """Conteo agregado: TODOS los sites de `meal_plans FOR UPDATE` en
    los archivos auditados deben tener advisory lock antes. Defensa
    contra falsa cobertura (e.g., remover sites del array sin notar
    que un nuevo site se añadió sin protección)."""
    total = 0
    unprotected = []
    for file_path in _FILES_TO_AUDIT:
        text = file_path.read_text(encoding="utf-8")
        for offset in _find_for_update_sites(text):
            total += 1
            if not _advisory_called_before(text, offset):
                line_no = _line_of_offset(text, offset)
                unprotected.append(f"{file_path.name}:{line_no}")
    assert total >= 4, (
        f"Esperaba ≥4 sites de `meal_plans FOR UPDATE` (1 /shift-plan + "
        f"3 en cron_tasks: T1, T2, bg-refill); encontré {total}. "
        f"¿Refactor eliminó algún site? Actualizar este test."
    )
    assert not unprotected, (
        f"Sites sin advisory lock antes: {unprotected}"
    )


# ---------------------------------------------------------------------------
# 3. /shift-plan refactor preservó comportamiento downstream
# ---------------------------------------------------------------------------
def test_shift_plan_resolves_plan_id_before_advisory():
    """[P2-LOCK-2] El refactor de `/shift-plan` introduce un SELECT
    de `id` SIN lock antes del advisory. Verificar que ambos statements
    están presentes y en el orden correcto. Defensa contra rollback
    accidental al patrón pre-P2-LOCK-2."""
    text = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
    # Buscar el bloque del /shift-plan endpoint.
    shift_block_match = re.search(
        r"def api_shift_plan\([\s\S]+?(?=\ndef |\Z)", text
    )
    assert shift_block_match, "`api_shift_plan` no encontrado en routers/plans.py"
    block = shift_block_match.group(0)

    # Debe haber un `SELECT id FROM meal_plans` (sin FOR UPDATE) ANTES
    # del advisory lock.
    select_id_match = re.search(
        r"SELECT\s+id\s+FROM\s+meal_plans\b", block, re.IGNORECASE
    )
    assert select_id_match, (
        "`/shift-plan` no contiene un `SELECT id FROM meal_plans` sin lock. "
        "El refactor P2-LOCK-2 lo añadió como step 1 — su ausencia sugiere "
        "rollback. Restaurar el split: SELECT id → advisory → FOR UPDATE."
    )

    advisory_match = _ADVISORY_CALL_RE.search(block)
    assert advisory_match, (
        "`/shift-plan` no llama advisory lock (alias `_pXX_acquire_lock` "
        "o raw `acquire_meal_plan_advisory_lock`)"
    )

    # SELECT id debe venir ANTES del advisory.
    assert select_id_match.start() < advisory_match.start(), (
        "`/shift-plan` invoca el advisory lock ANTES del SELECT id — "
        "orden incorrecto. P2-LOCK-2 requiere: SELECT id → advisory → FOR UPDATE."
    )


def test_bg_refill_resolves_plan_id_before_advisory():
    """[P2-LOCK-2] Mismo invariante para `_background_shift_plan_for_user`
    (la función que el cron P0-2 invoca para usuarios inactivos; replica
    la lógica de `api_shift_plan` sin autenticación HTTP). El nombre
    histórico "bg-refill" en comentarios se refiere a esta función.

    El SELECT id (sin lock) debe estar dentro de la función Y antes del
    advisory lock.
    """
    text = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
    bg_block_match = re.search(
        r"def _background_shift_plan_for_user\([\s\S]+?(?=\ndef |\Z)", text
    )
    assert bg_block_match, (
        "`_background_shift_plan_for_user` no encontrado en cron_tasks.py. "
        "¿Renombrada? Actualizar este test."
    )
    block = bg_block_match.group(0)

    # Buscar el primer SELECT id real, después del `with conn.transaction()`
    # (saltea menciones en docstring del módulo si las hubiera).
    tx_start = block.find("with conn.transaction()")
    code_after_tx = block[tx_start:] if tx_start >= 0 else block
    select_id_match = re.search(
        r"SELECT\s+id\s+FROM\s+meal_plans\b", code_after_tx, re.IGNORECASE
    )
    assert select_id_match, (
        "`_background_shift_plan_for_user` no contiene `SELECT id FROM "
        "meal_plans` (sin lock) tras el `with conn.transaction()`. Refactor "
        "P2-LOCK-2 ausente — restaurar split SELECT id → advisory → FOR UPDATE."
    )

    advisory_match = _ADVISORY_CALL_RE.search(code_after_tx)
    assert advisory_match, (
        "`_background_shift_plan_for_user` no llama advisory lock"
    )

    assert select_id_match.start() < advisory_match.start(), (
        "`_background_shift_plan_for_user` invoca advisory antes del "
        "SELECT id — orden incorrecto."
    )
