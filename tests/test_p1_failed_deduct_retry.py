"""[P1-FAILED-DEDUCT-RETRY · 2026-05-22] Tests del cron processor que
reintenta filas pendientes en `failed_inventory_deductions`.

Cierre del gap "tabla write-only" complementario al alert P0-5
(`_alert_failed_inventory_deductions_backlog`) que solo emitía warning
sin procesar la cola — `attempts` se mantenía en 0 forever en producción.

Verificado en BD prod 2026-05-22:
  - 3 rows acumuladas con `attempts=0` desde 2026-05-20.
  - Causa raíz: `parse_failed_or_invalid_qty` que P1-PANTRY-INFER cierra.
  - Sin processor, el alert seguiría disparando sin reducir backlog.

Convención cross-link (P2-HIST-AUDIT-14): slug `p1_failed_deduct_retry`
matchea este archivo.

Tooltip-anchor: P1-FAILED-DEDUCT-RETRY-PROCESSOR.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS_PY = _BACKEND_ROOT / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_TASKS_PY.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — presencia del processor + anchor
# ===========================================================================

def test_tooltip_anchor_present(cron_src: str):
    """Anchor `P1-FAILED-DEDUCT-RETRY-PROCESSOR` debe vivir en el código."""
    assert "P1-FAILED-DEDUCT-RETRY-PROCESSOR" in cron_src, (
        "P1-FAILED-DEDUCT-RETRY regresión: tooltip-anchor removido. "
        "Si renombraste la función, actualizar el anchor + este test."
    )


def test_processor_function_defined(cron_src: str):
    """`_process_failed_inventory_deductions_queue` debe estar definida."""
    assert re.search(
        r"^def\s+_process_failed_inventory_deductions_queue\s*\(",
        cron_src,
        re.MULTILINE,
    ), (
        "P1-FAILED-DEDUCT-RETRY regresión: función "
        "`_process_failed_inventory_deductions_queue` removida o renombrada."
    )


def test_processor_reads_table(cron_src: str):
    """El processor debe SELECT de `failed_inventory_deductions` filtrando
    por `attempts <` para no procesar rows ya dead-lettered."""
    fn_re = re.compile(
        r"def\s+_process_failed_inventory_deductions_queue\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(cron_src)
    assert m is not None
    body = m.group(0)
    assert "FROM failed_inventory_deductions" in body, (
        "Processor no consulta la tabla `failed_inventory_deductions`."
    )
    assert re.search(r"WHERE\s+attempts\s*<", body, re.IGNORECASE), (
        "Processor no filtra `WHERE attempts <` — sin esto procesaría rows "
        "dead-lettered en cada tick (loop infinito)."
    )


def test_processor_invokes_infer_typical_portion(cron_src: str):
    """El retry debe usar `_infer_typical_portion` para asignar porción
    típica a items con qty=0 (sin esto, los 3 incidentes del 2026-05-20/22
    seguirían atascados)."""
    fn_re = re.compile(
        r"def\s+_process_failed_inventory_deductions_queue\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(cron_src)
    assert m is not None
    body = m.group(0)
    assert "_infer_typical_portion" in body, (
        "P1-FAILED-DEDUCT-RETRY regresión: processor no invoca "
        "`_infer_typical_portion`. Sin la inferencia, los items legacy con "
        "qty=0 seguirán atascados en la cola."
    )


def test_processor_does_not_call_deduct_recursively(cron_src: str):
    """[P1-FAILED-DEDUCT-RETRY anti-loop] El processor NO debe llamar
    `deduct_consumed_meal_from_inventory` porque esa función vuelve a
    insertar en `failed_inventory_deductions` si falla → loop.

    Debe llamar directamente `add_or_update_inventory_item` con la qty
    inferida."""
    fn_re = re.compile(
        r"def\s+_process_failed_inventory_deductions_queue\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(cron_src)
    assert m is not None
    body = m.group(0)
    assert "deduct_consumed_meal_from_inventory(" not in body, (
        "P1-FAILED-DEDUCT-RETRY anti-loop regresión: processor invoca "
        "`deduct_consumed_meal_from_inventory()` recursivamente. Esa "
        "función inserta en `failed_inventory_deductions` cuando falla; "
        "llamarla desde el processor genera nuevas rows mientras procesa "
        "viejas — backlog inflado en loop. Usar "
        "`add_or_update_inventory_item` directamente."
    )
    assert "add_or_update_inventory_item(" in body, (
        "Processor no llama `add_or_update_inventory_item` — sin esto la "
        "deducción no llega a `user_inventory`."
    )


def test_processor_uses_distinct_mutation_marker(cron_src: str):
    """Las deducciones del retry deben marcarse `mutation_type=
    'consumption_replay'` (no `consumption`) para distinguirlas en audits
    del path en vivo."""
    fn_re = re.compile(
        r"def\s+_process_failed_inventory_deductions_queue\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(cron_src)
    assert m is not None
    body = m.group(0)
    assert '"consumption_replay"' in body, (
        "P1-FAILED-DEDUCT-RETRY regresión: el processor debe usar "
        "`mutation_type='consumption_replay'` para que las filas resultantes "
        "sean distinguibles en audits del path en vivo (que usa "
        "`mutation_type='consumption'`)."
    )


def test_processor_increments_attempts_on_partial_fail(cron_src: str):
    """Si el row no se pudo recuperar completo → `UPDATE attempts + 1`."""
    fn_re = re.compile(
        r"def\s+_process_failed_inventory_deductions_queue\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(cron_src)
    assert m is not None
    body = m.group(0)
    # SQL puede estar split entre dos string-literals adyacentes (Python concat
    # implícito). Permitir cualquier whitespace + quote chars entre tokens.
    assert re.search(
        r'UPDATE\s+failed_inventory_deductions[\s"\']+SET\s+attempts',
        body,
        re.IGNORECASE,
    ), (
        "P1-FAILED-DEDUCT-RETRY regresión: processor no incrementa "
        "`attempts` cuando un row no se procesa completamente. Sin esto, "
        "los rows irrecuperables nunca pasan al dead-letter path."
    )


def test_processor_dead_letters_after_max_attempts(cron_src: str):
    """Tras `MEALFIT_FAILED_DEDUCTIONS_MAX_ATTEMPTS` → DELETE forzado +
    log WARN `DEAD-LETTER`. Sin esto el cron procesaría los mismos rows
    irrecuperables forever."""
    fn_re = re.compile(
        r"def\s+_process_failed_inventory_deductions_queue\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(cron_src)
    assert m is not None
    body = m.group(0)
    assert "DEAD-LETTER" in body, (
        "P1-FAILED-DEDUCT-RETRY regresión: dead-letter path removido. "
        "Sin esto los rows irrecuperables nunca se eliminan y el cron los "
        "procesa cada tick forever."
    )
    assert re.search(r"DELETE\s+FROM\s+failed_inventory_deductions", body, re.IGNORECASE), (
        "Processor no contiene `DELETE FROM failed_inventory_deductions` — "
        "los rows nunca se limpian (ni los recuperados ni los dead-lettered)."
    )


# ===========================================================================
# Sección 2 — knobs operacionales
# ===========================================================================

def test_kill_switch_knob_referenced(cron_src: str):
    """`MEALFIT_FAILED_DEDUCTIONS_RETRY_ENABLED` — kill switch obligatorio."""
    assert "MEALFIT_FAILED_DEDUCTIONS_RETRY_ENABLED" in cron_src, (
        "P1-FAILED-DEDUCT-RETRY regresión: kill switch "
        "`MEALFIT_FAILED_DEDUCTIONS_RETRY_ENABLED` removido. Es el "
        "mecanismo de rollback operacional sin redeploy si el processor "
        "causa contención."
    )


def test_interval_knob_referenced(cron_src: str):
    """`MEALFIT_FAILED_DEDUCTIONS_RETRY_INTERVAL_MIN` ajusta frecuencia."""
    assert "MEALFIT_FAILED_DEDUCTIONS_RETRY_INTERVAL_MIN" in cron_src, (
        "Knob de interval del retry processor removido."
    )


def test_batch_knob_referenced(cron_src: str):
    """`MEALFIT_FAILED_DEDUCTIONS_RETRY_BATCH` limita rows por tick."""
    assert "MEALFIT_FAILED_DEDUCTIONS_RETRY_BATCH" in cron_src, (
        "Knob de batch size removido."
    )


def test_max_attempts_knob_referenced(cron_src: str):
    """`MEALFIT_FAILED_DEDUCTIONS_MAX_ATTEMPTS` controla el dead-letter."""
    assert "MEALFIT_FAILED_DEDUCTIONS_MAX_ATTEMPTS" in cron_src, (
        "Knob de max_attempts removido — sin esto el dead-letter no tiene "
        "umbral configurable."
    )


# ===========================================================================
# Sección 3 — registro en el scheduler
# ===========================================================================

def test_job_registered_in_scheduler(cron_src: str):
    """`process_failed_inventory_deductions_queue` debe estar registrado
    en `register_plan_chunk_scheduler` para que corra automáticamente."""
    # La sección de registro debe contener el id del job.
    assert 'id="process_failed_inventory_deductions_queue"' in cron_src, (
        "P1-FAILED-DEDUCT-RETRY regresión: job "
        "`process_failed_inventory_deductions_queue` NO registrado en el "
        "scheduler. Sin esto el processor nunca corre — cron silenciosamente "
        "muerto. Verificar `register_plan_chunk_scheduler` en cron_tasks.py."
    )


def test_scheduler_registration_uses_interval_trigger(cron_src: str):
    """El registro debe usar `interval` trigger con `minutes=` (no cron
    fijo) — frecuencia ajustable vía knob."""
    # Buscar el bloque del job
    sched_block_re = re.compile(
        r'_process_failed_inventory_deductions_queue\s*,\s*"interval"\s*,\s*'
        r'minutes\s*=',
        re.DOTALL,
    )
    assert sched_block_re.search(cron_src), (
        "P1-FAILED-DEDUCT-RETRY regresión: el registro del job no usa "
        "`interval` trigger con `minutes=` — frecuencia debe ser ajustable "
        "vía `MEALFIT_FAILED_DEDUCTIONS_RETRY_INTERVAL_MIN` en runtime."
    )
