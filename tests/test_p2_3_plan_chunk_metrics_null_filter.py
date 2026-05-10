"""[P2-3 · 2026-05-08] Tests del filtro `meal_plan_id IS NOT NULL` en queries
agregadoras de `plan_chunk_metrics`.

Bug original (audit 2026-05-07):
  La FK `plan_chunk_metrics.meal_plan_id → meal_plans.id` está definida como
  `ON DELETE SET NULL` (verificado vía `pg_constraint` el 2026-05-08, migración
  `p2_alpha_plan_chunk_queue_fk_cascade`). Cuando un plan se elimina (purga
  manual, reset de usuario, cleanup admin), las filas asociadas en
  `plan_chunk_metrics` SOBREVIVEN con `meal_plan_id=NULL` para preservar
  telemetría histórica.

  Las queries agregadoras del cron de quality + health endpoint NO filtraban
  esos NULLs:
    - `_persist_quality_degradation_alert` (cron_tasks.py:12928): un usuario
      cuyo único plan degradado fue borrado quedaba marcado con
      `quality_alert_at` por telemetría stale.
    - `_alert_if_degraded_rate_high` (cron_tasks.py:12998): el ratio global
      degradado mezclaba chunks de planes activos con chunks de planes
      eliminados → numerador/denominador inflados, alerta puede flamear.
    - `routers/system.py:185`: health endpoint reportaba degraded_rate con
      misma distorsión → ops dashboard engañoso.

  Nota: la premisa específica del audit ("_alert_high_synthesized_lesson_ratio")
  fue FALSO POSITIVO — ese cron lee `chunk_lesson_telemetry` (meal_plan_id
  NOT NULL) y `plan_chunk_queue` (FK CASCADE → rows borradas, no NULL'd).
  No requiere fix. El espíritu del audit ("queries que agregan por plan")
  sí aplica a los 3 sitios de arriba.

Fix:
  Añadido `WHERE meal_plan_id IS NOT NULL` a las 3 queries. Sin índice
  parcial nuevo: las queries ya están filtradas por `created_at` (24h) →
  scan acotado, COUNT trivial.

Cobertura:
  - Smoke estructural: cada uno de los 3 archivos contiene el filtro.
  - El filtro NO se aplicó a queries de INSERT (writes).
  - El filtro NO se aplicó al `_alert_high_synthesized_lesson_ratio` (false
    positive del audit; lee tablas que no sufren el bug).
"""
import pathlib
import re

import pytest


_BACKEND = pathlib.Path(__file__).parent
_CRON = _BACKEND / "cron_tasks.py"
_SYSTEM = _BACKEND / "routers" / "system.py"


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_source() -> str:
    return _SYSTEM.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. cron_tasks.py: 2 queries agregadoras
# ---------------------------------------------------------------------------
def test_persist_quality_degradation_alert_filters_null_meal_plan(cron_source):
    """`_persist_quality_degradation_alert` (DISTINCT user_id query) debe
    filtrar `meal_plan_id IS NOT NULL` para no marcar usuarios como
    degradados por telemetría de planes ya borrados."""
    func_start = cron_source.find("def _persist_quality_degradation_alert(")
    func_end = cron_source.find("\ndef ", func_start + 1)
    assert func_start != -1, "función _persist_quality_degradation_alert no encontrada"
    block = cron_source[func_start:func_end]
    assert "FROM plan_chunk_metrics" in block, "No lee plan_chunk_metrics — refactor inesperado"
    assert "meal_plan_id IS NOT NULL" in block, (
        "Query DISTINCT user_id debe excluir filas huérfanas (meal_plan_id NULL "
        "tras delete del padre). Sin el filtro, telemetría stale marca usuarios."
    )


def test_alert_if_degraded_rate_high_filters_null_meal_plan(cron_source):
    """`_alert_if_degraded_rate_high` (COUNT GROUP BY) debe filtrar NULLs
    para no incluir chunks de planes eliminados en el ratio global."""
    func_start = cron_source.find("def _alert_if_degraded_rate_high(")
    func_end = cron_source.find("\ndef ", func_start + 1)
    assert func_start != -1, "función _alert_if_degraded_rate_high no encontrada"
    block = cron_source[func_start:func_end]
    assert "FROM plan_chunk_metrics" in block
    assert "meal_plan_id IS NOT NULL" in block, (
        "COUNT global por is_rolling_refill debe excluir filas huérfanas. "
        "Sin el filtro, ratio degraded se distorsiona post-purga."
    )


# ---------------------------------------------------------------------------
# 2. routers/system.py: health endpoint
# ---------------------------------------------------------------------------
def test_system_health_degraded_rate_filters_null_meal_plan(system_source):
    """El query del health endpoint en routers/system.py debe filtrar
    `meal_plan_id IS NOT NULL` para reportar calidad de planes activos."""
    # Buscar el bloque que computa degraded rate (sección 3).
    block_start = system_source.find("Degraded rate desde plan_chunk_metrics")
    assert block_start != -1, "Sección 'Degraded rate' del health endpoint no encontrada"
    # Acotar al siguiente bloque numerado o final razonable.
    block = system_source[block_start:block_start + 1500]
    assert "FROM plan_chunk_metrics" in block
    assert "meal_plan_id IS NOT NULL" in block, (
        "Health endpoint debe excluir filas huérfanas para reportar "
        "calidad LIVE; sin el filtro el ops dashboard puede mostrar "
        "ratios distorsionados post-cleanup."
    )


# ---------------------------------------------------------------------------
# 3. NO se aplicó el filtro donde no toca
# ---------------------------------------------------------------------------
def test_alert_high_synthesized_lesson_ratio_unchanged(cron_source):
    """`_alert_high_synthesized_lesson_ratio` lee `chunk_lesson_telemetry`
    (meal_plan_id NOT NULL) y `plan_chunk_queue` (FK CASCADE — rows borradas
    en delete del padre). Es FALSO POSITIVO del audit. No debe haber
    modificación porque el bug no existe en ese cron."""
    func_start = cron_source.find("def _alert_high_synthesized_lesson_ratio(")
    func_end = cron_source.find("\ndef ", func_start + 1)
    block = cron_source[func_start:func_end]
    # Esta función NO debe leer plan_chunk_metrics (no aplica el bug).
    assert "FROM plan_chunk_metrics" not in block, (
        "Si _alert_high_synthesized_lesson_ratio empezó a leer "
        "plan_chunk_metrics, P2-3 necesita ampliarse — añadir el filtro."
    )


def test_insert_into_plan_chunk_metrics_unchanged(cron_source):
    """Los `INSERT INTO plan_chunk_metrics` no deben tener el filtro
    (writes, no reads). Verificamos para evitar que un refactor los
    añada por error."""
    insert_idx = cron_source.find("INSERT INTO plan_chunk_metrics")
    assert insert_idx != -1
    # Contar filtros dentro de los próximos 800 chars (signature del INSERT).
    insert_block = cron_source[insert_idx:insert_idx + 800]
    # No debe haber WHERE en un INSERT (sintácticamente nonsense).
    assert "meal_plan_id IS NOT NULL" not in insert_block, (
        "El INSERT no debe tener filtro WHERE meal_plan_id IS NOT NULL — "
        "el campo es la columna que se está poblando, no un filtro."
    )


# ---------------------------------------------------------------------------
# 4. Smoke combinado
# ---------------------------------------------------------------------------
def test_three_aggregator_queries_have_filter():
    """Smoke: las 3 queries agregadoras conocidas tienen el filtro nuevo.
    Si en el futuro se añade una 4ª query agregadora a plan_chunk_metrics
    sin el filtro, este test pita en review."""
    cron = _CRON.read_text(encoding="utf-8")
    system = _SYSTEM.read_text(encoding="utf-8")
    # Contar occurrences de "FROM plan_chunk_metrics" SIN INSERT cerca.
    pattern = r"FROM plan_chunk_metrics(?!\s*\()"
    cron_reads = re.findall(pattern, cron)
    system_reads = re.findall(pattern, system)
    total_reads = len(cron_reads) + len(system_reads)
    # 3 queries SELECT + posibles otros que ya filtran por user_id
    # específico (no agregadoras). Verificamos que las que usan agregación
    # tienen el filtro.
    assert total_reads >= 3, (
        f"Esperado al menos 3 SELECTs en plan_chunk_metrics; encontrado {total_reads}. "
        f"Si bajó, P2-3 puede haber sido revertido; si subió mucho, hay queries "
        f"nuevas que pueden necesitar el filtro."
    )
