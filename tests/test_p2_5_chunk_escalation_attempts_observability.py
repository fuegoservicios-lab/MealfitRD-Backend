"""[P2-5 · 2026-05-08] Tests de observabilidad de `attempts` en escalación
de chunks pausados indefinidamente.

Hallazgo del audit (2026-05-07):
  La premisa específica ("Phase 2 puede mover a dead_letter chunks que aún
  tenían `attempts < CHUNK_MAX_FAILURE_ATTEMPTS`, off-by-one") era FALSO
  POSITIVO. Verificación:
    - `pending_user_action` con `_pause_reason='missing_prior_lessons'` se
      setea en cron_tasks.py:16108 SIN incrementar `attempts`. Es un estado
      ortogonal "stuck waiting for human", no un retry agotado.
    - `_alert_chunks_paused_indefinitely` escala basándose en
      `paused_hours >= CHUNK_INDEFINITE_PAUSE_ESCALATE_HOURS` (24h por
      defecto) + un último intento de unblock. La invariante temporal es
      por diseño, no por presupuesto de attempts.
    - `attempts` y `pending_user_action`/`missing_prior_lessons` son state
      machines orthogonales. No hay off-by-one que arreglar.

  PERO existe una mejora defensiva genuina: al escalar a dead_letter, el
  log y el alert metadata NO incluían el valor de `attempts`. Para forensics
  ("¿este chunk había tenido retries parciales antes?"), el SRE necesita
  ese dato para diagnosticar root-cause.

Fix:
  1. SELECT incluye `q.attempts` en el query de candidatos.
  2. `chunk_attempts` extraído por iteración como variable local.
  3. `alert_metadata["attempts"] = chunk_attempts` en la fila de Phase 1.
  4. Log de Phase 2 (escalación a dead_letter) incluye `attempts={N}`.
  5. Sin cambio de comportamiento — el gating sigue siendo temporal.

Cobertura:
  - SELECT contiene `q.attempts`.
  - `chunk_attempts` se extrae como variable local.
  - alert_metadata incluye campo `attempts`.
  - El log de escalación incluye `attempts=` en el formato.
  - El comportamiento de gating (paused_hours < escalate_age_hours: continue)
    NO se ve modificado por `attempts`.
"""
import pathlib
import re

import pytest


# [Stale-fix P1-CHUNK-LEARN-3 · 2026-05-29] Era `.parent / "cron_tasks.py"` →
# resolvía a tests/cron_tasks.py (inexistente) → 6 errores de setup. cron_tasks.py
# vive en backend/ (parent.parent del test). Bug pre-existente de path.
_CRON = pathlib.Path(__file__).resolve().parent.parent / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def alert_function_block(cron_source) -> str:
    """Aísla el bloque de `_alert_chunks_paused_indefinitely`."""
    start = cron_source.find("def _alert_chunks_paused_indefinitely(")
    assert start != -1, "función _alert_chunks_paused_indefinitely no encontrada"
    end = cron_source.find("\ndef ", start + 1)
    return cron_source[start:end]


# ---------------------------------------------------------------------------
# 1. SELECT incluye attempts
# ---------------------------------------------------------------------------
def test_select_includes_attempts_column(alert_function_block):
    """El SELECT de candidatos debe incluir `q.attempts` para que esté
    disponible en forensics. P2-5 lo añadió como columna observable."""
    # Buscar la query del SELECT principal (la que filtra _pause_reason).
    select_match = re.search(
        r'SELECT.*?FROM plan_chunk_queue q.*?WHERE q\.status\s*=\s*\'pending_user_action\'',
        alert_function_block,
        re.DOTALL,
    )
    assert select_match is not None, "Query de candidatos no encontrada en el bloque"
    select_block = select_match.group(0)
    assert "q.attempts" in select_block, (
        "El SELECT debe incluir `q.attempts` para que el cron tenga el valor "
        "disponible al construir alert_metadata + log de escalación. P2-5."
    )


# ---------------------------------------------------------------------------
# 2. Variable local chunk_attempts
# ---------------------------------------------------------------------------
def test_chunk_attempts_extracted_as_local_var(alert_function_block):
    """Tras el fetch, debe existir una variable local `chunk_attempts`
    extraída de `row.get("attempts")` para uso downstream."""
    assert re.search(
        r'chunk_attempts\s*=\s*int\(\s*row\.get\(\s*["\']attempts["\']\s*\)\s*or\s*0\s*\)',
        alert_function_block,
    ), (
        "Debe existir `chunk_attempts = int(row.get('attempts') or 0)` para "
        "asegurar que el campo se referencia con un nombre claro y casteado."
    )


# ---------------------------------------------------------------------------
# 3. alert_metadata incluye attempts
# ---------------------------------------------------------------------------
def test_alert_metadata_includes_attempts(alert_function_block):
    """El dict alert_metadata persistido en system_alerts debe incluir
    `attempts`. Sin esto, el dashboard ops queda ciego al estado de retries."""
    # Buscar el dict alert_metadata.
    metadata_match = re.search(
        r'alert_metadata\s*=\s*\{[^}]*?\}',
        alert_function_block,
        re.DOTALL,
    )
    assert metadata_match is not None, "alert_metadata dict no encontrado"
    metadata_block = metadata_match.group(0)
    assert '"attempts"' in metadata_block, (
        "alert_metadata debe incluir el campo `attempts` para que el SRE "
        "vea el valor en system_alerts.metadata sin tener que cruzar tablas."
    )
    assert "chunk_attempts" in metadata_block, (
        "El campo attempts debe asignarse desde la variable local "
        "`chunk_attempts`, no desde un literal."
    )


# ---------------------------------------------------------------------------
# 4. Log de escalación incluye attempts
# ---------------------------------------------------------------------------
def test_escalation_log_includes_attempts(alert_function_block):
    """El log de Phase 2 (ESCALATED) debe incluir `attempts={N}` para
    forensics post-mortem. P2-5."""
    # Buscar el bloque del logger.error con [P1-CHUNKS-3/ESCALATED]. Acotar
    # al bloque del logger.error completo (hasta el cierre del paréntesis).
    escalated_idx = alert_function_block.find("[P1-CHUNKS-3/ESCALATED]")
    assert escalated_idx != -1, "Tag [P1-CHUNKS-3/ESCALATED] no encontrado"
    # Buscar específicamente el logger.error (no el del UNBLOCK-OK ni
    # otros). Tomamos los próximos 600 chars desde el primer match
    # tras la palabra "dead-lettered".
    dead_lettered_idx = alert_function_block.find("dead-lettered", escalated_idx)
    assert dead_lettered_idx != -1, "Bloque ESCALATED+dead-lettered no encontrado"
    log_block = alert_function_block[dead_lettered_idx:dead_lettered_idx + 600]
    assert "attempts=" in log_block, (
        f"El log [P1-CHUNKS-3/ESCALATED] debe contener `attempts={{chunk_attempts}}` "
        f"para forensics. P2-5: distinguir chunks llegados directo a stuck "
        f"(attempts=0) de aquellos con retries parciales antes. "
        f"Bloque inspeccionado: {log_block[:200]!r}..."
    )
    assert "chunk_attempts" in log_block, (
        "El log debe interpolar la variable `chunk_attempts` (no un literal)."
    )


# ---------------------------------------------------------------------------
# 5. Gating temporal preserved
# ---------------------------------------------------------------------------
def test_temporal_gating_unchanged_by_attempts(alert_function_block):
    """P2-5 NO modifica la lógica de gating — la decisión de saltar a
    Phase 2 sigue siendo `paused_hours < escalate_age_hours`. `attempts`
    es puramente observable."""
    # Verificar que el guard temporal sigue presente.
    assert re.search(
        r'if\s+paused_hours\s*<\s*escalate_age_hours\s*:\s*\n\s*continue',
        alert_function_block,
    ), (
        "El guard `if paused_hours < escalate_age_hours: continue` debe "
        "preservarse — P2-5 NO cambia comportamiento, solo añade observabilidad."
    )
    # Verificar que NO hay un guard nuevo del estilo
    # `if chunk_attempts < CHUNK_MAX_FAILURE_ATTEMPTS: continue`.
    assert not re.search(
        r'if\s+chunk_attempts\s*<\s*CHUNK_MAX_FAILURE_ATTEMPTS',
        alert_function_block,
    ), (
        "P2-5 NO debe introducir gating en attempts — los state machines son "
        "ortogonales. El audit original había caracterizado mal la relación."
    )


# ---------------------------------------------------------------------------
# 6. Smoke: no rompimos otras señales del cron
# ---------------------------------------------------------------------------
def test_other_log_lines_intact(alert_function_block):
    """Las otras líneas de log (UNBLOCK-OK, ALERT, resumen final) deben
    seguir siendo emitidas — P2-5 solo toca el log de escalación."""
    expected_log_tags = [
        "[P1-CHUNKS-3/UNBLOCK-OK]",
        "[P1-CHUNKS-3/ALERT]",
        "[P1-CHUNKS-3] Resumen tick",
    ]
    for tag in expected_log_tags:
        assert tag in alert_function_block, (
            f"Log tag {tag!r} ausente — P2-5 puede haber roto otra señal."
        )
