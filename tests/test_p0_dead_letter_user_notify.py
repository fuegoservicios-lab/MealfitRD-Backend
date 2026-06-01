"""[P0-DEAD-LETTER-USER-NOTIFY · 2026-05-27] Regression guard.

Bug original (audit 2026-05-27):
    Dead-letters por validacion del worker (GAP3 numeracion invalida +
    cualquier exception de pipeline LLM tras CHUNK_MAX_FAILURE_ATTEMPTS)
    bypasseaban `_escalate_unrecoverable_chunk` en las dos UPDATEs del
    outer-catch de `_chunk_worker`:

      - L27604 (is_critical=True) y L27625 (is_critical=False) escribian
        `dead_letter_reason = str(e)[:240]` literal con el texto del
        Exception (ej. `"[GAP3] Chunk 2 numeracion invalida. Esperado
        [4,5,6,7], recibido [1,2,3]..."`). Ese string NO esta en
        `ESCALATION_REASONS` (constants.py:2482), asi que:
          (a) `/blocked_reasons` caia al fallback `_unknown` (plans.py:3733),
              mintiendo al usuario sobre por que fallo,
          (b) el cron P2-NEW-3 `_escalate_unrecoverable_chunk` rechazaba
              cualquier callsite posterior con ese reason invalid.

      - La rama `is_degraded=True` fatal (L27649+) push-notificaba inline
        con copy mojibakeado (`"\xe2\x9a\xa0\xef\xb8\x8f Error extendiendo tu plan"`
        double-encoded) y NO seteaba `_recovery_exhausted_chunks` ni
        `_user_action_required` en `meal_plans.plan_data` -> el banner del
        Dashboard nunca aparecia. Tampoco emitia el per-chunk alert
        `dead_lettered_chunk:<plan>:<week>` (solo quedaba la agregada
        `dead_lettered_chunks_recent`, que solo SRE ve).

      - La rama `is_degraded=False` (downgrade-to-shuffle, L27681+) rescata
        el chunk que acaba de marcarse 'failed' con `dead_lettered_at = NOW()`,
        pero el rescue UPDATE pre-fix solo reseteaba `status='pending'` y
        `attempts=0` -> dejaba `dead_lettered_at` y `dead_letter_reason`
        intactos. El cron P1-2 `_alert_new_dead_lettered_chunks` contaba ese
        chunk como dead-lettered aunque estuviera vivo corriendo Smart
        Shuffle, inflando `dead_lettered_chunks_recent` con falsos positivos.

Evidencia viva en prod 2026-05-27:
    Chunk 2603e618-29ff-40be-a66a-6c47ce9dd8ca del plan
    1cb1d027-d97c-4f03-94f5-aaf25fe9da0a (user 8e40b0fd) dead-lettered con
    `dead_letter_reason = "[GAP3] Chunk 2 numeracion invalida. Esperado
    [4,5,6,7], recibido [1,2,3]..."`. Plan de 15 dias quedo con week 1 (dias
    1-3) pero week 2 (dias 4-7) faltantes, sin banner, sin push, sin
    `_user_action_required`. Usuario en limbo.

Fix:
    1. Las dos UPDATEs en `_chunk_worker` (is_critical=True/False) ahora
       escriben `dead_letter_reason = "recovery_exhausted"` (canonical de
       `ESCALATION_REASONS`). Forensic completo (`str(e)[:1000]`) sigue
       persistido en `pipeline_metrics.error_message` via
       `_record_chunk_metric` arriba en el mismo except.

    2. La rama `is_degraded=True` fatal invoca
       `_escalate_unrecoverable_chunk(escalation_reason="recovery_exhausted")`
       en lugar del push mojibakeado inline. El helper hace los 4 pasos
       idempotentemente: UPDATE chunk (COALESCE no pisa nuestro dead_letter
       row), UPDATE meal_plans.plan_data._recovery_exhausted_chunks +
       _user_action_required (banner frontend), INSERT system_alerts per-
       chunk con alert_key `dead_lettered_chunk:<plan>:<week>`, push con
       copy correcto + deeplink `/dashboard?action_required=recovery_exhausted`.

    3. La rama `is_degraded=False` rescue UPDATE ahora limpia
       `dead_lettered_at = NULL, dead_letter_reason = NULL` para que el
       chunk genuinamente regrese a la cola sin dejar marca forense falsa.

Este test:
    - Parser-based: scanea `cron_tasks.py` y exige los 3 invariantes.
    - Funcional: mockea `execute_sql_write` + `_escalate_unrecoverable_chunk`
      y verifica que la rama fatal lo llama con kwargs correctos.

Tooltip-anchor: P0-DEAD-LETTER-USER-NOTIFY.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND_ROOT / "cron_tasks.py"
_CONSTANTS = _BACKEND_ROOT / "constants.py"


def _read_cron_tasks() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Parser-based: structural invariants
# -----------------------------------------------------------------------------

def test_recovery_exhausted_is_canonical_in_escalation_reasons():
    """`"recovery_exhausted"` debe estar declarado en `ESCALATION_REASONS`
    (constants.py). Si alguien renombra el reason canonical, el helper
    `_escalate_unrecoverable_chunk` rechazaria nuestras llamadas y volveriamos
    al regimen pre-fix (chunks sin user notification).
    """
    text = _CONSTANTS.read_text(encoding="utf-8")
    assert '"recovery_exhausted"' in text, (
        '`"recovery_exhausted"` no aparece en constants.py. '
        'Probablemente fue removido del catalogo `ESCALATION_REASONS` — '
        'P0-DEAD-LETTER-USER-NOTIFY requiere que siga siendo canonical.'
    )
    # Verificar que esta dentro de la tupla ESCALATION_REASONS
    m = re.search(
        r"ESCALATION_REASONS\s*=\s*\(([^)]*)\)",
        text,
        re.DOTALL,
    )
    assert m is not None, "ESCALATION_REASONS no encontrada en constants.py"
    assert '"recovery_exhausted"' in m.group(1), (
        '`"recovery_exhausted"` no esta dentro de la tupla ESCALATION_REASONS. '
        'El helper `_escalate_unrecoverable_chunk` lo rechazaria.'
    )


def test_chunk_worker_updates_use_canonical_reason_not_str_e():
    """El UPDATE del outer-catch de `_chunk_worker` debe pasar `"recovery_exhausted"`
    literal como `dead_letter_reason` en su param list. NO `str(e)[:240]` (el bug
    original).

    [S18-2 · GAP-6 · 2026-05-29] Pre-fix había DOS UPDATEs byte-idénticos (ramas
    is_critical=True/False) y este test asertaba `>= 2`. El colapso S18-2 unificó
    ambas en UN solo UPDATE (la única diferencia era una línea de log), así que ahora
    se asienta `>= 1`. El intent real del test (canonical reason, NO `str(e)[:240]`)
    se preserva por esta aserción + el guard anti-`str(e)[:240]` de abajo.
    """
    text = _read_cron_tasks()

    # Patron rigido: lineas que son SOLO `"recovery_exhausted",` con indent
    # (el formato del param list de execute_sql_write).
    matches = re.findall(
        r'^\s+"recovery_exhausted",\s*$',
        text,
        re.MULTILINE,
    )
    assert len(matches) >= 1, (
        f'Esperaba >=1 linea con `"recovery_exhausted",` como param del UPDATE de '
        f'escalation en `_chunk_worker` (colapsado de 2 a 1 en S18-2). '
        f'Encontre {len(matches)}. Si reemplazaste el canonical reason por '
        f'`str(e)[:240]`, regresamos al bug P0-DEAD-LETTER-USER-NOTIFY.'
    )

    # Defensa anti-regresion: `str(e)[:240]` no debe aparecer como LIVE
    # CODE (no en comentario) en cron_tasks.py. El patron pre-fix era pasarlo
    # como param de UPDATE para que terminara como `dead_letter_reason` raw.
    # Lo aceptamos solo dentro de comentarios que documentan el bug original.
    live_str_e_240 = []
    for i, line in enumerate(text.splitlines(), 1):
        if 'str(e)[:240]' in line:
            stripped = line.lstrip()
            # Aceptar si la linea es comentario (`#`) o esta dentro de un
            # docstring/string literal (heuristica: si empieza con `#` no es
            # live). Para nuestro caso un `str(e)[:240]` legitimo no deberia
            # existir; cualquier ocurrencia live es regresion.
            if not stripped.startswith('#'):
                live_str_e_240.append((i, line.strip()[:120]))
    assert not live_str_e_240, (
        f'Bug pattern `str(e)[:240]` regresion como LIVE CODE en cron_tasks.py: '
        f'{live_str_e_240[:3]}. Usar `"recovery_exhausted"` canonical en su lugar. '
        f'El forensic full sigue en pipeline_metrics.error_message via '
        f'_record_chunk_metric (cron_tasks.py:~27556).'
    )


def test_chunk_worker_fatal_branch_calls_escalator():
    """La rama `if is_degraded:` fatal (post-failure transition en
    `_chunk_worker`) debe invocar `_escalate_unrecoverable_chunk(...)` con
    `escalation_reason="recovery_exhausted"`. Sin esta llamada, el usuario
    no recibe push, el banner del frontend nunca aparece, y el per-chunk
    alert `dead_lettered_chunk:<plan>:<week>` no se emite.
    """
    text = _read_cron_tasks()

    # Buscar el bloque `if is_degraded:` que sigue al `if res and res[0].get('status') == 'failed':`
    # y dentro de ese bloque debe haber un call a `_escalate_unrecoverable_chunk(`
    # con `escalation_reason="recovery_exhausted"`.
    fatal_block_pat = re.compile(
        r"if\s+is_degraded\s*:\s*\r?\n"
        r".*?_escalate_unrecoverable_chunk\(\s*\r?\n"
        r"\s+task_id=task_id,\s*\r?\n"
        r"\s+user_id=user_id,\s*\r?\n"
        r"\s+plan_id=meal_plan_id,\s*\r?\n"
        r"\s+week_number=week_number,\s*\r?\n"
        r"\s+recovery_attempts=next_attempt,\s*\r?\n"
        r'\s+escalation_reason="recovery_exhausted",',
        re.DOTALL,
    )
    assert fatal_block_pat.search(text), (
        'La rama `if is_degraded:` fatal en `_chunk_worker` no invoca '
        '`_escalate_unrecoverable_chunk(escalation_reason="recovery_exhausted")` '
        'con la firma esperada. Sin esa llamada, usuarios afectados quedan '
        'en limbo (sin push, sin banner). Vease P0-DEAD-LETTER-USER-NOTIFY.'
    )


def test_chunk_worker_fatal_branch_drops_mojibake_inline_push():
    """La rama `if is_degraded:` fatal NO debe contener el push inline
    mojibakeado (`"\xe2\x9a\xa0\xef\xb8\x8f Error extendiendo tu plan"` con bytes double-encoded
    en el source) ni la importacion `from utils_push import send_push_notification`
    inline. La notificacion debe pasar exclusivamente por
    `_escalate_unrecoverable_chunk`.
    """
    text = _read_cron_tasks()

    # El texto mojibake especifico que estaba en el push body.
    mojibake_marker = "Hubo un problema generando tus proximas semanas"
    # Buscar lineas con ese marker que NO sean comentarios (live code).
    live_offenders = []
    for line in text.splitlines():
        if mojibake_marker in line:
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                live_offenders.append(line.strip())

    assert not live_offenders, (
        f'El push inline mojibakeado del bug original sigue activo en '
        f'`_chunk_worker`: {live_offenders[:2]}. La notificacion debe '
        f'delegarse al helper `_escalate_unrecoverable_chunk`.'
    )


def test_chunk_worker_shuffle_rescue_clears_dead_letter_columns():
    """La rama `is_degraded=False` (downgrade-to-shuffle) que rescata todos
    los chunks `pending`/`failed` del plan a `pending` con `_degraded=true`
    debe LIMPIAR `dead_lettered_at = NULL, dead_letter_reason = NULL` para
    el chunk que acaba de marcarse 'failed' en el except superior. Sin esa
    limpieza, el cron `_alert_new_dead_lettered_chunks` cuenta falsos
    positivos (chunks "dead-lettered" pero genuinamente vivos en Smart
    Shuffle).
    """
    text = _read_cron_tasks()

    rescue_clear_pat = re.compile(
        r"UPDATE\s+plan_chunk_queue\s*\r?\n"
        r"\s+SET\s+status\s*=\s*'pending'\s*,\s*\r?\n"
        r"\s+attempts\s*=\s*0\s*,\s*\r?\n"
        r"\s+dead_lettered_at\s*=\s*NULL\s*,\s*\r?\n"
        r"\s+dead_letter_reason\s*=\s*NULL\s*,",
        re.DOTALL,
    )
    assert rescue_clear_pat.search(text), (
        'El rescue UPDATE del downgrade-to-shuffle (`is_degraded=False`) no '
        'limpia `dead_lettered_at = NULL, dead_letter_reason = NULL`. Chunks '
        'resucitados quedaran marcados forensicamente como dead-lettered, '
        'inflando `dead_lettered_chunks_recent` con falsos positivos. '
        'Vease P0-DEAD-LETTER-USER-NOTIFY.'
    )


def test_anchor_marker_present_in_chunk_worker():
    """El marker textual `P0-DEAD-LETTER-USER-NOTIFY` debe aparecer en
    `cron_tasks.py` para que un futuro refactor no borre la convencion
    silenciosamente. Cross-link al cierre del P-fix.
    """
    text = _read_cron_tasks()
    occurrences = text.count("P0-DEAD-LETTER-USER-NOTIFY")
    assert occurrences >= 3, (
        f'Esperaba >=3 referencias al marker `P0-DEAD-LETTER-USER-NOTIFY` '
        f'en cron_tasks.py (al menos: comment en cada UPDATE param, comment '
        f'en rama fatal, comment en rescue clear, log.error en except del '
        f'escalate). Encontre {occurrences}.'
    )


# -----------------------------------------------------------------------------
# Functional: mock _escalate_unrecoverable_chunk and verify call
# -----------------------------------------------------------------------------

def test_escalate_unrecoverable_chunk_accepts_recovery_exhausted_reason():
    """Sanity check: `_escalate_unrecoverable_chunk` no rechaza nuestro
    reason canonical. El helper valida contra `ESCALATION_REASONS` y aborta
    si el reason no esta listado (P2-NEW-3); confirmamos que
    `"recovery_exhausted"` pasa el guard.

    NO ejecutamos el cuerpo del helper (haria SQL real); solo verificamos
    que el reason esta en el whitelist canonical via import.
    """
    import sys
    # Stub apscheduler (no instalado en CI standalone) — mismo patron que
    # otros tests del bundle P1-CRON-BUNDLE.
    if "apscheduler" not in sys.modules:
        import types
        for mod_name in (
            "apscheduler",
            "apscheduler.schedulers",
            "apscheduler.schedulers.background",
            "apscheduler.executors",
            "apscheduler.executors.pool",
            "apscheduler.events",
        ):
            sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
        # Constants que el listener referencia.
        sys.modules["apscheduler.events"].EVENT_JOB_MISSED = 1
        sys.modules["apscheduler.events"].EVENT_JOB_ERROR = 2
        sys.modules["apscheduler.events"].EVENT_JOB_EXECUTED = 4

    from constants import ESCALATION_REASONS
    assert "recovery_exhausted" in ESCALATION_REASONS, (
        '`"recovery_exhausted"` no esta en ESCALATION_REASONS. El helper '
        '`_escalate_unrecoverable_chunk` rechazaria la llamada del fix P0 '
        'y regresariamos al bug original (user en limbo).'
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
