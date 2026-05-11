"""[P1-NEW-2 · 2026-05-10] Regression guard: el UPDATE de escalation
en `_chunk_worker` (cron_tasks.py ~L22276 y ~L22296) tiene CAS guard
`AND attempts = %s` contra `_pickup_attempts`.

Bug original (audit 2026-05-10):
    `_chunk_worker` captura `_pickup_attempts = int(task["attempts"] or 0)`
    al recoger el chunk (cron_tasks.py:16797). Si el chunk falla y
    cae al except handler (~L22250+), el UPDATE de escalation incrementaba
    attempts vía `COALESCE(attempts, 0) + 1` con `WHERE id = %s` simple.

    Race: si entre el pickup y el except handler el zombie rescue
    (cron de stale-locks) ya incrementó attempts y otro worker
    reclamó el chunk, nuestra UPDATE clobbearía:
      - Incremento doble de attempts → escalation prematura a dead_letter.
      - Potencial sobreescritura de `dead_letter_reason` del worker B.
      - Disparo erróneo del bloque zombie-cascade (status='partial' +
        cancel-future-chunks + push), confundiendo al usuario.

Fix:
    Añadir `AND attempts = %s` al WHERE de ambas UPDATEs (critical y
    normal path) + pasar `_pickup_attempts` como CAS token. Si el
    rowcount=0 (CAS displaced), log warning y NO ejecutar zombie-cascade
    porque el otro worker es el dueño actual.

Cobertura (parser-based, no DB):
    1. Ambos UPDATEs llevan `WHERE id = %s AND attempts = %s` al CASE
       de escalation.
    2. `_pickup_attempts` aparece en el tuple de params de ambos.
    3. Existe rama `if not res:` con log `CAS-DISPLACED` que aborta
       limpiamente sin ejecutar zombie-cascade.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


def _read_source() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Ambos UPDATEs de escalation tienen CAS guard
# ---------------------------------------------------------------------------
def test_escalation_updates_have_cas_attempts_guard():
    """Los dos UPDATEs de escalation (critical + normal) llevan
    `WHERE id = %s AND attempts = %s` para guard CAS contra
    `_pickup_attempts`."""
    src = _read_source()
    # Pattern: SET attempts = COALESCE(attempts, 0) + 1 + ... +
    # WHERE id = %s AND attempts = %s.
    pattern = re.compile(
        r"UPDATE\s+plan_chunk_queue\s+"
        r"SET\s+attempts\s*=\s*COALESCE\(attempts,\s*0\)\s*\+\s*1[^;]+?"
        r"WHERE\s+id\s*=\s*%s\s+AND\s+attempts\s*=\s*%s",
        re.DOTALL,
    )
    matches = pattern.findall(src)
    # Esperamos al menos 2 (critical path + normal path en _chunk_worker).
    assert len(matches) >= 2, (
        f"Esperaba >=2 UPDATEs con `WHERE id = %s AND attempts = %s` en "
        f"`SET attempts = COALESCE(attempts, 0) + 1` (escalation paths "
        f"critical + normal en _chunk_worker). Encontrados: {len(matches)}. "
        f"Sin el CAS guard, dos workers pueden duplicar el incremento de "
        f"attempts si el zombie rescue compite con el worker original."
    )


def test_escalation_updates_pass_pickup_attempts():
    """Ambos call sites pasan `_pickup_attempts` como último param del
    tuple de params, alineado con `AND attempts = %s` del WHERE."""
    src = _read_source()
    # Cuenta ocurrencias de `_pickup_attempts,` en la sección del
    # except handler (~L22250-L22310). Verificación laxa: que el
    # símbolo aparezca al menos 2 veces como param de UPDATEs.
    # Trim al rango razonable: desde `is_critical = lag_seconds > 86400`
    # hasta el final del except handler.
    start_match = re.search(r"is_critical\s*=\s*lag_seconds\s*>\s*86400", src)
    assert start_match is not None, (
        "No encuentro el marker `is_critical = lag_seconds > 86400` que "
        "delimita el except handler de escalation."
    )
    end_match = re.search(
        r"\[GAP 4 DE 30 DÃAS FIX\s*/\s*GAP 2 IMPLEMENTATION\]",
        src,
    )
    assert end_match is not None, (
        "No encuentro el marker `[GAP 4 DE 30 DÃAS FIX / GAP 2 "
        "IMPLEMENTATION]` que marca el final del bloque escalation."
    )
    section = src[start_match.start():end_match.start()]
    # `_pickup_attempts` debe aparecer al menos 2 veces como param (al
    # final de cada tuple de params, antes del `), returning=True)`).
    occurrences = section.count("_pickup_attempts,")
    assert occurrences >= 2, (
        f"Esperaba >=2 referencias a `_pickup_attempts,` (param de los 2 "
        f"UPDATEs de escalation). Encontradas: {occurrences}. Sin el CAS "
        f"token correcto el WHERE clause no aplica."
    )


# ---------------------------------------------------------------------------
# 2. Rama CAS-displaced existe y aborta sin zombie-cascade
# ---------------------------------------------------------------------------
def test_cas_displaced_branch_present():
    """Existe el `if not res:` con log `CAS-DISPLACED` y aborta limpiamente
    SIN ejecutar la cascada zombie (status='partial', cancel-future-chunks,
    push). Si esta rama desaparece, una CAS-failed se trata como res-vacío
    y silenciosamente no logueamos el displacement."""
    src = _read_source()
    # Buscar el marker P1-NEW-2/CAS-DISPLACED en el log warning.
    assert "[P1-NEW-2/CAS-DISPLACED]" in src, (
        "No encuentro el log `[P1-NEW-2/CAS-DISPLACED]` que debe avisar "
        "cuando el CAS guard rechazó la UPDATE de escalation. Sin ese log "
        "perdemos observabilidad del race vs zombie rescue."
    )


def test_zombie_cascade_only_runs_when_res_truthy():
    """El bloque zombie-cascade (`if res and res[0].get('status') == 'failed':`)
    sigue gated por `res and ...` — esto significa que cuando CAS falla
    (res vacío) el bloque NO ejecuta, evitando push/cancel cascada errónea."""
    src = _read_source()
    pattern = re.compile(
        r"if\s+res\s+and\s+res\[0\]\.get\(['\"]status['\"]\)\s*==\s*['\"]failed['\"]\s*:"
    )
    assert pattern.search(src) is not None, (
        "El guard `if res and res[0].get('status') == 'failed':` desapareció. "
        "Sin ese guard, un CAS-failed (res vacío) pasaría inadvertido y "
        "podríamos disparar push/cancel cascada sobre un chunk que otro "
        "worker está procesando."
    )
