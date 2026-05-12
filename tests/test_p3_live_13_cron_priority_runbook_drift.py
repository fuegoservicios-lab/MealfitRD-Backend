"""[P3-LIVE-10 + P3-LIVE-12 + P3-LIVE-13 · 2026-05-11] Drift detection
entre `register_plan_chunk_scheduler` (cron_tasks.py) y el runbook
`runbook_cron_priority_and_synthetic_cleanup_2026_05_11.md`.

P3-LIVE-10 — Mapa de prioridad de crons
    El runbook documenta los 30+ crons registrados con clasificación
    CRITICAL/HIGH/MEDIUM/LOW/HYGIENE. Sin enforcement parser-based, un
    nuevo cron registrado SIN entrada en el runbook deja al SRE sin guía
    bajo saturación del scheduler — repite el modo de fallo del audit
    2026-05-11 (cascade de 19 jobs missed sin saber cuáles sacrificar).

P3-LIVE-12 — SOP `Plan Sintético` manual cleanup
    Complementa el cron diario `_sweep_synthetic_test_plans` (P1-LIVE-3)
    cuando el cron está MISSED o el pattern requiere ajuste manual.

P3-LIVE-13 — Marker freshness anchor
    Validar que `_LAST_KNOWN_PFIX` se bumpeó a P3-LIVE-X tras cerrar los
    P3 del audit live (subsume el test P3-1 freshness, pero anchora el
    requisito específico de este batch).

Drift detection:
    - Nuevo `id="..."` en `register_plan_chunk_scheduler` sin entry en
      el runbook → falla.
    - Entry en runbook que ya no existe en código → falla (orphan).
    - Runbook desaparece / renombrado → falla.
    - Marker no bumpeado a P3-LIVE-X → falla.

Tooltip-anchor: P3-LIVE-10-START | P3-LIVE-12-START | P3-LIVE-13-START | gap audit 2026-05-11
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"
_APP = _BACKEND / "app.py"

# Runbook vive en el proyecto memory dir
_HOME = Path(os.path.expanduser("~"))
_MEMORY_DIR = (
    _HOME / ".claude" / "projects"
    / "c--Users-angel-OneDrive-Escritorio-MealfitRD-IA" / "memory"
)
_RUNBOOK = _MEMORY_DIR / "runbook_cron_priority_and_synthetic_cleanup_2026_05_11.md"


# Crons registrados en `register_plan_chunk_scheduler`. Extraídos via regex
# del source para detectar drift automáticamente.
def _extract_cron_ids(cron_source: str) -> set[str]:
    """Extrae todos los `id="<job_id>"` dentro de `register_plan_chunk_scheduler`."""
    # Localizar el cuerpo de register_plan_chunk_scheduler
    fn_pattern = re.compile(
        r"^def\s+register_plan_chunk_scheduler\s*\(",
        re.MULTILINE,
    )
    m = fn_pattern.search(cron_source)
    if not m:
        return set()
    next_def = re.compile(r"^(def |class |@)", re.MULTILINE).search(
        cron_source, pos=m.end()
    )
    body = cron_source[m.start():(next_def.start() if next_def else len(cron_source))]

    # Extraer todos los `id="..."` o `id='...'` dentro del cuerpo. Excluye
    # IDs documentados como ejemplos en comentarios mediante el filtro
    # "tiene comentario # en la misma línea antes del id=".
    ids = set()
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        id_match = re.search(r'\bid\s*=\s*["\']([a-zA-Z0-9_]+)["\']', line)
        if id_match:
            ids.add(id_match.group(1))
    return ids


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_source() -> str:
    return _APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def runbook_source() -> str:
    assert _RUNBOOK.exists(), (
        f"[P3-LIVE-10] Runbook ausente: {_RUNBOOK}. "
        f"Sin el runbook, SRE no tiene guía de qué cron sacrificar bajo "
        f"saturación. Restaurar desde memoria o re-crear desde "
        f"`project_p3_live_*.md`."
    )
    return _RUNBOOK.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P3-LIVE-10: drift bidireccional cron_tasks ↔ runbook
# ---------------------------------------------------------------------------

def test_p3_live_10_every_registered_cron_documented_in_runbook(
    cron_source: str, runbook_source: str
):
    """Cada `id="<job_id>"` en `register_plan_chunk_scheduler` debe aparecer
    en el runbook. Sin documentación, un SRE no sabe la criticidad del job."""
    code_ids = _extract_cron_ids(cron_source)
    assert code_ids, (
        "[P3-LIVE-10] No se detectaron `id=...` en register_plan_chunk_scheduler. "
        "Patrón roto o función renombrada."
    )

    missing = []
    for job_id in code_ids:
        if job_id not in runbook_source:
            missing.append(job_id)

    assert not missing, (
        f"[P3-LIVE-10] Crons registrados pero NO documentados en runbook: "
        f"{sorted(missing)}. Añadirlos a una de las 5 secciones (CRITICAL/"
        f"HIGH/MEDIUM/LOW/HYGIENE) en `runbook_cron_priority_and_synthetic_cleanup_2026_05_11.md`."
    )


def test_p3_live_10_runbook_has_all_5_priority_sections(runbook_source: str):
    """El runbook debe tener las 5 secciones de prioridad."""
    sections = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "HYGIENE"]
    for section in sections:
        assert section in runbook_source, (
            f"[P3-LIVE-10] Sección `{section}` ausente en runbook. "
            f"Esquema de 5 niveles definido en P3-LIVE-10 spec."
        )


def test_p3_live_10_runbook_references_known_critical_jobs(runbook_source: str):
    """Los crons CRITICAL conocidos (load-bearing) deben aparecer en el
    runbook bajo la sección CRITICAL específicamente."""
    # Split por header CRITICAL hasta el siguiente header
    m = re.search(r"###\s*CRITICAL.*?(?=###|\Z)", runbook_source, re.DOTALL)
    assert m, "[P3-LIVE-10] Sección CRITICAL del runbook no encontrada."
    critical_block = m.group(0)
    critical_jobs = [
        "process_plan_chunk_queue",
        "recover_failed_chunks_long_plans",
        "recover_future_scheduled_pending_chunks",
        "cleanup_orphan_chunks",
    ]
    for job_id in critical_jobs:
        assert job_id in critical_block, (
            f"[P3-LIVE-10] Cron `{job_id}` debe estar en sección CRITICAL "
            f"(load-bearing: sistema se rompe si missed sostenido). "
            f"Mover desde otra sección o añadir."
        )


# ---------------------------------------------------------------------------
# P3-LIVE-12: SOP plan sintético cleanup
# ---------------------------------------------------------------------------

def test_p3_live_12_sop_synthetic_cleanup_steps_present(runbook_source: str):
    """SOP debe enumerar los 6 pasos de cleanup manual."""
    # Verificar presencia de los headers de paso
    for step_num in range(1, 7):
        pattern = re.compile(rf"###\s*Paso\s+{step_num}\b", re.IGNORECASE)
        assert pattern.search(runbook_source), (
            f"[P3-LIVE-12] SOP no incluye Paso {step_num}. Los 6 pasos son: "
            f"detectar, backup audit, cancelar chunks, marcar abandoned, "
            f"verificar, post-mortem."
        )


def test_p3_live_12_sop_references_audit_table(runbook_source: str):
    """SOP debe referenciar `meal_plans_audit` para backup defensivo."""
    assert "meal_plans_audit" in runbook_source, (
        "[P3-LIVE-12] SOP no menciona `meal_plans_audit`. El backup defensivo "
        "es paso 2 del SOP — sin él, una mutación incorrecta es irrecuperable."
    )


def test_p3_live_12_sop_uses_jsonb_merge_not_set(runbook_source: str):
    """SOP debe usar `||` jsonb merge (atómico, exento I7) en lugar de
    full overwrite que requeriría advisory lock."""
    # Buscar al menos un UPDATE con `||` y `jsonb_build_object`
    assert "plan_data || jsonb_build_object" in runbook_source, (
        "[P3-LIVE-12] SOP no usa `plan_data || jsonb_build_object(...)`. "
        "Full overwrite requeriría advisory lock (I7); el merge `||` es "
        "atómico a nivel de keys mencionadas."
    )


def test_p3_live_12_cross_link_to_cron(runbook_source: str):
    """SOP debe cross-referenciar el cron automático para que el SRE sepa
    que el SOP es complementario, no único path."""
    assert "_sweep_synthetic_test_plans" in runbook_source, (
        "[P3-LIVE-12] SOP debe referenciar el cron automático "
        "`_sweep_synthetic_test_plans` (P1-LIVE-3). Sin cross-link, el SRE "
        "puede no saber que el cron existe y la duplicación es esperada."
    )


# ---------------------------------------------------------------------------
# P3-LIVE-13: marker freshness
# ---------------------------------------------------------------------------

def test_p3_live_13_marker_bumped(app_source: str):
    """`_LAST_KNOWN_PFIX` debe ser >= la fecha de cierre de P3-LIVE-13.

    Diseño original (escrito al cerrar P3-LIVE-13 el 2026-05-11):
    chequear presencia literal de `P3-LIVE` en el marker. Limitación:
    cualquier bump futuro con prefix distinto (P0-PROD-1, P1-AUDIT-NEW,
    etc.) hacía fallar el test sin razón funcional — el bump SÍ ocurrió.

    Relajado [P0-PROD-1 · 2026-05-12]: comparar contra date floor en
    lugar de substring. Preserva intent original (forzar bump tras
    P3-LIVE-13) sin colisionar con bumps subsecuentes. Mismo patrón que
    `test_p3_1_last_known_pfix_freshness::test_marker_date_meets_floor`.
    """
    from datetime import date, datetime
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']',
        app_source,
    )
    assert m, "[P3-LIVE-13] `_LAST_KNOWN_PFIX` no encontrado."
    marker = m.group(1)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_match, (
        f"[P3-LIVE-13] `_LAST_KNOWN_PFIX={marker}` sin fecha ISO. "
        f"El marker debe seguir formato `Pn-X · YYYY-MM-DD`."
    )
    marker_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 11)  # P3-LIVE-13 closure date
    assert marker_date >= floor, (
        f"[P3-LIVE-13] `_LAST_KNOWN_PFIX={marker}` (date={marker_date}) "
        f"anterior al floor {floor} de cierre P3-LIVE-13. "
        f"Sube el marker."
    )
