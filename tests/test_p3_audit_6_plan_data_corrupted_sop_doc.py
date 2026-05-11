"""[P3-AUDIT-6 · 2026-05-10] Anchor test del SOP de
`plan_data_corrupted:<plan_id>:<field_name>` en CLAUDE.md.

Bug original (audit 2026-05-10):
  La tabla de `system_alerts` decía que `plan_data_corrupted:*` era Manual,
  pero la fila no explicaba CÓMO. El operador SRE veía el alert, sabía que
  era Manual, pero no tenía SOP — riesgo de mutar `meal_plans.plan_data`
  sin backup, sin entender qué campo está corrupto, o sin cerrar el alert.

Fix:
  Sub-sección dedicada en CLAUDE.md ("SOP: resolver
  `plan_data_corrupted:<plan_id>:<field_name>`") con 7 pasos: extracción del
  plan_id/field_name del alert_key → SELECT del field → backup defensivo →
  decisión rollback vs hotfix → aplicar fix → UPDATE resolved_at → post-mortem.

Este test ancla la existencia de la sección y de los pasos clave. Si alguien
elimina la sección o rompe la numeración, falla en CI con copy explicativo.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"


def test_sop_section_exists():
    """La sub-sección con SOP debe existir bajo "Política de `system_alerts`
    resolution"."""
    src = _CLAUDE_MD.read_text(encoding="utf-8")
    header = "### SOP: resolver `plan_data_corrupted:<plan_id>:<field_name>`"
    assert header in src, (
        f"Falta el header '{header}' en CLAUDE.md. "
        f"Sin SOP, el operador SRE recibe alert manual sin guía y "
        f"puede mutar meal_plans.plan_data sin backup."
    )


def test_sop_steps_complete():
    """El SOP debe enumerar al menos 7 pasos. Si bajan, alguien quitó
    parte del procedimiento."""
    src = _CLAUDE_MD.read_text(encoding="utf-8")
    section_start = src.find(
        "### SOP: resolver `plan_data_corrupted:<plan_id>:<field_name>`"
    )
    assert section_start >= 0
    section_end = src.find("\n### ", section_start + 1)
    if section_end < 0:
        section_end = len(src)
    section = src[section_start:section_end]

    # Pasos numerados: "1. ", "2. ", ..., "7. ". Markdown puede renderizar
    # con o sin sangría; el contenido fuente usa el formato estricto.
    steps = [f"{n}. " for n in range(1, 8)]
    missing = [s for s in steps if s not in section]
    assert not missing, (
        f"El SOP de plan_data_corrupted debe tener al menos 7 pasos. "
        f"Faltan: {missing}. Si reordenaste, actualiza este test."
    )


def test_sop_mentions_backup_before_mutation():
    """El SOP DEBE mencionar backup defensivo ANTES de mutación. La regla
    operacional core: nunca tocar plan_data sin snapshot previo."""
    src = _CLAUDE_MD.read_text(encoding="utf-8")
    section_start = src.find(
        "### SOP: resolver `plan_data_corrupted:<plan_id>:<field_name>`"
    )
    section_end = src.find("\n### ", section_start + 1)
    if section_end < 0:
        section_end = len(src)
    section = src[section_start:section_end].lower()

    assert "backup" in section, (
        "El SOP debe mencionar 'backup' explícitamente ANTES del paso de "
        "mutación. Sin esa instrucción defensiva, un operador puede pisar "
        "plan_data corruptos sin posibilidad de rollback."
    )
    assert "antes" in section or "before" in section, (
        "El SOP debe enfatizar que el backup va ANTES de la mutación, no "
        "como side-effect."
    )


def test_sop_includes_resolved_at_update():
    """El SOP debe incluir el UPDATE explícito a `system_alerts.resolved_at`
    porque el productor solo emite, no resuelve. Sin este paso final, la
    alert queda viva indefinidamente y futuras corrupciones del MISMO field
    no re-emiten (ON CONFLICT pisa pero el operador no ve el delta)."""
    src = _CLAUDE_MD.read_text(encoding="utf-8")
    section_start = src.find(
        "### SOP: resolver `plan_data_corrupted:<plan_id>:<field_name>`"
    )
    section_end = src.find("\n### ", section_start + 1)
    if section_end < 0:
        section_end = len(src)
    section = src[section_start:section_end]

    assert "UPDATE system_alerts" in section, (
        "El SOP debe incluir el `UPDATE system_alerts SET resolved_at = "
        "NOW()` como paso explícito. Sin este paso, la alert no se cierra "
        "y la dashboard pierde la señal de 'incidente resuelto'."
    )
    assert "resolved_at" in section
