"""[P3-SKELETON-FIDELITY-CRITIQUE-AWARE · 2026-05-16] Threshold dinámico
del check SKELETON FIDELITY según si self_critique modificó el día.

Síntoma observado plan post-reset 2026-05-16 21:49-22:04:
  - Skeleton asignó a Día 1: protein_pool = ['Soya/Tofu', 'Claras de huevo', 'Queso Mozzarella']
  - Self-critique sugirió: "Cambiar las claras de la cena por queso para
    reducir la repetición de este staple"
  - LLM corrector removió Soya/Tofu Y Queso Mozzarella
  - SKELETON FIDELITY (threshold hardcoded >=2 missing) → rechazo fatal
  - Reviewer CB OPEN para gemini-3.1-flash-lite → fail-closed
  - Plan crítico abortado, usuario sin plan utilizable

Root cause: el check de fidelity NO distinguía entre:
  A) day_generator ignoró skeleton (bug real → rechazar)
  B) self_critique reemplazó proteínas legítimamente (NO es bug → tolerar)

Pre-fix: ambos casos → threshold >=2 → ambos rechazados.
Post-fix: día con `_critique_applied=True` → threshold >=3 (tolera 1-2
reemplazos legítimos); día sin marker → threshold >=2 (preserva detección
del bug original P0-6 que motivó el check).

El marker `_critique_applied=True` se inyecta en `corrected_day` cuando
`_correct_single_day` retorna exitosamente desde el corrector LLM.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix #1: marker `_critique_applied` se inyecta en days corregidos
# ---------------------------------------------------------------------------


def test_critique_applied_marker_set_in_corrected_day():
    """Cuando `_correct_single_day` retorna exitosamente desde el corrector
    LLM, el `corrected_day` DEBE marcarse con `_critique_applied=True`.
    Sin este marker, el check downstream no sabe qué días tolerar."""
    # Slice del bloque de éxito de _correct_single_day
    idx = _GRAPH.find("Día {day_num} corregido exitosamente")
    assert idx > 0, "Log de éxito de self_critique no encontrado."
    # Ventana de ~2000 chars antes (el comentario explicativo + marker
    # están al inicio del bloque, antes del set y el log).
    block = _GRAPH[max(0, idx - 2000):idx + 200]
    assert 'corrected_day["_critique_applied"] = True' in block, (
        "El marker `_critique_applied=True` NO se inyecta en `corrected_day` "
        "tras una corrección exitosa. Sin esto, P3-SKELETON-FIDELITY-CRITIQUE-AWARE "
        "queda sin efecto — el check downstream no puede distinguir días "
        "modificados de no-modificados."
    )
    assert "P3-SKELETON-FIDELITY-CRITIQUE-AWARE" in block, (
        "Marker textual P3-SKELETON-FIDELITY-CRITIQUE-AWARE ausente cerca "
        "del set — un refactor cosmético podría borrar el por qué."
    )


# ---------------------------------------------------------------------------
# Fix #2: SKELETON FIDELITY check usa threshold dinámico
# ---------------------------------------------------------------------------


def test_skeleton_fidelity_threshold_is_dynamic():
    """El check de SKELETON FIDELITY debe usar threshold >=3 cuando
    `day.get('_critique_applied')` es True, >=2 cuando False/missing."""
    # Localizar el bloque del check
    idx = _GRAPH.find("def _run_assembly_validations")
    assert idx > 0
    # Tomar las ~3000 chars que siguen (cubre el bloque de skeleton fidelity)
    block = _GRAPH[idx:idx + 4000]

    # Anchor del marker:
    assert "P3-SKELETON-FIDELITY-CRITIQUE-AWARE" in block, (
        "Marker P3-SKELETON-FIDELITY-CRITIQUE-AWARE ausente del check."
    )

    # Debe leer `day.get("_critique_applied")` para decidir threshold:
    assert 'day.get("_critique_applied")' in block, (
        "El check no lee `day.get('_critique_applied')` — siempre usará "
        "el threshold antiguo (>=2) para todos los días."
    )

    # Debe haber un threshold ternario 3/2:
    assert "_missing_threshold = 3 if" in block, (
        "Threshold dinámico no implementado. Esperado: "
        "`_missing_threshold = 3 if _critique_applied_for_day else 2`."
    )

    # El check usa el threshold variable, NO hardcoded >=2:
    assert "len(missing_proteins) >= _missing_threshold" in block, (
        "El check usa `>= _missing_threshold` (variable). Si vuelve al "
        "hardcoded `>= 2`, el fix queda sin efecto para días con critique."
    )


def test_old_hardcoded_threshold_2_removed():
    """Defensa anti-regresión: la línea vieja `if len(missing_proteins) >= 2:`
    NO debe estar activa en el check (comentarios históricos OK)."""
    # Buscar la línea de assignment activa específicamente
    idx = _GRAPH.find("def _run_assembly_validations")
    block = _GRAPH[idx:idx + 4000]
    # Pattern del código viejo: línea exacta con `>= 2:`
    bad = re.compile(r"^\s*if len\(missing_proteins\) >= 2:", re.MULTILINE)
    assert not bad.search(block), (
        "La línea vieja `if len(missing_proteins) >= 2:` reapareció — "
        "revierte el fix dinámico P3-SKELETON-FIDELITY-CRITIQUE-AWARE."
    )


# ---------------------------------------------------------------------------
# Funcional smoke: lógica del threshold (sin importar el módulo entero)
# ---------------------------------------------------------------------------


def _threshold_for(critique_applied: bool) -> int:
    """Mirror de la lógica del fix: 3 si critique_applied, 2 si no."""
    return 3 if critique_applied else 2


def test_threshold_3_when_critique_applied():
    """Día con `_critique_applied=True` tolera 2/3 proteínas missing
    (caso del user 2026-05-16: skeleton dio 3 proteínas, critique removió 2,
    pero 1 quedó). Solo flag si 3/3 missing (regla más estricta)."""
    assert _threshold_for(True) == 3, "Threshold con critique debe ser 3."
    # Caso del usuario: 2 proteínas missing, critique aplicado → NO flag
    missing_count = 2
    assert missing_count < _threshold_for(True), (
        "Caso del usuario (2 missing + critique aplicado) debería tolerarse."
    )
    # Caso extremo: 3 proteínas missing → flag (todas removidas = bug)
    missing_count = 3
    assert missing_count >= _threshold_for(True), (
        "3 proteínas missing con critique sigue siendo critical "
        "(no quedó NINGUNA proteína asignada)."
    )


def test_threshold_2_when_no_critique():
    """Día sin self_critique mantiene threshold estricto >=2 (preserva
    detección de bug del day_generator que ignora skeleton)."""
    assert _threshold_for(False) == 2, "Threshold sin critique debe ser 2."
    # Caso original P0-6: 2 missing sin critique → flag
    missing_count = 2
    assert missing_count >= _threshold_for(False), (
        "Día sin critique con 2 missing debe seguir siendo critical "
        "(preserva detección del bug que motivó el check inicial)."
    )
