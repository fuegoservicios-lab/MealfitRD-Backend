"""[P3-RECOVERY-BYPASS-FORM-CHECK · 2026-05-16] Cierre del segundo gap
donde el usuario volvía tras cerrar tab durante generación, PendingPipelineRecovery
navegaba a /plan correctamente, pero Plan.jsx redirigía DE VUELTA a /assessment
porque formData estaba incompleto (race con hidratación de localStorage).

Síntoma reportado por usuario:
> "no funciono cuando entre de nuevo me redirigió al formulario"

Log backend confirmó que el cancel automático YA NO se disparaba
(P3-BEFOREUNLOAD-NO-CANCEL del fix anterior funcionando), KV permanecía
`status=generating`, polling de /pending-status visible repetidamente
desde el frontend. Pero user landed en /assessment.

Causa raíz: Plan.jsx tiene 3 lugares que checkean `findFirstIncompleteField(formData)`:
  1. useEffect `processPlan` (línea ~493) — early return si formData incompleto.
  2. useEffect `navigate-to-assessment` (línea ~994) — naviga a /assessment.
  3. Render condicional (línea ~1046) — renderiza LoadingScreen mientras
     el navigate se ejecuta.

El problema: cuando el user vuelve, formData hidrata async desde localStorage.
Durante el primer render, formData puede estar default (vacío) → los 3
checks fallan → navigate('/assessment') dispara → user fuera de la pantalla
de carga aunque el plan se está generando en backend.

Fix: bypass los 2 useEffects cuando `localStorage.mealfit_plan_in_progress`
está presente. Razón: la validación de form existe para evitar SSE con
datos vacíos. En recovery mode NO disparamos SSE (el backend ya está
generando), así que la validación no aplica.

El render condicional NO se cambió porque ya rinde LoadingScreen (no es
una redirección).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLAN = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Plan.jsx"
).read_text(encoding="utf-8")


def test_marker_present():
    assert "P3-RECOVERY-BYPASS-FORM-CHECK" in _PLAN, (
        "Marker P3-RECOVERY-BYPASS-FORM-CHECK ausente en Plan.jsx — un "
        "refactor cosmético podría borrar el bypass sin signal."
    )


def test_processPlan_bypasses_form_check_when_flag_set():
    """El useEffect `processPlan` (que dispara SSE) debe BYPASS el check
    de form incompleto cuando hay `mealfit_plan_in_progress` flag. Sin
    esto, processPlan early-returns y nunca llega al pre-flight de
    /pending-status — el user queda en limbo."""
    # Slice del useEffect processPlan: desde la primera línea hasta processPlan().
    idx = _PLAN.find("const processPlan = async () =>")
    assert idx > 0
    # Buscamos hacia atrás hasta encontrar el useEffect
    pre_block = _PLAN[max(0, idx - 5000):idx]
    # Debe contener: `_hasInProgressFlag` definido leyendo localStorage
    assert "_hasInProgressFlag" in pre_block, (
        "Variable `_hasInProgressFlag` no definida antes de processPlan. "
        "Sin ella el bypass no aplica."
    )
    assert "mealfit_plan_in_progress" in pre_block, (
        "El bypass no lee `localStorage.mealfit_plan_in_progress` — no "
        "puede detectar recovery mode."
    )
    # El check debe usar el flag para bypass:
    # Pattern: `if (!_hasInProgressFlag && findFirstIncompleteField(formData)) return;`
    assert "!_hasInProgressFlag && findFirstIncompleteField" in pre_block, (
        "El check de formData NO está combinado con `!_hasInProgressFlag`. "
        "Sin esto, processPlan early-returns aunque el flag esté seteado, "
        "y nunca dispara el pre-flight de /pending-status."
    )


def test_navigate_useeffect_bypasses_when_flag_set():
    """El useEffect que navega a /assessment cuando form incompleto DEBE
    también bypass cuando `mealfit_plan_in_progress` está set. Sin esto,
    aunque processPlan logre mostrar loading, este useEffect lo desplaza.

    Anchor único de ese useEffect específico: `const missing = findFirstIncompleteField(formData);`
    seguido de `if (!missing) return;` — eso es solo del useEffect del navigate.
    """
    # Anchor único del useEffect en cuestión:
    anchor = "const missing = findFirstIncompleteField(formData)"
    idx = _PLAN.find(anchor)
    assert idx > 0, f"Anchor `{anchor}` no encontrado."
    # Tomar el bloque del useEffect (hasta el cierre `}, [...])`)
    end = _PLAN.find("}, [loadingSensitive, formData, navigate])", idx)
    assert end > 0, "Cierre del useEffect no encontrado."
    block = _PLAN[idx:end + 50]

    assert "mealfit_plan_in_progress" in block, (
        "El useEffect del navigate-to-assessment no lee "
        "`localStorage.mealfit_plan_in_progress`. Sin esto, recovery mode "
        "es ignorado y el navigate dispara aunque el plan se esté generando."
    )
    assert "P3-RECOVERY-BYPASS-FORM-CHECK" in block, (
        "Marker P3-RECOVERY-BYPASS-FORM-CHECK ausente en el useEffect — "
        "un refactor podría remover el bypass sin signal."
    )


def test_return_pattern_before_navigate():
    """El bypass debe ser un EARLY RETURN, no solo un toast skip. Sin
    return, el navigate se dispara igualmente."""
    anchor = "const missing = findFirstIncompleteField(formData)"
    idx = _PLAN.find(anchor)
    assert idx > 0
    end = _PLAN.find("}, [loadingSensitive, formData, navigate])", idx)
    block = _PLAN[idx:end + 50]

    # Patrón: dentro del bloque hay un `if (...mealfit_plan_in_progress...)) return;`
    assert re.search(
        r"if\s*\(\s*localStorage\.getItem\(['\"]mealfit_plan_in_progress['\"]\)\s*\)\s*return",
        block,
    ), (
        "Falta `if (localStorage.getItem('mealfit_plan_in_progress')) return;` "
        "antes del navigate. Sin el `return`, el bypass es no-op."
    )
