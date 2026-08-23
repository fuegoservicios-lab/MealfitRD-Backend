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

import pytest

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _PLAN
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


# [P1-I18N-DASHBOARD · 2026-08-15] El toast del useEffect («Falta completar: …»)
# pasa ahora por el traductor, así que `t` entró en el array de dependencias y el
# ancla textual `}, [loadingSensitive, formData, navigate])` dejó de existir.
# Lo vigilado aquí es LÓGICA, no copy, y está intacto: el bypass por
# `mealfit_plan_in_progress` sigue siendo un early return antes del navigate.
# Se reancla el CIERRE del effect por CONTENIDO del array de deps (las tres deps
# originales siguen exigidas, en cualquier orden; deps añadidas se toleran) en vez
# de por su grafía exacta — así el slice sigue apuntando a ESE useEffect y no a
# otro, que es lo que el ancla textual garantizaba.
_DEPS_CLOSE_RE = re.compile(r"\}\s*,\s*\[(?P<deps>[^\]]*)\]\s*\)\s*;")


def _navigate_effect_block() -> str:
    """Slice del useEffect `navigate-to-assessment`, desde su anchor único
    (`const missing = findFirstIncompleteField(formData)`) hasta el cierre
    `}, [...]);` cuyo array de deps contiene loadingSensitive/formData/navigate."""
    anchor = "const missing = findFirstIncompleteField(formData)"
    idx = _PLAN.find(anchor)
    assert idx > 0, f"Anchor `{anchor}` no encontrado."
    m = _DEPS_CLOSE_RE.search(_PLAN, idx)
    assert m, "Cierre del useEffect (`}, [...]);`) no encontrado tras el anchor."
    deps = m.group("deps")
    for dep in ("loadingSensitive", "formData", "navigate"):
        assert re.search(rf"\b{dep}\b", deps), (
            f"El cierre hallado no es el del useEffect navigate-to-assessment: "
            f"falta `{dep}` en sus deps ({deps!r}). El slice apuntaría a otro bloque."
        )
    return _PLAN[idx:m.end()]


def test_navigate_useeffect_bypasses_when_flag_set():
    """El useEffect que navega a /assessment cuando form incompleto DEBE
    también bypass cuando `mealfit_plan_in_progress` está set. Sin esto,
    aunque processPlan logre mostrar loading, este useEffect lo desplaza.

    Anchor único de ese useEffect específico: `const missing = findFirstIncompleteField(formData);`
    seguido de `if (!missing) return;` — eso es solo del useEffect del navigate.
    """
    block = _navigate_effect_block()

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
    block = _navigate_effect_block()

    # Patrón: dentro del bloque hay un `if (...mealfit_plan_in_progress...)) return;`
    guard = re.search(
        r"if\s*\(\s*(?:localStorage\.getItem|safeLocalStorageGet)\(['\"]mealfit_plan_in_progress['\"][^)]*\)\s*\)\s*return",  # [re-anclado 2026-08-18: wrapper seguro]
        block,
    )
    assert guard, (
        "Falta `if (localStorage.getItem('mealfit_plan_in_progress')) return;` "
        "antes del navigate. Sin el `return`, el bypass es no-op."
    )
    # [P1-I18N-DASHBOARD · 2026-08-15] El early return sólo sirve si llega ANTES
    # del navigate: colocado después es sintácticamente válido y semánticamente
    # inerte (el usuario ya salió de la pantalla de carga).
    nav = block.find("navigate('/assessment'")
    assert nav > 0, "El `navigate('/assessment', ...)` del useEffect desapareció."
    assert guard.start() < nav, (
        "El bypass está DESPUÉS del navigate — llega tarde, es no-op."
    )
