"""[P3-UPDATE-PLATOS-REQUIRES-PANTRY · 2026-05-17] El botón "Actualizar
platos" del Dashboard se bloquea cuando la Nevera tiene menos de
`PANTRY_MIN_ITEMS_FOR_UPDATE` alimentos.

Síntoma reportado por usuario:
> "quiero que el boton de actualizar platos este bloqueado si la nevera
>  esta vacia, ya que no tiene sentido actualizar platos si la nevera
>  esta vacia ya que solo deberia actualizar platos con los alimentos
>  dentro de la nevera, tiene que haber un minimo de alimentos en la
>  nevera para que el boton se desbloquee"

Decisión: threshold = 3 (recomendado en AskUserQuestion). Con 0-2 items
el regenerador de platos no puede construir variedad significativa.

Fix:
  1. Const módulo `PANTRY_MIN_ITEMS_FOR_UPDATE = 3` en `Dashboard.jsx`.
  2. `pantryItemCount` derivado de `liveInventory` (null si no cargó);
     `isPantryTooEmpty = !isLoadingInventory && pantryItemCount !== null
     && pantryItemCount < THRESHOLD`. Fail-open mientras carga (no
     bloquear si no sabemos).
  3. Botón "Actualizar platos":
     - `onClick`: si `isPantryTooEmpty`, toast.info con CTA "Ir a Nevera"
       (navega a `/pantry`) y `return` (no abre el modal).
     - `style`: fondo gris #E2E8F0 + cursor not-allowed cuando bloqueado.
     - `label`: "Llena tu Nevera" en lugar de "Actualizar platos".
     - `icon`: Lock en lugar de Wand2.
     - `aria-disabled` + `title` para accesibilidad.
  4. Precedencia con otros estados desactivados:
     `isLimitReached` > `isPlanExpired` > `isPantryTooEmpty` > default.
"""
from __future__ import annotations

import pytest

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FRONT_SRC = _BACKEND_ROOT.parent / "frontend" / "src"
@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _DASHBOARD, _PANTRY_GATE
    _DASHBOARD = (_FRONT_SRC / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    _PANTRY_GATE = (_FRONT_SRC / "utils" / "pantryGate.js").read_text(encoding="utf-8")

# [P3-AUDIT-V5-TESTDEBT · 2026-07-30] `P1-SWAP-PANTRY-GATE` extrajo el umbral y el gate a
# `utils/pantryGate.js` (lógica PURA, testeable de verdad). Estos anchors seguían apuntando al
# Dashboard, donde ya no vive ninguna de las dos cosas — rojo por un refactor CORRECTO, no por
# una regresión. Se re-anclan al nuevo hogar; lo que protegen (umbral en rango + fail-open) es
# idéntico y de hecho más fuerte, porque el helper resuelve por `typeof !== 'number'` y así
# cubre también `undefined` y `NaN`, no solo `null`.


def test_marker_present():
    assert "P3-UPDATE-PLATOS-REQUIRES-PANTRY" in _DASHBOARD, (
        "Marker P3-UPDATE-PLATOS-REQUIRES-PANTRY ausente — un refactor podría "
        "borrar el gate y reintroducir el flow sin sentido (modal abre con "
        "nevera vacía)."
    )


def test_threshold_const_declared_at_module_scope():
    """`PANTRY_MIN_ITEMS_FOR_UPDATE` debe estar a nivel módulo (no dentro
    del componente) para poder tunearse sin afectar render."""
    import re as _re
    match = _re.search(
        r"export const PANTRY_MIN_ITEMS_FOR_UPDATE\s*=\s*(\d+);", _PANTRY_GATE
    )
    assert match, (
        "Const `PANTRY_MIN_ITEMS_FOR_UPDATE = <N>;` ausente en utils/pantryGate.js."
    )
    threshold = int(match.group(1))
    assert 1 <= threshold <= 12, (
        f"Threshold {threshold} fuera de rango razonable [1, 12]. Si subió mucho más es "
        "probablemente un bug; si bajó a 0 el gate no aplica."
    )
    assert "PANTRY_MIN_ITEMS_FOR_UPDATE" in _DASHBOARD, (
        "el Dashboard debe seguir consumiendo la const del módulo, no un número hardcodeado")


def test_is_pantry_too_empty_derived_fail_open():
    """`isPantryTooEmpty` debe ser fail-open: false cuando `pantryItemCount`
    es null. Solo true cuando SABEMOS que hay < threshold.

    [P3-PLAN-BTN-STABLE · 2026-05-19] El gate `!isLoadingInventory` fue
    REMOVIDO intencionalmente para eliminar el flash verde→gris en el primer
    paint. El fail-open ahora se logra vía fallback al `cachedPantryCount`:
    si no hay cache, `pantryItemCount` queda `null` → `isPantryTooEmpty=false`.
    Es decir, la misma garantía fail-open, implementada por null-check en vez
    de por el flag de loading."""
    # Anchor a la declaración derivada.
    idx = _DASHBOARD.find("const isPantryTooEmpty")
    assert idx > 0, (
        "Variable derivada `isPantryTooEmpty` ausente. El gate del botón "
        "depende de ella."
    )
    window = _DASHBOARD[idx : idx + 400]
    # Debe usar el threshold (no un número hardcodeado que divergiría del label/title).
    assert "PANTRY_MIN_ITEMS_FOR_UPDATE" in window, (
        "`isPantryTooEmpty` no usa la const `PANTRY_MIN_ITEMS_FOR_UPDATE` — "
        "número hardcodeado divergerá del label/title (que sí lo interpolan)."
    )
    # [P3-AUDIT-V5-TESTDEBT · 2026-07-30] El fail-open ya no es un `pantryItemCount !== null`
    # inline: vive en `computePantryGate`. Se afirma sobre el helper, que es donde está la regla.
    assert "computePantryGate" in window, (
        "el gate debe delegar en el helper puro, no re-implementar la condición inline")
    _fn = _PANTRY_GATE[_PANTRY_GATE.index("export function computePantryGate("):]
    assert "typeof pantryItemCount !== 'number'" in _fn and "return false" in _fn, (
        "`computePantryGate` no es fail-open: bloquearía el botón cuando el inventario no cargó "
        "o el fetch falló (el usuario vería el estado bloqueado sin razón visible)."
    )
    assert "Number.isNaN" in _fn, "un NaN colado tampoco puede leerse como 'nevera vacía'"


def test_button_onclick_guards_pantry_before_modal():
    """El onClick del botón "Actualizar platos" DEBE chequear `isPantryTooEmpty`
    antes de `setShowUpdatePlanModal(true)`, sino el modal abre con nevera
    vacía (defeats el fix).

    [P3-LLENA-NEVERA-DIRECT-CTA · 2026-05-27] El comment-anchor cambió de
    `Gate antes del credit check` (diseño toast.info inicial) al marker del
    rediseño direct-CTA. El guard sigue: cuando la Nevera está vacía/escasa, el
    onClick navega DIRECTO a /pantry y hace `return;` antes de abrir el modal.
    Lo que perdura es el contrato: `if (isPantryTooEmpty) { ...; return; }`
    ANTES de `setShowUpdatePlanModal(true)`."""
    # Anchor al marker del rediseño direct-CTA dentro del onClick.
    anchor = "P3-LLENA-NEVERA-DIRECT-CTA"
    onclick_pos = _DASHBOARD.find(anchor)
    assert onclick_pos > 0, (
        "Anchor `P3-LLENA-NEVERA-DIRECT-CTA` ausente — el comment del guard "
        "fue borrado. Verificar que el `if (isPantryTooEmpty)` sigue dentro "
        "del onClick del botón."
    )
    # Ventana relevante: 2000 chars desde el anchor.
    onclick_window = _DASHBOARD[onclick_pos : onclick_pos + 2000]
    if_pos = onclick_window.find("if (isPantryTooEmpty)")
    assert if_pos > 0, (
        "Guard `if (isPantryTooEmpty)` ausente tras el comment del onClick. "
        "Sin él, el modal abre incluso con Nevera vacía."
    )
    # El guard debe tener `return;` para no caer al setShowUpdatePlanModal.
    guard_block = onclick_window[if_pos : if_pos + 1500]
    assert "return;" in guard_block, (
        "Guard `if (isPantryTooEmpty)` no tiene `return;` — la ejecución "
        "cae al modal igual."
    )
    # Y `setShowUpdatePlanModal(true)` debe venir DESPUÉS del guard.
    modal_pos_in_onclick = onclick_window.find("setShowUpdatePlanModal(true)")
    assert modal_pos_in_onclick > 0 and if_pos < modal_pos_in_onclick, (
        "REGRESIÓN: `setShowUpdatePlanModal(true)` está ANTES del guard o "
        "fuera de la ventana esperada. El gate no aplica."
    )


def test_blocked_button_label_and_lock_icon():
    """Cuando la Nevera está vacía/escasa, el botón muestra 'Ir a mi Nevera'
    y el icono Refrigerator (no Wand2 que sería el estado normal).

    [P3-LLENA-NEVERA-DIRECT-CTA · 2026-05-27] El estado pantry-vacía dejó de
    ser un botón gris disabled con label 'Llena tu Nevera' + icono Lock
    (visualmente bloqueado pero técnicamente clickeable → UX confuso). Ahora
    es un CTA REAL azul que navega directo a /pantry, con label 'Ir a mi
    Nevera' e icono Refrigerator. Sigue siendo un estado distinto del normal
    (Wand2 / 'Actualizar platos'), gateado por `isPantryTooEmpty`."""
    assert "'Ir a mi Nevera'" in _DASHBOARD, (
        "Label 'Ir a mi Nevera' ausente — el botón pantry-vacía mostraría "
        "'Actualizar platos' (label engañoso, parece que actualiza con nevera "
        "vacía)."
    )
    # El icono Refrigerator se renderiza condicionalmente cuando isPantryTooEmpty.
    assert "isPantryTooEmpty" in _DASHBOARD
    # Buscar Refrigerator dentro de un ternario relacionado a isPantryTooEmpty.
    import re as _re
    near_fridge = _re.search(
        r"isPantryTooEmpty[\s\S]{0,200}<Refrigerator size=\{18\}",
        _DASHBOARD,
    )
    assert near_fridge, (
        "Icono <Refrigerator /> no aparece cerca de `isPantryTooEmpty` — el "
        "botón pantry-vacía seguiría con el icono Wand2 (no comunica 've a "
        "llenar la Nevera')."
    )


def test_toast_cta_navigates_to_pantry():
    """El gate de Nevera-vacía DEBE llevar al usuario a la Nevera, sino queda
    atascado sin saber cómo llenarla.

    [P3-LLENA-NEVERA-DIRECT-CTA · 2026-05-27] Antes el gate mostraba un
    `toast.info` con sub-CTA "Ir a Nevera" (`navigate('/pantry')`). El
    rediseño lo simplificó a navegación DIRECTA: clickear el botón
    pantry-vacía hace `navigate('/dashboard/pantry')` en el acto (sin toast
    intermedio). La ruta canónica además se anidó bajo `/dashboard/pantry`.
    El intent perdura (y es más directo): el path de pantry-vacía conduce a
    la Nevera."""
    assert "navigate('/dashboard/pantry')" in _DASHBOARD, (
        "Navigate hacia /dashboard/pantry ausente — sin él el usuario con "
        "Nevera vacía no tiene cómo llegar a añadir alimentos."
    )
    # El navigate directo debe vivir dentro del guard `if (isPantryTooEmpty)`
    # del onClick, anclado al marker del rediseño direct-CTA.
    anchor = "P3-LLENA-NEVERA-DIRECT-CTA"
    onclick_pos = _DASHBOARD.find(anchor)
    assert onclick_pos > 0
    window = _DASHBOARD[onclick_pos : onclick_pos + 1500]
    if_pos = window.find("if (isPantryTooEmpty)")
    assert if_pos > 0, "Guard `if (isPantryTooEmpty)` ausente en el onClick."
    guard_block = window[if_pos : if_pos + 200]
    assert "navigate('/dashboard/pantry')" in guard_block, (
        "El guard de nevera-vacía no navega directo a /dashboard/pantry — "
        "el rediseño direct-CTA (P3-LLENA-NEVERA-DIRECT-CTA) lo exige dentro "
        "del `if (isPantryTooEmpty)`."
    )
