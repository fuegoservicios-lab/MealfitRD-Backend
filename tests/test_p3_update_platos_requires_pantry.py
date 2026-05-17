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

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


def test_marker_present():
    assert "P3-UPDATE-PLATOS-REQUIRES-PANTRY" in _DASHBOARD, (
        "Marker P3-UPDATE-PLATOS-REQUIRES-PANTRY ausente — un refactor podría "
        "borrar el gate y reintroducir el flow sin sentido (modal abre con "
        "nevera vacía)."
    )


def test_threshold_const_declared_at_module_scope():
    """`PANTRY_MIN_ITEMS_FOR_UPDATE` debe estar a nivel módulo (no dentro
    del componente) para poder tunearse sin afectar render."""
    # Aceptamos cualquier número >= 1, default actual 3.
    import re as _re
    match = _re.search(
        r"const PANTRY_MIN_ITEMS_FOR_UPDATE\s*=\s*(\d+);", _DASHBOARD
    )
    assert match, (
        "Const `PANTRY_MIN_ITEMS_FOR_UPDATE = <N>;` ausente en Dashboard.jsx. "
        "Debe declararse a nivel módulo (fuera del componente)."
    )
    threshold = int(match.group(1))
    assert 1 <= threshold <= 10, (
        f"Threshold {threshold} fuera de rango razonable [1, 10]. Si subió "
        "a >10 es probablemente un bug; si bajó a 0 el gate no aplica."
    )


def test_is_pantry_too_empty_derived_fail_open():
    """`isPantryTooEmpty` debe ser fail-open: false mientras `isLoadingInventory`
    o `pantryItemCount === null`. Solo true cuando SABEMOS que hay < threshold."""
    # Anchor a la declaración derivada.
    idx = _DASHBOARD.find("const isPantryTooEmpty")
    assert idx > 0, (
        "Variable derivada `isPantryTooEmpty` ausente. El gate del botón "
        "depende de ella."
    )
    window = _DASHBOARD[idx : idx + 400]
    # Debe verificar `!isLoadingInventory` (fail-open mientras carga)
    assert "!isLoadingInventory" in window, (
        "`isPantryTooEmpty` no verifica `!isLoadingInventory` — bloquearía "
        "el botón durante el initial load (UX confusa)."
    )
    # Debe verificar que pantryItemCount no es null (fail-open si fetch falló)
    assert "pantryItemCount !== null" in window, (
        "`isPantryTooEmpty` no verifica que pantryItemCount no sea null — "
        "podría bloquear el botón cuando el fetch del inventario falló "
        "(usuario verá Locked sin razón visible)."
    )
    # Debe usar el threshold
    assert "PANTRY_MIN_ITEMS_FOR_UPDATE" in window, (
        "`isPantryTooEmpty` no usa la const `PANTRY_MIN_ITEMS_FOR_UPDATE` — "
        "número hardcodeado divergerá del label/toast (que sí lo interpolan)."
    )


def test_button_onclick_guards_pantry_before_modal():
    """El onClick del botón "Actualizar platos" DEBE chequear `isPantryTooEmpty`
    antes de `setShowUpdatePlanModal(true)`, sino el modal abre con nevera
    vacía (defeats el fix)."""
    # Anchor único al comment del onClick (no se repite en const ni en derived
    # state). Si alguien remueve el comment + el guard junto, falla aquí.
    anchor = "Gate antes del credit check"
    onclick_pos = _DASHBOARD.find(anchor)
    assert onclick_pos > 0, (
        "Anchor `Gate antes del credit check` ausente — el comment del guard "
        "fue borrado. Verificar que el `if (isPantryTooEmpty)` sigue dentro "
        "del onClick del botón."
    )
    # Ventana relevante: 2000 chars desde el anchor (cubre el toast con
    # template strings que es voluminoso).
    onclick_window = _DASHBOARD[onclick_pos : onclick_pos + 2000]
    if_pos = onclick_window.find("if (isPantryTooEmpty)")
    assert if_pos > 0, (
        "Guard `if (isPantryTooEmpty)` ausente tras el comment del onClick. "
        "Sin él, el modal abre incluso con Nevera vacía."
    )
    # El guard debe tener `return;` para no caer al setShowUpdatePlanModal.
    # El return puede estar lejos del `if` por el toast con template strings.
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
    """Cuando bloqueado por nevera vacía, el botón muestra 'Llena tu Nevera'
    y el icono Lock (no Wand2 que sería el estado normal)."""
    assert "'Llena tu Nevera'" in _DASHBOARD, (
        "Label 'Llena tu Nevera' ausente — el botón bloqueado mostraría "
        "'Actualizar platos' (label engañoso, parece clickable)."
    )
    # El icono Lock se renderiza condicionalmente cuando isPantryTooEmpty
    assert "isPantryTooEmpty" in _DASHBOARD
    # Buscar Lock dentro de un ternario relacionado a isPantryTooEmpty
    # (heurística laxa: existe `<Lock size={18}` cerca del `isPantryTooEmpty`)
    import re as _re
    # Captura aprox: ` isPantryTooEmpty ... <Lock`
    near_lock = _re.search(
        r"isPantryTooEmpty[\s\S]{0,200}<Lock size=\{18\}",
        _DASHBOARD,
    )
    assert near_lock, (
        "Icono <Lock /> no aparece cerca de `isPantryTooEmpty` — el botón "
        "bloqueado seguiría con el icono Wand2 (no comunica 'bloqueado')."
    )


def test_toast_cta_navigates_to_pantry():
    """El toast de Nevera-vacía DEBE incluir CTA hacia `/pantry`, sino el
    usuario queda atascado sin saber cómo llenar la nevera."""
    assert "navigate('/pantry')" in _DASHBOARD, (
        "Navigate hacia /pantry ausente — sin CTA en el toast el usuario "
        "no sabe que tiene que ir a la Nevera para añadir alimentos."
    )
    # Defensa adicional: el toast debe usar variant info (no error/success)
    # para no asustar al usuario. Anchor único al comment del onClick.
    anchor = "Gate antes del credit check"
    onclick_pos = _DASHBOARD.find(anchor)
    assert onclick_pos > 0
    window = _DASHBOARD[onclick_pos : onclick_pos + 1500]
    assert "toast.info(" in window, (
        "Toast del gate de nevera-vacía no usa variant `info` — usar error "
        "asustaría al usuario, success sería confuso. Mantener `toast.info`."
    )
