"""[P3-PANTRY-ADD-UX · 2026-05-18] Ancla la UX interactiva del flujo
"añadir alimento a la nevera" en `Pantry.jsx`.

Bug original (UX feedback usuario 2026-05-18):
    El modal "Registrar Nuevo Alimento" hacía click→add directo con
    `quantity=1` y `unit=default_unit` del catálogo maestro. El usuario
    no podía registrar "1 botella de vinagre" ni "2 libras de pollo"
    sin tener que editar después desde el counter inline del item ya
    añadido (UX cíclica + estado intermedio confuso).

Fix:
    `handleAddNewItem(masterItem, customQty=1, customUnit=null)` acepta
    qty + unit explícitos. El modal expone un mini-form inline (counter
    de cantidad + pills de unidades comunes) al hacer click en el item;
    el usuario confirma con un botón grande "Añadir N <unit> a la nevera".

    El path "+1 con default_unit" sigue funcionando para callers que no
    pasen overrides (backward-compatible).

Por qué parser-based:
    El picker es UI puramente frontend; correr Playwright para un toggle
    de UX no aporta más señal que verificar:
      1. signature del handler acepta los 2 overrides
      2. UI tiene los componentes clave (counter + pills + boton final)
      3. marker presente para que el slug matchee este file
        (cross-link enforced por test_p2_hist_audit_14)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


def _read_pantry() -> str:
    assert _PANTRY_JSX.exists(), f"Esperado: {_PANTRY_JSX}"
    return _PANTRY_JSX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Marker presente — slug `p3_pantry_add_ux` debe matchear este file
#    (test_p2_hist_audit_14_marker_test_link enforza el cross-link).
# ---------------------------------------------------------------------------
def test_pantry_marker_present():
    src = _read_pantry()
    assert "P3-PANTRY-ADD-UX" in src, (
        "Marker `P3-PANTRY-ADD-UX` no encontrado en Pantry.jsx. "
        "El marker ancla la UX interactiva del flujo add-item."
    )


# ---------------------------------------------------------------------------
# 2. handleAddNewItem acepta `customQty` y `customUnit` explícitos.
#    Sin estos params el usuario no puede registrar "1 botella de vinagre".
# ---------------------------------------------------------------------------
def test_handle_add_new_item_accepts_qty_and_unit():
    src = _read_pantry()
    # Buscar la signature: `handleAddNewItem = async (masterItem, customQty = 1, customUnit = null)`
    pattern = re.compile(
        r"handleAddNewItem\s*=\s*async\s*\(\s*masterItem\s*,\s*customQty\s*=\s*1\s*,\s*customUnit\s*=\s*null\s*\)",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "Signature de `handleAddNewItem` debe aceptar `(masterItem, customQty=1, customUnit=null)`. "
        "Sin overrides el flujo retrocede al click→+1 directo y el usuario "
        "no puede customizar cantidad/unidad antes de añadir."
    )


# ---------------------------------------------------------------------------
# 3. handleAddNewItem sanitiza qty (clamp [1, 999]) para evitar inputs
#    adversarios (-1, 999999, "abc", null, NaN).
# ---------------------------------------------------------------------------
def test_qty_sanitization_clamp():
    src = _read_pantry()
    # Heurística: presencia de `Math.max(1, Math.min(999, ...))` sobre customQty.
    # Permite refactorizar el formato exacto del clamp con tal que ambos boundaries
    # estén en una sola expresión cerca del nombre del param.
    assert re.search(
        r"Math\.max\s*\(\s*1\s*,\s*Math\.min\s*\(\s*999\s*,",
        src,
    ), (
        "qty del picker debe estar clampada al rango [1, 999] dentro de "
        "`handleAddNewItem` para evitar inputs adversarios (-1, NaN, 1e9)."
    )


# ---------------------------------------------------------------------------
# 4. States del picker inline declarados (pickerForId + pickerQty + pickerUnit).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state_name", [
    "pickerForId",
    "pickerQty",
    "pickerUnit",
])
def test_picker_states_declared(state_name):
    src = _read_pantry()
    pattern = re.compile(
        rf"const\s*\[\s*{state_name}\s*,\s*set[A-Z]\w*\s*\]\s*=\s*useState\("
    )
    assert pattern.search(src), (
        f"State `{state_name}` no declarado con `useState`. "
        "Los 3 states son load-bearing para el picker inline: "
        "pickerForId identifica el item en config, qty y unit son los "
        "valores que el usuario ajusta antes de confirmar."
    )


# ---------------------------------------------------------------------------
# 5. Constante COMMON_PURCHASE_UNITS expone unidades de compra típicas
#    (botella, libra, paquete, etc.) que el usuario reconoce sin pensar.
# ---------------------------------------------------------------------------
def test_common_purchase_units_includes_realistic_options():
    src = _read_pantry()
    # Extraer el bloque del array para verificar que incluye los esenciales.
    m = re.search(
        r"COMMON_PURCHASE_UNITS\s*=\s*\[(.*?)\]",
        src,
        re.DOTALL,
    )
    assert m, "COMMON_PURCHASE_UNITS array no encontrado."
    body = m.group(1).lower()
    # El usuario citó explícitamente "botella" y "aceite/vinagre" como
    # ejemplos canónicos — botella DEBE estar. libra y paquete son
    # mínimos para el resto de la canasta dominicana.
    for required in ("botella", "libra", "paquete", "unidad"):
        assert f"'{required}'" in body or f'"{required}"' in body, (
            f"COMMON_PURCHASE_UNITS debe incluir `{required}`. "
            f"Sin estas unidades el picker fuerza al usuario al "
            f"default_unit del catálogo (poco intuitivo para envases reales)."
        )


# ---------------------------------------------------------------------------
# 6. El render del item-resultado tiene los 3 elementos UX clave:
#    counter (Cantidad), unit pills (¿Cómo viene?), botón final.
# ---------------------------------------------------------------------------
def test_picker_ui_has_required_elements():
    src = _read_pantry()
    # Label visible del counter.
    assert "Cantidad" in src, "Label 'Cantidad' del counter ausente."
    # Label de las unit pills (solo aparece cuando item NEW, no existing).
    assert "¿Cómo viene?" in src, "Label '¿Cómo viene?' de las unit pills ausente."
    # CTA del botón final — heurística por verbo + plural; permite
    # variantes copy futuras siempre que mantenga el patrón "Añadir/Sumar … nevera".
    assert re.search(r"(Añadir|Sumar)\s.*\bnevera", src), (
        "Botón final de confirmación debe leer 'Añadir … nevera' o "
        "'Sumar … nevera' para que el usuario entienda el efecto."
    )


# ---------------------------------------------------------------------------
# 7. Copy del modal renovado: cuerpo del título refleja la acción
#    ("Añade a tu Nevera") en vez del label técnico antiguo
#    ("Registrar Nuevo Alimento"). Si alguien revive el copy viejo,
#    falla aquí para forzar reconsiderar la UX.
# ---------------------------------------------------------------------------
def test_modal_title_uses_friendly_copy():
    src = _read_pantry()
    assert "Añade a tu Nevera" in src, (
        "El título del modal debe ser amigable ('Añade a tu Nevera'). "
        "El label técnico anterior ('Registrar Nuevo Alimento') sonaba a "
        "formulario administrativo, no a acción cotidiana."
    )


# ---------------------------------------------------------------------------
# 8. [P3-PANTRY-ADD-UX-INSERT · 2026-05-18] Anti-regresión del upsert con
#    onConflict que apuntaba a una constraint inexistente. Postgres devuelve
#    42P10 ('there is no unique or exclusion constraint matching the ON
#    CONFLICT specification') cuando ON CONFLICT no matchea ningún UNIQUE
#    real — la tabla `user_inventory` solo tiene UNIQUE (user_id,
#    ingredient_name, unit), NO (user_id, master_ingredient_id).
# ---------------------------------------------------------------------------
def test_no_upsert_on_user_inventory():
    src = _read_pantry()
    # Buscar el patrón rota: .upsert(...) sobre user_inventory cerca del
    # handleAddNewItem. Si vuelve a aparecer, falla con el contexto del bug.
    assert ".upsert(" not in src or "onConflict: 'user_id,master_ingredient_id'" not in src, (
        "Detectado upsert con onConflict='user_id,master_ingredient_id' sobre "
        "user_inventory. La tabla NO tiene esa constraint UNIQUE — Postgres "
        "responde 42P10 y el INSERT falla con 400 Bad Request al usuario. "
        "Usa INSERT plano y maneja 23505 (unique_violation) como hace "
        "`handleRestoreDepleted`."
    )


def test_handle_add_handles_23505_unique_violation():
    src = _read_pantry()
    # El bloque de handleAddNewItem debe tener un check de '23505' (la fila
    # ya existe por (user_id, ingredient_name, unit) — race con otra pestaña
    # o legacy row sin master_id). Heurística: encontrar 'handleAddNewItem'
    # y verificar que su body contiene '23505' dentro de los siguientes 4000
    # caracteres (cubre el bloque entero).
    idx = src.find("handleAddNewItem = async")
    assert idx >= 0, "handleAddNewItem signature no encontrada."
    body_window = src[idx : idx + 4000]
    assert "23505" in body_window, (
        "handleAddNewItem debe manejar el código '23505' (unique_violation) "
        "que retorna Postgres cuando un INSERT viola la constraint "
        "(user_id, ingredient_name, unit). Sin este manejo, una race "
        "condition entre pestañas o un legacy row sin master_id explota "
        "como toast.error genérico."
    )


# ---------------------------------------------------------------------------
# 9. Chip de unidad muestra solo la unidad capitalize, NO el prefijo técnico
#    "Unidad base:". Si alguien revive el copy viejo el chip vuelve a sonar
#    a label de DB en vez de pieza de UI.
# ---------------------------------------------------------------------------
def test_unit_chip_uses_clean_copy():
    src = _read_pantry()
    # Anti-regresión del copy viejo.
    assert "Unidad base: {item.unit}" not in src, (
        "El chip de unidad NO debe leer 'Unidad base: X' — es jerga de DB. "
        "Usa solo {item.unit} y deja que la CSS .nevera-item-unit-tag lo "
        "capitalice."
    )
    # CSS debe declarar capitalize para que '{item.unit}' (lowercase) se
    # muestre como 'Cartón'/'Botella' sin tocar los strings.
    assert "text-transform: capitalize" in src, (
        "`.nevera-item-unit-tag` debe usar `text-transform: capitalize` para "
        "que la unidad cruda lowercase quede presentable. Sin esto, el chip "
        "muestra 'cartón' en vez de 'Cartón'."
    )


# ---------------------------------------------------------------------------
# 10. [P3-PANTRY-QTY-EDIT · 2026-05-18] Editor de cantidad exacta:
#     - states declarados
#     - botón clickable invoca setQtyEditItem(item)
#     - modal renderiza con título "Ajustar cantidad"
#     - 5 atajos rápidos (1, 2, 5, 10, 20) presentes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state_name", [
    "qtyEditItem",
    "qtyEditValue",
    "qtyEditSaving",
])
def test_qty_edit_states_declared(state_name):
    src = _read_pantry()
    pattern = re.compile(
        rf"const\s*\[\s*{state_name}\s*,\s*set[A-Z]\w*\s*\]\s*=\s*useState\("
    )
    assert pattern.search(src), (
        f"State `{state_name}` no declarado con useState. Es necesario "
        "para el editor de cantidad exacta (P3-PANTRY-QTY-EDIT)."
    )


def test_qty_edit_modal_present():
    src = _read_pantry()
    assert "Ajustar cantidad" in src, (
        "El modal de ajuste exacto debe tener el header 'Ajustar cantidad'. "
        "Sin este título, el usuario no entiende la naturaleza del modal."
    )
    assert "P3-PANTRY-QTY-EDIT" in src, (
        "Marker P3-PANTRY-QTY-EDIT ausente del bloque del modal. "
        "Si alguien edita la sección, debe ver el contexto del fix."
    )


def test_qty_edit_has_quick_presets():
    src = _read_pantry()
    # Los presets son atajos para "tengo 5/10/20" sin tener que tap-incrementar.
    # Heurística: array literal `[1, 2, 5, 10, 20]` cerca del modal.
    assert re.search(r"\[\s*1\s*,\s*2\s*,\s*5\s*,\s*10\s*,\s*20\s*\]", src), (
        "Atajos rápidos [1, 2, 5, 10, 20] esperados en el editor de "
        "cantidad. Cubren los casos típicos del catálogo dominicano "
        "(1 ítem, 1 par, media docena, 'pack grande')."
    )


def test_qty_edit_warns_when_setting_zero():
    src = _read_pantry()
    # Antes de guardar 0, el modal debe advertir que el item será marcado
    # agotado — sino el usuario que solo querría 'reducir a 0 temporal' es
    # sorprendido por la lógica de agotamiento (que dispara _addDepleted).
    assert "agotado" in src and "qtyEditValue === 0" in src, (
        "El editor debe advertir que setear cantidad a 0 marca el item "
        "como agotado. Sin warning, el usuario no anticipa el side-effect "
        "(la fila desaparece + entra a la lista de agotados)."
    )


# ---------------------------------------------------------------------------
# 11. [P3-PANTRY-MINUS-DISABLED · 2026-05-18] El botón '-' del counter NO
#     debe transformarse en trash icon cuando qty<=1 — eso duplicaba el
#     path de eliminación contra el botón 'Agotar' debajo del counter.
#     Ambos invocaban handleDeleteItem → _addDepleted (semánticamente
#     iguales), generando confusión.
# ---------------------------------------------------------------------------
def test_no_trash_icon_in_counter_at_min_quantity():
    src = _read_pantry()
    # Anti-regresión: el patrón del ternario viejo `quantity <= 1 ? <trash> : <minus>`.
    # Heurística: NO debe haber `handleUpdateQuantity(item.id, 0)` invocado
    # como onClick directo desde el counter (eso era el trash icon).
    forbidden = re.compile(
        r"onClick\s*=\s*\{\s*\(\s*\)\s*=>\s*handleUpdateQuantity\(\s*item\.id\s*,\s*0\s*\)\s*\}"
    )
    assert not forbidden.search(src), (
        "Detectado onClick directo a `handleUpdateQuantity(item.id, 0)` "
        "en el counter — eso era el trash icon viejo que duplicaba el "
        "botón 'Agotar' debajo. La eliminación debe ir exclusivamente "
        "por handleDeleteItem desde 'Agotar'."
    )


def test_minus_button_disabled_at_min_quantity():
    src = _read_pantry()
    # El botón `-` debe declarar `disabled={item.quantity <= 1}` para que
    # el user vea visualmente que llegó al mínimo (cursor not-allowed +
    # color gris) en lugar de ofrecer una acción destructiva escondida.
    assert re.search(r"disabled\s*=\s*\{\s*item\.quantity\s*<=\s*1\s*\}", src), (
        "El botón '-' del counter debe declarar "
        "`disabled={item.quantity <= 1}` cuando llega a qty mínima. "
        "Esto deja la eliminación exclusivamente por 'Agotar'."
    )
