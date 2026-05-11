"""[P3-AUDIT-8 . 2026-05-10] Backend anchor del wiring del helper
`_recalcShoppingListAfterPantryChange` en `frontend/src/pages/Pantry.jsx`.

Bug original (audit 2026-05-10):
  Solo `confirmDeleteAll` ejecutaba el flow de recalculate-shopping-list.
  Tras add/delete individual el Dashboard mostraba la lista cacheada.

Fix (P3-AUDIT-8, frontend):
  Helper SSOT en Pantry.jsx invocado en 4 call sites:
    1. handleDeleteItem (path principal post-delete)
    2. handleDeleteItem undo callback (post-restore)
    3. handleAddNewItem (post-insert)
    4. confirmDeleteAll (que ANTES inlineaba el flow)

Por qué un anchor backend:
  - El test frontend principal vive en
    `frontend/src/__tests__/Pantry.p3_audit_8_recalc_after_change.test.js`
    (8 tests Vitest con drift detection).
  - El test cross-link `test_p2_hist_audit_14_marker_test_link.py` exige que
    el slug del `_LAST_KNOWN_PFIX` matchee al menos un archivo en
    `backend/tests/`. Sin este anchor, bumpear el marker a P3-AUDIT-8
    rompería ese contrato.
  - El anchor también sirve como gate cross-language: si alguien borra el
    helper del Pantry.jsx, el test backend falla en CI antes del frontend.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANTRY = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


def _read_pantry() -> str:
    assert _PANTRY.exists(), f"Pantry.jsx no encontrado en {_PANTRY}"
    return _PANTRY.read_text(encoding="utf-8")


def test_pantry_jsx_exists():
    """Sanity: el archivo Pantry.jsx existe en su path canónico."""
    assert _PANTRY.exists(), (
        f"`{_PANTRY.relative_to(_REPO_ROOT)}` no encontrado. ¿Renombre del path "
        f"`frontend/src/pages/`? Actualiza este test."
    )


def test_helper_defined():
    """El helper SSOT `_recalcShoppingListAfterPantryChange` debe estar
    definido en Pantry.jsx."""
    src = _read_pantry()
    assert re.search(
        r"const\s+_recalcShoppingListAfterPantryChange\s*=\s*async",
        src,
    ), (
        "El helper `_recalcShoppingListAfterPantryChange` fue removido o "
        "renombrado en Pantry.jsx. Sin el helper, los 4 call sites quedan "
        "huérfanos y el Dashboard mostraría lista cacheada tras "
        "add/delete individual."
    )


def test_helper_invoked_4_times():
    """El helper debe invocarse exactamente 4 veces (delete, delete-undo,
    deleteAll, add). Regex distingue invocaciones (`name(`) de la
    declaración (`name = async`)."""
    src = _read_pantry()
    # Matchea solo invocaciones, no la declaración (que tiene `= async` entre
    # el nombre y el `(`).
    invocation_re = re.compile(r"_recalcShoppingListAfterPantryChange\s*\(")
    matches = invocation_re.findall(src)
    assert len(matches) == 4, (
        f"Esperaba 4 invocaciones de `_recalcShoppingListAfterPantryChange` "
        f"(delete, delete-undo, deleteAll, add); encontré {len(matches)}. "
        f"Si añadiste/quitaste un call site, actualiza este test Y el "
        f"frontend test `Pantry.p3_audit_8_recalc_after_change.test.js`."
    )


def test_endpoint_path_in_helper():
    """El helper debe invocar el endpoint `/api/plans/recalculate-shopping-list`.
    Si alguien renombra el endpoint sin actualizar el helper, el recálculo
    falla silencioso y el bug P3-AUDIT-8 vuelve."""
    src = _read_pantry()
    assert "recalculate-shopping-list" in src, (
        "El path `recalculate-shopping-list` no aparece en Pantry.jsx. "
        "¿Renombre del endpoint o eliminación del helper? Verifica "
        "`backend/routers/plans.py` y actualiza el helper en consecuencia."
    )


def test_handleUpdateQuantity_does_not_invoke_helper():
    """Anti-regresión: `handleUpdateQuantity` NO debe invocar el helper.
    Los qty changes (especialmente el burst del velocímetro turbo) ya se
    reflejan vía `increment_inventory_quantity` + el PDF live-fetch; llamar
    al recalc por cada delta dispararía N HTTP innecesarios."""
    src = _read_pantry()
    # Aísla el cuerpo de handleUpdateQuantity.
    start = src.find("handleUpdateQuantity = async")
    assert start >= 0, "No se encontró `handleUpdateQuantity = async` en Pantry.jsx"
    # El cuerpo termina antes de `// Activación del "Velocímetro"`
    # (delimitador conocido — si renombran el comentario, ajusta).
    end_marker = src.find("// Activación del", start)
    assert end_marker >= 0, (
        "Delimitador `// Activación del` no encontrado tras handleUpdateQuantity. "
        "Si el comentario fue renombrado, actualiza este test."
    )
    body = src[start:end_marker]
    # NO debe haber invocación del helper dentro de este cuerpo.
    assert "_recalcShoppingListAfterPantryChange(" not in body, (
        "Anti-regresión P3-AUDIT-8: `handleUpdateQuantity` invoca el helper. "
        "Esto dispara N HTTP por la ráfaga del velocímetro turbo. Mover la "
        "invocación a `handleAddNewItem` / `handleDeleteItem` (cambios de SET, "
        "no de cantidad)."
    )
