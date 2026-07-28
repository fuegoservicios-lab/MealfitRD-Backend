"""[P3-DEPLETED-CONTRACT-ANCHORS · 2026-07-28] Consolidación tras la muerte de la card "breathe".

## Qué pasó

`test_p3_depleted_card_breathe.py` (9 rojos / 1 verde) anclaba el LAYOUT de las cards de
"Agotados" del diseño esquiomórfico de la Nevera: flex-column, padding "que respira", subclases
top-row/info, ancho del botón de restaurar, minmax del grid. Ese diseño fue reemplazado a
propósito por `cafb430 P3-PANTRY-FRIDGE-REDESIGN` (2026-06-24) — mismo destino que los 4 archivos
de la nevera cerrados en el mismo barrido (ver test_p3_pantry_redesign_anchors.py).

## Lo que SIGUE vivo — la función, no la estética

El estado "Agotados" es dominio real y sobrevivió entero: marcar un item como agotado lo saca de
la nevera SIN tocar `user_inventory` en caliente, se persiste en `user_depleted_items` (BD,
cross-device — la fuente de verdad) con espejo en localStorage, y se puede restaurar. Eso es lo
que este archivo ancla.

tooltip-anchor: P3-DEPLETED-CONTRACT-ANCHORS
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
_PLANS_PY = Path(__file__).resolve().parent.parent / "routers" / "plans.py"


def _pantry() -> str:
    return _PANTRY_JSX.read_text(encoding="utf-8")


def test_estado_agotados_existe():
    src = _pantry()
    assert re.search(r"const\s+\[depletedItems,\s*setDepletedItems\]\s*=\s*useState", src), (
        "el estado de Agotados desapareció del Pantry — la función entera, no solo la card"
    )


def test_clave_canonica_de_dedup():
    """`_depletedKey` es el dedup por identidad de item; sin él, marcar dos veces duplica."""
    src = _pantry()
    assert "_depletedKey" in src
    assert src.count("_depletedKey(") >= 3, "los caminos de alta/baja deben pasar por la clave"


def test_fuente_de_verdad_es_la_bd_cross_device():
    """La tabla `user_depleted_items` es la SSOT (cross-device); localStorage es espejo. Si el
    frontend deja de hablar con la tabla, los agotados vuelven a ser por-dispositivo."""
    assert "user_depleted_items" in _pantry()
    assert "user_depleted_items" in _PLANS_PY.read_text(encoding="utf-8"), (
        "el backend perdió el endpoint/tabla de agotados"
    )


def test_espejo_local_con_lazy_init():
    src = _pantry()
    assert "mealfit_depleted_items" in src, "el espejo localStorage desapareció"


def test_restaurar_devuelve_el_item():
    """Restaurar = sacar de depletedItems (el filter por clave). Sin este camino, 'agotado'
    sería un borrado disfrazado."""
    src = _pantry()
    assert re.search(r"depletedItems\.filter\(\s*e\s*=>\s*_depletedKey\(e\)\s*!==\s*k\)", src), (
        "el camino de restaurar (filter por _depletedKey) desapareció"
    )
