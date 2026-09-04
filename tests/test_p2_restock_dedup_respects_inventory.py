"""[P2-RESTOCK-DEDUP-RESPECTS-INVENTORY · 2026-09-04] «Ya compré lo que faltaba» no puede ser un no-op silencioso.

El dueño borró de la Nevera calamar, espárragos, yogur y pimienta para una prueba, la lista los
volvió a pedir («Ya compré lo que faltaba (5)»), pulsó el botón y «no pasó nada»: el dedup por
ciclo de `/restock` (P3-RESTOCK-STALE-DEDUP, 7 días) saltó los 5 porque `restocked_items` los
tenía de esa mañana, devolvió 200 con «todos ya estaban registrados» y el cliente celebró
«¡Ingredientes ingresados!» con la Nevera intacta.

El dedup existe para no SUMAR dos veces una compra que SIGUE en la Nevera. Ahora salta solo lo
que sigue presente (`user_inventory.quantity > 0`); lo borrado o consumido a cero se re-compra.
La respuesta lleva `added` y `skipped` y el cliente dice la verdad cuando no sumó nada.
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _restock_body() -> str:
    start = _SRC.index('@router.post("/restock")')
    end = _SRC.find("\n@router.", start + 10)
    return _SRC[start:end if end != -1 else len(_SRC)]


def test_dedup_skips_only_items_still_present_in_the_pantry():
    body = _restock_body()
    assert "SELECT ingredient_name FROM user_inventory WHERE user_id = %s AND quantity > 0" in body
    assert "if _present_keys is None or key in _present_keys:" in body
    assert "rebought.append(name)" in body
    # sin inventario legible (fallo de lectura) el dedup vuelve al comportamiento previo, no a sumar todo
    assert "_present_keys = None" in body


def test_both_returns_tell_the_client_what_was_added():
    body = _restock_body()
    assert '"added": 0, "skipped": skipped_dupes' in body
    assert '"added": len(persisted_names)' in body
    assert '"skipped": skipped_dupes,' in body


def test_frontend_is_honest_when_nothing_was_added():
    dash = (_BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx")
    if not dash.exists():
        return
    src = dash.read_text(encoding="utf-8")
    assert "if (data.added === 0) {" in src
    assert "Ya tenías registrados estos ingredientes esta semana; no se sumaron otra vez." in src
