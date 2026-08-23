"""[P3-I18N-TOOLCALL-MODIFIED-ITEMS-SIN-RED · 2026-08-23] `mark_shopping_list_purchased`
ganó en `P2-I18N-TOOLCALL-NOMBRE-SIN-CANONICALIZAR` una red para `excluded_items`: cuenta
lo que CASÓ con la lista y avisa al agente de lo que no. `modified_items` se quedó sin
ella: un ítem modificado que no casa con la lista entra a la Nevera como EXTRA con el
nombre tal cual — «3 lbs of chicken» si el usuario chatea en inglés — una fila que
`pantry_names_match` no resolverá nunca contra las recetas en español, y el mensaje decía
«se modificaron 1 ítems» como si hubiera cambiado la cantidad del pollo.

Se mide la CONDUCTA de la tool real con la lista y el restock como dobles.
"""
from __future__ import annotations

import pytest

_MARKER = "P3-I18N-TOOLCALL-MODIFIED-ITEMS-SIN-RED"


@pytest.fixture
def nevera(monkeypatch):
    import tools
    import db_inventory
    import shopping_calculator

    persistido = {}

    def _restock(user_id, items):
        persistido["items"] = list(items)
        return (True, [str(i) for i in items])

    monkeypatch.setattr(db_inventory, "restock_inventory", _restock)
    monkeypatch.setattr(tools, "get_latest_meal_plan", lambda uid: {"id": "p1", "plan_data": {"days": []}})
    monkeypatch.setattr(
        shopping_calculator, "get_shopping_list_delta",
        lambda uid, plan, structured=True: [
            {"display_string": "1 lb de Pollo"},
            {"display_string": "2 unidades de Aguacate"},
        ],
    )
    return persistido


def _marcar(**kw) -> str:
    import tools
    fn = tools.mark_shopping_list_purchased
    fn = getattr(fn, "func", fn)  # langchain @tool → función cruda
    return fn("11111111-1111-1111-1111-111111111111", **kw)


def test_un_modified_que_no_casa_se_declara_como_extra_y_alerta_al_agente(nevera):
    msg = _marcar(modified_items=["3 lbs of chicken"])
    assert "[ALERTA INTERNA PARA LA IA]: Estos ítems modificados NO casaron" in msg, (
        f"el ítem que no casó entra a la Nevera en silencio: {msg!r} [{_MARKER}]")
    assert "3 lbs of chicken" in msg
    assert "Se modificaron 0 ítems de la lista y se añadieron 1 extras" in msg, msg
    # Sigue entrando (es la conducta del EXTRA), y el pollo de la lista sigue ahí.
    assert "3 lbs of chicken" in nevera["items"] and "1 lb de Pollo" in nevera["items"]


def test_un_modified_que_casa_se_cuenta_y_no_alerta(nevera):
    msg = _marcar(modified_items=["2 lb de Pollo"])
    assert "Estos ítems modificados NO casaron" not in msg, msg
    assert "Se modificaron 1 ítems de la lista)" in msg, msg
    # El pollo de la lista se sustituye por la cantidad nueva.
    assert "2 lb de Pollo" in nevera["items"] and "1 lb de Pollo" not in nevera["items"]


def test_la_red_de_excluded_sigue_intacta(nevera):
    msg = _marcar(excluded_items=["Dragonfruit"])  # «Avocado» ya casa: el normalizador conoce alias ingleses
    assert "NO encontré en la lista de" in msg
    assert "Se excluyeron 0 ítems" in msg
