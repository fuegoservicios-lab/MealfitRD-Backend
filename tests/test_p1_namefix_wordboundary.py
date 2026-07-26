"""[P1-NAMEFIX-WORDBOUNDARY · 2026-07-26] 'agua' vive dentro de 'aguacate'.

`_reflect_added_protein_in_name` refleja en el nombre del plato la proteína que el cerrador
añadió, para que **un plato nunca esconda su proteína principal** (P2-DISH-COHERENCE). Antes de
renombrar comprueba si la proteína ya está representada… por SUBCADENA:

    if any(t in name_low for t in sig_tokens)

En el plan vivo `0afa0ed5`, el desayuno

    "Revoltillo Dominicano con Casabe Crujiente y **Agua**cate Fresco"

recibió **120 g de atún en agua**. Los tokens significativos de "Atún en agua" son
`['atun', 'agua']`, y `'agua'` está dentro de `'aguacate'` → el guard concluyó que la proteína
ya estaba en el nombre y no renombró. Resultado: un revoltillo dominicano con 120 g de atún
escondidos en la lista de ingredientes, que es exactamente lo que este código existe para
impedir.

**Sexta instancia de la misma colisión en una sola sesión**: `'sal'`⊂`'salsa'`,
`'pollo'`⊂`'repollo'`, `'ajo'`⊂`'abajo'`, `'batido'` como adjetivo, `'pina'`⊂`'espinaca'`, y
ahora `'agua'`⊂`'aguacate'`. En español las palabras de comida son cortas y se anidan; el `in`
de subcadena sobre nombres de alimentos es un bug latente por defecto.

tooltip-anchor: P1-NAMEFIX-WORDBOUNDARY
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g
from constants import strip_accents


def _reflejar(nombre_plato, proteina):
    meal = {"name": nombre_plato}
    cambio = g._reflect_added_protein_in_name(meal, proteina, strip_accents)
    return cambio, meal["name"]


# ───────────── 1. el caso vivo ─────────────

def test_el_atun_del_revoltillo_ya_no_se_esconde():
    cambio, nombre = _reflejar(
        "Revoltillo Dominicano con Casabe Crujiente y Aguacate Fresco", "Atún en agua")
    assert cambio is True, "el atún debe reflejarse: 'agua' no es 'aguacate'"
    assert "Atún" in nombre, nombre


def test_aguacate_no_satisface_a_agua():
    """El corazón del bug, aislado."""
    cambio, _ = _reflejar("Tostada con Aguacate", "Atún en agua")
    assert cambio is True


# ───────────── 2. la idempotencia SIGUE viva ─────────────

def test_si_la_proteina_ya_esta_no_duplica():
    """Contrato original: 'Res Molida' + 'Carne de res' no debe volverse
    'Res Molida y Carne de Res'."""
    cambio, nombre = _reflejar("Res Molida a la Plancha", "Carne de res")
    assert cambio is False
    assert nombre == "Res Molida a la Plancha"


@pytest.mark.parametrize("plato,proteina", [
    ("Queso Blanco Guisado", "Queso mozzarella"),
    ("Pollo al Horno", "Pollo"),
    ("Ensalada de Atún", "Atún en agua"),
    ("Camarones al Ajillo", "Camarones"),
])
def test_no_renombra_cuando_ya_esta_representada(plato, proteina):
    cambio, nombre = _reflejar(plato, proteina)
    assert cambio is False, f"{plato} + {proteina} → {nombre}"


def test_el_plural_tambien_cuenta_como_presente():
    """"Camarón" en un plato llamado "Camarones": la frontera acepta el plural."""
    cambio, _ = _reflejar("Camarones al Ajillo", "Camarón")
    assert cambio is False


# ───────────── 3. el renombrado sigue bien formado ─────────────

def test_conector_y_cuando_el_nombre_ya_usa_con():
    _, nombre = _reflejar("Revoltillo con Casabe", "Atún en agua")
    assert " y Atún en Agua" in nombre, nombre


def test_conector_con_cuando_no_lo_usa():
    _, nombre = _reflejar("Revoltillo Dominicano", "Atún en agua")
    assert " con Atún en Agua" in nombre, nombre


def test_preserva_el_nombre_completo_de_la_proteina():
    """Contrato NAMEFIX: 'carne de res' → 'Carne de Res', no el truncado 'Carne De'."""
    _, nombre = _reflejar("Bowl Criollo", "carne de res")
    assert "Carne de Res" in nombre, nombre


# ───────────── 4. bordes y ancla de clase ─────────────

@pytest.mark.parametrize("plato,proteina", [("", "Atún"), ("Plato", ""), ("Plato", None)])
def test_entradas_vacias_no_rompen(plato, proteina):
    meal = {"name": plato}
    assert g._reflect_added_protein_in_name(meal, proteina, strip_accents) is False


def test_no_queda_el_in_de_subcadena_como_camino_principal():
    """Ancla de la CLASE. El `in` sobrevive SOLO como fail-open dentro del `except`; si vuelve
    a ser el camino normal, 'agua' vuelve a casar con 'aguacate'."""
    import inspect
    cuerpo = inspect.getsource(g._reflect_added_protein_in_name)
    assert "_re.search" in cuerpo, "debe comparar con frontera de palabra"
    i_try = cuerpo.index("try:", cuerpo.index("sig_tokens = ["))
    i_exc = cuerpo.index("except Exception:", i_try)
    principal = cuerpo[i_try:i_exc]
    assert "t in name_low" not in principal, \
        "el camino principal no puede volver a la subcadena"
