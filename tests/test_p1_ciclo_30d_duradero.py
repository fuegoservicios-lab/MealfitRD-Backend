# -*- coding: utf-8 -*-
"""[P1-CICLO-30D-DURADERO · 2026-09-09] El producto vende UNA compra al mes; la biblioteca
estaba escrita para comprar cada semana.

## Lo que se midió

El dueño preguntó si el código podía elegir los platos y cuadrar las macros. Se midió
construyendo menús de 30 días contra sus objetivos reales (2.100 kcal · 123 g proteína ·
271 g carbos · 58 g grasa), banda [0,90-1,12] en los CUATRO macros, ración escalable
[0,6× - 1,6×] y sin repetir plato dentro de 7 días:

| capa acumulativa | antes (144 plantillas) | después (178) |
|---|---|---|
| los 4 macros en banda | 8/8 | 8/8 |
| + dieta vegetariana | **4/8** | **8/8** |
| + alergia a lácteos | 8/8 | 8/8 |
| + durabilidad sin congelador desde el día 8 | **0/8**, atasco en el día 13-17 | **8/8** |

**El sustrato siempre dio para las macros.** Lo que faltaba era comida que aguantara: 91
de las 144 plantillas caducaban en la primera semana (48 duran 3 días, 43 duran 7), y del
día 15 en adelante quedaban 11 desayunos / 13 almuerzos / 10 meriendas / **9 cenas**. Con
la regla de no repetir en 7 días hacen falta 7 por franja y sobraban dos: cero margen.

## Por qué la durabilidad de un plato es la de su peor ingrediente

Una ramita de cilantro fresco arrastra el guiso entero a 3 días. No es un defecto del
clasificador —se comprobó: 85 de 143 alimentos proteicos aguantan ≥21 días y no hay un
solo desacuerdo con la vida útil del catálogo— sino la aritmética correcta: el día 20 de
una compra única no queda cilantro que picar.

## Los 34 platos añadidos

Cenas y almuerzos sobre proteína de despensa (soya texturizada 365 d, sardinas / arenque /
atún / bacalao 180 d, seis legumbres secas 180 d, quesos curados 45 d) con víveres y
vegetales de guarda (papa 45 d, auyama 45 d, batata 30 d, ñame 30 d, repollo 45 d,
zanahoria 45 d, cebolla y ajo 60 d). Desayunos y meriendas sobre avena, casabe, maní,
granola, leches de larga vida, pasas y dátiles.

**Ninguno inventado**: cada constituyente lleva el nombre EXACTO del catálogo, que es la
regla que el propio `build_dish_constituents_do.py` se puso. La nutrición, la logística y
el riesgo los deriva el compilador; aquí no se escribió ni una caloría a mano.
"""
import json
from pathlib import Path

import pytest

_REG = (Path(__file__).resolve().parents[1] / "data" / "registry" /
        "dish_registry_do_v1.json")
_SLOTS = ("desayuno", "almuerzo", "merienda", "cena")
# Sin repetir plato en 7 días hacen falta 7 por franja en todo momento. El suelo se pone
# en 12 —casi el doble— porque además cada elección tiene que caber en la banda de macros:
# con exactamente 7 la combinatoria se agota y el mes no se completa (medido: con 9 cenas
# se atascaba en el día 13-17).
_SUELO_DURADERO = 12
_SUELO_VEGETARIANO = {"desayuno": 20, "almuerzo": 18, "merienda": 20, "cena": 20}
_CARNE = {"pollo", "pescado", "res", "cerdo", "pavo", "atun", "camarones", "mixta"}


def _plantillas():
    d = json.loads(_REG.read_text(encoding="utf-8"))
    return [t for t in d["templates"] if (t.get("nutrition_per_serving") or {}).get("kcal")]


_T = _plantillas()


def _aguanta(t, dia_0based):
    lg = t.get("logistics") or {}
    return bool(lg.get("pantry_only")) or float(lg.get("days_fresh_min") or 0) >= dia_0based + 1


@pytest.mark.parametrize("slot", _SLOTS)
def test_queda_comida_para_el_final_del_ciclo_de_30_dias(slot):
    """Del día 15 al 30 de una compra ÚNICA sin congelador tiene que quedar surtido.

    Es la capa que estaba en 0 de 8: no porque el motor no supiera elegir, sino porque a
    esas alturas del mes ya no quedaba nada que aguantara.
    """
    vivos = [t for t in _T if slot in (t.get("slots") or []) and _aguanta(t, 29)]
    assert len(vivos) >= _SUELO_DURADERO, (
        f"solo {len(vivos)} plantillas de {slot} aguantan hasta el día 30 sin nevera "
        f"(mínimo {_SUELO_DURADERO}): el ciclo mensual de una sola compra se queda sin "
        f"menú a mitad de mes")


@pytest.mark.parametrize("slot", _SLOTS)
def test_un_vegetariano_tiene_de_donde_elegir(slot):
    """El almuerzo vegetariano era el hueco: 12 plantillas, de las que varias llevan huevo."""
    veg = [t for t in _T if slot in (t.get("slots") or []) and t.get("protein") not in _CARNE]
    assert len(veg) >= _SUELO_VEGETARIANO[slot], (
        f"solo {len(veg)} plantillas de {slot} son aptas para vegetariano "
        f"(mínimo {_SUELO_VEGETARIANO[slot]})")


def test_los_platos_duraderos_no_esconden_un_ingrediente_perecedero():
    """La durabilidad declarada de un plato tiene que ser la de su PEOR constituyente.

    Si el compilador dejara de tomar el mínimo, un guiso con cilantro fresco se anunciaría
    como de despensa y el usuario lo cocinaría el día 20 con un ingrediente que se le
    echó a perder el día 3.
    """
    from pantry_durability import classify

    fallos = []
    for t in _T:
        lg = t.get("logistics") or {}
        declarado = lg.get("days_fresh_min")
        cons = t.get("constituents") or []
        if declarado is None or not cons:
            continue
        real = min(classify(c.get("canonical") or c.get("name"))["days_fresh"] for c in cons)
        if float(declarado) > real:
            fallos.append((t["name"], declarado, real))
    assert not fallos, (
        f"plantillas que se declaran más duraderas que su peor ingrediente: {fallos[:5]}")


def test_el_registro_declara_sus_exclusiones_en_vez_de_inventar():
    """La regla de la biblioteca: lo que el catálogo no tiene se deja FUERA y se dice.

    Un plato que resolviera sus ingredientes a ojo metería un alimento fantasma en la lista
    de compras — que es exactamente la clase de defecto que costó RD$476 de pavo en el plan
    del 09-sep.
    """
    d = json.loads(_REG.read_text(encoding="utf-8"))
    stats = d.get("stats") or {}
    assert stats.get("resolution_pct", 0) >= 99.0, (
        f"la resolubilidad del registro cayó a {stats.get('resolution_pct')}%")
    assert stats.get("templates_with_unknown_nutrient", 0) == 0, (
        "hay plantillas con nutrientes desconocidos: sus macros serían una conjetura")
