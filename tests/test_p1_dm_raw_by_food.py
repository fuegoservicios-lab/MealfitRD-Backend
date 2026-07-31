"""[P1-DM-RAW-BY-FOOD · 2026-07-31] (audit solver+seeder v6 · P1 / F9) El resolvedor de doble
proteína principal era el ÚLTIMO pase que seguía tocando `ingredients_raw` por índice ciego.

`_meal_double_main_resolve` quita del plato la 2ª especie animal y re-escala la primaria. Sobre
`ingredients_raw` hacía dos cosas por índice:

    _lockstep = isinstance(raw, list) and len(raw) == len(ings)
    ...
    del ings[idx_sec]
    if _lockstep:
        del raw[idx_sec]                      # (1) borrado por índice
    ...
    meal["ingredients_raw"][_ip] = ...        # (2) re-escalado por índice, sin mirar _lockstep

"Mismo largo" nunca fue evidencia de "mismo orden" — este repo ya lo midió: el 93,5% de las comidas
tiene largos iguales y solo el 48,1% de ESAS son paralelas por índice, porque el reconciliador
display↔raw reconstruye raw como `[conservadas] + [añadidas]` (preserva el largo, cambia el orden).
Sus hermanos ya se pasaron al mapeo por ALIMENTO (`_sync_one_raw_line`, `_rescale_raw_by_food`,
`_raw_display_parallel_by_food`); este pase se quedó atrás — la asimetría "el fix aterrizó en una
superficie y no en sus hermanas" que domina el historial del repo.

Consecuencia con raw rotado: se borra de la lista de compras un alimento que el plato SÍ lleva
(el arroz) mientras la 2ª proteína que se quitó del plato SIGUE comprándose, y el factor de
recuperación engorda la línea equivocada.

El contrato correcto ya existe y es conservador: índice SOLO con paralelismo verificado; si no, por
alimento; 0 o >1 coincidencias ⇒ no tocar nada. Preferimos raw sin cambiar (que el reconciliador
sabe cerrar) a raw cambiado en la línea equivocada, que nadie detecta.

Anchor de producción: P1-DM-RAW-BY-FOOD.
"""
import re
from pathlib import Path

import pytest

GO = Path(__file__).resolve().parent.parent / "graph_orchestrator.py"


class _FakeDB:
    """'N g de <alimento>' → macros proporcionales. Solo proteína, que es lo que usa el pase."""

    _P100 = {"pollo": 23.0, "res": 22.0, "arroz blanco": 2.7, "aceite de oliva": 0.0}

    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+(.+?)\s*$", str(s), re.I)
        if not m:
            return {}
        gramos = float(m.group(1).replace(",", "."))
        food = m.group(2).strip().lower()
        if food not in self._P100:
            return {}
        return {"protein": self._P100[food] * gramos / 100.0, "kcal": gramos, "carbs": 0.0, "fats": 0.0}


def _plato_rotado():
    """display y raw del MISMO largo pero en orden distinto — el caso normal medido."""
    return {
        "name": "Pollo guisado",
        "ingredients": [
            "200 g de pollo",
            "150 g de arroz blanco",
            "120 g de res",
            "10 g de aceite de oliva",
        ],
        # raw rotado: la res va primera, el aceite último
        "ingredients_raw": [
            "120 g de res",
            "200 g de pollo",
            "150 g de arroz blanco",
            "10 g de aceite de oliva",
        ],
        "recipe": ["Mise en place.", "Cocina el pollo."],
    }


def _correr(meal):
    from graph_orchestrator import _meal_double_main_resolve
    dias = [{"meals": [meal]}]
    n = _meal_double_main_resolve(dias, db=_FakeDB())
    return n, meal


def _linea_de(lista, alimento):
    return [s for s in lista if alimento in str(s).lower()]


# --------------------------------------------------------------- el bug

def test_no_borra_de_raw_el_alimento_equivocado():
    """Con raw rotado, `del raw[idx_sec]` se llevaba el ARROZ en vez de la res."""
    n, meal = _correr(_plato_rotado())
    assert n >= 1, "el pase debe haber actuado sobre este plato (dos principales)"
    raw = meal["ingredients_raw"]

    assert _linea_de(raw, "arroz"), (
        f"el arroz desapareció de ingredients_raw: se borró por índice la línea equivocada. raw={raw}"
    )
    assert not _linea_de(raw, "res"), (
        f"la 2ª proteína sigue en ingredients_raw: se quitó del plato pero se seguiría comprando. raw={raw}"
    )


def test_no_escala_en_raw_el_alimento_equivocado():
    """El factor de recuperación debe caer sobre el POLLO de raw, nunca sobre el aceite."""
    _n, meal = _correr(_plato_rotado())
    raw = meal["ingredients_raw"]

    aceite = _linea_de(raw, "aceite")
    assert aceite, f"el aceite desapareció de raw: {raw}"
    assert aceite[0].strip().startswith("10 "), (
        f"el aceite fue re-escalado: el factor cayó en la línea equivocada. aceite={aceite[0]!r}"
    )

    arroz = _linea_de(raw, "arroz")
    assert arroz and arroz[0].strip().startswith("150 "), (
        f"el arroz fue re-escalado: el factor cayó en la línea equivocada. arroz={arroz}"
    )

    pollo = _linea_de(raw, "pollo")
    assert pollo, f"el pollo desapareció de raw: {raw}"
    gramos = float(re.match(r"\s*(\d+(?:[.,]\d+)?)", pollo[0]).group(1).replace(",", "."))
    assert gramos > 200, (
        f"el pollo de raw no creció ({gramos} g): el display subió y la lista de compras no, "
        f"que es la desincronización que este pase debe evitar"
    )


def test_display_sigue_correcto():
    """Regresión: lo que ya hacía bien sobre `ingredients` no puede cambiar."""
    _n, meal = _correr(_plato_rotado())
    ings = meal["ingredients"]
    assert not _linea_de(ings, "res"), f"la 2ª principal debe salir del plato: {ings}"
    assert _linea_de(ings, "arroz"), f"el arroz debe seguir en el plato: {ings}"
    pollo = _linea_de(ings, "pollo")
    gramos = float(re.match(r"\s*(\d+(?:[.,]\d+)?)", pollo[0]).group(1).replace(",", "."))
    assert gramos > 200, f"la primaria debe re-escalarse proteína-conservada: {pollo}"


def test_listas_paralelas_siguen_funcionando():
    """Control: con raw REALMENTE paralelo el resultado no cambia (el fix no rompe el caso bueno)."""
    meal = _plato_rotado()
    meal["ingredients_raw"] = list(meal["ingredients"])  # paralelo de verdad
    _n, meal = _correr(meal)
    raw = meal["ingredients_raw"]
    assert not _linea_de(raw, "res"), f"la res debe salir de raw: {raw}"
    assert _linea_de(raw, "arroz"), f"el arroz debe quedarse: {raw}"
    pollo = _linea_de(raw, "pollo")
    gramos = float(re.match(r"\s*(\d+(?:[.,]\d+)?)", pollo[0]).group(1).replace(",", "."))
    assert gramos > 200, f"el pollo de raw debe crecer: {pollo}"


def test_raw_de_largo_distinto_no_se_toca_por_indice():
    """Largos distintos = seguro NO paralelas. Antes el re-escalado escribía igual (solo miraba `_ip < len`)."""
    meal = _plato_rotado()
    meal["ingredients_raw"] = ["10 g de aceite de oliva", "150 g de arroz blanco"]  # sin pollo ni res
    _n, meal = _correr(meal)
    raw = meal["ingredients_raw"]
    assert raw[0].strip().startswith("10 "), f"el aceite no puede re-escalarse: {raw}"
    assert raw[1].strip().startswith("150 "), f"el arroz no puede re-escalarse: {raw}"


def test_alimento_ambiguo_en_raw_no_se_toca():
    """Si el alimento aparece en 2 líneas de raw no se adivina: se deja como está."""
    meal = _plato_rotado()
    meal["ingredients_raw"] = [
        "120 g de res", "100 g de pollo", "100 g de pollo", "150 g de arroz blanco",
    ]
    _n, meal = _correr(meal)
    raw = meal["ingredients_raw"]
    pollos = _linea_de(raw, "pollo")
    assert all(p.strip().startswith("100 ") for p in pollos), (
        f"con el alimento duplicado en raw no se puede elegir cuál escalar: {raw}"
    )


def test_sin_ingredients_raw_no_crashea():
    meal = _plato_rotado()
    meal.pop("ingredients_raw")
    n, meal = _correr(meal)
    assert n >= 1
    assert not _linea_de(meal["ingredients"], "res")


# --------------------------------------------------------------- anclaje estructural

def test_el_pase_no_usa_indice_ciego_sobre_raw():
    """tooltip-anchor de producción: P1-DM-RAW-BY-FOOD"""
    src = GO.read_text(encoding="utf-8", errors="ignore")
    i = src.index("def _meal_double_main_resolve")
    cuerpo = src[i: src.index("\ndef ", i + 10)]
    codigo = "\n".join(l for l in cuerpo.splitlines() if not l.lstrip().startswith("#"))

    assert "del raw[idx_sec]" not in codigo, (
        "el borrado por índice sobre raw sigue vivo: con raw rotado se lleva otro alimento"
    )
    assert 'meal["ingredients_raw"][_ip]' not in codigo, (
        "el re-escalado por índice sobre raw sigue vivo"
    )
    assert "_sync_one_raw_line" in codigo, (
        "debe reusar el helper by-food que ya usan los pases hermanos"
    )
