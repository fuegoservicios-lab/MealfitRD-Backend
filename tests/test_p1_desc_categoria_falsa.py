# -*- coding: utf-8 -*-
"""[P1-DESC-CATEGORIA-FALSA · 2026-09-09] El plátano no es un tubérculo.

## De dónde sale

El dueño lo dijo de pasada, después de preguntar por el mango de su plan vivo `0871ea93`:
*«El plátano maduro no es un tubérculo, eso es otro detalle»*. Tenía razón dos veces — el plato era
de plátano **verde**, y ni el verde ni el maduro lo son: son el fruto de una musácea. En RD entran
en «víveres», que es una categoría CULINARIA y mezcla tubérculos de verdad (yautía, ñame, batata,
yuca) con cosas que no lo son (plátano, guineo, panapén, auyama).

## El tamaño del arreglo va medido

```
comidas con `desc` en planes vivos ......... 72
dicen «tubérculo» .......................... 1
y el plato no lleva ninguno ................ 1   ← el suyo
```

Uno de 72. Por eso esto es una lima y no una taxonomía botánica: vigila UNA afirmación y sólo la
borra cuando es falsa. Ampliarlo a un diccionario de categorías sería construir maquinaria para un
caso — y *un guard ruidoso acaba apagado*, que es la frontera que ya se midió en
`P1-CLASH-HUEVO-Y-VIVERES`.

## El instrumento, otra vez

La primera sonda buscó la clave `description`. El campo es `desc`. Devolvió «0 casos» de un defecto
que sí estaba, y sólo se destapó al listar las claves reales de una comida en vez de asumirlas.
*Una sonda que pregunta por una clave inexistente no mide cero: no mide.*
"""
import pytest

import desc_sin_categoria_falsa as dcf


# ── El caso del dueño, textual ───────────────────────────────────────────────
def test_el_plato_del_dueno_pierde_la_afirmacion_falsa():
    m = {"name": "Plátano verde horneado con huevo, queso fresco y mango",
         "ingredients": ["½ plátano verde mediano", "2 huevos", "15 g de queso blanco"],
         "desc": ("Desayuno dominicano de tubérculo, horneado y práctico para un día laboral "
                  "caluroso; aporta energía sostenida y proteína para comenzar el reto personal.")}
    assert dcf.limpia_desc(m) is True
    assert "tubércul" not in m["desc"].lower()
    assert m["desc"].startswith("Desayuno dominicano"), (
        f"la frase quedó rota al borrar: {m['desc']!r}")
    assert "  " not in m["desc"] and " ," not in m["desc"]


def test_es_idempotente():
    m = {"name": "Plátano verde horneado", "ingredients": ["1 plátano verde"],
         "desc": "Desayuno dominicano de tubérculo, horneado."}
    assert dcf.limpia_desc(m) is True
    assert dcf.limpia_desc(m) is False


# ── LA frontera: cuando la ficha dice la verdad, no se toca ──────────────────
@pytest.mark.parametrize("ing", ["200 g de Yautía", "150 g de Ñame", "180 g de Batata",
                                 "200 g de Yuca", "200 g de Papa", "120 g de Mapuey"])
def test_si_el_plato_SI_lleva_tuberculo_la_ficha_se_respeta(ing):
    d = "Cena dominicana de tubérculo, sencilla y saciante."
    m = {"name": "Guiso criollo", "ingredients": [ing], "desc": d}
    assert dcf.limpia_desc(m) is False
    assert m["desc"] == d, "borró una afirmación que era CIERTA"


@pytest.mark.parametrize("nombre,ing", [
    ("Mangú de plátano verde con cebolla", "200 g de Plátano verde"),
    ("Guineo verde hervido con aceite", "180 g de Guineo verde"),
    ("Auyama guisada con cebolla", "250 g de Auyama"),
    ("Casabe con aguacate", "40 g de Casabe"),          # casabe ES yuca: cuenta como tubérculo
])
def test_la_frontera_platano_guineo_auyama_vs_casabe(nombre, ing):
    m = {"name": nombre, "ingredients": [ing], "desc": "Plato de tubérculo criollo."}
    tocado = dcf.limpia_desc(m)
    if "casabe" in nombre.lower():
        assert tocado is False, "el casabe se hace de yuca: la afirmación es cierta"
    else:
        assert tocado is True, f"«{nombre}» no lleva ningún tubérculo y la ficha lo afirmaba"


def test_una_ficha_sin_la_afirmacion_no_se_toca():
    d = "Desayuno dominicano práctico, con energía sostenida."
    m = {"name": "Mangú con huevo", "ingredients": ["200 g de Plátano verde"], "desc": d}
    assert dcf.limpia_desc(m) is False and m["desc"] == d


@pytest.mark.parametrize("m", [None, {}, "no soy un plato", {"desc": ""}, {"desc": None},
                               {"desc": 42}, {"desc": "de tubérculo"}])
def test_entradas_raras_no_revientan(m):
    try:
        dcf.limpia_desc(m)
    except Exception as e:                                             # noqa: BLE001
        pytest.fail(f"reventó con {m!r}: {e!r}")


def test_nunca_deja_una_desc_vacia():
    """Borrar de menos es recuperable; dejar la ficha en blanco es un hueco en la pantalla."""
    m = {"name": "Plátano hervido", "ingredients": ["200 g de Plátano verde"], "desc": "Tubérculo"}
    dcf.limpia_desc(m)
    assert m["desc"], "la ficha se quedó vacía"


# ── El plan entero, y el cableado ────────────────────────────────────────────
def test_limpia_plan_recorre_todas_las_comidas():
    plan = {"days": [
        {"meals": [{"name": "Mangú", "ingredients": ["200 g de Plátano verde"],
                    "desc": "Desayuno de tubérculo."},
                   {"name": "Yuca hervida", "ingredients": ["200 g de Yuca"],
                    "desc": "Cena de tubérculo."}]},
        {"meals": [{"name": "Guineo hervido", "ingredients": ["200 g de Guineo verde"],
                    "desc": "Merienda de tubérculo."}]},
    ]}
    assert dcf.limpia_plan(plan) == 2, "o se saltó una comida, o tocó la que decía la verdad"
    assert "tubércul" not in plan["days"][0]["meals"][0]["desc"].lower()
    assert "tubérculo" in plan["days"][0]["meals"][1]["desc"], "la yuca SÍ es un tubérculo"


@pytest.mark.parametrize("basura", [None, {}, {"days": None}, {"days": [None]},
                                    {"days": [{"meals": "no soy lista"}]}])
def test_limpia_plan_es_fail_open(basura):
    assert dcf.limpia_plan(basura) == 0


def test_esta_cableado_en_el_UNICO_chokepoint():
    """Va en el escudo pre-INSERT, no en un camino.

    `P1-PROTEIN-FLOOR-LAST-WORD` se desplegó el MISMO día cableado sólo en el merge del chunk y
    nació inerte para el bloque inicial: *cablear un paso en un solo camino es no cablearlo*.
    """
    import inspect
    import pathlib

    src = pathlib.Path(inspect.getfile(dcf)).with_name("db_plans.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _finalize_plan_data_for_insert")[1]
    assert "desc_sin_categoria_falsa" in cuerpo, (
        "la pasada salió del escudo pre-INSERT: vuelve a nacer inerte para algún camino")
