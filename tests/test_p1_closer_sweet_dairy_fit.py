"""[P1-CLOSER-SWEET-DAIRY-FIT + P1-CLOSER-NO-SPREAD-PLUS-CHEESE · 2026-07-26]

El dueño señaló una merienda: **"Apio Relleno con Mantequilla de Maní y Queso Cottage"**
(80 g de apio, 41 g de mantequilla de maní, 85 g de queso cottage). Al medirla salió que no
era un caso aislado sino un patrón, sobre 184 comidas de los 60 planes más recientes:

    queso cottage EN EL NOMBRE ......... 37 / 184  = 20.1%   (1 de cada 5 platos)
    mantequilla de maní en el nombre ... 22 / 184  = 12.0%
    LOS DOS a la vez ................... 15 / 184  =  8.2%
    nombres con ≥1 conjunción " y " .... 174 / 184 = 94.6%   (24% con dos o tres)

Y el cottage aparecía una y otra vez en **85 g clavados** — porción fija, no receta.

## Por qué pasaba (no era el modelo alucinando)

1. El cerrador de proteína añade una fuente cuando la comida queda corta y **renombra el
   plato para no esconderla** (P2-DISH-COHERENCE, graph_orchestrator.py:15304). Deliberado.
2. El pool de candidatos llega ordenado SOLO por densidad (`_safe_high_density_proteins`:
   `out.sort(key=protein/kcal, reverse=True)`). Medido en el catálogo vivo:

       Queso cottage   0.1779 prot/kcal   (12.4 g / 70 kcal)
       Yogurt          0.1743 prot/kcal   (10.3 g / 59 kcal)

   Un **2%**. Con eso el cottage ganaba siempre y se pegaba a todo plato dulce corto de
   proteína: batidos de guineo, tostadas francesas, mango fresco, rellenos de maní.
3. El filtro de coherencia existía pero exime a los lácteos por diseño ("extensores
   legítimos", graph_orchestrator.py:15693), así que nunca los frenaba.

## Qué se cambió

**Un reorden, no un filtro.** Dentro de una banda de margen respecto al más denso, decide el
ajuste culinario (en es-DO el yogurt va con fruta; el cottage no). Fuera de la banda sigue
mandando la densidad. Los dos siguen en el pool ⇒ **el piso de proteína se cumple igual**,
solo cambia con qué alimento — que es justo lo que el código advertía que no se podía
sacrificar.

Y una regla nueva en la SSOT `_dish_coherence_filter`: si el protagonista ya es una pasta
grasa untable, no se apila además un queso. Bloquea el queso, NO el yogurt ni el huevo (un
batido de maní con yogurt es normal; dos pastas de untar en el mismo relleno no).

tooltip-anchor: P1-CLOSER-SWEET-DAIRY-FIT
"""
from __future__ import annotations

import pytest

import graph_orchestrator as go
from constants import strip_accents


class _Info:
    """Stand-in de `NutritionInfo` con lo único que mira el reorden."""

    def __init__(self, name, protein, kcal):
        self.name = name
        self.protein = protein
        self.kcal = kcal


# Números REALES del catálogo (medidos 2026-07-26 con IngredientNutritionDB).
COTTAGE = _Info("Queso cottage", 12.4, 70)
YOGURT = _Info("Yogurt", 10.3, 59)
YOGURT_GRIEGO = _Info("Yogurt griego entero", 8.8, 94)
RICOTTA = _Info("Queso Ricotta", 7.5, 151)


def _par(info):
    return (info, strip_accents(info.name.lower()))


def _nombres(pool):
    return [n for (_i, n) in pool]


# ───────────── 1. el caso vivo, con los números reales ─────────────

def test_la_ventaja_del_cottage_es_del_2_por_ciento():
    """Si esta diferencia creciera de verdad, el reorden dejaría de ser gratis y habría que
    revisar la decisión. Anclarla hace visible ese día."""
    d_cot = go._sweet_dairy_density(COTTAGE)
    d_yog = go._sweet_dairy_density(YOGURT)
    assert d_cot > d_yog, "si el yogurt pasa al cottage, este fix es innecesario"
    assert (d_cot - d_yog) / d_cot < 0.05, "la ventaja del cottage debe seguir siendo marginal"


def test_el_yogurt_gana_al_cottage_dentro_de_la_banda():
    """El corazón del fix: 2% de densidad no debe comprar un defecto culinario."""
    fuera = go._reorder_sweet_dairy_by_fit([_par(COTTAGE), _par(YOGURT)], 0.10)
    assert _nombres(fuera)[0] == "yogurt"


def test_con_el_pool_completo_del_catalogo():
    pool = [_par(COTTAGE), _par(YOGURT), _par(YOGURT_GRIEGO), _par(RICOTTA)]
    out = _nombres(go._reorder_sweet_dairy_by_fit(pool, 0.10))
    assert out[0] == "yogurt"
    assert out[1] == "queso cottage", "el cottage sigue siendo la segunda opción, no se elimina"
    # los que están MUY por debajo conservan su orden por densidad
    assert out[2:] == ["yogurt griego entero", "queso ricotta"]


# ───────────── 2. la densidad sigue mandando cuando importa ─────────────

def test_una_diferencia_real_de_densidad_gana_al_ajuste():
    """El ajuste culinario solo desempata. Un lácteo con MUCHA más proteína no se degrada:
    si esto se rompe, el fix estaría costando proteína de verdad."""
    denso = _Info("Queso cottage", 30.0, 70)      # muy por encima de la banda
    out = _nombres(go._reorder_sweet_dairy_by_fit([_par(denso), _par(YOGURT)], 0.10))
    assert out[0] == "queso cottage"


def test_margen_cero_equivale_a_no_reordenar():
    out = _nombres(go._reorder_sweet_dairy_by_fit([_par(COTTAGE), _par(YOGURT)], 0.0))
    assert out[0] == "queso cottage"


# ───────────── 3. fail-open y bordes ─────────────

@pytest.mark.parametrize("pool", [[], [_par(YOGURT)]])
def test_pool_trivial_se_devuelve_igual(pool):
    assert go._reorder_sweet_dairy_by_fit(pool, 0.10) == pool


def test_datos_invalidos_no_rompen_la_cadena():
    malo = _Info("Queso raro", "no-numero", None)
    out = go._reorder_sweet_dairy_by_fit([_par(malo), _par(YOGURT)], 0.10)
    assert len(out) == 2, "no debe perder candidatos ante datos sucios"


def test_densidad_defensiva():
    assert go._sweet_dairy_density(None) == 0.0
    assert go._sweet_dairy_density(_Info("x", 10, 0)) == 0.0


# ───────────── 4. no apilar queso sobre una pasta de untar ─────────────

def _filtro(nombre_plato, ingredientes=None):
    meal = {"name": nombre_plato, "meal": "Merienda",
            "ingredients": ingredientes or []}
    return go._dish_coherence_filter(meal, strip_accents)


def test_el_apio_relleno_ya_no_admite_queso():
    """El caso exacto que lo destapó."""
    ok = _filtro("Apio Relleno con Mantequilla de Maní")
    assert ok("queso cottage") is False
    assert ok("queso ricotta") is False


def test_pero_si_admite_yogurt_y_huevo():
    """Bloquear TODO el lácteo dejaría la comida sin forma de cerrar el piso de proteína,
    y un batido de maní con yogurt es una merienda dominicana normal."""
    ok = _filtro("Batido de Mantequilla de Maní y Guineo")
    assert ok("yogurt") is True
    assert ok("huevos") is True


def test_un_plato_sin_pasta_untable_no_cambia():
    ok = _filtro("Mango Fresco con Linaza")
    assert ok("queso cottage") is True


def test_la_pasta_cuenta_por_el_NOMBRE_no_por_los_ingredientes():
    """Simétrico a `_cheese_dish`: en el nombre es el relleno; en la lista puede ser topping.
    Si se mirara la lista, una cucharada de maní vetaría el queso de cualquier plato."""
    ok = _filtro("Yogurt con Frutas", ingredientes=["1 cda de mantequilla de maní"])
    assert ok("queso cottage") is True


# ───────────── 5. knobs de rollback ─────────────

def test_knobs_existen_y_son_del_tipo_correcto():
    assert isinstance(go.CLOSER_SWEET_DAIRY_FIT_ENABLED, bool)
    assert isinstance(go.CLOSER_NO_SPREAD_PLUS_CHEESE, bool)
    assert 0.0 <= go.CLOSER_SWEET_DAIRY_FIT_MARGIN <= 1.0


def test_el_reorden_esta_cableado_en_el_pool_dulce():
    """El helper puede ser perfecto y no llamarse nunca — es exactamente el modo de fallo que
    dejó inerte P1-CAPPED-STAPLE-HONESTY. Se ancla el callsite."""
    from pathlib import Path
    src = (Path(go.__file__).resolve()).read_text(encoding="utf-8")
    i = src.index("_sweet_dairy.append((_info_sd, _ndlow))")
    bloque = src[i:src.index("_pool_sweet = ", i)]
    assert "_reorder_sweet_dairy_by_fit(" in bloque, "el reorden no se aplica al pool dulce"
    assert "CLOSER_SWEET_DAIRY_FIT_ENABLED" in bloque, "debe respetar el knob de rollback"


def test_la_regla_del_spread_vive_en_la_SSOT():
    """Tenerla en el cerrador y no en `_dish_coherence_filter` reproduciría el bug de paridad
    que originó la SSOT (el rescate de proteína quedó sin blindar y por ahí entraron 4 bolt-on)."""
    import inspect
    cuerpo = inspect.getsource(go._dish_coherence_filter)
    assert "spread_dish" in cuerpo
    assert "_SPREAD_PROTAGONIST_HINT" in cuerpo
