# -*- coding: utf-8 -*-
"""[P1-CANDIDATO-CON-PRECIO · 2026-09-09] El precio entra en la ELECCIÓN del plato.

## Qué defiende esto

Medido sobre planes vivos: con presupuesto `low` el prompt pide «evita cortes premium, mariscos
caros» y el modelo puso **Cangrejo 2 lb = RD$958**, el 16 % de la compra. Es el mismo resultado que
la otra instrucción del prompt («elige EXACTAMENTE uno del catálogo»: 0 de 12 platos, tres planes
seguidos). Pedir no es imponer, así que la restricción baja al selector, que es determinista.

La política ya traía el presupuesto compilado (`effective["budget"]["tier"]`, Fase 2) y **quien
elige los platos no lo leía**: `horizon.py` no nombraba `budget` ni una vez.

## Las dos reglas que estos tests existen para que nadie deshaga

1. **Sin precio ⇒ `None`, nunca una suma parcial.** Un total parcial SUBESTIMA, y lo subestimado
   asciende en el ranking: el filtro promovería justo el plato que no puede ver. Modo de fallo
   invertido.
2. **La poda es relativa y NUNCA vacía.** *Un filtro que descarta todo es ciego, no preciso.*

Y una tercera, silenciosa: el coste **no viaja** en el dict del candidato. Los candidatos se fijan
al run y entran en `slice_hash` → `input_hash`; meter ahí un precio ataría la huella de un plan al
cron de inflación.
"""
import pytest

import dish_cost as dc


# ── 1 · el precio de una fila del catálogo ───────────────────────────────────
def test_price_per_lb_manda_cuando_existe():
    assert dc.precio_por_lb({"price_per_lb": 120.0, "price_per_unit": 999.0}) == 120.0


def test_por_unidad_con_peso_de_envase():
    # 100 RD$ el envase de 453,59 g = 1 lb ⇒ 100 RD$/lb
    p = dc.precio_por_lb({"price_per_unit": 100.0, "container_weight_g": dc.G_POR_LB})
    assert p == pytest.approx(100.0, rel=1e-6)


def test_por_unidad_natural_el_caso_del_aguacate():
    """El aguacate se vende POR PIEZA y su peso vive en `density_g_per_unit`, no en un envase.

    Sin esta rama, 18 plantillas quedaban «sin precio» —medido— y, como lo que no tiene precio no
    se poda, el filtro se desactivaba justo donde más falta hace.
    """
    p = dc.precio_por_lb({"price_per_unit": 50.0, "container_weight_g": None,
                          "density_g_per_unit": 226.796185})   # media libra
    assert p == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("fila", [
    {}, None, "no soy una fila",
    {"price_per_lb": 0}, {"price_per_lb": None},
    {"price_per_unit": 50.0},                                   # sin peso: no se puede afirmar
    {"price_per_unit": 50.0, "container_weight_g": 0},
    {"price_per_lb": "carísimo"},
])
def test_sin_precio_afirmable_devuelve_None(fila):
    assert dc.precio_por_lb(fila) is None


# ── 2 · el coste de una ración ───────────────────────────────────────────────
_PRECIOS = {"arroz blanco": 45.0, "pollo": 120.0, "sal": 20.0}


def _tpl(*cons):
    return {"template_id": "t", "name": "n",
            "constituents": [{"canonical": c, "grams": g} for c, g in cons]}


def test_costo_racion_suma_los_gramos_crudos():
    t = _tpl(("Arroz blanco", dc.G_POR_LB), ("Pollo", dc.G_POR_LB / 2))
    assert dc.costo_racion(t, _PRECIOS) == pytest.approx(45.0 + 60.0, abs=0.01)


def test_costo_racion_resuelve_por_ingredient_id_si_el_nombre_no_casa():
    t = {"constituents": [{"name": "Arrocito", "ingredient_id": "arroz blanco", "grams": dc.G_POR_LB}]}
    assert dc.costo_racion(t, _PRECIOS) == pytest.approx(45.0, abs=0.01)


def test_UN_constituyente_sin_precio_anula_el_coste_entero():
    """LA regla. Si esto se relaja a «suma lo que puedas», el filtro se invierte.

    La plantilla de abajo lleva 1 lb de arroz (RD$45) y 1 lb de langosta (sin precio). Sumar sólo
    lo costeable daría RD$45 —más barata que el arroz con pollo— y la langosta ASCENDERÍA.
    """
    t = _tpl(("Arroz blanco", dc.G_POR_LB), ("Langosta", dc.G_POR_LB))
    assert dc.costo_racion(t, _PRECIOS) is None

    parcial = sum(_PRECIOS[c["canonical"].lower()] * (c["grams"] / dc.G_POR_LB)
                  for c in t["constituents"] if c["canonical"].lower() in _PRECIOS)
    barata = dc.costo_racion(_tpl(("Arroz blanco", dc.G_POR_LB), ("Pollo", dc.G_POR_LB)), _PRECIOS)
    assert parcial < barata, ("el escenario ya no demuestra la inversión; revísalo antes de "
                              "tocar la regla")


@pytest.mark.parametrize("t", [None, {}, {"constituents": []}, {"constituents": ["texto"]}])
def test_plantilla_inservible_no_tiene_coste(t):
    assert dc.costo_racion(t, _PRECIOS) is None


def test_sin_tabla_de_precios_nada_tiene_coste():
    assert dc.costo_racion(_tpl(("Arroz blanco", 100.0)), {}) is None


# ── 3 · la fracción por nivel de presupuesto ─────────────────────────────────
def test_low_poda_mas_que_medium():
    assert dc.fraccion_asequible("low") < dc.fraccion_asequible("medium")


@pytest.mark.parametrize("tier", ["high", "unlimited", "", None, "  ", "loquesea"])
def test_sin_restriccion_no_se_poda(tier):
    """Quien no tiene la restricción no debe perder variedad por ella."""
    assert dc.fraccion_asequible(tier) is None


def test_custom_va_con_medium_y_no_escribe_una_segunda_tabla():
    """El monto exacto se compara contra el suelo en la Fase 2 (`plan_policy`). Duplicar aquí esa
    aritmética es la segunda tabla que `P1-DIET-CANON-SSOT` prohíbe."""
    assert dc.fraccion_asequible("custom") == dc.fraccion_asequible("medium")


# ── 4 · la poda (función PURA: aquí vive la frontera) ────────────────────────
def _pares(costes):
    return [({"id": i}, c) for i, c in enumerate(costes)]


def test_poda_quita_la_cola_cara_y_conserva_el_orden():
    pares = _pares([10, 90, 20, 80, 30, 70, 40, 60, 50, 100])
    vivos = dc.poda_por_presupuesto(pares, "low", minimo=3)     # 60 % de 10 = 6
    # los 6 más baratos son 10,20,30,40,50,60 → índices 0,2,4,6,8,7; el ORDEN de salida es el de
    # entrada, no el de precio: ordenar aquí pisaría el hash de ARQ27-P1-04 y mataría la variedad.
    assert [v["id"] for v in vivos] == [0, 2, 4, 6, 7, 8], (
        "o se coló un caro, o la poda reordenó el conjunto")


def test_lo_que_no_tiene_precio_sobrevive_siempre():
    pares = [({"id": "caro"}, 900.0), ({"id": "barato"}, 1.0), ({"id": "misterio"}, None),
             ({"id": "b2"}, 2.0), ({"id": "b3"}, 3.0), ({"id": "c2"}, 800.0)]
    vivos = [v["id"] for v in dc.poda_por_presupuesto(pares, "low", minimo=3)]
    assert "misterio" in vivos, "se podó lo que no se sabe: la regla 1 al revés"
    assert "caro" not in vivos


def test_jamas_devuelve_vacio():
    for tier in ("low", "medium", "custom", "high", None):
        for n in range(0, 12):
            vivos = dc.poda_por_presupuesto(_pares(list(range(n))), tier, minimo=3)
            assert bool(vivos) == bool(n), f"conjunto vaciado: tier={tier} n={n}"


def test_con_menos_costeados_que_el_minimo_no_se_opina():
    pares = [({"id": 0}, 5.0), ({"id": 1}, 500.0), ({"id": 2}, None), ({"id": 3}, None)]
    assert len(dc.poda_por_presupuesto(pares, "low", minimo=3)) == 4


def test_sin_restriccion_la_poda_es_la_identidad():
    pares = _pares([1, 2, 3, 4, 5, 900])
    for tier in ("high", "unlimited", None, ""):
        assert dc.poda_por_presupuesto(pares, tier) == [it for it, _ in pares]


def test_la_poda_no_muta_su_entrada():
    pares = _pares([10, 90, 20, 80, 30])
    copia = list(pares)
    dc.poda_por_presupuesto(pares, "low", minimo=3)
    assert pares == copia


def test_dos_candidatos_identicos_no_se_confunden():
    """Se poda por ÍNDICE, no por identidad de objeto: dos dicts iguales son dos candidatos."""
    a, b = {"id": "x"}, {"id": "x"}
    pares = [(a, 1.0), (b, 999.0), ({"id": "y"}, 2.0), ({"id": "z"}, 3.0), ({"id": "w"}, 4.0)]
    vivos = dc.poda_por_presupuesto(pares, "low", minimo=3)
    assert len(vivos) == 3 and vivos[0] is a
    assert not any(v is b for v in vivos), "se salvó el caro por parecerse al barato"


# ── 5 · el cableado en el CandidateSet ───────────────────────────────────────
def _fake_precios(monkeypatch, caro_id):
    """Tabla en la que TODO cuesta poco salvo los constituyentes de `caro_id`."""
    import dish_registry as dr
    snap = dr.load_registry("DO") or {}
    caro = next(t for t in (snap.get("templates") or []) if t["template_id"] == caro_id)
    caros = {str(c.get("canonical") or c.get("name")).lower() for c in caro["constituents"]}
    tabla = {}
    for t in snap.get("templates") or []:
        for c in t.get("constituents") or []:
            nm = str(c.get("canonical") or c.get("name")).lower()
            tabla[nm] = 9000.0 if nm in caros else 1.0
    monkeypatch.setattr(dc, "tabla_de_precios", lambda catalogo=None: tabla)
    return caro


def test_template_candidates_excluye_el_caro_con_presupuesto_bajo(monkeypatch):
    import dish_registry as dr
    if not dr.load_registry("DO"):
        pytest.skip("sin snapshot DO en este entorno")
    base = dr.template_candidates("DO", "almuerzo", k=500)
    if len(base) < 8:
        pytest.skip("conjunto demasiado pequeño para que la poda tenga sentido")
    caro = _fake_precios(monkeypatch, base[0]["template_id"])

    con_low = dr.template_candidates("DO", "almuerzo", k=500, budget_tier="low")
    sin_tier = dr.template_candidates("DO", "almuerzo", k=500)
    ids_low = {c["template_id"] for c in con_low}
    assert caro["template_id"] not in ids_low, "el plato caro sobrevivió al presupuesto ajustado"
    assert caro["template_id"] in {c["template_id"] for c in sin_tier}
    assert len(con_low) < len(sin_tier)


def test_presupuesto_amplio_no_recorta(monkeypatch):
    import dish_registry as dr
    if not dr.load_registry("DO"):
        pytest.skip("sin snapshot DO en este entorno")
    base = dr.template_candidates("DO", "almuerzo", k=500)
    if len(base) < 8:
        pytest.skip("conjunto demasiado pequeño")
    _fake_precios(monkeypatch, base[0]["template_id"])
    for tier in ("high", "unlimited"):
        assert len(dr.template_candidates("DO", "almuerzo", k=500, budget_tier=tier)) == len(base)


def test_el_knob_apagado_devuelve_la_conducta_previa(monkeypatch):
    import dish_registry as dr
    if not dr.load_registry("DO"):
        pytest.skip("sin snapshot DO en este entorno")
    base = dr.template_candidates("DO", "almuerzo", k=500)
    if len(base) < 8:
        pytest.skip("conjunto demasiado pequeño")
    _fake_precios(monkeypatch, base[0]["template_id"])
    monkeypatch.setenv("MEALFIT_CANDIDATE_PRICE_FILTER", "0")
    assert dr.template_candidates("DO", "almuerzo", k=500, budget_tier="low") == base


def test_el_coste_NO_viaja_en_el_dict_del_candidato(monkeypatch):
    """Los candidatos se fijan al run y entran en `slice_hash` → `input_hash`. Un precio ahí ataría
    la huella de un plan al cron de inflación: el mismo plan cambiaría de huella sin tocarse."""
    import dish_registry as dr
    if not dr.load_registry("DO"):
        pytest.skip("sin snapshot DO en este entorno")
    base = dr.template_candidates("DO", "almuerzo", k=500)
    if len(base) < 8:
        pytest.skip("conjunto demasiado pequeño")
    _fake_precios(monkeypatch, base[0]["template_id"])
    for c in dr.template_candidates("DO", "almuerzo", k=500, budget_tier="low"):
        for k in c:
            assert "cost" not in k.lower() and "precio" not in k.lower() and "_rd" not in k.lower(), (
                f"el candidato expone {k!r}: eso entra en slice_hash")


def test_falla_de_precios_no_tumba_los_candidatos(monkeypatch):
    """Fail-open: sin catálogo, sin poda — la conducta previa, nunca un conjunto vacío."""
    import dish_registry as dr
    if not dr.load_registry("DO"):
        pytest.skip("sin snapshot DO en este entorno")
    def _revienta(catalogo=None):
        raise RuntimeError("catálogo caído")
    monkeypatch.setattr(dc, "tabla_de_precios", _revienta)
    base = dr.template_candidates("DO", "almuerzo", k=500)
    assert dr.template_candidates("DO", "almuerzo", k=500, budget_tier="low") == base


# ── 6 · quien elige lee el presupuesto que la política ya compiló ────────────
def test_horizon_pasa_el_tier_a_los_candidatos():
    """`effective["budget"]["tier"]` existe desde la Fase 2 y `horizon.py` no lo nombraba ni una
    vez. Este test ancla el puente: si alguien lo quita, el filtro nace inerte —que es exactamente
    cómo `P1-PROTEIN-FLOOR-LAST-WORD` se desplegó muerto el mismo día."""
    import inspect

    import horizon

    src = inspect.getsource(horizon)
    assert "budget_tier=" in src, (
        "horizon dejó de pasar el presupuesto al CandidateSet: el filtro de precio queda inerte")
    assert src.count("budget_tier=") >= 2, (
        "sólo UNO de los dos call sites de horizon pasa el tier — «cablear un paso en un solo "
        "camino es no cablearlo» (P1-PROTEIN-FLOOR-LAST-WORD)")


def test_el_dia_determinista_tambien_pasa_el_tier():
    import inspect

    import deterministic_day

    assert "budget_tier=" in inspect.getsource(deterministic_day.build_day_for_skeleton), (
        "el día determinista es el ÚNICO camino donde el candidato se convierte en plato sin que "
        "el modelo pueda ignorarlo: sin el tier aquí, el filtro sólo aconseja")


def test_el_tier_sale_de_la_politica_COMPILADA_no_del_campo_crudo():
    """La Fase 2 no copia `budget`: resuelve `custom` y relaja el modo donde no hay precios.

    Medido: en los 5 planes vivos del dueño `plan_data` NO persiste `form_data`, y lo único que
    prueba que el presupuesto llegó es `_plan_policy.effective.budget.tier`. Colgar el filtro de
    una clave que no se puede verificar es exactamente como se despliega algo inerte.
    """
    from deterministic_day import _tier_presupuesto as tp

    assert tp({"_plan_policy_effective": {"budget": {"tier": "low"}}, "budget": "high"}) == "low", (
        "ganó el campo crudo sobre lo compilado: se pierden `custom` resuelto y las relajaciones")
    assert tp({"budget": "medium"}) == "medium"                      # respaldo
    assert tp({"_plan_policy_effective": {"budget": {}}, "budget": "low"}) == "low"
    for vacio in ({}, None, {"budget": ""}, {"_plan_policy_effective": None}):
        assert tp(vacio) is None
    assert tp({"_plan_policy_effective": "no soy un dict", "budget": "low"}) == "low"
