"""[P1-COHERENCE-BASE-QTY · 2026-07-26] El guard comparaba dos idiomas distintos.

El coherence guard contrasta la suma de las recetas contra la lista de compras emparejando por
**(alimento, UNIDAD)**. Las recetas hablan en `g` / `taza` / `cda`; la lista, tras
`apply_smart_market_units`, solo guardaba `market_qty` / `market_unit` — *pote*, *sobre*,
*paquete*, *mazo*. Sin unidad común no hay pareja, y `expected_qty` salía **0.0 para todos**:

    Miel                pote     esperado=0.0  lista=1.0  ratio=inf  ->  unknown
    Orégano dominicano  sobre    esperado=0.0  lista=1.0  ratio=inf  ->  unknown
    Ciruela             paquete  esperado=0.0  lista=1.0  ratio=inf  ->  unknown

Medido con el guard REAL sobre el plan vivo `01d63a5b`: **41 divergencias, 39 `unknown`** por
esta causa. Los `{'unknown': 32}` y `{'unknown': 42}` de los planes anteriores son lo mismo.

**El guard no estaba detectando incoherencias: estaba comparando pote contra gramo y llamando
"desconocido" al resultado.** Por eso nunca bloqueaba pese a estar en modo `block`.

El propio docstring de `compare_expected_vs_aggregated` ya lo avisaba — *"el caller es
responsable de construir `aggregated` ANTES de la conversión `apply_smart_market_units`"*—
pero el caller recibe la lista ya convertida. Preservar la cantidad base en el item es la
forma no invasiva de cumplir ese contrato sin reordenar el pipeline.

⚠️ LIMITACIÓN: los planes YA persistidos no tienen `base_qty`, así que siguen comparándose por
unidad de mercado (fallback). El efecto se mide sobre planes NUEVOS. No se reescribe historia.

tooltip-anchor: P1-COHERENCE-BASE-QTY
"""
from __future__ import annotations

import pytest

import shopping_calculator as sc


# ───────────── 1. la conversión preserva la base ─────────────

def test_apply_smart_market_units_devuelve_la_base():
    r = sc.apply_smart_market_units("Miel", 0.5, "cdta", 12.0)
    assert r.get("base_qty") == 12.0
    assert r.get("base_unit") == "cdta"


def test_la_ruta_por_PESO_tambien_produce_base():
    """⚠️ El aggregator tiene DOS rutas y solo una trae `raw_qty`:

        por unidades:  apply_smart_market_units(name, 0.0, u,   q,   ...)
        por peso:      apply_smart_market_units(name, lbs, 'lb', 0.0, ...)

    Mi primera versión solo miraba `raw_qty`, así que en la ruta de peso guardaba 0.0 y el
    extractor lo descartaba. Medido sobre el plan vivo fbe53a5b: **2 de 48 items** tenían base.
    El fix quedaba inerte justo donde más items hay — mismo modo de fallo que dejó muerto
    P1-CAPPED-STAPLE-HONESTY, y solo se ve MIDIENDO el resultado, no leyendo el código.
    """
    r = sc.apply_smart_market_units("Pechuga de pollo", 1.5, "lb", 0.0)
    assert r.get("base_unit") == "g"
    assert r.get("base_qty") == pytest.approx(1.5 * 453.592, rel=1e-3)


def test_las_dos_rutas_del_helper():
    assert sc._coherence_base_fields(2.0, "unidad", 0.0) == {"base_qty": 2.0, "base_unit": "unidad"}
    assert sc._coherence_base_fields(0.0, "lb", 1.5)["base_unit"] == "g"


@pytest.mark.parametrize("args", [(0.0, "lb", 0.0), ("x", None, None), (None, None, None),
                                  (-1, "g", -1)])
def test_el_helper_es_fail_safe(args):
    """Sin datos utilizables devuelve {} y el item sale sin base → el extractor cae al
    comportamiento previo. Nunca revienta la cadena del aggregator."""
    assert sc._coherence_base_fields(*args) == {}


def test_la_base_no_pisa_las_claves_de_mercado():
    """Aditivo: quien lee `market_*` no se entera del cambio."""
    r = sc.apply_smart_market_units("Miel", 0.5, "cdta", 12.0)
    for k in ("name", "market_qty", "market_qty_numeric", "market_unit",
              "display_qty", "display_string", "confidence_score"):
        assert k in r, k


@pytest.mark.parametrize("raw_qty,unit", [(736.0, "g"), (2.0, "unidad"), (0.25, "taza")])
def test_distintas_unidades_base(raw_qty, unit):
    r = sc.apply_smart_market_units("Atún en agua", raw_qty / 453.6, unit, raw_qty)
    assert r.get("base_qty") == raw_qty
    assert r.get("base_unit") == unit


# ───────────── 2. el extractor la prefiere ─────────────

def test_prefiere_la_base_cuando_existe():
    out = sc._extract_aggregated_food_dict([{
        "name": "Miel", "base_qty": 12.0, "base_unit": "g",
        "market_qty_numeric": 1.0, "market_unit": "pote",
    }])
    assert out == {"Miel": {"g": 12.0}}, "debe comparar en gramos, no en potes"


def test_fallback_a_mercado_en_listas_legacy():
    """Los planes ya persistidos no tienen `base_qty`. No deben romperse."""
    out = sc._extract_aggregated_food_dict([{
        "name": "Miel", "market_qty_numeric": 1.0, "market_unit": "pote",
    }])
    assert out == {"Miel": {"pote": 1.0}}


@pytest.mark.parametrize("base_qty", [0, -3, None, "no-numero"])
def test_base_invalida_cae_al_fallback(base_qty):
    out = sc._extract_aggregated_food_dict([{
        "name": "Miel", "base_qty": base_qty, "base_unit": "g",
        "market_qty_numeric": 1.0, "market_unit": "pote",
    }])
    assert out == {"Miel": {"pote": 1.0}}, base_qty


def test_sin_base_unit_tampoco_la_usa():
    out = sc._extract_aggregated_food_dict([{
        "name": "Miel", "base_qty": 12.0, "base_unit": None,
        "market_qty_numeric": 1.0, "market_unit": "pote",
    }])
    assert out == {"Miel": {"pote": 1.0}}


# ───────────── 3. el efecto: la pareja aparece ─────────────

def test_receta_y_lista_por_fin_se_emparejan():
    """Antes: expected {Miel: {cdta: 2}} vs list {Miel: {pote: 1}} → sin unidad común,
    expected_qty=0 → `unknown`. Ahora ambos lados hablan en cdta y la comparación es real."""
    expected = {"Miel": {"cdta": 2.0}}
    agg = sc._extract_aggregated_food_dict([{
        "name": "Miel", "base_qty": 2.0, "base_unit": "cdta",
        "market_qty_numeric": 1.0, "market_unit": "pote",
    }])
    divs = sc.compare_expected_vs_aggregated(expected, agg, tolerance=0.10)
    assert divs == [], f"cantidades idénticas no deben divergir: {divs}"


def test_una_divergencia_REAL_sigue_reportandose():
    """El fix no silencia el guard: si la lista trae la mitad, se reporta."""
    expected = {"Miel": {"cdta": 4.0}}
    agg = sc._extract_aggregated_food_dict([{
        "name": "Miel", "base_qty": 2.0, "base_unit": "cdta",
        "market_qty_numeric": 1.0, "market_unit": "pote",
    }])
    divs = sc.compare_expected_vs_aggregated(expected, agg, tolerance=0.10)
    assert len(divs) == 1
    # El comparador pre-normaliza a la unidad base del mismo sistema físico (4 cdta -> 20 g,
    # P1-NEW-10), así que no se afirman los números crudos sino la RELACIÓN: la lista trae la
    # mitad de lo que piden las recetas. Mi primera versión fijaba 4.0/2.0 y fallaba por eso.
    assert divs[0]["actual_qty"] == pytest.approx(divs[0]["expected_qty"] / 2.0)


# ───────────── 4. ancla ─────────────

def test_el_extractor_mira_base_antes_que_mercado():
    """Se busca en el CÓDIGO, no en el docstring: el docstring menciona `market_qty_numeric`
    en su primera línea y mi primera versión de este test comparaba contra esa posición."""
    import inspect
    src = inspect.getsource(sc._extract_aggregated_food_dict)
    cuerpo = src[src.index('"""', src.index('"""') + 3) + 3:]   # tras cerrar el docstring
    i_base = cuerpo.index('item.get("base_qty")')
    i_mkt = cuerpo.index('item.get("market_qty_numeric")')
    assert i_base < i_mkt, "la base debe evaluarse ANTES que la unidad de mercado"


# ───────────── 5. normalización a gramos (P1-COHERENCE-GRAM-NORM) ─────────────
#
# Preservar la base en la lista fue necesario pero NO suficiente: el lado de las RECETAS
# seguía en medidas caseras. Medido sobre el plan vivo fbe53a5b:
#
#     lado recetas:  taza×8, unidad×15, cda×9, cdta×8, g×11, pizca, diente, hoja, rebanada
#     lado lista:    g
#
# Solo los 11 que ya venían en gramos podían casar. Normalizando AMBOS lados:
#
#     39 divergencias (todas unknown, expected_qty=0.0)  ->  8, con señal real
#         Pescado  esp=349.71  lista=574.71  ratio=1.64   <- discrepancia genuina
#         Plátano  esp=0.0     lista=697.2                <- fantasma genuino

def test_normaliza_medidas_caseras_a_gramos():
    out = sc._normalize_food_dict_to_grams({"Avena": {"taza": 1.0}})
    assert "g" in out.get("Avena", {}), out
    assert out["Avena"]["g"] > 0


def test_suma_varias_unidades_del_mismo_alimento():
    out = sc._normalize_food_dict_to_grams({"Aceite de oliva": {"cda": 2.0, "g": 10.0}})
    g = out.get("Aceite de oliva", {}).get("g")
    assert g and g > 10.0, "debe sumar los gramos convertidos a los que ya venían en g"


def test_lo_inconvertible_conserva_su_unidad():
    """Mejor una divergencia sin pareja que una conversión inventada. `convert_amount` en modo
    strict devuelve None cuando falta la densidad — no se rellena con un valor plausible."""
    out = sc._normalize_food_dict_to_grams({"Sal": {"pizca": 3.0}})
    assert out.get("Sal") == {"pizca": 3.0}


def test_gramos_pasan_intactos():
    assert sc._normalize_food_dict_to_grams({"Pollo": {"g": 250.0}}) == {"Pollo": {"g": 250.0}}


@pytest.mark.parametrize("entrada", [{}, None, "no-dict", {"X": None}, {"X": {"g": "abc"}}])
def test_normalizador_fail_safe(entrada):
    assert isinstance(sc._normalize_food_dict_to_grams(entrada), dict)


def test_se_normalizan_LOS_DOS_lados():
    """Ancla: normalizar solo uno crea el sesgo inverso (recetas en g contra lista en taza)."""
    from pathlib import Path
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("expected_canonical = _normalize_food_dict_to_grams(")
    bloque = src[i:i + 260]
    assert "aggregated_canonical = _normalize_food_dict_to_grams(" in bloque
