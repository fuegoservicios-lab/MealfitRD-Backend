"""[P1-RESOLVE-PAREN-GRAMS · 2026-07-26] Dos parsers del sistema discrepaban 7,5× sobre la misma línea.

Medido en producción con el catálogo real:

    "1 lechosa mediana (198g)"
        grams_from_ingredient_string  →   198 g   (lee el paréntesis)
        _resolve_line_food_grams      →  1500 g   (resolvía una lechosa ENTERA)

`_resolve_line_food_grams` nunca leía la anotación entre paréntesis: resolvía siempre por
conteo × densidad del catálogo.

## Por qué importa

Ese resolvedor alimenta el reconciliador display↔raw (`P1-DISPLAY-RAW-QTY-RECONCILE` /
`P1-RECONCILE-LAST-WORD`) y el tracer de desalineación. Medir 1500 g donde el motor de macros lee
198 g hace que el reconciliador vea divergencias que no existen — o que se pierda las que sí.

Y el paréntesis es justo lo que anota `_bigfruit_bare_count_serving` para **corregir** la magnitud
de una fruta grande ("1 lechosa"=711 kcal → "1 lechosa (200g)"=71 kcal). Ignorarlo desandaba esa
corrección aguas abajo.

## Precedencia

Los gramos LÍDER mandan sobre el paréntesis, igual que en `P1-PAREN-GRAMS-CAP`:
`"75 g de pollo (aprox. 80 g cocido)"` son **75** — el líder es el peso crudo (lo que se compra) y
el paréntesis el cocido. Es una diferencia deliberada con `grams_from_ingredient_string`, que ahí
devuelve 80.
"""
import pytest

import graph_orchestrator as go


@pytest.fixture(autouse=True)
def _cache_limpio():
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()


# ───────────── 1. el helper de precedencia ─────────────

@pytest.mark.parametrize("linea,esperado", [
    ("1 lechosa mediana (198g)", 198.0),
    ("1 mapuey mediano (200g)", 200.0),
    ("½ conejo (aprox. 358 g en piezas)", 358.0),
    ("6½ láminas de casabe (95 g)", 95.0),
    ("2½ papas medianas (378.92g)", 378.92),
    ("1 lechosa mediana (198g) ≈ 1/8 de la fruta", 198.0),
    ("1 lechosa (~205 gr)", 205.0),
])
def test_lee_la_masa_del_parentesis(linea, esperado):
    assert go._paren_grams_in_line(linea) == pytest.approx(esperado)


@pytest.mark.parametrize("linea", [
    "75 g de costilla de cerdo",
    "75 g de pollo (aprox. 80 g cocido)",       # el LÍDER manda: 75, no 80
    "110g de atún en agua (1 lata)",
])
def test_los_gramos_LIDER_ganan(linea):
    assert go._paren_grams_in_line(linea) is None


@pytest.mark.parametrize("linea", [
    "1 cebolla", "Sal al gusto", "2½ dientes de ajo",
    "1 taza de repollo (rallado)", "½ naranja (jugo)",
])
def test_sin_masa_declarada_devuelve_None(linea):
    assert go._paren_grams_in_line(linea) is None


def test_fail_safe():
    assert go._paren_grams_in_line(None) is None
    assert go._paren_grams_in_line(12345) is None


# ───────────── 2. el resolvedor la usa ─────────────

def test_el_resolvedor_respeta_el_parentesis(monkeypatch):
    """Sin catálogo real se fuerza la resolución del alimento; lo que se afirma es que la masa
    del paréntesis GANA sobre lo que devuelva el conteo × densidad."""
    monkeypatch.setattr(go, "_phantom_resolve_food", lambda *_a, **_k: ("lechosa", "Lechosa"))
    monkeypatch.setattr(go, "_dup_merge_line_to_grams", lambda *_a, **_k: 1500.0)
    food, g = go._resolve_line_food_grams("1 lechosa mediana (198g)", cheap=True)
    assert food == "lechosa"
    assert g == pytest.approx(198.0), "el paréntesis debe ganar sobre el conteo × densidad"


def test_sin_parentesis_conserva_el_conteo(monkeypatch):
    monkeypatch.setattr(go, "_phantom_resolve_food", lambda *_a, **_k: ("lechosa", "Lechosa"))
    monkeypatch.setattr(go, "_dup_merge_line_to_grams", lambda *_a, **_k: 1500.0)
    _f, g = go._resolve_line_food_grams("1 lechosa mediana", cheap=True)
    assert g == pytest.approx(1500.0)


def test_no_inventa_masa_si_el_alimento_no_resuelve(monkeypatch):
    """Sin alimento no hay par (food, grams) que devolver: el paréntesis solo no basta."""
    monkeypatch.setattr(go, "_phantom_resolve_food", lambda *_a, **_k: None)
    assert go._resolve_line_food_grams("1 xyzzy mediano (198g)", cheap=True) == (None, None)


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(go, "RESOLVE_PAREN_GRAMS", False)
    monkeypatch.setattr(go, "_phantom_resolve_food", lambda *_a, **_k: ("lechosa", "Lechosa"))
    monkeypatch.setattr(go, "_dup_merge_line_to_grams", lambda *_a, **_k: 1500.0)
    _f, g = go._resolve_line_food_grams("1 lechosa mediana (198g)", cheap=True)
    assert g == pytest.approx(1500.0)
