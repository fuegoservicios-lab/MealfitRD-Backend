# -*- coding: utf-8 -*-
"""[ARQ27-P1-02 · 2026-09-06] El embudo medía su propia tabla, no el selector.

`scripts/coverage_funnel.py` responde «¿cuántos platos sobreviven de verdad a los filtros de un
usuario?», y su etapa de dieta usaba `_DIET_EXCLUDES`: una tabla PROPIA de familias de `protein` que
cada dieta excluye. La etiqueta `protein` dice qué protagoniza el plato, jamás qué contiene, así que
una plantilla `none` con lácteos o una `mixta` con jamón pasaban su filtro y el vegano las veía como
elegibles. Sobre ES/almuerzo vegano el recuento caía de 26 a 5 al ejecutar la guarda real.

Un embudo que no usa el filtro del producto no mide el producto: mide su propia tabla, y su número
tranquiliza sobre algo que nadie comprobó. Este archivo existe para que las dos mitades no vuelvan a
divergir en silencio — la paridad es el criterio de cierre del gap, palabra por palabra: «counts de
embudo y selector idénticos para los mismos inputs y versiones».

También ancla lo que el criterio pide además de los contadores: **la lista de IDs y motivos por
caída**. Un contador dice que se perdieron veinte platos; la lista dice cuáles, que es lo único con
lo que se puede actuar.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import dish_registry as DR  # noqa: E402


def _cargar_embudo():
    ruta = _BACKEND / "scripts" / "coverage_funnel.py"
    spec = importlib.util.spec_from_file_location("coverage_funnel", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CF = _cargar_embudo()

PAISES = ["DO", "PR", "MX", "CO", "ES", "US"]
SLOTS = ["desayuno", "almuerzo", "merienda", "cena"]
_DIETAS = {"vegana": "vegan", "vegetariana": "vegetarian", "omnivora": "balanced"}


def _etapa(res, nombre):
    for n, v in res["etapas"]:
        if n == nombre:
            return v
    raise AssertionError(f"el embudo perdió la etapa {nombre!r}: {[n for n, _ in res['etapas']]}")


# ── paridad embudo ↔ selector ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cc", PAISES)
@pytest.mark.parametrize("dieta_es,dieta_canon", list(_DIETAS.items()))
def test_el_embudo_y_el_selector_cuentan_lo_mismo(cc, dieta_es, dieta_canon):
    """El criterio de cierre. Se compara en la etapa de DIETA porque es hasta ahí donde los dos
    aplican los mismos filtros: el embudo añade después mercado y conservación, que el selector
    recibe por separado."""
    for slot in SLOTS:
        emb = CF.embudo(cc, slot, dieta_es, (), None, "limited")
        sel = DR.template_candidates(cc, slot, None, k=999, diet=dieta_canon)
        assert _etapa(emb, "dieta") == len(sel), (
            f"{cc}/{slot}/{dieta_es}: embudo {_etapa(emb, 'dieta')} vs selector {len(sel)}")


def test_la_tabla_propia_de_dieta_ya_no_existe():
    """Si alguien la reintroduce, el embudo vuelve a medirse a sí mismo. Que la paridad de arriba
    falle depende de que los datos la delaten; que la tabla no exista, no."""
    assert not hasattr(CF, "_DIET_EXCLUDES"), (
        "volvió `_DIET_EXCLUDES`: la dieta se resuelve con `_diet_pool_item_banned`, el guard SSOT")
    fuente = (_BACKEND / "scripts" / "coverage_funnel.py").read_text(encoding="utf-8")
    assert "_diet_pool_item_banned" in fuente


@pytest.mark.parametrize("cc", ["DO", "ES"])
def test_la_alergia_tambien_cuenta_igual(cc):
    """La otra mitad del embudo compartida con el selector: `exclude_allergens`."""
    for slot in SLOTS:
        emb = CF.embudo(cc, slot, "omnivora", ("lacteos", "lactosa"), None, "limited")
        sel = DR.template_candidates(cc, slot, None, k=999, exclude_allergens=("lacteos", "lactosa"))
        assert _etapa(emb, "alergias") == len(sel), f"{cc}/{slot}"


# ── IDs y motivos, no solo contadores ─────────────────────────────────────────────────────────
def test_cada_caida_conserva_id_y_nombre():
    emb = CF.embudo("DO", "almuerzo", "vegana", (), None, "limited")
    assert emb["caidas"], "el embudo dejó de decir QUÉ cayó"
    assert "dieta" in emb["caidas"], "ninguna caída atribuida a la dieta en DO/almuerzo vegano"
    for etapa, items in emb["caidas"].items():
        for it in items:
            assert it.get("id") and it.get("name"), f"caída sin identificar en {etapa}: {it}"


def test_las_caidas_cuadran_con_los_contadores():
    """Lo que entra menos lo que cae es lo que queda. Sin esta identidad, la lista podría estar
    contando otra cosa que los contadores y las dos mitades volverían a divergir."""
    for cc in PAISES:
        emb = CF.embudo(cc, "cena", "vegetariana", ("gluten",), None, "limited")
        etapas = emb["etapas"]
        for i in range(1, len(etapas)):
            nombre, quedan = etapas[i]
            antes = etapas[i - 1][1]
            caidos = len(emb["caidas"].get(nombre, []))
            assert antes - caidos == quedan, f"{cc}/{nombre}: {antes} - {caidos} ≠ {quedan}"


# ── etapas que el gap pedía añadir ────────────────────────────────────────────────────────────
def test_el_embudo_tiene_la_etapa_de_mercado():
    emb = CF.embudo("DO", "almuerzo", "omnivora", (), None, "limited")
    nombres = [n for n, _ in emb["etapas"]]
    assert any(n.startswith("mercado") for n in nombres), nombres


def test_sin_catalogo_la_etapa_de_mercado_se_declara_omitida(monkeypatch):
    """No se cuela como un cero. La etapa se nombra «(omitido)» y no descarta nada — la misma
    distinción ausente/vacío que ARQ27-P1-07 hace en la política."""
    monkeypatch.setattr(CF, "_mercado", lambda country: None)
    emb = CF.embudo("DO", "almuerzo", "omnivora", (), None, "limited")
    nombres = [n for n, _ in emb["etapas"]]
    assert "mercado (omitido)" in nombres
    assert _etapa(emb, "mercado (omitido)") == _etapa(emb, "dieta")


def test_la_etapa_de_datos_exigidos_solo_corre_si_se_pide():
    sin = [n for n, _ in CF.embudo("CO", "almuerzo", "omnivora", (), None, "limited")["etapas"]]
    assert "datos exigidos" not in sin
    con = CF.embudo("CO", "almuerzo", "omnivora", (), None, "limited",
                    requiere_nutrientes=("phosphorus_mg",))
    assert "datos exigidos" in [n for n, _ in con["etapas"]]


def test_estan_los_cruces_dificiles_que_el_gap_pedia():
    """«medir sin soja, sin gluten y cruces». Un escenario que no está en la lista no se mide, y lo
    que no se mide es exactamente donde vivían los ceros."""
    etiquetas = {e[0] for e in CF.ESCENARIOS}
    for esperado in ("vegano sin gluten", "vegano sin soja", "vegano sin soja ni gluten"):
        assert esperado in etiquetas, f"falta el cruce «{esperado}»"
    assert "soya" in CF._ALLERGY_SETS and "soya + gluten" in CF._ALLERGY_SETS


def test_un_candidato_del_embudo_es_elegible_no_viable():
    """La palabra importa: el embudo no mide solver, precio ni cuotas. Que su cabecera lo siga
    diciendo es lo que impide que su número se lea como una promesa de entrega."""
    fuente = (_BACKEND / "scripts" / "coverage_funnel.py").read_text(encoding="utf-8")
    assert "elegible" in fuente and "no es servible" in fuente
