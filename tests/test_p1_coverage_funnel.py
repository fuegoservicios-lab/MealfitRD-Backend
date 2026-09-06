# -*- coding: utf-8 -*-
"""[P1-COVERAGE-FUNNEL · 2026-09-05] Trabajo 4 del plan corto de la 2.6: medir cuántos platos sobreviven de
verdad a los filtros de un usuario, para poder distinguir «faltan platos» de «se eligen mal».

Este test vigila el INSTRUMENTO, no las cifras. Si mañana hay más plantillas los números cambian y sigue verde;
si el guion vuelve a medir de mentira, cae. Las dos formas de medir de mentira ya ocurrieron mientras se
escribía, y las dos daban una tabla perfectamente creíble:

  · leer el mercado de `country` (que no existe en el perfil; se llama `market_default`) hacía que las SEIS
    bibliotecas cargaran el snapshot por defecto y la tabla saliera seis veces idéntica;
  · filtrar alérgenos por `dairy` cuando el registry los escribe `lacteos`/`lactosa` dejaba el filtro inerte y
    las columnas de «sin lácteos» salían iguales a las de «sin alergias».
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SCRIPT = _BACKEND / "scripts" / "coverage_funnel.py"
_SNAPSHOTS = (_BACKEND / "data" / "registry" / "dish_registry_es_v1.json").exists()

pytestmark = pytest.mark.skipif(not _SNAPSHOTS, reason="snapshots no compilados")


def _modulo():
    spec = importlib.util.spec_from_file_location("coverage_funnel", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def res():
    return _modulo().correr()


def test_mide_las_seis_bibliotecas_y_no_seis_veces_la_misma(res):
    """La trampa que ya cayó: sin mercado, las seis cargan el snapshot por defecto y la tabla es idéntica."""
    perfiles = res["perfiles"]
    assert len(perfiles) == 6, perfiles.keys()
    mercados = {p["country"] for p in perfiles.values()}
    assert None not in mercados, "un perfil sin mercado mide la biblioteca equivocada"
    assert len(mercados) == 6, f"seis bibliotecas, seis mercados distintos: {mercados}"
    bases = {pid: p["escenarios"]["base"]["almuerzo"]["elegibles"] for pid, p in perfiles.items()}
    assert len(set(bases.values())) > 1, f"seis cifras idénticas es la firma de medir el mismo snapshot: {bases}"


def test_los_alergenos_filtran_de_verdad(res):
    """«sin lácteos» tiene que quitar platos. Si empata con «sin alergias», el filtro está inerte."""
    for pid, p in res["perfiles"].items():
        con = p["escenarios"]["vegetariano"]["desayuno"]["elegibles"]
        sin = p["escenarios"]["veg + sin lácteos"]["desayuno"]["elegibles"]
        assert sin <= con, pid
    quita_algo = any(p["escenarios"]["veg + sin lácteos"]["desayuno"]["elegibles"]
                     < p["escenarios"]["vegetariano"]["desayuno"]["elegibles"]
                     for p in res["perfiles"].values())
    assert quita_algo, "en ninguna biblioteca quitar los lácteos quita un desayuno: el filtro no filtra"


def test_el_embudo_solo_puede_estrecharse(res):
    """Cada etapa es un filtro: ninguna puede devolver más candidatos que la anterior."""
    for pid, p in res["perfiles"].items():
        for etiqueta, por_slot in p["escenarios"].items():
            for slot, d in por_slot.items():
                ns = [n for _, n in d["etapas"]]
                assert ns == sorted(ns, reverse=True), f"{pid}/{etiqueta}/{slot}: {d['etapas']}"
                assert d["elegibles"] == ns[-1]


def test_la_dieta_ordena_como_se_espera(res):
    """Vegano ⊆ vegetariano ⊆ base: si se invierte, el filtro de dieta mira la columna equivocada."""
    for pid, p in res["perfiles"].items():
        for slot in ("desayuno", "almuerzo", "merienda", "cena"):
            base = p["escenarios"]["base"][slot]["elegibles"]
            veg = p["escenarios"]["vegetariano"][slot]["elegibles"]
            vegano = p["escenarios"]["vegano"][slot]["elegibles"]
            assert vegano <= veg <= base, f"{pid}/{slot}: base={base} veg={veg} vegano={vegano}"


def test_la_conservacion_estrecha_el_dia_25(res):
    """Sin congelador, el día 25 tiene que dejar menos platos que el día sin restricción."""
    for pid, p in res["perfiles"].items():
        base = p["escenarios"]["base"]["cena"]["elegibles"]
        d25 = p["escenarios"]["día 25 sin congelador"]["cena"]["elegibles"]
        assert d25 < base, f"{pid}: el día 25 sin congelador no quita nada ({d25} vs {base})"


def test_el_documento_existe_y_dice_lo_que_no_mide():
    doc = (_BACKEND / "docs" / "coverage_funnel.md")
    assert doc.exists()
    txt = doc.read_text(encoding="utf-8")
    assert "no necesariamente servible" in txt, "el doc distingue elegible de servible"
    assert "market_default" in txt and "lacteos" in txt, "las dos trampas quedan escritas"
