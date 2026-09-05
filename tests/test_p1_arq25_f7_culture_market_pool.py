"""[P1-ARQ25-F7-CULTURE · 2026-09-05 · subfase H] El pool de mercado del sembrador es la ASIGNACIÓN OBLIGATORIA del
día. Medido en la prueba A (mercado US + cocina dominicana 0,7): el pool US tenía 10 carbos (bagels, frijoles horneados,
sin arroz, plátano ni yuca) y contaba ranch/barbacoa/kétchup como «grasas», así que los días dominicanos salieron
como «pollo BBQ sobre frijoles horneados». Ahora los pools beta suman básicos universales, sacan los condimentos
y, cuando la cocina no es la del mercado, se sesgan a los constituyentes del registry de esa cocina.
Opt-in por kwargs: la firma sin kwargs sigue byte-idéntica (los tests de F2 la fijan).
"""
import re
from pathlib import Path

import pytest

import constants

_BACKEND = Path(__file__).resolve().parents[1]


def _names(seq):
    return [(r.get("name") if isinstance(r, dict) else str(r)) for r in (seq or [])]


def test_a_sin_kwargs_el_pool_beta_es_byte_identico():
    p, c, v, f = constants._get_fast_filtered_catalogs((), (), "", country="US")
    assert set(c) == set(constants.COUNTRY_POOLS["US"]["carbs"])
    assert "Aderezo ranch" in v, "sin opt-in nada cambia (contrato de F2)"


def test_b_con_market_extras_el_pool_us_tiene_basicos_y_sin_condimentos(monkeypatch):
    monkeypatch.delenv("MEALFIT_MARKET_POOL_UNIVERSAL", raising=False)
    p, c, v, f = constants._get_fast_filtered_catalogs((), (), "", country="US", market_extras=True)
    for must in ("Arroz blanco", "Plátano verde", "Yuca", "Avena", "Lentejas"):
        assert must in c, must
    for must in ("Aceite de oliva", "Aguacate", "Repollo"):
        assert must in v, must
    for gone in ("Aderezo ranch", "Salsa barbacoa", "Kétchup", "Mostaza", "Sazonador para tacos"):
        assert gone not in v, f"{gone} no es una grasa del día"
    assert "Bagels" in c, "los básicos de US se conservan; el sesgo cultural es lo que los aparta"
    assert len(c) > len(constants.COUNTRY_POOLS["US"]["carbs"])
    # knob apagado ⇒ pool beta tal cual
    monkeypatch.setenv("MEALFIT_MARKET_POOL_UNIVERSAL", "0")
    p2, c2, v2, f2 = constants._get_fast_filtered_catalogs((), (), "", country="US", market_extras=True)
    assert set(c2) == set(constants.COUNTRY_POOLS["US"]["carbs"])


def test_c_mercado_us_con_cocina_dominicana_sesga_el_pool_a_lo_criollo(monkeypatch):
    import dish_registry as dr
    if not Path(dr.snapshot_path("do")).exists():
        pytest.skip("snapshots no compilados")
    monkeypatch.delenv("MEALFIT_MARKET_POOL_UNIVERSAL", raising=False)
    p, c, v, f = constants._get_fast_filtered_catalogs((), (), "", country="US", market_extras=True, culture_country="DO")
    assert "Bagels" not in c and "Frijoles horneados" not in c, c
    for must in ("Arroz blanco", "Plátano verde", "Yuca", "Habichuelas rojas"):
        assert must in c, must
    assert "Pechuga de pollo" in p and "Huevo" in p
    assert len(c) >= 5 and len(v) >= 5 and len(p) >= 5
    # misma cocina que el mercado ⇒ sin sesgo (solo básicos)
    p3, c3, v3, f3 = constants._get_fast_filtered_catalogs((), (), "", country="US", market_extras=True, culture_country="US")
    assert "Bagels" in c3
    # las restricciones siguen aplicando sobre el pool sesgado (vegano sin pollo)
    pv, cv, vv, fv = constants._get_fast_filtered_catalogs((), (), "vegano", country="US", market_extras=True, culture_country="DO")
    assert not [x for x in pv if "pollo" in str(x).lower() or "res" == str(x).lower()]


def test_d_los_tres_call_sites_de_produccion_optan_por_los_basicos():
    ai = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "market_extras=True, culture_country=_variety_culture" in ai
    assert "_variety_culture = _ccffd_variety(form_data)" in ai
    ag = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "market_extras=True, culture_country=_swap_culture" in ag
    cr = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert re.search(r"country=country,\s*\n\s*market_extras=True", cr), "el camino degradado también"


def test_e_los_basicos_universales_existen_en_el_catalogo_y_no_son_condimentos():
    import json
    names = set()
    for lib in ("do", "es", "us", "mx", "co", "pr"):
        import dish_registry as dr
        p = Path(dr.snapshot_path(lib))
        if p.exists():
            snap = json.loads(p.read_text(encoding="utf-8"))
            names |= {c.get("canonical") for t in snap["templates"] for c in t.get("constituents") or []}
    if not names:
        pytest.skip("snapshots no compilados")
    universal = {x for k in constants.UNIVERSAL_MARKET_STAPLES for x in constants.UNIVERSAL_MARKET_STAPLES[k]}
    faltan = sorted(universal - names)
    assert not faltan, f"básicos que ningún plato del registry usa (¿nombre fuera del catálogo?): {faltan}"
    assert not (universal & constants.MARKET_POOL_CONDIMENTS_EXCLUDED)
