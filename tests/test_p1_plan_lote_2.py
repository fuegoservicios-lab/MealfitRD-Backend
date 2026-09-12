"""[P1-PLAN-LOTE-2 · 2026-09-11] Segundo lote del plan de pendientes (`docs/plan_pendientes_2026_09_11.md`).

  B6  Memoria ENTRE días del día determinista: los platos servidos en los 6 días anteriores (los de este
      run y, si el bloque continúa un plan, los ya entregados) van al FINAL de los elegibles cuando agotaron
      el tope de repetición de la política (`balanced` ⇒ 2 por 7 días). Se prefiere, no se descarta.
  D1  «1 sobre» pesa lo que pesa el sobre (`density_g_per_unit`), no la caja (`container_weight_g`).
  D2  «Cebolla en polvo» es su propia fila del catálogo: el chain no la colapsa a la cebolla fresca.
  D3  El tope de condimentos reescribe también `display_string`, la frase que ve el usuario.
  G2  `plan_tier`: 'free' (defecto viejo del esquema) se lee como 'gratis'; migración normaliza y ancla CHECK.
  G3  La purga de cuenta borra la identidad en `neon_auth."user"` (cascade a session/account) y olvida el
      positivo cacheado del guard P1-AUTH-CUENTA-BORRADA.
  F7  Fósforo con fuente citable para Borojó y Hoja santa; las otras tres filas siguen en NULL a sabiendas.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# B6 · memoria entre días
# ---------------------------------------------------------------------------

@pytest.fixture
def dd():
    import deterministic_day
    return deterministic_day


_CAT = {
    "Pollo": {"name": "Pollo", "kcal_per_100g": 165, "protein_g_per_100g": 31, "carbs_g_per_100g": 0,
              "fats_g_per_100g": 3.6, "sodium_mg_per_100g": 74},
    "Arroz": {"name": "Arroz", "kcal_per_100g": 130, "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28,
              "fats_g_per_100g": 0.3, "sodium_mg_per_100g": 1},
}


def _tpl(tid, nombre):
    return {"template_id": tid, "name": nombre, "slots": ["almuerzo"], "protein": "pollo", "status": "ok",
            "constituents": [{"name": "Pollo", "grams": 200}, {"name": "Arroz", "grams": 200}],
            "nutrition_per_serving": {"sodium_mg": 150},
            "logistics": {"prep_minutes_source": "receta", "prep_minutes_est": 40}}


_POR_ID = {f"tpl_{k}": _tpl(f"tpl_{k}", f"Pollo con arroz ({k})") for k in "abc"}
_OBJ = {"kcal": 700, "protein_g": 60, "carbs_g": 60, "fats_g": 15}


def test_b6_los_saturados_van_al_final_y_la_rotacion_gira_sobre_los_frescos(dd):
    base = dd.elegir_plantillas(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo")
    assert [t["template_id"] for t, _ in base] == ["tpl_a", "tpl_b", "tpl_c"], "sin memoria, el orden del score"
    con = dd.elegir_plantillas(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo",
                               saturados={"tpl_a": 2}, max_rep=2)
    assert [t["template_id"] for t, _ in con] == ["tpl_b", "tpl_c", "tpl_a"], "tpl_a agotó su cuota: al final, no fuera"
    rot = dd.elegir_plantillas(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo", rotacion=1,
                               saturados={"tpl_a": 2}, max_rep=2)
    assert [t["template_id"] for t, _ in rot] == ["tpl_c", "tpl_b", "tpl_a"], "la rotación no devuelve al saturado a la cabeza"
    bajo = dd.elegir_plantillas(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo", saturados={"tpl_a": 1}, max_rep=2)
    assert [t["template_id"] for t, _ in bajo][0] == "tpl_a", "una sola vez en la ventana no satura"


def test_b6_todos_saturados_no_deja_sin_plato(dd):
    todos = dd.elegir_plantillas(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo",
                                 saturados={"tpl_a": 5, "tpl_b": 5, "tpl_c": 5}, max_rep=2)
    assert len(todos) == 3, "quedarse sin día por no repetir es peor que repetir"


def test_b6_la_ventana_cuenta_los_ultimos_seis_dias_por_id_y_por_nombre(dd, monkeypatch):
    monkeypatch.setattr(dd, "_dias_previos_persistidos", lambda fd, uid: [])
    memoria = []
    for i in range(8):
        memoria.append({"day": i + 1, "meals": [{"_template_id": "tpl_a"}]})
    memoria.append({"day": 9, "meals": [{"name": "Pollo con arroz (b)"}]})   # sin id: por nombre
    conteo = dd._conteo_ventana(memoria, 9, {}, None, _POR_ID)
    assert conteo == {"tpl_a": 5, "tpl_b": 1}, "6 días de ventana: 5 de tpl_a + el de nombre"
    assert dd._conteo_ventana(None, 3) == {}
    assert dd._conteo_ventana([], 0) == {}


def test_b6_el_bloque_que_continua_un_plan_carga_lo_entregado_una_sola_vez(dd, monkeypatch):
    llamadas = []

    def _previos(fd, uid):
        llamadas.append(uid or (fd or {}).get("user_id"))
        return [{"day": 1, "meals": [{"_template_id": "tpl_c"}]}, {"day": 2, "meals": [{"_template_id": "tpl_c"}]}]
    monkeypatch.setattr(dd, "_dias_previos_persistidos", _previos)
    memoria = []
    c1 = dd._conteo_ventana(memoria, 14, {"user_id": "u1"}, None, _POR_ID)
    c2 = dd._conteo_ventana(memoria, 15, {"user_id": "u1"}, None, _POR_ID)
    assert c1 == {"tpl_c": 2} and c2 == {"tpl_c": 2}
    assert llamadas == ["u1"], "la base se consulta UNA vez por run"
    assert all(d.get("_persistido") for d in memoria)
    # el primer bloque (offset 0) no toca la base
    memoria2 = []
    dd._conteo_ventana(memoria2, 0, {"user_id": "u1"}, None, _POR_ID)
    assert llamadas == ["u1"] and memoria2 == []


def test_b6_sin_memoria_nada_cambia(dd):
    assert dd._max_repeticion_7d({}) == 2
    assert dd._max_repeticion_7d({"_plan_policy_effective": {"recurrence": {"global_mode": "routine"}}}) >= 2
    assert dd._conteo_ventana(None, 5) == {}


def test_b6_build_day_lee_y_actualiza_la_memoria(dd, monkeypatch):
    import dish_registry as dr
    import shopping_calculator as sc
    import recipe_library as rl
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY", "true")
    monkeypatch.setattr(dr, "template_candidates", lambda country, slot, family=None, **kw:
                        [{"template_id": tid, "name": t["name"]} for tid, t in _POR_ID.items()])
    monkeypatch.setattr(dr, "templates_by_id", lambda country: dict(_POR_ID))
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_CAT.values()))
    monkeypatch.setattr(rl, "recipe_for_dish_name", lambda name, country="DO": ["Cuece el pollo.", "Sirve."])
    monkeypatch.setattr(dd, "verifica_comida", lambda meal, fd, cat: [])
    monkeypatch.setattr(dd, "_dias_previos_persistidos", lambda fd, uid: [])
    nut = {"target_calories": "700 kcal", "macros": {"protein": "60g", "carbs": "60g", "fats": "15g"}}
    esqueleto = {"day": 1, "protein_pool": ["Pollo"], "meal_types": ["Almuerzo"]}
    memoria = []
    servidos = []
    for n in range(1, 5):
        dia = dd.build_day_for_skeleton(nut, {"user_id": "u1"}, dict(esqueleto, day=n), n, memoria=memoria)
        assert dia is not None and dia["_day_source"] == "deterministic"
        servidos.append(dia["meals"][0]["_template_id"])
    armados = [d for d in memoria if not d.get("_persistido")]
    assert len(armados) == 4 and [d["day"] for d in armados] == [1, 2, 3, 4], "cada día armado entra en la memoria"
    assert not any(d.get("_persistido") for d in memoria), "primer bloque (offset 0): la base no se toca"
    # con tope 2 por 7 días y tres plantillas, ninguna puede servirse tres veces en cuatro días seguidos
    assert max(servidos.count(t) for t in set(servidos)) <= 2, servidos
    assert len(set(servidos)) >= 2


def test_b6_el_grafo_comparte_la_memoria_entre_los_dias_del_run():
    src = _src("graph_orchestrator.py")
    assert "_det_prev: list = []" in src
    assert "_det_day(nutrition, form_data, skel_day, day_num, memoria=_det_prev) or await _generate_day_hedged" in src
    assert len(src.splitlines()) <= 53100


# ---------------------------------------------------------------------------
# D1 · el sobre pesa lo que pesa el sobre
# ---------------------------------------------------------------------------

def test_d1_to_grams_prefiere_la_densidad_del_sobre():
    from nutrition_db import IngredientNutritionDB, NutritionInfo
    import inspect
    db = IngredientNutritionDB.__new__(IngredientNutritionDB)
    campos = {k: None for k in inspect.signature(NutritionInfo).parameters}
    campos.update({"name": "Sazón con culantro y achiote", "density_g_per_unit": 5.0, "container_weight_g": 40.0})
    info = NutritionInfo(**campos)
    assert db.to_grams(1, "sobre", info) == 5.0
    assert db.to_grams(3, "sobres", info) == 15.0
    campos["density_g_per_unit"] = None
    assert db.to_grams(2, "sobre", NutritionInfo(**campos)) == 80.0, "sin densidad por unidad, el envase como siempre"
    campos["container_weight_g"] = None
    assert db.to_grams(1, "sobre", NutritionInfo(**campos)) is None


def test_d1_el_agregador_normaliza_el_sobre_por_su_densidad():
    src = _src("shopping_calculator.py")
    assert "_SACHET_UNITS = frozenset({'sobre', 'sobres', 'sobrecito', 'sobrecitos'})" in src
    i = src.find("is_container_alias = (u_lower == db_container) or (u_lower in _CONTAINER_UNIT_ALIASES)")
    assert i > 0
    bloque = src[i:i + 900]
    assert "if u_lower in _SACHET_UNITS and _dens_u > 0:" in bloque
    assert "effective_g = _dens_u" in bloque
    assert "_fallback_container_weight_g(master_item.get(\"category\"))" in bloque, "el respaldo por categoría sigue ahí"


# ---------------------------------------------------------------------------
# D2 · un polvo no compra su alimento fresco
# ---------------------------------------------------------------------------

def test_d2_cebolla_en_polvo_no_es_cebolla():
    import shopping_calculator as sc
    assert sc.canonicalize_cebolla("Cebolla en polvo") is None
    assert sc.canonicalize_cebolla("cebolla deshidratada") is None
    assert sc.canonicalize_cebolla("cebolla roja") == "Cebolla", "las variedades frescas siguen fundiéndose"
    assert sc.canonicalize_cebolla("cebollín") == "Cebollín"
    mapa = {"Cebolla en polvo": {"name": "Cebolla en polvo"}, "Cebolla": {"name": "Cebolla"}}
    assert sc.canonicalize_shopping_food_name("Cebolla en polvo", mapa) == "Cebolla en polvo"
    assert sc.canonicalize_shopping_food_name("cebolla morada", mapa) == "Cebolla"
    assert sc.canonicalize_shopping_food_name("Ajo en polvo", {"Ajo en polvo": {"name": "Ajo en polvo"}}) == "Ajo en polvo"


# ---------------------------------------------------------------------------
# D3 · la frase que ve el usuario dice la cantidad capada
# ---------------------------------------------------------------------------

def test_d3_el_tope_reescribe_display_string(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_shoplist_sanity_cap_enabled", lambda: True)
    tope = sc._condiment_package_cap(7)
    qty = float(tope + 2)
    obj = {"name": "Orégano", "market_qty_numeric": qty, "market_qty": str(int(qty)), "market_unit": "frasco",
           "display_qty": f"{int(qty)} frascos", "display_string": f"{int(qty)} frascos de Orégano",
           "estimated_cost_rd": 90.0 * qty}
    assert sc._apply_condiment_sanity_cap(obj, {"container_weight_g": 90}, "Despensa", 7) is True
    unidad = "frasco" if tope == 1 else "frascos"
    assert obj["display_qty"] == f"{tope} {unidad}"
    assert obj["display_string"] == f"{tope} {unidad} de Orégano"
    assert obj["estimated_cost_rd"] == pytest.approx(90.0 * tope)
    # display_string con otra forma: se sustituye el número inicial y el plural cuando queda en 1
    obj2 = {"name": "Orégano", "market_qty_numeric": qty, "market_qty": str(int(qty)), "market_unit": "frasco",
            "display_qty": "otra cosa", "display_string": f"{int(qty)} frascos (90 g) de Orégano"}
    assert sc._apply_condiment_sanity_cap(obj2, {"container_weight_g": 90}, "Despensa", 7) is True
    assert obj2["display_string"].startswith(f"{tope} {unidad} (90 g) de Orégano")


# ---------------------------------------------------------------------------
# G2 · plan_tier canónico
# ---------------------------------------------------------------------------

def test_g2_free_se_lee_como_gratis(monkeypatch):
    import llm_provider as lp
    import db
    monkeypatch.setattr(db, "get_user_plan_tier", lambda uid: "free")
    lp._TIER_CACHE.clear()
    assert lp.get_user_tier("11111111-1111-1111-1111-111111111111") == "gratis"
    lp._TIER_CACHE.clear()
    monkeypatch.setattr(db, "get_user_plan_tier", lambda uid: "plus")
    assert lp.get_user_tier("22222222-2222-2222-2222-222222222222") == "plus"
    lp._TIER_CACHE.clear()


def test_g2_migracion_normaliza_default_y_ancla_check():
    a = (_BACKEND / "migrations" / "p1_plan_tier_gratis_2026_09_11.sql").read_text(encoding="utf-8")
    b = (_ROOT / "migrations" / "p1_plan_tier_gratis_2026_09_11.sql")
    if b.exists():
        assert b.read_text(encoding="utf-8") == a, "SSOT dual-dir: las dos copias deben ser idénticas"
    assert "SET plan_tier = 'gratis' WHERE plan_tier = 'free'" in a
    assert "SET DEFAULT 'gratis'" in a
    assert "user_profiles_plan_tier_canonical" in a and "DROP CONSTRAINT IF EXISTS" in a
    assert "RAISE EXCEPTION" in a, "sanity antes de la CHECK"


# ---------------------------------------------------------------------------
# G3 · la purga borra la identidad
# ---------------------------------------------------------------------------

def test_g3_la_purga_borra_en_neon_auth_y_olvida_el_positivo_cacheado():
    src = _src("routers/system.py")
    assert "_storage_client" not in src, "el cliente legacy (siempre None) ya no decide el borrado de auth"
    assert 'DELETE FROM neon_auth."user" WHERE id = %s RETURNING id' in src
    assert "forget_auth_row_alive(body.user_id)" in src
    import db_profiles as dp
    dp._AUTH_ROW_ALIVE_IDS.add("zombi")
    dp.forget_auth_row_alive("zombi")
    assert "zombi" not in dp._AUTH_ROW_ALIVE_IDS
    dp.forget_auth_row_alive(None)   # nunca revienta


# ---------------------------------------------------------------------------
# F7 · fósforo con fuente
# ---------------------------------------------------------------------------

def test_f7_fosforo_solo_con_fuente_citable():
    a = (_BACKEND / "migrations" / "p1_fosforo_borojo_hoja_santa_2026_09_11.sql").read_text(encoding="utf-8")
    assert "name = 'Borojó' AND phosphorus_mg_per_100g IS NULL" in a and "160.0" in a
    assert "name = 'Hoja santa' AND phosphorus_mg_per_100g IS NULL" in a and "38.0" in a
    for sin_dato in ("Achiote", "Chontaduro", "Champús"):
        assert f"name = '{sin_dato}'" not in a, f"{sin_dato} sigue en NULL a sabiendas: no hay valor citable"
    assert "nutrition_source_ref" in a


def test_marker_bumpeado():
    """El marker avanza con cada lote; lo anclado es el comentario del lote en `app.py`."""
    import app
    assert "[P1-PLAN-LOTE-2 · 2026-09-11]" in _src("app.py")
    # [P1-PLAN-LOTE-13 · 2026-09-12] «no anterior a este lote», no «igual a hoy»: el pin de la fecha y del prefijo
    # `P1-PLAN-` rompía 12 tests el primer día en que otro P-fix bumpeaba el marker.
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-11", app._LAST_KNOWN_PFIX
