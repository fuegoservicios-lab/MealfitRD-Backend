# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-221 · 2026-09-24] La compra única cubre el mes también en lo que tenía tope.

El dueño: «si elijo 30 días y quiero hacer la compra de golpe, debería darme alimentos para comprar una sola vez
durante todo el mes». Su plan real de 30 días, recalculado con la proyección del lote 215, seguía pidiendo cuatro
segundas idas: cebolla 2,55 de 4,9 kg («alcanza ~15 de 30 días»), batata 22/30, habichuelas secas 19/30 y leche de
soya 25/30. Eran los topes de realismo de mayo (P5-VEG-CAP, P6-LEGUMES-DRY-CAP, P6-LACTEOS-PERISHABLE-CAP), nacidos
contra listas mensuales EXTRAPOLADAS; en una compra única la lista es la Σ exacta del mes y el tope sólo deja al
usuario sin comida. Ahora lo que aguanta el ciclo (vida útil del catálogo ≥ días del ciclo) sale sin tope.

Y sin congelador, el fresco dice para cuántos días es: el filete de 32 oz salía «alcanza ~14 días» (su vida
CONGELADO) a un usuario que no congela. La Nevera virtual de los bloques 2+ (lote 216) ofrece sólo lo que llega a sus
días. Con «Nada» de tiempo, el relleno del piso de ganancia muscular es casabe (listo), no batata ni arroz por cocer;
y V7f deja de llamar «sin cocción» a la auyama del microondas «hasta que esté tierna» y al maduro dorado — el
reparador del lote 68 les añadía una SEGUNDA cocción (3 de 3 disparos de la batería real eran falsos).
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import shopping_calculator as sc  # noqa: E402

_MASTER = {
    "Cebolla": {"name": "Cebolla", "category": "Vegetales", "shelf_life_days": 60, "density_g_per_unit": 150,
                "default_unit": "unidad"},
    "Tomate": {"name": "Tomate", "category": "Vegetales", "shelf_life_days": 7, "density_g_per_unit": 100,
               "default_unit": "unidad"},
}


def _agregar(compra_unica: bool, monkeypatch):
    monkeypatch.setattr(sc, "_build_shopping_master_map", lambda *a, **k: dict(_MASTER))
    lineas = [f"{150 * 30} g de Cebolla", f"{100 * 30} g de Tomate"]
    return sc.aggregate_and_deduct_shopping_list(lineas, [], structured=True, multiplier=1.0, num_days=30,
                                                  cycle_days=30, compra_unica=compra_unica)


def _gramos(res, nombre):
    for it in res or []:
        if isinstance(it, dict) and str(it.get("name")) == nombre:
            return it
    return None


def test_compra_unica_el_duradero_no_se_recorta(monkeypatch):
    sc._CAPS_APPLIED_LAST_RUN.clear()
    res = _agregar(True, monkeypatch)
    razones = {(c.get("food"), c.get("reason")) for c in sc._CAPS_APPLIED_LAST_RUN}
    assert ("Cebolla", "P5-VEG-CAP") not in razones, razones      # vida 60 ≥ 30: sin tope
    assert ("Tomate", "P5-VEG-CAP") in razones, razones           # vida 7: el tope sigue
    assert _gramos(res, "Cebolla") is not None


def test_sin_compra_unica_el_tope_sigue(monkeypatch):
    sc._CAPS_APPLIED_LAST_RUN.clear()
    _agregar(False, monkeypatch)
    assert any(c.get("food") == "Cebolla" and c.get("reason") == "P5-VEG-CAP" for c in sc._CAPS_APPLIED_LAST_RUN)


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_SINGLE_TRIP_DURABLES_UNCAPPED", "false")
    sc._CAPS_APPLIED_LAST_RUN.clear()
    _agregar(True, monkeypatch)
    assert any(c.get("food") == "Cebolla" and c.get("reason") == "P5-VEG-CAP" for c in sc._CAPS_APPLIED_LAST_RUN)


def test_los_duraderos_se_apartan_antes_de_los_topes_y_vuelven_despues():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    i = src.index("tooltip-anchor: P1-PLAN-LOTE-221-DURADEROS-SIN-TOPE")
    assert i < src.index("# [P3-HERB-CAP] Cap defensivo de hierbas frescas")
    j = src.index("# [P1-PLAN-LOTE-221] los duraderos de la compra única vuelven")
    assert src.index('_record_cap_applied(_name, _old, _units[_unit_key], "P6-BROTHS-CAP")') < j
    assert j < src.index("[2026-05-06 PROTEIN-UNIT-FALLBACK]")


def test_sin_congelador_el_fresco_dice_para_cuantos_dias(monkeypatch):
    item = {"name": "Filete de pescado blanco", "category": "Proteínas", "is_perishable": True,
            "shelf_life_days": 14, "pkg_cover_ratio": 5.538, "_compra_unica": 30,
            "_compra_unica_sin_congelador": True, "display_qty": "1 paquete (32 Oz · Genérico)"}
    out = sc._build_hybrid_shopping_list([dict(item)], [dict(item)])
    dq = out[0]["display_qty"]
    assert dq == "1 paquete (32 Oz · Genérico) · para los primeros 3 días — sin congelador no aguanta más", dq
    # con congelador (sin la marca), la nota de siempre
    item.pop("_compra_unica_sin_congelador")
    out = sc._build_hybrid_shopping_list([dict(item)], [dict(item)])
    assert "alcanza ~14 días" in out[0]["display_qty"]


def test_el_sello_lleva_el_congelador(monkeypatch):
    import compra_unica as cu
    monkeypatch.setattr(cu, "ciclo_de", lambda _pd: 30)
    plan = {"_plan_policy": {"effective": {"shopping": {"freezer_mode": "none", "main_cycle_days": 30}}}}
    res = cu.sellar_lista([{"name": "Pollo"}], plan)
    assert res[0]["_compra_unica"] == 30 and res[0]["_compra_unica_sin_congelador"] is True
    plan["_plan_policy"]["effective"]["shopping"]["freezer_mode"] = "limited"
    res = cu.sellar_lista([{"name": "Pollo"}], plan)
    assert "_compra_unica_sin_congelador" not in res[0]


# ─────────────────────────────── la Nevera virtual del bloque: sólo lo que llega a sus días

def test_la_nevera_virtual_no_ofrece_lo_que_ya_no_aguanta():
    import compra_unica as cu
    single = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"},
              "diet": {"type": "balanced", "allergies": []}}
    lista = [{"name": n, "_compra_unica": 30} for n in (
        "Atún en agua", "Sardinas en lata", "Garbanzos", "Huevo", "Arroz blanco", "Casabe", "Cebolla",
        "Pechuga de pollo", "Filete de pescado blanco", "Yogurt")]

    def _q(sql, params=(), **k):
        return {"activa": lista, "mensual": lista}

    def _nevera(dia):
        fd = {"_plan_policy_effective": single, "_days_offset": dia, "current_pantry_ingredients": []}
        return cu.nevera_virtual(fd, task_id=7, user_id="u1", consultar=_q).get("current_pantry_ingredients") or []

    bloque_2, bloque_6 = _nevera(2), _nevera(21)
    assert "Pechuga de pollo" in bloque_2 and "Filete de pescado blanco" in bloque_2   # días 1-3: aún frescos
    for fresco in ("Pechuga de pollo", "Filete de pescado blanco", "Yogurt"):
        assert fresco not in bloque_6, bloque_6                                        # día 22: ya no
    assert {"Atún en agua", "Huevo", "Arroz blanco", "Casabe"} <= set(bloque_6)


# ─────────────────────────────── el piso kcal de ganancia muscular con «Nada» de tiempo: casabe, no batata ni arroz

_NUT = {"macros": {"protein_g": 135, "carbs_g": 334, "fats_g": 69}}   # 2497 kcal


def _meal(slot, name, cals, carbs, prot, fats, ingredients=None):
    ings = list(ingredients or [f"{prot}g de proteina"])
    return {"meal": slot, "name": name, "cals": cals, "carbs": carbs, "protein": prot, "fats": fats,
            "ingredients": ings, "ingredients_raw": list(ings),
            "recipe": ["Mise en place: prepara.", "El Toque de Fuego: cocina.", "Montaje: sirve."]}


def test_sin_tiempo_el_piso_se_rellena_con_casabe(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_PCT", 0.95)
    fd = {"mainGoal": "gain_muscle", "cookingTime": "none"}
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Wrap de pollo", 600, 60, 45, 15, ["150 g de pollo", "1 tortilla integral"]),
        _meal("Cena", "Tortilla de sardinas", 450, 30, 30, 12, ["1 lata de sardinas", "2 huevos"]),
    ]}]
    assert go._repair_gainmuscle_day_kcal(days, _NUT, fd) > 0
    todas = [i for m in days[0]["meals"] for i in m["ingredients"]]
    pasos = [s for m in days[0]["meals"] for s in m["recipe"]]
    assert any(go._GM_READY_LINE_RE.match(i) for i in todas), todas
    assert not any("arroz" in i.lower() or "batata" in i.lower() for i in todas), todas
    assert not any("Cuece" in s for s in pasos), pasos
    assert any(s.startswith("🫓 Acompaña con el casabe") for s in pasos), pasos
    assert all(int(i.split()[0]) <= go._GM_READY_MAX_G for i in todas if go._GM_READY_LINE_RE.match(i))


def test_sin_tiempo_un_plato_con_su_casabe_no_recibe_otro(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    fd = {"mainGoal": "gain_muscle", "cookingTime": "none"}
    days = [{"day": 1, "meals": [
        _meal("Cena", "Sardinas con casabe", 450, 30, 30, 12, ["1 lata de sardinas", "2 tortas de casabe"])]}]
    assert go._repair_gainmuscle_day_kcal(days, _NUT, fd) == 0
    assert days[0]["meals"][0]["ingredients"] == ["1 lata de sardinas", "2 tortas de casabe"]


def test_el_tiempo_de_cocina_viaja_con_el_plan_hasta_la_pasada_final():
    """El escudo pre-INSERT y los bloques 2+ no tienen el formulario: su pasada final del piso volvía a poner arroz."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'result["_cooking_time"] = str(form_data.get("cookingTime")).strip().lower()' in src
    assert "cooking_time=None) -> tuple:" in src
    assert '{"mainGoal": main_goal, "cookingTime": cooking_time}' in src
    dbp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert 'target_macros=_tm_ins, cooking_time=_ct_ins)' in dbp
    assert '_ct_ins = _pd.get("_cooking_time") or' in dbp


def test_la_pasada_final_del_escudo_usa_casabe(monkeypatch):
    import graph_orchestrator as go
    capturado = {}

    def _fake(days, nutrition, form_data, db=None, **kw):
        capturado.update(form_data)
        return 0
    monkeypatch.setattr(go, "_repair_gainmuscle_day_kcal", _fake)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_FLOOR_FINAL_REFILL", True)
    days = [{"day": 1, "meals": [_meal("Almuerzo", "Wrap de pollo", 600, 60, 45, 15)]}]
    go.finalize_plan_data_coherence(days, main_goal="Ganancia muscular",
                                    target_macros={"protein_g": 135, "carbs_g": 334, "fats_g": 69},
                                    cooking_time="none")
    assert capturado.get("cookingTime") == "none", capturado


def test_con_tiempo_sigue_la_guarnicion_de_siempre(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    days = [{"day": 1, "meals": [_meal("Almuerzo", "Pollo guisado con habichuelas", 700, 60, 50, 20)]}]
    assert go._repair_gainmuscle_day_kcal(days, _NUT, {"mainGoal": "gain_muscle", "cookingTime": "30min"}) > 0
    assert any("arroz blanco cocido" in i for i in days[0]["meals"][0]["ingredients"])


# ─────────────────────────────── V7f: los tres falsos «sin cocción» de la batería real «Nada» de tiempo

def _sin_coccion(ingredientes, pasos):
    import json
    import culinary_coherence as cc
    filas = json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                       .read_text(encoding="utf-8"))["catalogo_filas"]
    idx = cc.build_culinary_index(filas)
    return [f for f, _ in cc.alimentos_sin_coccion({"ingredients": ingredientes, "recipe": pasos}, idx)]


def test_auyama_al_microondas_hasta_tierna_esta_cocida():
    assert _sin_coccion(["120 g de auyama"], [
        "Mise en place: corta la auyama en cubos.",
        "El Toque de Fuego: coloca la auyama en un recipiente apto para microondas con 1 cucharada de agua y "
        "caliéntala 4-5 min hasta que esté tierna.",
        "Montaje: sirve la auyama con el queso."]) == []


def test_maduro_dorado_esta_cocido():
    assert _sin_coccion(["65 g de plátano maduro", "3 huevos"], [
        "Mise en place: pela el plátano maduro y córtalo en rodajas finas.",
        "El Toque de Fuego: calienta el aceite, dora las rodajas de plátano maduro 2 min por lado, bátelas con los "
        "huevos y cuaja la tortilla 2-3 min.",
        "Montaje: sirve la tortilla."]) == []
    assert _sin_coccion(["120 g de plátano maduro"], [
        "El Toque de Fuego: dora las láminas de plátano maduro 3 minutos por lado hasta que estén tibias y "
        "caramelizadas por fuera y suaves por dentro.",
        "Montaje: sirve con queso."]) == []


def test_la_yuca_dorada_4_minutos_sigue_sin_cocer():
    """La regla del lote 62 sigue en pie para lo duro: dorar la yuca 4 minutos no la cuece."""
    assert _sin_coccion(["200 g de yuca"], ["Calienta la yuca en la plancha 4 minutos, girándola para dorarla.",
                                             "Sirve la yuca con cebolla."]) == ["Yuca"]


def test_la_concordancia_no_estropea_un_sustantivo_que_no_conoce():
    import humanize_ingredients as hi
    assert hi._fix_display_grammar("2 tortas pequeñas de casabe") == "2 tortas pequeñas de casabe"
    assert hi._fix_display_grammar("3½ tortas pequeñas de casabe") == "3½ tortas pequeñas de casabe"
    assert hi._fix_display_grammar("2 tortas pequeño de casabe") == "2 tortas pequeñas de casabe"   # repara lo viejo
    # lo conocido sigue igual
    assert hi._fix_display_grammar("2 guineos mediano") == "2 guineos medianos"
    assert hi._fix_display_grammar("1 Lechosa mediano") == "1 lechosa mediana"
    assert hi._fix_display_grammar("2½ tomate picado") == "2½ tomates picados"


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 221
