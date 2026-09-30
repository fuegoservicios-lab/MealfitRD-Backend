"""[P1-PLAN-LOTE-950 · 2026-09-30] El coach sabe qué comidas de hoy faltan y cuáles ya pasaron; plato mixto → foto.

El dueño, 29-sep: a las 20:01, 22:27 y 22:42 (RD) el coach cerró tres veces con «Cuando almuerces, cuéntame qué
comiste» (copiaba su coletilla anterior). Pidió: «no me mandaste tu almuerzo de hoy; si comiste, mándame foto y procedo
a contabilizar las macros», y «si explico que me comí algo complejo y no lo detallo, que diga: mándame foto».
"""
from __future__ import annotations

import coach_day_context as cdc

_DESAYUNO = [{"meal_type": "desayuno", "calories": 490, "protein": 26}]


def test_de_noche_el_almuerzo_ya_paso_y_toca_la_cena():
    t = cdc.estado_de_las_comidas(_DESAYUNO, 22.45)
    assert "desayuno — anotado" in t and "almuerzo — SIN anotar y su hora ya pasó" in t
    assert "cena — sin anotar, es la que toca ahora" in t and "que toca AHORA (cena)" in t
    assert "«No me mandaste tu almuerzo de hoy; si lo comiste, mándame una foto y lo contabilizo»" in t
    assert "NUNCA copies el cierre de tu respuesta anterior" in t


def test_a_mediodia_el_almuerzo_es_el_que_toca_y_no_se_regana_por_nada():
    t = cdc.estado_de_las_comidas(_DESAYUNO, 13.5)
    assert "almuerzo — sin anotar, es la que toca ahora" in t and "(almuerzo)" in t
    assert "No me mandaste" not in t


def test_con_la_cena_anotada_no_se_invita_a_comer_ni_se_confunde_con_el_almuerzo():
    t = cdc.estado_de_las_comidas(_DESAYUNO + [{"meal_type": "cena"}], 22.45)
    assert "no le toca ninguna comida" in t and "toca AHORA (" not in t


def test_la_merienda_no_se_reclama_y_los_nombres_en_ingles_cuentan():
    t = cdc.estado_de_las_comidas([{"meal_type": "breakfast"}, {"meal_type": "lunch"}], 19.0)
    assert "desayuno — anotado" in t and "almuerzo — anotado" in t
    assert "merienda — SIN anotar y su hora ya pasó" in t and "No me mandaste" not in t


def test_sin_hora_madrugada_o_turno_nocturno_no_afirma_nada():
    assert cdc.estado_de_las_comidas(_DESAYUNO, None) == ""
    assert cdc.estado_de_las_comidas(_DESAYUNO, 2.0) == ""
    assert cdc.estado_de_las_comidas(_DESAYUNO, 22.0, "night_shift") == ""


def test_va_dentro_del_bloque_de_lo_que_falta_hoy():
    form = {"dailyCalories": 2500, "proteinTarget": 134}
    plan = {"calories": 2500, "macros": {"protein": "134g", "carbs": "300g", "fats": "80g"}}
    t = cdc.build_day_gap_context(form, plan, _DESAYUNO, 22.45)
    assert "LO QUE LE FALTA HOY" in t and "COMIDAS DE HOY" in t and "(cena)" in t


def test_la_regla_del_plato_mixto_esta_en_los_dos_prompts_y_en_voz():
    from prompts import chat_agent as ca
    for f in (ca.build_tools_instructions, ca.build_tools_instructions_stream):
        assert "PLATO MIXTO SIN DETALLE → FOTO" in f("u1")
    assert "Mándame una foto del plato para ser más preciso" in ca._plato_mixto_bullet()
    src = open(ca.__file__, encoding="utf-8").read()
    assert "pídele que te mande una foto del plato cuando pueda para ser más preciso" in src
    assert "(salvo el PLATO MIXTO SIN DETALLE: ahí se pide la foto)" in src
