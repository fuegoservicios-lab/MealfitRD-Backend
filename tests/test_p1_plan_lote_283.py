# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-283 · 2026-09-25] La guía de compra quincenal/mensual no contradice el tiempo de cocina ni el
congelador, y «Nada» de tiempo no ofrece nada «ya cocido» que alguien tenga que cocinar otro día.

Perfil del dueño (30 días de una vez, sin congelador, «Nada», nunca por tandas): la guía mensual pedía «granos secos,
arroz, tubérculos, proteínas congelables … instrucciones de congelación» y el tope de tiempo ofrecía «pollo ya cocido»;
el modelo lo resolvía con «arroz blanco cocido y refrigerado», «auyama ya cocida», «pescado ya cocido», «garbanzos
secos»."""
from __future__ import annotations

import graph_orchestrator as go
import horizon
from prompts.plan_generator import build_grocery_duration_context

_DUENO = {"groceryDuration": "monthly", "cookingTime": "none", "freezerMode": "none", "batchCooking": "never"}


def test_el_perfil_del_dueno_no_recibe_ordenes_contrarias():
    ctx = build_grocery_duration_context(_DUENO)
    for prohibido in ("granos secos", "tubérculos", "proteínas congelables", "instrucciones de congelación"):
        assert prohibido not in ctx, (prohibido, ctx)
    assert "EN LATA" in ctx and "nunca legumbres secas" in ctx and "«ya cocido»" in ctx, ctx
    assert "SIN CONGELADOR" in ctx and "no escribas «congela»" in ctx, ctx


def test_sin_respuestas_el_texto_de_siempre():
    ctx = build_grocery_duration_context({"groceryDuration": "monthly"})
    assert ("Usa predominantemente: granos secos, arroz, avena, tubérculos (yuca, batata, plátano verde),\n"
            "proteínas congelables (pollo, carne, pescado empacado al vacío), leche en polvo o UHT, huevos.\n"
            "Para cualquier perecedero, incluye instrucciones de congelación en la receta.\n") in ctx
    q = build_grocery_duration_context({"groceryDuration": "biweekly", "freezerMode": "full", "cookingTime": "1hour"})
    assert "Planifica congelación para proteínas frescas.\n" in q
    assert build_grocery_duration_context({"groceryDuration": "weekly", "cookingTime": "none"}) == ""


def test_sin_congelador_con_tiempo():
    ctx = build_grocery_duration_context({"groceryDuration": "monthly", "cookingTime": "30min", "freezerMode": "none"})
    assert "granos secos" in ctx and "SIN CONGELADOR" in ctx
    assert "congelables" not in ctx and "instrucciones de congelación" not in ctx
    q = build_grocery_duration_context({"groceryDuration": "biweekly", "freezerMode": "none"})
    assert "Planifica congelación" not in q and "SIN CONGELADOR" in q


def test_nada_de_tiempo_con_congelador():
    ctx = build_grocery_duration_context({"groceryDuration": "monthly", "cookingTime": "none", "freezerMode": "full"})
    assert "EN LATA" in ctx and "granos secos" not in ctx and "instrucciones de congelación" in ctx
    q = build_grocery_duration_context({"groceryDuration": "biweekly", "cookingTime": "none"})
    assert "EN LATA" in q and "Planifica congelación" in q


def test_el_tope_de_tiempo_no_ofrece_ya_cocido():
    r = horizon.cooking_time_rule({"cookingTime": "none"})
    assert r.startswith("NO TIENE TIEMPO para cocinar: cada comida se prepara en 10 minutos o menos"), r
    assert "`prep_time` de cada comida ≤ 10 min." in r
    assert "pollo ya cocido" not in r and "nunca legumbres secas" in r and "microondas" in r, r
    dias = [{"day": 1, "meals": [{"meal": "Cena", "name": "Guiso de lentejas", "prep_time": "25 min"}]}]
    msgs = go._detect_prep_time_issues(dias, {"cookingTime": "none"})
    assert msgs and "pollo ya cocido" not in msgs[0] and "legumbres de lata" in msgs[0], msgs
