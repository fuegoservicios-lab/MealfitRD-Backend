# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-132 · 2026-09-20] El coach resuelve lo que el usuario NECESITA.

El encargo del dueño: «quiero llevar el agente IA chat al 100 % de inteligencia… imagínate que al usuario le falta proteína
y calorías en el día y son las 9 de la noche, y le dice al agente "tengo proteína en mi casa": el agente debe pedir pruebas
para registrar (¿déjame ver la marca de la proteína?), y si lo ve mal por la hora debe dar la recomendación… que el usuario
pueda decirle "dame una receta para comer hoy" o "dame una comida para el desayuno" y el agente resuelva de manera 100 %
eficiente de acuerdo a lo que el usuario necesita».

Lo que había: las kcal consumidas y las macros «(meta N g)» — la RESTA la hacía el modelo, y sin diario no recibía meta
ninguna; una receta pedida en el chat salía de memoria, con macros inventadas y sin mirar la Nevera; y la foto de un pote de
proteína caía en «alimentos sueltos» y el chat ofrecía meterla a la Nevera.

Cuatro piezas:
  1. `coach_day_context.build_day_gap_context` — «LO QUE LE FALTA HOY»: la resta hecha, la hora, y cómo se cierra un día
     a ESA hora. De noche calibrado contra la evidencia: una batida a las 9 pm NO es una deshora; lo que pesa es lo
     pesado/frito, la cafeína y el líquido justo antes de acostarse. Un diario casi vacío de noche = falta REGISTRAR.
  2. Tool `proponer_comida` — reutiliza el día determinista (candidatos del registro, escala a macros con la tabla de
     alimentos, escáner culinario + backstop clínico) para UNA comida, con la receta escrita y la cobertura de la Nevera.
  3. Reglas Q-T del prompt (compartidas por las 4 constantes): comida a pedido, usar las cifras, PRUEBA antes de anotar
     un producto de proteína de envase (única excepción a «pasado = registra»), y la hora sin mitos.
  4. Visión: cuarto tipo `etiqueta` — la tabla nutricional se LEE, no se estima.

El humo SIN IA del lote (`scripts/coach_battery/propuesta_smoke.py`) cazó tres fallos antes de gastar un céntimo: las
metas del contador usan `protein_g` y no `protein` (salía «meta 2050 kcal» sin proteína); V3 acusa a todo ingrediente
que ningún paso menciona, y sin receta adjunta ANTES de verificar morían los 42 candidatos; y verificar 40 platos costaba
4-13 s por llamada (ahora se verifica solo lo que se devuelve: ~1,5 s)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REG = _BACKEND / "data" / "registry" / "dish_registry_do_v1.json"

FD = {"weight": 134, "weightUnit": "lb", "height": 173, "age": 21, "gender": "male", "mainGoal": "gain_muscle",
      "activityLevel": "moderate"}
PLAN = {"calories": 2000, "macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}}
DIARIO = [{"meal_type": "desayuno", "meal_name": "Mangú", "calories": 500, "protein": 25, "carbs": 60, "healthy_fats": 18},
          {"meal_type": "almuerzo", "meal_name": "Arroz con pollo", "calories": 700, "protein": 40, "carbs": 85, "healthy_fats": 18}]


@pytest.fixture(scope="module")
def cdc():
    import coach_day_context
    return coach_day_context


# ───────────────────────── 1. Lo que falta hoy ─────────────────────────

def test_las_metas_aceptan_las_dos_formas_de_clave(cdc):
    """El plan guarda `protein: "150g"`; `get_nutrition_targets`, `protein_g: 134`. El humo del lote dio «meta 2050
    kcal» SIN proteína porque solo se leía la primera."""
    m = cdc.metas_del_dia({}, PLAN)
    assert (m["kcal"], m["protein_g"], m["carbs_g"], m["fats_g"]) == (2000, 150, 200, 60)
    m2 = cdc.metas_del_dia({}, {"calories": 2050, "macros": {"protein_g": 134, "carbs_g": 251, "fats_g": 57}})
    assert m2["protein_g"] == 134 and m2["fats_g"] == 57


def test_sin_datos_reales_no_hay_metas_ni_bloque(cdc):
    """Un invitado sin formulario no recibe una meta inventada (la guarda de `_daily_goal_context`)."""
    assert cdc.metas_del_dia({}, None) is None
    assert cdc.metas_del_dia({"weight": 150}, None) is None
    assert cdc.build_day_gap_context({}, None, [], 21.0) == ""


def test_la_resta_viene_hecha(cdc):
    out = cdc.build_day_gap_context({}, PLAN, DIARIO, 14.5)
    assert "le faltan ~800 kcal" in out and "~85 g de proteína" in out
    assert "Son las 14:30" in out
    assert "🌙" not in out, "a las 2:30 pm no hay guía de noche"


def test_de_noche_la_guia_esta_calibrada_no_prohibe_la_proteina(cdc):
    out = cdc.build_day_gap_context({}, PLAN, DIARIO, 21.08)
    assert "🌙 CÓMO SE CIERRA EL DÍA A ESTA HORA" in out
    assert "NO es una deshora" in out and "no la prohíbas" in out
    assert "cafeína" in out and "pre-entrenos" in out
    assert "~600 kcal" in out, "el cierre razonable es el 30 % de la meta (2000 → 600)"
    assert "mañana se sigue normal" in out


def test_turno_nocturno_no_recibe_la_guia_de_noche(cdc):
    assert cdc.momento_del_dia(21.0, "night_shift") == "turno_nocturno"
    assert "🌙" not in cdc.build_day_gap_context({}, PLAN, DIARIO, 21.0, "night_shift")


def test_un_diario_corto_de_noche_es_falta_de_registro(cdc):
    out = cdc.build_day_gap_context({}, PLAN, [], 21.2)
    assert "le falta REGISTRAR" in out
    assert "le falta REGISTRAR" not in cdc.build_day_gap_context({}, PLAN, [], 8.2), "a las 8 am un diario vacío es lo normal"


def test_pasarse_y_cubrir_la_proteina_se_dicen(cdc):
    lleno = DIARIO + [{"calories": 1000, "protein": 100, "carbs": 80, "healthy_fats": 40}]
    out = cdc.build_day_gap_context({}, PLAN, lleno, 20.5)
    assert "ya se pasó de su meta por ~200 kcal" in out and "ya cubrió su proteína del día" in out


def test_el_knob_apaga_el_bloque(cdc, monkeypatch):
    monkeypatch.setenv("MEALFIT_CHAT_DAY_GAP_BLOCK", "false")
    assert cdc.build_day_gap_context({}, PLAN, DIARIO, 21.0) == ""


# ───────────────────────── 2. La comida a medida ─────────────────────────

def test_la_franja_sale_de_la_hora_y_de_lo_que_falta(cdc):
    assert cdc.franja_por_hora(8.2) == "desayuno"
    assert cdc.franja_por_hora(13.0) == "almuerzo"
    assert cdc.franja_por_hora(21.1) == "cena"
    assert cdc.franja_por_hora(13.0, faltan=["cena"]) == "cena", "el almuerzo ya está registrado"


def test_el_objetivo_persigue_lo_que_falta_sin_salirse_de_la_franja(cdc):
    metas = cdc.metas_del_dia({}, PLAN)
    falta = cdc.falta_hoy(metas, cdc.consumido_hoy(DIARIO))          # 800 kcal, 85 g
    dia = cdc.objetivo_de_la_comida("cena", metas, falta, 18.0)
    assert dia["kcal"] == 800, "800 cabe en [0,6×, 1,4×] de una cena de 600"
    assert 60 <= dia["protein_g"] <= 72, "la proteína que falta, hasta 1,6× la de la franja"
    vacio = cdc.objetivo_de_la_comida("cena", metas, cdc.falta_hoy(metas, cdc.consumido_hoy([])), 18.0)
    assert vacio["kcal"] == pytest.approx(840), "un día vacío no convierte la cena en 2.000 kcal"


def test_con_el_diario_vacio_manda_la_racion_normal_y_con_registros_se_reparte(cdc):
    """Medido en la batería del lote: a las 8:10, con el día en blanco, el desayuno salía inflado a 1,4× (574 kcal en vez
    de 410) y el almuerzo de un diabético a 1.003. Un diario vacío suele ser «no registró», no «no comió»."""
    metas = cdc.metas_del_dia({}, PLAN)
    todo = cdc.falta_hoy(metas, cdc.consumido_hoy([]))
    des = cdc.objetivo_de_la_comida("desayuno", metas, todo, 8.2, faltan=["desayuno", "almuerzo", "cena"],
                                    hay_registros=False)
    assert des["kcal"] == pytest.approx(400) and des["protein_g"] == pytest.approx(30), "el 20 % de la franja, sin inflar"
    # con el desayuno registrado, lo que falta se reparte entre almuerzo (0,35) y cena (0,30): 1500 × 0,35/0,65
    falta = cdc.falta_hoy(metas, cdc.consumido_hoy(DIARIO[:1]))
    alm = cdc.objetivo_de_la_comida("almuerzo", metas, falta, 12.5, faltan=["almuerzo", "cena"])
    assert alm["kcal"] == pytest.approx(1500 * 0.35 / 0.65, rel=0.01)


def test_la_condicion_clinica_inclina_la_eleccion(cdc):
    assert cdc.riesgos_a_evitar({"medicalConditions": ["Diabetes tipo 2"]}) == {"glycemic_load_high", "sugar_high"}
    assert "sodium_high" in cdc.riesgos_a_evitar({"health_profile": {"medicalConditions": ["Hipertensión"]}})
    assert cdc.riesgos_a_evitar({"medicalConditions": ["Ninguna"]}) == set()


def test_como_queda_el_dia_viene_calculado(cdc):
    """En la batería el modelo hizo la resta de cabeza y dio 1.560 kcal donde eran 1.476."""
    c = {"name": "Avena", "calories": 574, "protein": "42g", "carbs": "67g", "fats": "15g", "_minutos": 10,
         "_lineas": [(70, "Avena")], "_tiene": [], "_falta": ["Avena"], "recipe": []}
    txt = cdc.formatear_propuestas([c], "desayuno", {"kcal": 574, "protein_g": 43}, con_nevera=False,
                                   falta={"kcal": 2050, "protein_g": 134})
    assert "Si se la come: le quedarían ~1476 kcal y ~92 g de proteína por cubrir." in txt
    assert "no hagas la resta tú" in txt
    assert "NO significa que no tenga comida en casa" in txt, "Nevera vacía en la app ≠ «toca comprar»"


def test_de_noche_la_comida_de_cierre_tiene_techo(cdc):
    """Medido en el humo: con 840 kcal por cerrar a las 21:05 proponía yuca con 237 g de cerdo (837 kcal) — lo contrario
    de lo que el bloque del día le aconseja al coach."""
    metas = cdc.metas_del_dia({}, PLAN)
    falta = cdc.falta_hoy(metas, cdc.consumido_hoy(DIARIO))
    noche = cdc.objetivo_de_la_comida("cena", metas, falta, 21.1, de_noche=True)
    assert noche["kcal"] == pytest.approx(600)
    assert noche["protein_g"] >= 60, "se recortan kcal, carbohidrato y grasa; la proteína se queda"
    pedido = cdc.objetivo_de_la_comida("cena", metas, falta, 21.1, kcal=900, de_noche=True)
    assert pedido["kcal"] == 900, "lo que el usuario pide explícitamente manda"


def test_los_gramos_se_leen_como_los_pesa_una_persona(cdc):
    assert cdc._linea_amigable(206.5, "Yuca") == "205 g de Yuca"
    assert cdc._linea_amigable(0.5, "Sal") == "una pizca de Sal"
    assert cdc._linea_amigable(4.6, "Ajo") == "5 g de Ajo"


def _catalogo_falso():
    fuera = {}
    for t in json.loads(_REG.read_text(encoding="utf-8")).get("templates") or []:
        for c in (t.get("constituents") or []):
            n = c.get("name")
            if n and n not in fuera:
                h = sum(ord(ch) for ch in n)
                fuera[n] = {"name": n, "kcal_per_100g": 80 + h % 300, "protein_g_per_100g": h % 25,
                            "carbs_g_per_100g": (h * 3) % 60, "fats_g_per_100g": (h * 7) % 20}
    return fuera


def test_propone_platos_del_registro_con_receta_y_solo_verifica_lo_que_devuelve(cdc, monkeypatch):
    if not _REG.exists():
        pytest.skip("sin snapshot del registry")
    import deterministic_day as dd
    import shopping_calculator as sc
    cat = _catalogo_falso()
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(cat.values()))
    llamadas = []

    def _verifica(meal, fd, catalogo):
        llamadas.append((meal.get("name"), bool(meal.get("recipe"))))
        return []

    monkeypatch.setattr(dd, "verifica_comida", _verifica)
    obj = {"kcal": 550, "protein_g": 35, "carbs_g": 60, "fats_g": 18}
    out = cdc.proponer_comidas({"country": "DO"}, "desayuno", obj, nevera_nombres=["Avena", "Huevo"], n=3)
    assert len(out) == 3 and len({c["name"] for c in out}) == 3
    assert len(llamadas) == 3, "verificar cuesta 0,3 s por plato: solo lo que se devuelve (eran 4-13 s por llamada)"
    assert all(con_receta for _n, con_receta in llamadas), "la receta va ANTES de verificar (V3 la necesita)"
    for c in out:
        assert abs(c["calories"] - 550) <= 25 and c["recipe"] and c["_meal_source"] == "coach_proposal"
    otra = cdc.proponer_comidas({"country": "DO"}, "desayuno", obj, excluir=[c["name"] for c in out], n=3)
    assert not ({c["name"] for c in otra} & {c["name"] for c in out}), "«dame otra» no repite"


def test_un_plato_que_no_pasa_la_verificacion_no_se_propone(cdc, monkeypatch):
    if not _REG.exists():
        pytest.skip("sin snapshot del registry")
    import deterministic_day as dd
    import shopping_calculator as sc
    cat = _catalogo_falso()
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(cat.values()))
    monkeypatch.setattr(dd, "verifica_comida", lambda meal, fd, c: ["alérgeno: maní"])
    obj = {"kcal": 550, "protein_g": 35, "carbs_g": 60, "fats_g": 18}
    assert cdc.proponer_comidas({"country": "DO"}, "desayuno", obj, n=3) == []
    txt = cdc.formatear_propuestas([], "desayuno", obj, con_nevera=False)
    assert "No inventes una receta «del catálogo»" in txt


def test_el_texto_para_el_modelo_dice_como_presentarlo(cdc):
    c = {"name": "Avena con claras", "calories": 574, "protein": "42g", "carbs": "67g", "fats": "15g", "_minutos": 10,
         "_lineas": [(71.4, "Avena"), (175.0, "Clara de huevo")], "_tiene": ["Avena"], "_falta": ["Clara de huevo"],
         "recipe": ["Cocina la avena.", "Añade las claras."]}
    txt = cdc.formatear_propuestas([c], "desayuno", {"kcal": 574, "protein_g": 43}, con_nevera=True)
    assert "574 kcal · proteína 42g" in txt and "70 g de Avena; 175 g de Clara de huevo" in txt
    assert "le falta: Clara de huevo" in txt and "Pasos: 1. Cocina la avena." in txt
    assert "NO la registres" in txt and "`excluir`" in txt
    vacia = cdc.formatear_propuestas([c], "cena", {"kcal": 600, "protein_g": 50}, con_nevera=False, solo_nevera=True)
    assert "su Nevera está VACÍA" in vacia


# ───────────────────────── 3. La herramienta y el prompt ─────────────────────────

def test_la_tool_vive_en_la_lista_y_su_knob_la_retira(monkeypatch):
    import tools
    assert "proponer_comida" in {t.name for t in tools.agent_tools}
    monkeypatch.setenv("MEALFIT_CHAT_MEAL_PROPOSAL_TOOL", "false")
    assert "proponer_comida" not in {t.name for t in tools._apply_chat_tool_knobs([tools.proponer_comida])}
    from prompts import chat_agent as ca
    assert "NO tienes herramienta para armar comidas" in ca.build_tools_instructions_stream("u")
    monkeypatch.setenv("MEALFIT_CHAT_MEAL_PROPOSAL_TOOL", "true")
    assert "- Usa `proponer_comida`" in ca.build_tools_instructions_stream("u")
    assert "- Usa `proponer_comida`" in ca.build_tools_instructions("u")


def test_las_reglas_de_resolver_van_en_las_cuatro_constantes():
    from prompts import chat_agent as ca
    for base in (ca.CHAT_SYSTEM_PROMPT_BASE, ca.CHAT_STREAM_SYSTEM_PROMPT_BASE, ca.CHAT_AGENT_INLINE_PROMPT,
                 ca.CHAT_STREAM_INLINE_PROMPT):
        assert base.endswith(ca._CHAT_RESOLVE_RULES), "que una edición no pueda arreglar 3 de 4"
    r = ca._CHAT_RESOLVE_RULES
    assert "Q. COMIDA O RECETA A PEDIDO" in r and "llama `proponer_comida` EN ESE TURNO" in r
    assert "S. PRUEBA ANTES DE ANOTAR UN PRODUCTO DE PROTEÍNA DE ENVASE" in r
    assert "foto de la tabla nutricional del pote, o dime la marca y cuántos scoops" in r
    assert "la comida casera o de la calle se sigue estimando y registrando en el mismo turno" in r
    assert "nunca le digas que «no puede» tomar proteína de noche" in r


def test_la_hora_del_bloque_sale_del_mismo_reloj_que_el_contexto_temporal(monkeypatch):
    from datetime import datetime, timezone
    from prompts import chat_agent as ca

    class _Reloj(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 21, 1, 5, tzinfo=timezone.utc)      # 21:05 en RD (UTC-4)

    monkeypatch.setattr(ca, "datetime", _Reloj)
    assert ca.hora_local_del_chat(240) == pytest.approx(21 + 5 / 60)
    assert ca.hora_local_del_chat(99999) == pytest.approx(21 + 5 / 60), "un huso basura cae a UTC-4, como su hermana"
    assert "21:05" in ca.build_temporal_context(tz_offset=240)


def test_los_dos_paths_del_chat_inyectan_el_bloque():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    llamada = "system_prompt += _day_gap_context_for_chat(form_data, plan_vigente, _diario_de_hoy, tz_offset, schedule_type)"
    assert src.count(llamada) == 2, "stream y no-stream: la divergencia entre ambos ya costó bugs"
    assert src.count("_diario_de_hoy = list(consumed_today or [])") == 2
    k = src.index("def _day_gap_context_for_chat(")
    assert "if diario_de_hoy is None:\n        return \"\"" in src[k:k + 1200].replace("\r\n", "\n"), \
        "sin diario legible no se afirma «le falta todo»"


# ───────────────────────── 4. La etiqueta ─────────────────────────

def test_la_etiqueta_se_lee_no_se_estima():
    import vision_agent as va
    assert va._MEAL_VISION_SCHEMA["properties"]["photo_kind"]["enum"] == ["plato", "items", "otro", "etiqueta"]
    assert "'etiqueta'" in va._MEAL_VISION_PROMPT and "NO estimes nada, LEE" in va._MEAL_VISION_PROMPT
    out = va._coerce_meal_scan({"photo_kind": "etiqueta", "is_food": True, "meal_name": "Gold Standard Whey",
                                "description": "Marca ON; porción 1 scoop (31 g); 120 kcal, 24 g proteína",
                                "calories": 120, "protein": 24, "carbs": 3, "healthy_fats": 1.5, "items": []})
    assert out["photo_kind"] == "etiqueta" and out["is_food"] is True and out["items"] == []
    assert (out["calories"], out["protein"], out["carbs"]) == (120, 24, 3), "las kcal LEÍDAS mandan: aquí no corre 4P+4C+9G"
    assert out["meal_name"] == "Gold Standard Whey" and out["label_read"] is True
    sin = va._coerce_meal_scan({"photo_kind": "etiqueta", "is_food": True, "meal_name": "ISO100",
                                "description": "Marca Dymatize", "calories": 0, "protein": 0, "carbs": 0,
                                "healthy_fats": 0, "items": []})
    assert sin["label_read"] is False and "No se lee la tabla nutricional" in sin["description"]


def test_el_chat_usa_las_cifras_de_la_etiqueta_y_no_la_manda_a_la_nevera():
    from prompts import chat_agent as ca
    ctx = ca.build_vision_context({"kind": "etiqueta", "description": "Marca ON; 1 scoop (31 g): 120 kcal, 24 g",
                                   "has_text": True})
    assert "ETIQUETA" in ctx and "POR PORCIÓN" in ctx and "cifras de la etiqueta × porciones" in ctx
    assert "NO lo ofrezcas para la Nevera" in ctx
    multi = ca.build_vision_context({"kind": "multi", "has_text": True, "items": [
        {"kind": "etiqueta", "description": "120 kcal por scoop"}, {"kind": "plato", "description": "arroz"}]})
    assert "ETIQUETA DE PRODUCTO" in multi
    assert "cifras de la etiqueta × porciones" in multi, \
        "el cliente manda SIEMPRE `multi`, también con una sola foto: la instrucción tiene que ir en esa rama"


def test_el_cliente_del_chat_deja_pasar_la_etiqueta():
    """`AgentPage.jsx` convertía todo tipo desconocido en 'plato': sin esto el coach recibía la etiqueta como un plato."""
    front = _BACKEND.parent / "frontend" / "src" / "pages" / "AgentPage.jsx"
    if not front.exists():
        pytest.skip("sin el repo del frontend al lado")
    src = front.read_text(encoding="utf-8")
    assert "item.kind === 'etiqueta' ? 'etiqueta'" in src


def test_la_bateria_tiene_los_casos_del_encargo():
    casos = {c["id"]: c for c in json.loads((_BACKEND / "scripts" / "coach_battery" / "battery.json")
                                            .read_text(encoding="utf-8"))["casos"]}
    assert casos["R1"]["turns"] == ["me falta proteina y tengo proteina en mi casa"] and casos["R1"]["hora"] == "21:05"
    assert "log_consumed_meal" in casos["R1"]["expect"]["no_tools"]
    assert casos["R5"]["expect"]["tools"] == ["proponer_comida"]
    assert casos["R9"]["expect"]["tools"] == ["log_consumed_meal"], "la comida casera se sigue registrando sin pruebas"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 132
