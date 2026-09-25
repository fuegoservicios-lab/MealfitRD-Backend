# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-220 · 2026-09-24] El tiempo de cocina de «sin tiempo» llega a quien reescribe un día o un plato.

Sobre las 85 corridas guardadas: con «30 min» y «1 hora», 0 platos fuera de tiempo en 894 comidas; con «sin tiempo»
(`cookingTime=none`, 10 min), 41 de 144 por encima del margen de la auditoría y 20 al doble o más. Primera hipótesis:
el prompt lo pedía y nadie lo corregía → el plato fuera de tiempo pasa a ser una incoherencia determinista del día
(piso de corrección) y el corrector la recibe literal. Con solo eso, 3 corridas reales: 16 de 36 fuera.

La corrida instrumentada enseñó la causa real: el generador del día entregó las 12 comidas dentro de 10 min (el
detector no disparó) y el CORRECTOR del self-critique —que no ve el formulario— las reescribió en 15-20 min al
arreglar variedad («Pinchos de pollo a la plancha con yuca hervida», 20). Ahora el tope viaja SIEMPRE a los tres que
reescriben: corrector del self-critique, regen quirúrgico y swap (`horizon.cooking_time_rule`, mismo texto que ve el
generador); el swap lo hidrata del perfil porque el frontend no lo manda.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import horizon  # noqa: E402


def _dias():
    return [
        {"day": 1, "meals": [
            {"meal": "Almuerzo", "name": "Arepitas rápidas de maíz", "prep_time": "10 min"},
            {"meal": "Cena", "name": "Guiso de lentejas con yuca hervida", "prep_time": "15 min"},
        ]},
        {"day": 3, "meals": [
            {"meal": "Almuerzo", "name": "Wrap integral con auyama asada", "prep_time": "25 min"},
            {"meal": "Cena", "name": "Bollitos de maíz rellenos", "prep_time": "30 min", "_prep_time_source": "defecto"},
        ]},
    ]


def test_sin_tiempo_marca_lo_que_pasa_el_margen():
    out = go._detect_prep_time_issues(_dias(), {"cookingTime": "none"})
    assert len(out) == 2, out                       # 10 min cumple; 15 y 25 pasan 12,5; el de minutos «de defecto» no se mide
    assert out[0].startswith("TIEMPO DE COCINA: Día 1, cena: «Guiso de lentejas con yuca hervida» declara 15 min")
    assert "Día 3, almuerzo" in out[1] and "(10 min por comida como máximo)" in out[1]


def test_con_tiempo_no_se_toca_nada():
    for ct in ("30min", "1hour", "plenty", "", None):
        assert go._detect_prep_time_issues(_dias(), {"cookingTime": ct}) == [], ct


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setattr(go, "CRITIQUE_PREP_TIME_ENABLED", False)
    assert go._detect_prep_time_issues(_dias(), {"cookingTime": "none"}) == []


def test_cableado_en_el_self_critique():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("async def self_critique_node(")
    cuerpo = src[i:src.index("\n@_node_label", i + 10)] if "\n@_node_label" in src[i + 10:] else src[i:]
    assert "_prep_time_issues = _detect_prep_time_issues(days, state.get(\"form_data\") or {})" in cuerpo
    assert "slot_issues.extend(_prep_time_issues)" in cuerpo
    # entra ANTES de construir el bloque de incoherencias (y del salto «limpio»)
    assert cuerpo.index("slot_issues.extend(_prep_time_issues)") < cuerpo.index("if slot_issues:")
    assert "PROBLEMA DETECTADO: {critique.suggestions}{_pt_block}" in cuerpo
    assert "tooltip-anchor: P1-PLAN-LOTE-220-TIEMPO-AL-CORRECTOR" in src


# ─────────────────────────────── la causa real: el tope a quien reescribe

def test_la_regla_es_el_texto_del_generador_sin_el_codigo():
    r = horizon.cooking_time_rule({"cookingTime": "none"})
    assert r.startswith("NO TIENE TIEMPO para cocinar: cada comida se prepara en 10 minutos o menos")
    assert "`prep_time` de cada comida ≤ 10 min." in r and "none =" not in r
    assert horizon.cooking_time_rule({"cookingTime": "30min"}) == (
        "cada comida en 30 minutos o menos en total; `prep_time` de cada comida ≤ 30 min.")
    assert "60 minutos" in horizon.cooking_time_rule({"cookingTime": " 1HOUR "})
    for sin_tope in ({"cookingTime": "plenty"}, {"cookingTime": ""}, {"cookingTime": "raro"}, {}, None):
        assert horizon.cooking_time_rule(sin_tope) == "", sin_tope
    # el generador ve el MISMO texto (un solo SSOT): la regla es su cola tras «none = »
    assert horizon.explain_form_codes_for_prompt({"cookingTime": "none"})["cookingTime"].endswith(r)


def test_los_dos_correctores_reciben_el_tope():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("async def self_critique_node(")
    critica = src[i:src.index("\n@_node_label", i + 10)]          # incluye el `_correct_single_day` anidado
    assert '_ct_rule = __import__("horizon").cooking_time_rule(state.get("form_data") or {})' in critica
    assert "{ctx['nutrition_context_minimal']}{_ct_block}\n{skeleton_block}" in critica
    j = src.index("async def surgical_marker_regen_node(")
    quirurgico = src[j:src.index("\nasync def ", j + 30)]
    assert '_ct_rule_sg = __import__("horizon").cooking_time_rule(form_data or {})' in quirurgico
    assert "{ctx['nutrition_context_minimal']}{_ct_block_sg}\n{skeleton_block}" in quirurgico
    assert "tooltip-anchor: P1-PLAN-LOTE-220-TIEMPO-A-LOS-CORRECTORES" in (_BACKEND / "horizon.py").read_text(
        encoding="utf-8")


def test_el_swap_recibe_el_tope_salvo_fin_de_semana():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("def swap_meal(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    k = cuerpo.index('_ct_rule_swap = __import__("horizon").cooking_time_rule(form_data)')
    assert 'if swap_reason != "weekend":' in cuerpo[k - 120:k]
    assert 'context_extras += f"\\n    - ⏱️ TIEMPO DE COCINA DEL USUARIO (obligatorio): {_ct_rule_swap}"' in cuerpo
    # y va ANTES de que context_extras entre al prompt
    assert k < cuerpo.index("context_extras=context_extras")


def test_el_swap_hidrata_el_tiempo_del_perfil(monkeypatch):
    import db
    from routers import plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda _u: {"health_profile": {"cookingTime": "none"}})
    body = {"allergies": []}
    rp._enrich_clinical_from_profile(body, "u-1")
    assert body["cookingTime"] == "none"
    body = {"allergies": [], "cookingTime": "30min"}            # el body gana si lo trae
    rp._enrich_clinical_from_profile(body, "u-1")
    assert body["cookingTime"] == "30min"


# ─────────────────────────────── las dos pasadas deterministas que alargaban un plato ya corregido

def test_el_paso_de_microondas_no_hereda_el_horno_del_nombre():
    meal = {"name": "Arepitas horneadas con huevo duro y ensalada fría", "recipe": [
        "Mise en place: corta el nabo en tiras finas.",
        "El Toque de Fuego: calienta las 2 arepitas horneadas ya preparadas en el microondas, sin usar la estufa.",
        "Montaje: sirve la ensalada con el huevo y las arepitas."]}
    assert go._inject_recipe_time_temp_defaults(meal) is True
    assert meal["recipe"][1].endswith("(~2-3 min en el microondas).") and "180 °C" not in meal["recipe"][1]
    # sin microondas en el paso, el horno del nombre sigue mandando (conducta previa)
    horno = {"name": "Arepitas horneadas", "recipe": ["El Toque de Fuego: hornea las arepitas hasta dorar."]}
    go._inject_recipe_time_temp_defaults(horno)
    assert "18-20 min a 180 °C" in horno["recipe"][0]


def test_el_clamp_no_usa_el_microondas():
    meal = {"name": "Pollo al horno", "recipe": [
        "El Toque de Fuego: calienta la salsa en el microondas y hornea el pollo 200 min a 180 °C."]}
    go._clamp_recipe_time_temp_outliers(meal)
    assert "18-20 min a 180 °C" in meal["recipe"][0] and "microondas 2-3" not in meal["recipe"][0]


def _viver_sin_cocer():
    return {"meal": "Cena", "name": "Ñame Guisado en Salsa Criolla",
            "ingredients": ["150 g de ñame pelado en cubos", "1 cdta de aceite de oliva"],
            "recipe": ["Mise en place: pela el ñame y córtalo en cubos.",
                       "El Toque de Fuego: calienta el aceite y sofríe la cebolla 4 minutos.",
                       "Montaje: sirve el guiso en un plato hondo."]}


def _catalogo():
    import json
    return json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                      .read_text(encoding="utf-8"))["catalogo_filas"]


def test_sin_tiempo_el_viver_se_cuece_en_cubos_pequenos():
    plan = {"days": [{"day": 1, "meals": [_viver_sin_cocer()]}]}
    assert go._auto_patch_uncooked_foods(plan, _catalogo(), form_data={"cookingTime": "none"}) == 1
    pasos = plan["days"][0]["meals"][0]["recipe"]
    nuevo = [s for s in pasos if str(s).startswith("🍠")]
    assert nuevo == ["🍠 Corta Ñame en cubos pequeños (1 cm) y hiérvelos 10-12 minutos, hasta que estén tiernos, "
                     "antes de servir."], pasos
    assert str(pasos[-1]).startswith("Montaje")
    # idempotente
    assert go._auto_patch_uncooked_foods(plan, _catalogo(), form_data={"cookingTime": "none"}) == 0
    # con tiempo, el paso de siempre
    plan2 = {"days": [{"day": 1, "meals": [_viver_sin_cocer()]}]}
    go._auto_patch_uncooked_foods(plan2, _catalogo(), form_data={"cookingTime": "30min"})
    assert any("al guiso y cocínalo 15-20 minutos" in str(s) for s in plan2["days"][0]["meals"][0]["recipe"])


def test_el_review_le_pasa_el_formulario_al_reparador():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '_auto_patch_uncooked_foods(plan, _gmi_uc(), form_data=state.get("form_data"))' in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 220
