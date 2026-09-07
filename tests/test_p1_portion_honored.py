# -*- coding: utf-8 -*-
"""[P1-PORTION-HONORED · 2026-09-07] La ración pedida deja de ser telemetría y levanta el techo.

`P1-ANCHOR-PORTION` hizo viajar la cantidad («desayuno 10 claras») y volvió ruidoso el recorte.
Esto la HONRA: el motor deja de cortar a 6 para quien la pidió, sin tocar a nadie más.

Medido antes de escribirlo, sobre 96 planes vivos: las claras entregadas son 1, 2, 3, 4, 5 y 6, y
ahí se cortan en seco. Esa distribución que muere justo en el tope es la firma del cap.

## Por qué hacían falta CUATRO sitios y no uno

Subir solo el cap del ensamblado habría sido inerte: `_cap_unrealistic_portions` corre DESPUÉS, al
persistir, y habría vuelto a recortar a 6. Y aunque ambos permitieran 10, el prompt seguiría
pidiendo «máximo 6 claras/día» y el modelo nunca las escribiría.

    1. cap explícito de claras (`assemble_plan_node`)      -> lee la política
    2. cap de realismo por conteo (al persistir)           -> override desde `_plan_policy`
    3. prompt del generador del día                        -> el número de la regla, sustituido
    4. sello `enforced` en `_plan_policy`                   -> para que (2) sepa si aplica

## Las tres decisiones que hacen esto seguro

**`max(default, pedido)`, nunca `pedido`.** Una petición sube el techo; jamás lo baja. Pedir «2
claras» no puede apretarle el margen al motor.

**Solo con la política EN VIGOR.** Mismo canary que gobierna el resto de la Fase 3: la primera
persona que reciba 10 claras será una a la que se le encendió a propósito.

**Solo unidades de pieza.** «150 g» no dice cuántas piezas caben; tratarlo como tal sería
`P1-UNKNOWN-UNIT-NOT-WHOLE` por la puerta de atrás.

## Y una que costó una vuelta

El override casaba por SUBCADENA y «Clara de huevo» subía también el tope de «huevo»: pedir 10
claras habría autorizado 10 huevos enteros. Es `"res" ⊂ "fresco"` una vez más. Se compara contra
el sustantivo CABECERA del ancla, que es justo lo que distingue las tres formas del huevo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import plan_policy as pp  # noqa: E402

_BASE = {"clara": 6.0, "huevo": 4.0, "papa": 3.0}


def _pol(nombre, qty, *, unit="unidad", enforced=True):
    return {"enforced": enforced, "effective": {"food_anchors": [{
        "name": nombre, "ingredient_id": pp.ingredient_id_for(nombre),
        "portion": {"qty": qty, "unit": unit}}]}}


# ── el tope efectivo ──────────────────────────────────────────────────────────────────────────
def test_la_racion_pedida_levanta_el_techo():
    eff = _pol("Clara de huevo", 10)["effective"]
    assert pp.portion_cap_for(eff, "Clara de huevo", 6.0, enforced=True) == 10.0


def test_pedir_MENOS_no_baja_el_techo():
    """Una petición sube; jamás aprieta. «2 claras» no puede quitarle margen al motor."""
    eff = _pol("Clara de huevo", 2)["effective"]
    assert pp.portion_cap_for(eff, "Clara de huevo", 6.0, enforced=True) == 6.0


def test_sin_la_politica_en_vigor_no_cambia_nada():
    """El canary es el interruptor de despliegue: quien está solo en modo sombra se mide, no se
    le cambia el plan."""
    eff = _pol("Clara de huevo", 10)["effective"]
    assert pp.portion_cap_for(eff, "Clara de huevo", 6.0, enforced=False) == 6.0


def test_una_racion_en_GRAMOS_no_toca_un_techo_de_piezas():
    eff = _pol("Pollo", 150, unit="g")["effective"]
    assert pp.portion_cap_for(eff, "Pollo", 3.0, enforced=True) == 3.0


def test_fail_safe_ante_cualquier_basura():
    for args in ((None, "Clara de huevo"), ({}, "Clara de huevo"), ({"food_anchors": 7}, "x"),
                 (_pol("Clara de huevo", 10)["effective"], None)):
        assert pp.portion_cap_for(args[0], args[1], 6.0, enforced=True) == 6.0


# ── el override del cap de realismo ───────────────────────────────────────────────────────────
def test_diez_claras_NO_autorizan_diez_huevos_enteros():
    """La trampa que costó una vuelta: casar por subcadena veía «huevo» dentro de «Clara de
    huevo». Pedir claras no puede levantar el techo del huevo entero — son tres alimentos
    distintos, con grasa y colesterol distintos."""
    out = pp.build_count_caps_override(_pol("Clara de huevo", 10), _BASE)
    assert out["clara"] == 10.0
    assert out["huevo"] == 4.0, out


def test_el_diccionario_global_NO_se_muta():
    """Mutarlo le cambiaría el techo a todo el proceso: la petición de una persona alteraría el
    plan de otra."""
    antes = dict(_BASE)
    pp.build_count_caps_override(_pol("Clara de huevo", 10), _BASE)
    assert _BASE == antes


def test_sin_canary_o_sin_politica_devuelve_None():
    assert pp.build_count_caps_override(_pol("Clara de huevo", 10, enforced=False), _BASE) is None
    assert pp.build_count_caps_override(None, _BASE) is None
    assert pp.build_count_caps_override({"enforced": True}, _BASE) is None


def test_si_nada_sube_devuelve_None_y_el_llamador_usa_el_global():
    assert pp.build_count_caps_override(_pol("Clara de huevo", 3), _BASE) is None


# ── el prompt ─────────────────────────────────────────────────────────────────────────────────
def test_el_prompt_pide_las_claras_que_la_persona_pidio():
    """Sin esto el tope subido sería inerte: el motor permitiría 10 y el prompt seguiría pidiendo
    6, así que el modelo nunca las escribiría."""
    from prompts.day_generator import build_day_generator_system_prompt, override_egg_white_limit
    base = build_day_generator_system_prompt()
    assert "máximo 6 claras/día" in base, "cambió la regla del prompt: revisa el sustituidor"
    assert "máximo 10 claras/día" in override_egg_white_limit(base, 10)


def test_el_prompt_no_se_toca_si_el_tope_no_sube():
    from prompts.day_generator import build_day_generator_system_prompt, override_egg_white_limit
    base = build_day_generator_system_prompt()
    for v in (6, 3, 0, None, "diez", -1):
        assert override_egg_white_limit(base, v) == base, v


def test_se_sustituye_el_NUMERO_y_no_se_añade_una_excepcion():
    """Dos instrucciones contradictorias en el mismo prompt es un modo de fallo conocido aquí:
    «una regla insatisfacible gasta reintentos y empeora el plato». Una sola regla."""
    from prompts.day_generator import build_day_generator_system_prompt, override_egg_white_limit
    salida = override_egg_white_limit(build_day_generator_system_prompt(), 12)
    assert salida.count("claras/día") == 1
    assert "máximo 6 claras/día" not in salida


# ── el cableado: los cuatro sitios ────────────────────────────────────────────────────────────
def _go() -> str:
    return (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")


def test_el_cap_explicito_de_claras_lee_la_politica():
    src = _go()
    assert '_pcf(form_data.get("_plan_policy_effective"), "Clara de huevo"' in src
    for viejo in ("if _new_count > MAX_EGG_WHITES_PER_MEAL",
                  "while _day_total > MAX_EGG_WHITES_PER_DAY"):
        assert viejo not in src, f"quedó un tope fijo sin leer la política: {viejo}"


def test_el_cap_de_realismo_acepta_el_override():
    src = _go()
    assert "def _cap_unrealistic_portions(days, db=None, *, count_caps=None)" in src
    assert "_CC = count_caps if isinstance(count_caps, dict) else _REALISM_COUNT_CAPS" in src
    assert "_REALISM_COUNT_CAPS.get(" not in src, (
        "un lookup se quedó leyendo el global y volvería a recortar a 6")


def test_el_persist_resuelve_el_override_desde_el_PLAN():
    """Al persistir ya no existe `form_data`; la política se lee del propio plan. Enhebrar
    `form_data` habría tocado media docena de firmas."""
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8", errors="ignore")
    assert "build_count_caps_override" in src and 'count_caps=_cc_ins' in src
    assert '_pd.get("_plan_policy")' in src


def test_la_politica_persistida_sella_si_estaba_en_vigor():
    """Sin este sello, el persist no puede distinguir un usuario del canary de uno en sombra."""
    src = (_BACKEND / "plan_policy.py").read_text(encoding="utf-8")
    assert 'compiled["enforced"] = bool((form_data or {}).get("_policy_enforced"))' in src


def test_los_TRES_retornos_del_prompt_pasan_por_el_override():
    """Uno sin cubrir deja un camino entero pidiendo 6 claras — y sería el camino silencioso."""
    src = _go()
    assert src.count("_eggw_prompt_limit(") >= 4   # def + 3 llamadas


# ── el módulo extraído ────────────────────────────────────────────────────────────────────────
def test_la_extraccion_respeta_la_regla_del_repo():
    """El techo de líneas no se sube: se extrae. Se eligió el candidato más barato del fichero
    (72 líneas, CERO dependencias del módulo, medido con AST), porque una extracción con
    dependencias habría provocado el import circular que descartó a las funciones del cap."""
    import upstream_errors
    from graph_orchestrator import _is_transient_upstream_error as reexportado
    assert reexportado is upstream_errors._is_transient_upstream_error
    assert "cero dependencias" in upstream_errors.__doc__.lower().replace("_", " ")
