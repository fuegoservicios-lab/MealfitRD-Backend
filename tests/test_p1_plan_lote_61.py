# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-61 · 2026-09-15] (lote 44 del plan 38-44) Las decisiones que el dueño delegó el 14-sep, aplicadas y medidas.

(C4) las tres recetas del dueño en la biblioteca DO, y `recipe_usage` entiende «otro tercio» y «el último tercio/cuarto»:
     ninguna receta queda en `revisar`;
(B7) el peso del exceso de carbohidrato del día determinista sigue al usuario del canario, el global no se toca, y el
     default del canario es INERTE (1.0): medido en el perfil del canario (ganancia), 2.0 empeora carbohidrato y grasa;
(G1) `MEALFIT_BILLING_VERIFY_AMOUNT` por defecto `block`, y sin NINGÚN cupón activo para el tier un override por debajo
     de la lista se bloquea; si hay cupones (o no se pudo saber) sigue siendo ambiguo: alerta sin bloquear;
(C3, E5) medidos y NO aplicados: la botella de claras cruza el umbral en 1 de 64 comidas del corpus fijo (su propio lote)
     y la fase B de la lista canónica no pasa su gate (21 de ≥ 30 planes);
docs, knobs y marker ≥ 61.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REG = _BACKEND / "data" / "registry"
_TPLS = ("tpl_14c76a1c346e", "tpl_a7799418aa9c", "tpl_fc758e30f5e7")

import recipe_usage as ru  # noqa: E402
import deterministic_day as dd  # noqa: E402


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────────── C4 · las recetas y el reparto

def test_las_tres_recetas_llevan_la_redaccion_del_dueno_y_su_procedencia():
    lib = _json(_REG / "recipe_library_do_v1.json")
    yani = " ".join(lib["por_id"]["tpl_14c76a1c346e"]["pasos"])
    assert "un tercio del aceite" in yani and "otro tercio del aceite" in yani and "el último tercio del aceite" in yani
    assert "la otra mitad del aceite" not in yani
    pollo = " ".join(lib["por_id"]["tpl_a7799418aa9c"]["pasos"])
    assert "un cuarto del aceite" in pollo and "el último cuarto del aceite" in pollo and "el resto del aceite" not in pollo
    lent = lib["por_id"]["tpl_fc758e30f5e7"]["pasos"]
    assert "auyama" in lent[0] and "auyama" in lent[3]
    assert "curacion_c4_2026_09_14" in lib["procedencia"] and "P1-PLAN-LOTE-61" in lib["revision"]
    assert lib["recetas"] == len(lib["por_id"]) == 193


def test_ninguna_receta_queda_en_revisar():
    ru.clear_caches()
    snap = ru.cargar_uso("DO")
    estados = {t: snap["por_id"][t]["estado"] for t in _TPLS}
    assert estados == {"tpl_14c76a1c346e": "estimada", "tpl_a7799418aa9c": "estimada", "tpl_fc758e30f5e7": "exacta"}, estados
    assert snap["resumen"]["estados"].get("revisar", 0) == 0
    for t in _TPLS:
        assert not [h for h in snap["por_id"][t]["hallazgos"] if h["tipo"] in ("sobre_asignado", "a_medias", "sin_uso")], t


def _uso(pasos, alimento="aceite vegetal", gramos=15):
    return ru.derivar_uso(pasos, [{"ingredient_id": "aceite_vegetal", "canonical": alimento, "grams": gramos}], None)


def test_otro_tercio_y_el_ultimo_tercio_suman_uno():
    e = _uso(["Mezcla la harina con un tercio del aceite vegetal.", "Unta la bandeja con otro tercio del aceite vegetal.",
              "Sofríe la cebolla con el último tercio del aceite vegetal."])
    fr = [sum(x["fraccion"] for x in paso) for paso in e["usa"]]
    assert [round(f, 4) for f in fr] == [0.3333, 0.3333, 0.3333] and not e["hallazgos"], e


def test_la_mitad_un_cuarto_y_el_ultimo_cuarto_suman_uno():
    e = _uso(["Sazona el pollo con la mitad del aceite vegetal.", "Mezcla la batata con un cuarto del aceite vegetal.",
              "Aliña la ensalada con el último cuarto del aceite vegetal."])
    fr = [sum(x["fraccion"] for x in paso) for paso in e["usa"]]
    assert [round(f, 4) for f in fr] == [0.5, 0.25, 0.25] and not e["hallazgos"], e


def test_el_ultimo_tercio_cierra_como_el_resto_y_no_como_un_tercio_fijo():
    e = _uso(["Añade la mitad del aceite vegetal.", "Termina con el último tercio del aceite vegetal."])
    assert [round(sum(x["fraccion"] for x in paso), 4) for paso in e["usa"]] == [0.5, 0.5], "cierra con lo que queda"
    assert "P1-PLAN-LOTE-61-TERCIOS" in _src("recipe_usage.py")


# ─────────────────────────────── B7 · el peso del canario

def test_el_peso_del_canario_sigue_al_usuario_y_no_toca_el_global(monkeypatch):
    for k in ("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS", "MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT",
              "MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_USERS", "Dueno-1, otro")
    assert dd._pesos_scorer() == (1.0, 1.0)
    assert dd._pesos_scorer("dueno-1") == (1.0, 1.0), "default inerte: medido, 2.0 empeora el perfil del canario"
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY", "2.0")
    assert dd._pesos_scorer("dueno-1") == (2.0, 1.0) and dd._pesos_scorer("DUENO-1 ") == (2.0, 1.0)
    assert dd._pesos_scorer("alguien") == (1.0, 1.0) and dd._pesos_scorer(None) == (1.0, 1.0), "el global no se toca"
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY", "9")
    assert dd._pesos_scorer("dueno-1") == (1.0, 1.0), "fuera del clamp [0.5, 5.0] ⇒ default"
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_USERS", "")
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY", "2.0")
    assert dd._pesos_scorer("dueno-1") == (1.0, 1.0), "lista vacía: nadie es canario"


def test_quien_arma_el_dia_pasa_los_pesos_del_usuario():
    src = _src("deterministic_day.py")
    assert "pesos=_pesos_scorer(_uid))" in src, "build_day_for_skeleton pasa los pesos de SU usuario"
    cuerpo = src[src.find("def elegir_plantillas("):]
    cuerpo = cuerpo[:cuerpo.find("\ndef ", 10)]
    assert "pesos: Optional[tuple] = None" in cuerpo and "if pesos:" in cuerpo
    assert "_w_cs, _w_fd = _pesos_scorer()" in cuerpo, "el ancla del lote 10 sigue: una lectura por franja"


# ─────────────────────────────── G1 · PayPal

def _billing():
    spec = importlib.util.spec_from_file_location("billing_l61", _BACKEND / "routers" / "billing.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sub(price):
    return {"plan_overridden": True, "billing_info": {}, "plan": {"billing_cycles": [
        {"tenure_type": "REGULAR", "pricing_scheme": {"fixed_price": {"value": price, "currency_code": "USD"}}}]}}


def _verificar(b, monkeypatch, *, hay_cupon, disc=None, price="0.01", mode=None):
    if mode is None:
        monkeypatch.delenv("MEALFIT_BILLING_VERIFY_AMOUNT", raising=False)
    else:
        monkeypatch.setenv("MEALFIT_BILLING_VERIFY_AMOUNT", mode)

    async def _validate(code, tier):
        return {"discount_percent": disc} if disc is not None else None

    async def _lista(*a, **k):
        return 9.99

    async def _hay(tier):
        return hay_cupon

    alertas = []
    monkeypatch.setattr(b, "_validate_discount_code", _validate)
    monkeypatch.setattr(b, "_fetch_plan_list_price", _lista)
    monkeypatch.setattr(b, "_active_coupon_exists_for_tier", _hay)
    monkeypatch.setattr(b, "_persist_billing_alert", lambda **kw: alertas.append(kw))
    from fastapi import HTTPException
    try:
        asyncio.run(b._verify_subscription_amount(sub_data=_sub(price), verified_plan_id="P-ULTRA", tier="ultra",
                                                  coupon_code="", access_token="t", paypal_api_base="x",
                                                  user_id="u1", subscription_id="s1"))
        return False, alertas
    except HTTPException as e:
        return e.status_code == 409, alertas


def test_sin_ningun_cupon_para_el_tier_el_override_se_bloquea_por_defecto(monkeypatch):
    b = _billing()
    bloqueado, alertas = _verificar(b, monkeypatch, hay_cupon=False)          # sin la variable: el default
    assert bloqueado is True and len(alertas) == 1
    assert alertas[0]["metadata"]["sin_cupon_posible"] is True and alertas[0]["alert_key"] == "billing_price_tampering:u1:s1"


def test_si_hay_cupones_o_no_se_sabe_sigue_siendo_ambiguo(monkeypatch):
    b = _billing()
    for hay in (True, None):
        bloqueado, alertas = _verificar(b, monkeypatch, hay_cupon=hay)
        assert bloqueado is False and len(alertas) == 1, hay                    # alerta, y el pago sigue
        assert alertas[0]["metadata"]["sin_cupon_posible"] is False


def test_en_warn_nunca_bloquea_y_el_precio_de_lista_no_se_toca(monkeypatch):
    b = _billing()
    assert _verificar(b, monkeypatch, hay_cupon=False, mode="warn")[0] is False
    bloqueado, alertas = _verificar(b, monkeypatch, hay_cupon=False, price="9.99")   # override al precio de lista
    assert bloqueado is False and len(alertas) == 1, "sin cupón válido se alerta, pero no está por debajo: no se bloquea"


def test_el_default_es_block_y_la_consulta_es_fail_cheap():
    src = _src("routers/billing.py")
    assert '_env_str("MEALFIT_BILLING_VERIFY_AMOUNT", "block"' in src
    cuerpo = src[src.find("async def _active_coupon_exists_for_tier"):]
    cuerpo = cuerpo[:cuerpo.find("\nasync def ", 10)]
    assert "return None" in cuerpo and "WHERE is_active = TRUE" in cuerpo and "_CUPON_GRACIA_H" in cuerpo
    assert "tooltip-anchor: P1-PLAN-LOTE-61-SIN-CUPON" in cuerpo
    assert 'if mode == "block" and (proven_underpaid or sin_cupon_posible):' in src


# ─────────────────────────────── docs, knobs, marker

def test_docs_knobs_y_marker():
    kr = _src("docs/knobs_reference.md")
    for k in ("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY", "MEALFIT_BILLING_VERIFY_AMOUNT"):
        assert k in kr, k
    assert "P1-PLAN-LOTE-61" in kr
    plan = _src("docs/plan_pendientes_2026_09_11.md")
    assert plan.count("P1-PLAN-LOTE-61") >= 3
    assert "P1-PLAN-LOTE-61" in _src("docs/plan_agente_lotes_38_43_2026_09_14.md")
    assert "P1-PLAN-LOTE-61" in _src("docs/deterministic_day.md")
    app = _src("app.py")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-\d\d-\d\d"', app, re.M)
    assert m and int(m.group(1)) >= 61
