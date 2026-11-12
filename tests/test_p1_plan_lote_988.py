"""[P1-PLAN-LOTE-988 · 2026-11-12] «Dos paquetes de las de ayer»: el sistema busca el registro y hace la cuenta.

Caso vivo del dueño (02:11 del 12-nov, hora RD): ayer (día biológico) anotó «3 galletas Crakeñas CLUB» = 106 kcal.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import referencia_a_registro as rr  # noqa: E402

AHORA = datetime(2026, 11, 12, 6, 11, tzinfo=timezone.utc)          # 02:11 RD
GALLETAS = {"meal_name": "3 galletas Crakeñas CLUB", "calories": 106, "protein": 2, "carbs": 17, "healthy_fats": 3,
            "ingredients": ["galletas saladas Crakeñas CLUB (3 galletas, 24 g)"],
            "consumed_at": "2026-11-11T05:19:58+00:00"}                   # 01:19 RD del 11 = día biológico 10
CENA_HOY = {"meal_name": "Cena con galletas de soda", "calories": 900, "consumed_at": "2026-11-12T00:35:00+00:00"}
TEXTO = "Me metí dos paquete de galletas de las de ayer"


def setup_function(_):
    rr._YA_AVISADO.clear()


def test_detecta_la_referencia_y_las_unidades():
    d = rr.detectar(TEXTO)
    assert d["dias"] == 1 and d["n"] == 2 and not d["cambios"] and "galleta" in d["tokens"]
    assert rr.detectar("otra como la de anoche")["n"] == 1
    assert rr.detectar("el mismo desayuno de ayer pero sin el atún")["cambios"]
    assert rr.detectar("me comí dos paquetes de galletas") is None
    assert rr.detectar("ayer cené sancocho") is None


def test_elige_la_fila_del_dia_biologico_anterior_no_la_de_hoy():
    r = rr.resolver("u", TEXTO, 240, ahora_utc=AHORA, filas=[CENA_HOY, GALLETAS])
    assert r and r[1]["meal_name"] == "3 galletas Crakeñas CLUB"


def test_bloque_con_la_cuenta_hecha():
    b = rr.bloque_para_chat("u", TEXTO, 240, ahora_utc=AHORA, filas=[GALLETAS])
    assert "212 kcal" in b and "SIN preguntar" in b and "3 galletas Crakeñas CLUB" in b


def test_sin_registro_que_case_no_dice_nada():
    assert rr.bloque_para_chat("u", "dos tazas de las de ayer de café", 240, ahora_utc=AHORA, filas=[GALLETAS]) == ""
    assert rr.bloque_para_chat("guest", TEXTO, 240, ahora_utc=AHORA, filas=[GALLETAS]) == ""


def test_rechazo_unico_si_la_cifra_no_cuadra():
    kw = dict(ahora_utc=AHORA, filas=[GALLETAS])
    assert rr.rechazo("u", TEXTO, 212, 240, ahora=1.0, **kw) is None          # cuadra
    txt = rr.rechazo("u", TEXTO, 400, 240, ahora=2.0, **kw)
    assert txt and "212 kcal" in txt and txt.startswith("⚠️ NO REGISTRADO TODAVÍA")
    assert rr.rechazo("u", TEXTO, 400, 240, ahora=3.0, **kw) is None          # el reintento pasa
    assert rr.rechazo("v", "dos como las de ayer pero sin sal", 400, 240, ahora=3.0,
                      ahora_utc=AHORA, filas=[{**GALLETAS, "meal_name": "galletas"}]) is None   # con cambios


def test_knob(monkeypatch):
    monkeypatch.setenv("MEALFIT_PRIOR_LOG_REFERENCE", "0")
    assert rr.bloque_para_chat("u", TEXTO, 240, ahora_utc=AHORA, filas=[GALLETAS]) == ""


def test_no_se_confunde_con_aviso_de_duplicado():
    from agent import _AVISOS_DE_DUPLICADO
    txt = rr.rechazo("w", TEXTO, 400, 240, ahora=9.0, ahora_utc=AHORA, filas=[GALLETAS])
    assert not any(a in txt for a in _AVISOS_DE_DUPLICADO)


def test_cableado():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert src.count('__import__("referencia_a_registro").bloque_para_chat(user_id, prompt, tz_offset)') == 2
    i_freno = src.index("tool_messages.append(ToolMessage(content=_freno, tool_call_id=tool_id))")
    assert i_freno < src.index("_rechazo_ref(_trusted_uid, _txt_ref, tool_args.get(\"calories\"), state.get(\"tz_offset\"))")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 988


def test_la_franja_entera_se_suma_cuando_no_nombra_alimento():
    ayer = "2026-11-11T13:00:00+00:00"   # 09:00 RD del 11 = día biológico 11... usamos un "ahora" del 12 a mediodía
    ahora = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)
    filas = [{"meal_name": "4 huevos hervidos con pan", "meal_type": "desayuno", "calories": 768, "protein": 40,
              "carbs": 50, "healthy_fats": 30, "consumed_at": ayer},
             {"meal_name": "Yogurt Ratio Protein", "meal_type": "desayuno", "calories": 170, "protein": 25,
              "carbs": 8, "healthy_fats": 4, "consumed_at": ayer},
             {"meal_name": "Arroz con pollo", "meal_type": "almuerzo", "calories": 700, "consumed_at": ayer}]
    r = rr.resolver("u", "Desayuné lo mismo de ayer", 240, ahora_utc=ahora, filas=filas)
    assert r and r[1]["calories"] == 938 and "Yogurt" in r[1]["meal_name"] and "huevos" in r[1]["meal_name"]
