# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1145 · 2026-11-21] Tu día termina cuando duermes, y «ayer» sin hora no cae a la hora de ahora.

Caso del dueño (21-nov, RD = UTC-4): registró a las 00:40-01:01 y, sin dormir, a las 4:22 dijo «ayer me comí otra
galleta de Koko». El contador había vuelto a 0 a las 4:00 y la galleta quedó a las 4:22 del VIERNES (ahora − 24 h).
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

import dia_extendido as de
import registro_de_otro_dia as rod

RD = 240
_BACKEND = Path(__file__).resolve().parent.parent


def _u(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _encendido(monkeypatch):
    monkeypatch.setenv("MEALFIT_DIA_TERMINA_AL_DORMIR", "1")


NOCHE = [_u("2026-11-21T04:40:00"), _u("2026-11-21T04:44:00"), _u("2026-11-21T05:01:00")]   # 0:40-1:01 RD


def test_sin_dormir_sigue_en_el_viernes():
    e = de.decidir(_u("2026-11-21T08:22:00"), RD, actividad=NOCHE)          # 4:22 RD
    assert e["extendido"] and e["dia"] == "2026-11-20" and e["motivo"] == "despierto"
    assert e["fin"] == "2026-11-21T08:00:00+00:00"                          # el corte natural (4:00 RD)
    assert _u("2026-11-21T08:22:00") < datetime.fromisoformat(e["hasta"])


def test_si_durmio_empieza_el_dia_nuevo():
    e = de.decidir(_u("2026-11-21T11:30:00"), RD, actividad=NOCHE)          # 7:30 RD: 6,5 h sin actividad
    assert not e["extendido"] and e["dia"] == "2026-11-21"
    assert e["puede_seguir"]                                                 # hubo noche: se le puede ofrecer


def test_quien_durmio_temprano_ni_se_entera():
    e = de.decidir(_u("2026-11-21T10:00:00"), RD, actividad=[_u("2026-11-21T02:00:00")])   # 22:00 RD
    assert not e["extendido"] and not e["puede_seguir"]


def test_empezar_gana_y_seguir_extiende_hasta_el_mediodia():
    ahora = _u("2026-11-21T08:22:00")
    e = de.decidir(ahora, RD, actividad=NOCHE, accion={"dia": "2026-11-20", "accion": "empezar"})
    assert not e["extendido"] and not e["puede_seguir"]
    e = de.decidir(_u("2026-11-21T14:00:00"), RD, actividad=[], accion={"dia": "2026-11-20", "accion": "seguir"})
    assert e["extendido"] and e["motivo"] == "pedido" and e["hasta"] == "2026-11-21T16:00:00+00:00"   # 12:00 RD
    # una acción de OTRO día no vale
    e = de.decidir(ahora, RD, actividad=[], accion={"dia": "2026-11-19", "accion": "seguir"})
    assert not e["extendido"]


def test_corte_personal_y_fuera_de_la_manana():
    e = de.decidir(_u("2026-11-21T10:30:00"), RD, actividad=[], termina_h=7)   # 6:30 RD, su día acaba a las 7
    assert e["extendido"] and e["motivo"] == "corte_personal"
    assert not de.decidir(_u("2026-11-21T11:30:00"), RD, actividad=[], termina_h=7)["extendido"]
    assert not de.decidir(_u("2026-11-21T18:00:00"), RD, actividad=NOCHE)["extendido"]   # 14:00 RD
    assert de.corte_personal("9") == 8 and de.corte_personal(None) == 4


def test_apagado_es_el_corte_de_siempre(monkeypatch):
    monkeypatch.setenv("MEALFIT_DIA_TERMINA_AL_DORMIR", "0")
    assert not de.decidir(_u("2026-11-21T08:22:00"), RD, actividad=NOCHE)["extendido"]


def test_estado_lee_la_base_y_hora_para_registro(monkeypatch):
    def _q(sql, params, fetch_all=False, fetch_one=False):
        if fetch_all:
            return [{"t": t} for t in NOCHE]
        return {}
    monkeypatch.setattr("db.get_user_profile", lambda uid: {"health_profile": {}}, raising=False)
    e = de.estado("11111111-1111-1111-1111-111111111111", _u("2026-11-21T08:22:00"), tz_offset_min=RD,
                  execute_sql_query=_q, usar_cache=False)
    assert e["extendido"] and e["dia"] == "2026-11-20"
    assert de.estado("guest", _u("2026-11-21T08:22:00"), tz_offset_min=RD)["extendido"] is False


def test_ayer_sin_hora_va_a_la_hora_tipica_de_su_franja():
    assert rod.hora_tipica("merienda", "2026-11-20", RD) == "2026-11-20T21:00:00+00:00"   # 17:00 RD
    assert rod.hora_tipica("cena", "2026-11-20", RD) == "2026-11-21T00:00:00+00:00"       # 20:00 RD
    ini, fin = rod.ventana_del_dia("2026-11-20", RD)
    assert ini.isoformat() == "2026-11-20T08:00:00+00:00" and fin.isoformat() == "2026-11-21T08:00:00+00:00"


def test_otra_galleta_se_suma_a_la_merienda_del_dia():
    fila = {"id": "m1", "meal_type": "merienda", "meal_name": "3 paquetes de papas Lay's, galletas Koko y queso",
            "calories": 652, "protein": 19, "carbs": 55, "healthy_fats": 42,
            "ingredients": ["galletas Koko de coco (1 paquete, 27 g)"]}
    s = rod.sumar(fila, meal_name="Galleta Koko de coco", calories=55, protein=1, carbs=7, healthy_fats=3,
                  ingredients=["Galleta Koko de coco (1 unidad, ≈9 g)"])
    assert (s["calories"], s["protein"], s["carbs"], s["healthy_fats"]) == (707, 20, 62, 45)
    assert s["meal_name"] == fila["meal_name"] + " + Galleta Koko de coco"
    assert len(s["ingredients"]) == 2
    assert rod.picoteo_del_dia("u", "cena", "2026-11-20", RD, execute_sql_query=lambda *a, **k: 1 / 0) is None


def test_cableado():
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "[P1-PLAN-LOTE-1145] Sigue en el día anterior" in tools
    assert "_rod.hora_tipica(_meal_type, _dia_destino" in tools
    assert "_rod2.picoteo_del_dia(" in tools
    assert '__import__("dia_extendido").dia_de_hoy(user_id, _off)' in tools
    assert 'hora_para_registro(user_id, meal_type=meal_type)' in (_BACKEND / "db_facts.py").read_text(encoding="utf-8")
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert '@router.get("/dia-activo")' in diary and '@router.post("/dia-activo")' in diary
    assert "dia_extendido" in (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
