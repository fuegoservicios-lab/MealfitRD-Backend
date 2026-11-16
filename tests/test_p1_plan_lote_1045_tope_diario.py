# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1045 · D5/A-05] Tope diario: las fotos cuentan, las cuentas de prueba quedan exentas y el aviso dice
la hora real de liberación (decisión del dueño, 16-nov)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chat_hard_turn as cht
import coach_tope_diario as ctd

_BACKEND = Path(__file__).resolve().parent.parent


def test_las_fotos_cuentan_en_el_gasto():
    src = (_BACKEND / "chat_hard_turn.py").read_text(encoding="utf-8")
    assert "(node LIKE 'chat%%' OR node = 'vision_scan')" in src
    assert ctd.FILTRO_NODOS_SQL == "(node LIKE 'chat%%' OR node = 'vision_scan')"


def test_cuenta_de_prueba_exenta(monkeypatch):
    import cuentas_prueba
    monkeypatch.setattr(cuentas_prueba, "marca_viva", lambda uid: {"user_id": uid} if uid == "u-prueba" else None)
    monkeypatch.setattr("admin_acceso.admin_ids", lambda: set())
    assert cht.user_exempt_from_chat_budget("u-prueba") is True
    assert cht.user_exempt_from_chat_budget("u-normal") is False


def test_libre_desde_resta_los_viejos_hasta_bajar_del_tope():
    t0 = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)
    filas = [{"c": 100_000, "t": t0}, {"c": 100_000, "t": t0 + timedelta(hours=2)},
             {"c": 100_000, "t": t0 + timedelta(hours=5)}]   # 0,30 en 24 h con tope 0,25
    q = lambda *a, **k: filas
    # quitar el 1.º deja 0,20 < 0,25 ⇒ libre cuando el 1.º cumple 24 h
    assert ctd.libre_desde("u", 0.25, execute_sql_query=q) == t0 + timedelta(hours=24)
    # con tope 0,15 hay que esperar también al 2.º
    assert ctd.libre_desde("u", 0.15, execute_sql_query=q) == t0 + timedelta(hours=26)


def test_libre_desde_falla_abierto():
    def rota(*a, **k):
        raise RuntimeError("db")
    assert ctd.libre_desde("u", 0.25, execute_sql_query=rota) is None


def test_la_hora_sale_en_hora_local_y_en_su_idioma():
    libre = datetime(2026, 11, 17, 18, 5, tzinfo=timezone.utc)
    assert ctd.frase_de_la_hora(libre, 240, "es") == "Podrás volver a escribirle a partir de las 14:05."
    assert ctd.frase_de_la_hora(libre, 240, "en") == "You'll be able to message the coach again from 14:05."
    assert ctd.frase_de_la_hora(None, 240, "es") == ""


def test_el_aviso_cambia_unas_horas_por_la_hora():
    import agent
    libre = datetime(2026, 11, 17, 18, 5, tzinfo=timezone.utc)
    for loc, idioma in (("es-DO", "es"), ("en-US", "en"), ("pt-BR", "pt"), ("fr-FR", "fr"), ("it-IT", "it")):
        txt = agent._budget_exceeded_graceful_reply({"locale": loc}, libre, 240)
        assert "14:05" in txt, loc
        assert agent._BUDGET_CAP_COLA_SIN_HORA[idioma] not in txt, loc
        assert agent._BUDGET_CAP_TEXTOS[idioma].endswith(agent._BUDGET_CAP_COLA_SIN_HORA[idioma]), loc
    assert agent._budget_exceeded_graceful_reply({"locale": "es-DO"}) == agent._BUDGET_CAP_TEXTOS["es"]
