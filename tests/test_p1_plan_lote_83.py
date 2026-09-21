# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-83 · 2026-09-17] El «¿ya cenaste?» de las 2:30 de la madrugada. A las 00:38 el dueño anotó la cena de
AYER; el registro quedó como una cena a las 00:38 del día 16, dentro de la franja de cena (17→3), la hora media pasó a
00:38 y el aviso (+1,5 h) cayó a las 2:08: el tick de las 2:30 no vio cena «hoy» (día 17) y preguntó. Dos cierres:
un registro de un día pasado no cuenta para la hora media, y horas de silencio antes de las 6:00 locales."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

# el día del dueño con todas las dependencias simuladas (fixture del lote 72, reutilizada tal cual)
_spec = importlib.util.spec_from_file_location("_t72_para_83", Path(__file__).with_name("test_p1_plan_lote_72.py"))
_t72 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t72)
dia = _t72.dia
_utc = _t72._utc


def test_a_las_2_30_de_la_madrugada_no_se_pregunta_por_la_cena(dia):
    # 06:30 UTC del 17 = 02:30 en RD; la «cena de ayer» anotada a las 00:38 sigue llegando por el stub del SQL
    dia.update(ahora=_utc(17, 6), registros={"Cena": ["00:38"]}, comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == [], "a las 2:30 de la madrugada salió «¿ya cenaste?»"
    assert dia["guardados"] == [] and dia["push"] == []


def test_la_cena_de_verdad_se_sigue_recordando_por_la_noche(dia):
    # 01:30 UTC del 18 = 21:30 en RD; cena habitual a las 19:37 → aviso a las 19:22 → sigue tocando en este tick por
    # la ventana de reintento de 3 h. [P1-PLAN-LOTE-150] Antes la hora era 21:07 (la habitual + 1,5 h); ahora el aviso
    # se ADELANTA 15 min. Lo que este test protege no cambia: la cena de verdad se recuerda de noche, y el silencio
    # de madrugada del lote 83 no se la come.
    dia.update(ahora=_utc(18, 1), registros={"Cena": ["19:40", "19:35"]}, comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Cena"]
    assert "7:22 PM" in dia["prompts"][0]


def test_el_silencio_es_un_knob_y_cero_lo_apaga(dia, monkeypatch):
    monkeypatch.setenv("MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR", "0")
    dia.update(ahora=_utc(17, 6), registros={"Cena": ["00:38"]}, comidas=[], avisos_hoy=[], mensajes=[])
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Cena"], "con el silencio en 0 vuelve la conducta anterior"


def test_el_resumen_de_las_23_no_se_ve_afectado(dia):
    dia.update(ahora=_utc(18, 3), registros={}, comidas=[], avisos_hoy=[], mensajes=[])   # 23:30 en RD
    dia["correr"]()
    assert dia["avisos_nuevos"] == ["Resumen del día"]


def test_un_registro_de_un_dia_pasado_no_entra_en_la_hora_media(monkeypatch):
    import db_facts
    capturado = {}

    def _q(query, params, **k):
        capturado["query"], capturado["params"] = query, params
        return [{"hr": 19, "mn": 40}]

    monkeypatch.setattr(db_facts, "connection_pool", object(), raising=True)
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda uid: 240, raising=True)
    monkeypatch.setattr(db_facts, "execute_sql_query", _q, raising=True)
    assert db_facts.get_avg_meal_hour("u-1", "Cena") == pytest.approx(19.67, abs=0.01)
    q = re.sub(r"\s+", " ", capturado["query"])
    assert "AND consumed_at >= created_at - interval '18 hours'" in q
    assert q.count("consumed_at - make_interval(mins => %s)") == 2   # contrato de P1-AVG-MEAL-HOUR-SIGN intacto
    p = capturado["params"]
    assert len(p) == 5 and p[0] == 240 and p[1] == 240


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 83
    doc = (_BACKEND / "docs" / "recordatorios_de_comida.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-83" in doc and "MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR" in doc
