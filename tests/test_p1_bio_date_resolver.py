"""[P1-BIO-DATE-RESOLVER · 2026-11-03] Test suite para la resolución biológica de fechas conversacionales.
Verifica que las expresiones 'anoche' y 'ayer cené' emitidas en la ventana de madrugada (00:00 - 03:59:59 local)
se anclen a la jornada biológica en curso (days_ago=0) y no se desfacen al día previo (days_ago=1).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import biological_date_resolver as BDR
import agent as A
from prompts import chat_agent


def test_es_ventana_madrugada():
    assert BDR.es_ventana_madrugada(0.0) is True
    assert BDR.es_ventana_madrugada(1.5) is True
    assert BDR.es_ventana_madrugada(3.99) is True
    assert BDR.es_ventana_madrugada(4.0) is False
    assert BDR.es_ventana_madrugada(14.0) is False
    assert BDR.es_ventana_madrugada(None) is False


def test_anoche_en_madrugada_resuelve_cero():
    """A la 01:30 AM, 'anoche cené pechuga' se refiere a la noche que acaba de terminar (days_ago=0)."""
    msgs = [
        SystemMessage(content="🕒 CONTEXTO TEMPORAL ACTUAL: Hoy es Martes, 3 de Noviembre de 2026. La hora local es 01:30 (formato 24 h)."),
        HumanMessage(content="Anoche cené pechuga a la plancha con ensalada"),
    ]
    res = BDR.corregir_days_ago_madrugada(msgs, days_ago_actual=1, hora_local=1.5, user_id="u-test")
    assert res == 0

    # Probando a través de _days_ago_named_by_user
    res_agent = A._days_ago_named_by_user(msgs, hora_local=1.5)
    assert res_agent == 0


def test_ayer_cene_en_madrugada_slot_libre_resuelve_cero(monkeypatch):
    """A las 02:00 AM, 'ayer cené...' con cena vacía en el día biológico se registra en la jornada activa (days_ago=0)."""
    monkeypatch.setattr(BDR, "tiene_slot_registrado_hoy", lambda uid, slot, tz=None: False)

    msgs = [
        HumanMessage(content="Ayer cené dos panes con queso y un vaso de leche"),
    ]
    res = BDR.corregir_days_ago_madrugada(msgs, days_ago_actual=1, hora_local=2.0, user_id="u-test", slot_pedido="cena")
    assert res == 0


def test_ayer_cene_en_madrugada_slot_ocupado_resuelve_uno(monkeypatch):
    """A las 02:00 AM, si la cena de hoy YA estaba registrada, 'ayer cené' se refiere a ayer real (days_ago=1)."""
    monkeypatch.setattr(BDR, "tiene_slot_registrado_hoy", lambda uid, slot, tz=None: True)

    msgs = [
        HumanMessage(content="Ayer cené dos panes con queso"),
    ]
    res = BDR.corregir_days_ago_madrugada(msgs, days_ago_actual=1, hora_local=2.0, user_id="u-test", slot_pedido="cena")
    assert res == 1


def test_dia_semana_explicito_en_madrugada():
    """A la 01:30 AM del martes (calendario), el día biológico en curso es lunes.
    - 'el lunes cené' -> days_ago=0
    - 'el domingo cené' -> days_ago=1
    - 'el sábado cené' -> days_ago=2
    """
    # Martes = weekday 1 en Python (lunes=0, martes=1, ..., domingo=6)
    wk_martes = 1

    msgs_lunes = [HumanMessage(content="El lunes cené pollo al horno")]
    assert BDR.corregir_days_ago_madrugada(
        msgs_lunes, days_ago_actual=None, hora_local=1.5, weekday_actual=wk_martes
    ) == 0

    msgs_domingo = [HumanMessage(content="El domingo cené una pizza con amigos")]
    assert BDR.corregir_days_ago_madrugada(
        msgs_domingo, days_ago_actual=None, hora_local=1.5, weekday_actual=wk_martes
    ) == 1

    msgs_sabado = [HumanMessage(content="El sábado almorcé chivo liniero")]
    assert BDR.corregir_days_ago_madrugada(
        msgs_sabado, days_ago_actual=None, hora_local=1.5, weekday_actual=wk_martes
    ) == 2


def test_anoche_y_ayer_fuera_de_madrugada_mantiene_dias_uno():
    """A las 10:00 AM o 14:00 PM, 'anoche' o 'ayer' deben mantener days_ago=1."""
    msgs_anoche = [HumanMessage(content="Anoche cené mangú")]
    assert BDR.corregir_days_ago_madrugada(msgs_anoche, days_ago_actual=1, hora_local=10.0) == 1

    msgs_ayer = [HumanMessage(content="Ayer cené mangú")]
    assert BDR.corregir_days_ago_madrugada(msgs_ayer, days_ago_actual=1, hora_local=14.0) == 1


def test_antier_en_madrugada():
    """A la 01:30 AM del martes, 'antier' para el humano es el domingo (2 días de calendario atrás).
    Respecto al lunes biológico, el domingo es 1 día atrás (days_ago=1)."""
    msgs = [HumanMessage(content="Antier me comí una sopa de pollo")]
    assert BDR.corregir_days_ago_madrugada(msgs, days_ago_actual=2, hora_local=1.5) == 1


def test_system_prompt_madrugada_contiene_directiva_de_ayer_y_anoche():
    """Verifica que el prompt en madrugada (00:00 - 03:59) advierte al modelo que 'anoche' y 'ayer cené'
    pertenecen a HOY con days_ago=0."""
    from datetime import datetime, timezone

    # 01:30 AM local RD (UTC-4 = 05:30 UTC)
    t_130am = datetime(2026, 11, 3, 5, 30, 0, tzinfo=timezone.utc)
    # Monkeypatch datetime.now en chat_agent para simular la hora
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chat_agent, "datetime", type("MockDT", (), {
            "now": staticmethod(lambda tz=None: t_130am),
            "strptime": datetime.strptime,
        }))
        prompt = chat_agent.build_temporal_context("2026-11-03", tz_offset=240)
        assert "Madrugada antes de las 04:00 AM" in prompt
        assert "days_ago=0" in prompt
        assert "anoche" in prompt.lower()
