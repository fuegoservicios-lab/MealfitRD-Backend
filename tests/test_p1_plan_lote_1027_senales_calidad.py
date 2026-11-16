"""[P1-PLAN-LOTE-1027 · A-16] Señales implícitas de calidad: registro corregido, borrado en el diario, «te equivocaste»."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import coach_senales_calidad as csc  # noqa: E402

_RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture
def escrituras(monkeypatch):
    import db_core
    vistas = []
    monkeypatch.delenv("MEALFIT_COACH_SENALES_CALIDAD", raising=False)
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p=None, **k: vistas.append((q, p)) or True)
    return vistas


@pytest.mark.parametrize("texto", [
    "Te equivocaste, eran 2 huevos", "no, eso no es lo que comí", "Está mal, era almuerzo", "lo anotaste mal",
    "No comí eso", "pusiste mal la cena", "you're wrong, it was rice", "NO TE CONFUNDISTE?",
])
def test_detecta_la_queja_al_empezar(texto):
    assert csc.es_queja(texto)


@pytest.mark.parametrize("texto", [
    "¿Esto está bien o me paso?", "me comí 3 tacos", "no tengo hambre", "no sé qué cenar", "", None,
    "hoy comí arroz y luego te digo si estuvo mal",   # no arranca con la queja
])
def test_no_la_ve_donde_no_esta(texto):
    assert not csc.es_queja(texto)


def test_registra_sin_texto_del_usuario_y_con_la_antiguedad(escrituras):
    fila = {"meal_type": "almuerzo", "source": "chat", "meal_name": "Pollo con alergia al maní",
            "created_at": datetime.now(timezone.utc) - timedelta(minutes=7)}
    assert csc.registrar("u1", "coach_corrigio", fila=fila)
    q, p = escrituras[0]
    assert "pipeline_metrics" in q and p[2] == csc.NODO
    meta = json.loads(p[3])
    assert meta == {"senal": "coach_corrigio", "franja": "almuerzo", "fuente": "chat", "minutos_desde_registro": 7}
    assert "maní" not in p[3] and "Pollo" not in p[3]


def test_borrado_en_diario_solo_cuenta_lo_que_anoto_el_coach(escrituras):
    assert not csc.borrado_en_diario("u1", {"source": "manual", "created_at": None})
    assert not csc.borrado_en_diario("u1", None)
    assert csc.borrado_en_diario("u1", {"source": "photo", "meal_type": "cena", "created_at": "2026-11-15T10:00:00Z"})
    assert len(escrituras) == 1


def test_invitado_apagado_o_base_caida_no_rompen(escrituras, monkeypatch):
    assert not csc.registrar("guest", "usuario_dice_error") and not csc.registrar(None, "usuario_dice_error")
    import db_core
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
    assert csc.registrar("u1", "usuario_dice_error") is False
    monkeypatch.setenv("MEALFIT_COACH_SENALES_CALIDAD", "0")
    assert csc.registrar("u1", "usuario_dice_error") is False


def test_resumen_calcula_la_tasa(monkeypatch):
    import db_core

    def q(sql, params=None, fetch_one=False, fetch_all=False):
        if "pipeline_metrics" in sql:
            return [{"senal": "coach_corrigio", "n": 3}, {"senal": "borrado_en_diario", "n": 1},
                    {"senal": "usuario_dice_error", "n": 2}]
        return {"n": 40}
    monkeypatch.setattr(db_core, "execute_sql_query", q)
    r = csc.resumen(168)
    assert r["registros_del_coach"] == 40 and r["tasa_registro_corregido_pct"] == 10.0
    assert r["senales"]["usuario_dice_error"] == 2


def test_cableado():
    tools = (_RAIZ / "tools.py").read_text(encoding="utf-8")
    assert tools.count('__import__("coach_senales_calidad").registrar(user_id, "coach_corrigio", fila=_fila)') == 2
    db = (_RAIZ / "db_facts.py").read_text(encoding="utf-8")
    # anotación sin `Optional`: test_live_diary_removed ejecuta la función en un namespace mínimo (5e445039)
    assert 'def delete_consumed_meal(user_id: str, meal_id: str, via: "str | None" = None) -> bool:' in db
    assert '__import__("coach_senales_calidad").borrado_en_diario(user_id, rows[0])' in db
    assert 'delete_consumed_meal(verified_user_id, meal_id, via="diario")' in (_RAIZ / "routers" / "diary.py").read_text(encoding="utf-8")
    import diario_contexto
    assert "source, created_at FROM removed" in diario_contexto.DELETE_WITH_REMOVAL_SQL
    ag = (_RAIZ / "agent.py").read_text(encoding="utf-8")
    assert ag.index('"usuario_dice_error"') > ag.index("[P1-PLAN-LOTE-1017 · A-10] Primera señal")
    assert '"senales_calidad": _senales' in (_RAIZ / "routers" / "system.py").read_text(encoding="utf-8")
