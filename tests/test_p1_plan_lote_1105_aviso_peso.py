"""[P1-PLAN-LOTE-1105 · 2026-11-18] «Quiero que también me pida cambiar el peso cada cierto tiempo, ej. cada 2 semanas»."""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import aviso_peso as a  # noqa: E402

HP = {"weight": 180, "weightUnit": "lb"}


def test_cada_dos_semanas_por_defecto_y_lo_que_elija():
    assert a.frecuencia({}) == 14
    assert [a.frecuencia({"aviso_peso_dias": v}) for v in (7, "30", 0, 5, "x")] == [7, 30, 0, 14, 14]


def test_toca_por_su_ultimo_pesaje_o_por_el_alta_si_no_hay_historial():
    hoy = date(2026, 11, 18)
    assert a.que_toca(hoy, HP, None, date(2026, 11, 10)) is None, "8 días desde el alta: aún no"
    assert a.que_toca(hoy, HP, None, date(2026, 11, 1)) == "pregunta"
    con = {**HP, "weight_history": [{"date": "2026-11-12", "weight": 181, "unit": "lb"}]}
    assert a.que_toca(hoy, con, None, date(2026, 1, 1)) is None, "se pesó hace 6 días por su cuenta: no se pregunta"
    assert a.que_toca(hoy, {**HP, "aviso_peso_dias": 0}, None, date(2026, 1, 1)) is None, "apagado"
    assert a.que_toca(hoy, {}, None, date(2026, 1, 1)) is None, "sin peso en el perfil no hay a qué comparar"


def test_un_solo_recordatorio_y_luego_nada_hasta_el_ciclo_siguiente():
    alta = date(2026, 11, 1)
    assert a.que_toca(date(2026, 11, 19), HP, {"preguntado": "2026-11-18"}, alta) is None
    assert a.que_toca(date(2026, 11, 20), HP, {"preguntado": "2026-11-18"}, alta) == "recordatorio"
    assert a.que_toca(date(2026, 11, 25), HP, {"preguntado": "2026-11-18", "recordado": True}, alta) is None
    assert a.que_toca(date(2026, 12, 2), HP, {"preguntado": "2026-11-18", "recordado": True}, alta) == "pregunta"


def test_el_peso_que_se_le_dice_al_coach_va_al_historial_y_da_la_tendencia():
    hp = {"weight": 183, "weight_history": [{"date": "2026-11-01", "weight": 183, "unit": "lb"}]}
    a.anotar_en_historial(hp, 181.8, "lb", date(2026, 11, 18))
    assert hp["weight_history"][-1] == {"date": "2026-11-18", "weight": 181.8, "unit": "lb"}
    a.anotar_en_historial(hp, 181.5, "lb", date(2026, 11, 18))
    assert len(hp["weight_history"]) == 2, "el mismo día se corrige, no se duplica"
    assert a.tendencia(hp) == "-1.5 lb en 17 días (desde 183.0 lb)"
    assert a.tendencia({"weight_history": [{"date": "2026-11-15", "weight": 80, "unit": "kg"},
                                           {"date": "2026-11-18", "weight": 79, "unit": "kg"}]}) == "", "3 días: ruido"


def test_textos_en_los_cinco_idiomas_con_los_dias():
    for loc in ("es-DO", "en-US", "pt-BR", "fr-FR", "it-IT"):
        t, c, m = a.textos(loc, 15)
        assert t and "15" in c and "15" in m
    assert "14" not in a.textos("es-DO", None)[2]


def test_el_cron_pregunta_a_las_8_y_guarda_el_estado(monkeypatch):
    import db_core
    import diary_targets
    import utils_push
    escrito, push, chat = [], [], []
    fila = {"id": "u1", "locale": "es-DO", "hp": dict(HP), "created_at": datetime(2026, 10, 1, tzinfo=timezone.utc)}

    def query(q, p=None, fetch_one=False, fetch_all=False):
        return [fila] if fetch_all else {}

    monkeypatch.setattr(db_core, "execute_sql_query", query)
    monkeypatch.setattr(db_core, "execute_sql_write", lambda q, p: escrito.append(p) or True)
    monkeypatch.setattr(diary_targets, "resolve_user_effective_off_min", lambda uid, tz: 240)
    monkeypatch.setattr(utils_push, "send_push_notification", lambda *x, **k: push.append(x) or True)
    monkeypatch.setattr(a, "_sesion_del_dia", lambda uid, ahora, off: chat.append(uid) or None)
    assert a.enviar_avisos_de_peso(datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)) == 1   # 8:00 en RD
    assert push and push[0][0] == "u1" and escrito and '"preguntado": "2026-11-18"' in escrito[0][1]
    push.clear()
    assert a.enviar_avisos_de_peso(datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)) == 0, "otra hora: nada"


def test_cableado():
    import crons_informe_y_progreso as c
    assert ("aviso_peso", 1, "aviso_peso", "enviar_avisos_de_peso") in c.JOBS
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert '__import__("aviso_peso").anotar_en_historial(_hp, _vp, _up or _hp.get("weightUnit"))' in src
    assert '__import__("aviso_peso").tendencia(_resultado)' in src
