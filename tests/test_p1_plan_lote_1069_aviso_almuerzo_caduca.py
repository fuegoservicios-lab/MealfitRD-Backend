"""[P1-PLAN-LOTE-1069 · 2026-11-17] El aviso del almuerzo sin respuesta caduca a las 16:00 (antes a las 19:00): a las
18:14 el dueño, sin haber comido, veía «Para almorzar, ¿te provoca…?» como lo primero del chat."""
from datetime import datetime, timezone
from unittest.mock import patch

import db_chat
import db_facts

AVISO = [{"id": "m-almuerzo", "role": "model",
          "content": "Para almorzar, ¿te provoca un filete de pescado a la plancha con una batata mediana?",
          "created_at": "2026-11-17T16:30:34+00:00"}]   # 12:30 RD


def _reconciliar(hora_utc, minuto=0):
    ahora = datetime(2026, 11, 17, hora_utc, minuto, tzinfo=timezone.utc)
    with patch.object(db_chat, "connection_pool", object()), \
         patch.object(db_chat, "execute_sql_query", return_value=AVISO), \
         patch.object(db_chat, "execute_sql_write", return_value=[{"id": "m-almuerzo"}]) as w, \
         patch.object(db_facts, "user_tz_offset_min", return_value=240), \
         patch.object(db_facts, "get_consumed_meals_today", return_value=[]):
        n = db_chat.reconcile_unanswered_proactive_messages("ses", user_id="usr", now_utc=ahora)
    return n, w


def test_a_las_6_14_pm_el_aviso_del_almuerzo_se_poda():
    n, w = _reconciliar(22, 14)   # 18:14 RD
    assert n == 1 and w.call_args[0][1] == ("ses", ["m-almuerzo"])


def test_a_las_3_30_pm_sigue_vigente():
    n, w = _reconciliar(19, 30)   # 15:30 RD
    assert n == 0 and not w.called


def test_a_las_4_pm_ya_caduco():
    n, _ = _reconciliar(20, 0)    # 16:00 RD
    assert n == 1
