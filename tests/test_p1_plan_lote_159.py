"""[P1-PLAN-LOTE-159 · 2026-09-22] El aviso de HOY va al chat de HOY.

REPORTE DEL DUEÑO (con captura): un mismo chat con la corrección de la cena de un día y, debajo
del separador «HOY», el aviso del desayuno de otro. «Está mezclando los días en un mismo chat, y
cada día tiene que tener su chat separado».

CAUSA. `get_active_users_for_proactive` elige la sesión MÁS RECIENTE de los últimos 3 días
(`DISTINCT ON (user_id) … ORDER BY created_at DESC`) sin mirar de qué DÍA es. A las 10:00 de hoy,
con el usuario sin haber abierto la app desde ayer, esa sesión es la de ayer — y ahí caía el
aviso.

La regla de «un chat por día» existía, pero solo en el cliente (`chatSessionDay.js`), y solo
corre con la app ABIERTA: a medianoche, con el móvil en el bolsillo, no renueva nada.

Y el daño no se deshacía solo. `debeRenovarse` no renueva una sesión cuyo último mensaje es de
HOY — es la guarda que impide partir una conversación viva. Al escribir el aviso dentro, el
servidor convertía la conversación de ayer en «la de hoy» y la dejaba pegada para siempre:
abrir la app ya no la separaba.

  *Cuando dos lados comparten una regla y solo uno la conoce, el que no la conoce no es neutral:
  la rompe para los dos.*

ARREGLO. `_sesion_del_dia_para_aviso` decide, con el reloj LOCAL del usuario (el mismo que ya
decide la hora del aviso), si la conversación elegida es la de hoy; si no, abre una.

Tooltip-anchor: P1-PLAN-LOTE-159
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import proactive_agent

_AHORA = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)   # 10:00 en RD (UTC-4 ⇒ offset 240)
_RD = 240


def _correr(ultima, session_id="ses-vieja", user_id="u-1", ahora=_AHORA, tz=_RD):
    """Corre el selector con `MAX(created_at)` fingido. Devuelve (sesión, sesiones creadas)."""
    creadas = []
    fila = {"ultima": ultima}
    with patch("proactive_agent.execute_sql_query", return_value=fila), \
         patch("proactive_agent.get_or_create_session",
               side_effect=lambda sid, user_id=None: creadas.append((sid, user_id))):
        elegida = proactive_agent._sesion_del_dia_para_aviso(session_id, user_id, ahora, tz)
    return elegida, creadas


def test_el_aviso_de_hoy_no_cae_en_la_conversacion_de_ayer():
    """EL CASO DE LA CAPTURA."""
    ayer_por_la_noche = _AHORA - timedelta(hours=16)      # ayer 22:00 local
    elegida, creadas = _correr(ayer_por_la_noche)
    assert elegida != "ses-vieja", (
        "El aviso sigue cayendo en el chat de ayer: el cliente dibujará su separador «HOY» en "
        "medio y esa conversación quedará pegada para siempre."
    )
    assert len(creadas) == 1 and creadas[0][1] == "u-1", "El chat del día se crea a su nombre."
    assert creadas[0][0] == elegida


def test_la_conversacion_viva_de_hoy_NO_se_parte():
    """Si ya se habló hoy, el aviso entra ahí: crear otra sería fabricar dos chats el mismo día."""
    hace_una_hora = _AHORA - timedelta(hours=1)
    elegida, creadas = _correr(hace_una_hora)
    assert elegida == "ses-vieja"
    assert creadas == []


def test_el_corte_es_por_el_ULTIMO_mensaje_no_por_el_nacimiento():
    """Una sesión abierta anoche en la que se sigue hablando a las 00:30 es la de HOY.

    Mirar `created_at` la partiría justo mientras el usuario escribe. Aquí el último mensaje es
    de hoy a las 00:30 local aunque la sesión naciera ayer: no se toca.
    """
    medianoche_y_media = datetime(2026, 9, 22, 4, 30, tzinfo=timezone.utc)   # 00:30 en RD
    elegida, creadas = _correr(medianoche_y_media)
    assert elegida == "ses-vieja"
    assert creadas == []


def test_la_frontera_es_el_dia_LOCAL_de_cada_usuario():
    """Las 02:00 UTC son todavía AYER en RD: el mismo instante decide distinto según el huso.

    Sin esto, un dominicano recibiría el «chat de hoy» a las 20:00 de su día anterior.
    """
    # Último mensaje: 22-sep 02:00 UTC = 21-sep 22:00 en RD (UTC-4).
    msg = datetime(2026, 9, 22, 2, 0, tzinfo=timezone.utc)
    # Visto desde RD es de AYER ⇒ chat nuevo.
    elegida_rd, creadas_rd = _correr(msg, tz=_RD)
    assert elegida_rd != "ses-vieja" and len(creadas_rd) == 1
    # Visto desde UTC (offset 0) es de HOY ⇒ se respeta.
    elegida_utc, creadas_utc = _correr(msg, tz=0)
    assert elegida_utc == "ses-vieja" and creadas_utc == []


def test_una_sesion_recien_abierta_y_vacia_se_respeta():
    """Es la que el propio cliente acaba de crear al renovar: otra dejaría dos chats vacíos."""
    elegida, creadas = _correr(None)
    assert elegida == "ses-vieja"
    assert creadas == []


def test_sin_sesion_no_se_inventa_ninguna():
    """P1-PLAN-LOTE-133: quien lleva días sin abrir el chat recibe el aviso EN PANTALLA, a
    propósito, sin que le escribamos en ningún chat."""
    with patch("proactive_agent.execute_sql_query") as q, \
         patch("proactive_agent.get_or_create_session") as c:
        assert proactive_agent._sesion_del_dia_para_aviso(None, "u-1", _AHORA, _RD) is None
    assert q.call_count == 0 and c.call_count == 0


def test_si_la_base_falla_el_aviso_no_se_pierde():
    """Fail-open: un aviso en el chat de ayer es peor que uno de hoy, y mucho mejor que ninguno."""
    with patch("proactive_agent.execute_sql_query", side_effect=RuntimeError("base caída")), \
         patch("proactive_agent.get_or_create_session") as c:
        assert proactive_agent._sesion_del_dia_para_aviso("ses-vieja", "u-1", _AHORA, _RD) == "ses-vieja"
    assert c.call_count == 0


def test_si_no_se_puede_crear_el_chat_del_dia_tampoco():
    """Misma razón: degradar al comportamiento anterior, nunca quedarse mudo."""
    ayer = _AHORA - timedelta(hours=16)
    with patch("proactive_agent.execute_sql_query", return_value={"ultima": ayer}), \
         patch("proactive_agent.get_or_create_session", side_effect=RuntimeError("no se pudo")):
        assert proactive_agent._sesion_del_dia_para_aviso("ses-vieja", "u-1", _AHORA, _RD) == "ses-vieja"


def test_el_bucle_del_cron_usa_el_selector():
    """El ancla: que la decisión esté enchufada donde se elige la sesión del aviso."""
    from pathlib import Path
    src = (Path(proactive_agent.__file__)).read_text(encoding="utf-8")
    i = src.find("_user_tz_off = user_tz_offset_min(user_id)")
    assert i != -1
    bloque = src[i:i + 2200]
    assert "_sesion_del_dia_para_aviso(session_id, user_id, _now_utc, _user_tz_off)" in bloque, (
        "El selector existe pero el bucle no lo llama: el aviso volvería al chat de ayer."
    )
