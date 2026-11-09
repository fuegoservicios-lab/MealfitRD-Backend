# -*- coding: utf-8 -*-
"""
Tests automatizados para:
- BB-044 / P1-CHAT-ROLLOVER-4AM: Alineación del reinicio automático del chat al corte del día biológico (4:00 AM).
  Garantiza que:
  1. Conversaciones de madrugada (00:00 - 03:59 AM) pertenezcan al día activo anterior.
  2. A partir de las 04:00 AM inicie el nuevo chat sin contaminar la jornada de ayer.
  3. Los avisos proactivos matutinos (ej. 8:30 AM) abran siempre un chat nuevo si el último mensaje fue de madrugada.
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from proactive_agent import _sesion_del_dia_para_aviso
from diary_chat_bridge import get_or_resolve_today_session_for_user


class TestBB044ChatRollover4AM:
    """Validación del corte de 4:00 AM para sesiones de chat y avisos proactivos."""

    def test_proactive_session_kept_during_late_night(self):
        """Un mensaje a las 11:30 PM y una consulta a las 01:00 AM pertenecen al mismo día biológico."""
        tz_off = 240  # UTC-4
        # 11:30 PM local = 03:30 UTC siguiente
        msg_utc = datetime(2026, 11, 8, 3, 30, tzinfo=timezone.utc)
        # Consulta a las 01:00 AM local = 05:00 UTC
        now_utc = datetime(2026, 11, 8, 5, 0, tzinfo=timezone.utc)

        with patch("proactive_agent.execute_sql_query", return_value={"ultima": msg_utc}):
            res = _sesion_del_dia_para_aviso("session-123", "user-abc", now_utc, tz_off)
            assert res == "session-123"

    def test_proactive_session_renewed_in_morning_after_late_night_chat(self):
        """Caso Moisés: si el usuario habló a las 12:01 AM (o 01:00 AM), el aviso de las 8:30 AM debe abrir chat nuevo."""
        tz_off = 240  # UTC-4
        # 12:01 AM local del 8 de nov = 04:01 UTC del 8 de nov (día biológico 7 de nov)
        msg_utc = datetime(2026, 11, 8, 4, 1, tzinfo=timezone.utc)
        # Aviso de desayuno a las 08:30 AM local del 8 de nov = 12:30 UTC del 8 de nov (día biológico 8 de nov)
        now_utc = datetime(2026, 11, 8, 12, 30, tzinfo=timezone.utc)

        with patch("proactive_agent.execute_sql_query", return_value={"ultima": msg_utc}), \
             patch("proactive_agent.get_or_create_session") as mock_create:
            res = _sesion_del_dia_para_aviso("session-old", "user-abc", now_utc, tz_off)
            assert res != "session-old"
            mock_create.assert_called_once()

    def test_diary_bridge_resolves_same_session_before_4am(self):
        """En el puente de sincronización, registros a las 02:00 AM y 03:30 AM usan la misma sesión biológica."""
        tz_off = 240
        # Sesión creada a las 02:00 AM local = 06:00 UTC
        created_utc = datetime(2026, 11, 8, 6, 0, tzinfo=timezone.utc)
        # Registro a las 03:30 AM local = 07:30 UTC
        now_utc = datetime(2026, 11, 8, 7, 30, tzinfo=timezone.utc)

        with patch("diary_chat_bridge._user_tz_offset", return_value=tz_off), \
             patch("diary_chat_bridge._now_utc", return_value=now_utc), \
             patch("diary_chat_bridge.execute_sql_query") as mock_sql:
            mock_sql.return_value = {"id": "sess-bio-today", "created_at": created_utc}

            res = get_or_resolve_today_session_for_user("user-xyz")
            assert res == "sess-bio-today"

    def test_diary_bridge_creates_new_session_after_4am(self):
        """En el puente de sincronización, si la última sesión fue a las 02:00 AM y se registra a las 08:00 AM, abre sesión nueva."""
        tz_off = 240
        # Sesión creada a las 02:00 AM local = 06:00 UTC (día biológico ayer)
        created_utc = datetime(2026, 11, 8, 6, 0, tzinfo=timezone.utc)
        # Registro a las 08:00 AM local = 12:00 UTC (día biológico hoy)
        now_utc = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)

        with patch("diary_chat_bridge._user_tz_offset", return_value=tz_off), \
             patch("diary_chat_bridge._now_utc", return_value=now_utc), \
             patch("diary_chat_bridge.execute_sql_query") as mock_sql, \
             patch("diary_chat_bridge.get_or_create_session") as mock_create:
            mock_sql.return_value = {"id": "sess-old-night", "created_at": created_utc}

            res = get_or_resolve_today_session_for_user("user-xyz")
            assert res != "sess-old-night"
            mock_create.assert_called_once()
