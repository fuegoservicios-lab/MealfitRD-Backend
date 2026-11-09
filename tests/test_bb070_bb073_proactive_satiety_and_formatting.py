# -*- coding: utf-8 -*-
"""
Tests automatizados para:
- BB-070: Curva de saciedad fisiológica proactiva y supresión de reclamo de almuerzo por brunch / desayuno pesado.
- BB-071: Formato limpio de macro-prefijo (sin «aprox. ~») y consolidación de vehículos en platos compuestos.
- BB-072: Reconciliación en Neon DB de registros y mensajes de chat.
- BB-073: Suite de regresión y cobertura continua.
"""
from datetime import datetime, timezone, timedelta
import pytest

from proactive_agent import _almuerzo_cubierto_por_brunch, _comida_muy_reciente
from prompts.chat_agent import _CHAT_VOICE_RULES, build_vision_context


class TestBB070ProactiveSatietyCurve:
    """Validación de curva de saciedad calórica y detección de comidas puente en el agente proactivo."""

    def test_heavy_late_breakfast_covers_lunch(self):
        """Caso Moisés: desayuno de 1078 kcal a las 10:50 AM cubre almuerzo fisiológicamente."""
        # Zona horaria -04:00 (user_tz_off = 240 minutos)
        # 10:50 AM local = 14:50 UTC
        consumed_at_utc = datetime(2026, 11, 8, 14, 50, tzinfo=timezone.utc)
        consumed = [
            {
                "meal_name": "Panecillos con queso cottage, 2 tortitas de res y 4 huevos hervidos",
                "calories": 1078,
                "consumed_at": consumed_at_utc.isoformat(),
            }
        ]
        assert _almuerzo_cubierto_por_brunch(consumed, user_tz_off=240) is True

    def test_explicit_brunch_covers_lunch(self):
        """Un brunch explícito de >= 500 kcal entre 10:30 y 14:00 cubre el almuerzo."""
        # 11:30 AM local = 15:30 UTC
        consumed_at_utc = datetime(2026, 11, 8, 15, 30, tzinfo=timezone.utc)
        consumed = [
            {
                "meal_name": "Brunch dominicano con mangú y huevos",
                "calories": 580,
                "consumed_at": consumed_at_utc.isoformat(),
            }
        ]
        assert _almuerzo_cubierto_por_brunch(consumed, user_tz_off=240) is True

    def test_standard_early_breakfast_does_not_cover_lunch(self):
        """Un desayuno estándar y temprano (400 kcal a las 8:00 AM) NO cubre el almuerzo."""
        # 08:00 AM local = 12:00 UTC
        consumed_at_utc = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
        consumed = [
            {
                "meal_name": "Avena con manzana y canela",
                "calories": 380,
                "consumed_at": consumed_at_utc.isoformat(),
            }
        ]
        assert _almuerzo_cubierto_por_brunch(consumed, user_tz_off=240) is False

    def test_empty_consumed_returns_false(self):
        assert _almuerzo_cubierto_por_brunch([], user_tz_off=240) is False
        assert _almuerzo_cubierto_por_brunch(None, user_tz_off=240) is False

    def test_comida_muy_reciente_dynamic_satiety_heavy_meal(self):
        """Comida de >= 850 kcal exige al menos 4.0 horas de separación."""
        # Comida a las 10:50 UTC
        t0 = datetime(2026, 11, 8, 10, 50, tzinfo=timezone.utc)
        consumed = [
            {
                "meal_name": "Desayuno hipercalórico",
                "calories": 1078,
                "consumed_at": t0.isoformat(),
            }
        ]
        # A las 13:00 UTC (2h 10m / 2.16h después) DEBE reportar como muy reciente (bloquea almuerzo)
        t_cron = datetime(2026, 11, 8, 13, 0, tzinfo=timezone.utc)
        assert _comida_muy_reciente(consumed, t_cron, min_horas=2.0) is True

        # A las 15:00 UTC (4h 10m después) ya NO es reciente
        t_tarde = datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc)
        assert _comida_muy_reciente(consumed, t_tarde, min_horas=2.0) is False

    def test_comida_muy_reciente_moderate_meal(self):
        """Comida de >= 600 kcal exige al menos 3.5 horas de separación."""
        t0 = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
        consumed = [
            {
                "meal_name": "Almuerzo regular",
                "calories": 680,
                "consumed_at": t0.isoformat(),
            }
        ]
        # A las 14:30 (2.5h después) -> True
        assert _comida_muy_reciente(consumed, datetime(2026, 11, 8, 14, 30, tzinfo=timezone.utc), min_horas=2.0) is True
        # A las 15:40 (3.66h después) -> False
        assert _comida_muy_reciente(consumed, datetime(2026, 11, 8, 15, 40, tzinfo=timezone.utc), min_horas=2.0) is False

    def test_comida_muy_reciente_light_meal(self):
        """Comida ligera respeta min_horas base."""
        t0 = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
        consumed = [
            {
                "meal_name": "Snack ligero",
                "calories": 250,
                "consumed_at": t0.isoformat(),
            }
        ]
        # A las 14:15 (2.25h después con min_horas=2.0) -> False
        assert _comida_muy_reciente(consumed, datetime(2026, 11, 8, 14, 15, tzinfo=timezone.utc), min_horas=2.0) is False


class TestBB071CleanFormattingAndVehicleConsolidation:
    """Verificación de que las directivas del prompt de chat_agent están activas e impiden redundancias."""

    def test_prompt_contains_clean_macro_prefix_rule(self):
        assert "[BB-071 / P1-CLEAN-MACRO-PREFIX" in _CHAT_VOICE_RULES
        assert "NUNCA combines ambos en la misma cifra" in _CHAT_VOICE_RULES

    def test_prompt_contains_vehicle_consolidation_rule(self):
        assert "[BB-071 / P1-COMPOSITE-PLATE-VEHICLE-CONSOLIDATION" in _CHAT_VOICE_RULES
        assert "NOMBRA EL VEHÍCULO UNA SOLA VEZ" in _CHAT_VOICE_RULES
        assert "Está TERMINANTEMENTE PROHIBIDO duplicar el vehículo" in _CHAT_VOICE_RULES

    def test_build_vision_context_no_aprox_tilde(self):
        ctx = build_vision_context("Foto de sándwich con huevos")
        assert "aprox. **~" not in ctx
