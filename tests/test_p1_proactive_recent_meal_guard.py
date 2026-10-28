# -*- coding: utf-8 -*-
"""[P1-MIN-MEAL-GAP] Tests para la separación mínima entre comidas y recordatorios proactivos.

Problema histórico (caso real 2026-10-27 usuario darianbobea229@gmail.com):
- El usuario registró un almuerzo tarde a las 06:23 p. m. (18:23) con ~495 kcal.
- A las 07:30 p. m. (19:30), apenas 1 hora y 7 minutos después, el coach proactivo disparó
  el recordatorio de Cena ("Ponle buen sabor a la cena: prepara una tortilla...").
- A una persona que acaba de almorzar hace 1 hora no se le debe sugerir cenar de inmediato;
  debe respetarse un umbral mínimo (default 2.5 horas para la cena) y posponer el aviso al
  siguiente tick dentro de la ventana de reintento.
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from proactive_agent import (
    _min_horas_separacion_comida,
    _comida_muy_reciente,
    _almuerzo_muy_reciente,
    _comida_ya_registrada,
)


def test_min_horas_separacion_defaults():
    assert _min_horas_separacion_comida("Cena") == 2.5
    assert _min_horas_separacion_comida("cena") == 2.5
    assert _min_horas_separacion_comida("Merienda") == 2.0
    assert _min_horas_separacion_comida("Almuerzo") == 2.0
    assert _min_horas_separacion_comida("Desayuno") == 2.0


def test_min_horas_separacion_knob_override(monkeypatch):
    monkeypatch.setenv("MEALFIT_PROACTIVE_MIN_GAP_CENA_H", "3.0")
    monkeypatch.setenv("MEALFIT_PROACTIVE_MIN_GAP_ALMUERZO_H", "1.5")
    assert _min_horas_separacion_comida("Cena") == 3.0
    assert _min_horas_separacion_comida("Almuerzo") == 1.5


def test_comida_muy_reciente_bloquea_si_menos_de_umbral():
    ahora = datetime(2026, 10, 27, 19, 30, tzinfo=timezone.utc)
    # Almuerzo registrado a las 18:23 (hace 1h 7m = 1.12 horas)
    comidas = [
        {
            "meal_type": "almuerzo",
            "meal_name": "Carne de res guisada con arroz",
            "calories": 495,
            "consumed_at": "2026-10-27T18:23:00Z",
        }
    ]
    # Con umbral de 2.5h para la cena, es muy reciente -> True
    assert _comida_muy_reciente(comidas, ahora, min_horas=2.5) is True


def test_comida_muy_reciente_permite_si_mas_de_umbral():
    ahora = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)
    # Almuerzo registrado a las 18:23 (hace 2h 37m = 2.61 horas)
    comidas = [
        {
            "meal_type": "almuerzo",
            "meal_name": "Carne de res guisada con arroz",
            "calories": 495,
            "consumed_at": "2026-10-27T18:23:00Z",
        }
    ]
    # Con umbral de 2.5h, ya pasaron 2.61h -> False (se puede enviar la cena)
    assert _comida_muy_reciente(comidas, ahora, min_horas=2.5) is False


def test_comida_muy_reciente_con_datetime_aware_y_naive():
    ahora = datetime(2026, 10, 27, 19, 30, tzinfo=timezone.utc)
    # Aware datetime
    comidas_aware = [{
        "meal_type": "almuerzo",
        "calories": 500,
        "consumed_at": datetime(2026, 10, 27, 18, 0, tzinfo=timezone.utc),
    }]
    assert _comida_muy_reciente(comidas_aware, ahora, min_horas=2.5) is True

    # Naive datetime
    comidas_naive = [{
        "meal_type": "almuerzo",
        "calories": 500,
        "consumed_at": datetime(2026, 10, 27, 18, 0),
    }]
    assert _comida_muy_reciente(comidas_naive, ahora, min_horas=2.5) is True


def test_comida_muy_reciente_clock_skew_tolerancia():
    ahora = datetime(2026, 10, 27, 19, 30, 0, tzinfo=timezone.utc)
    # Servidor/cliente con 30 segundos de adelanto en consumed_at
    comidas = [{
        "meal_type": "merienda",
        "calories": 150,
        "consumed_at": datetime(2026, 10, 27, 19, 30, 30, tzinfo=timezone.utc),
    }]
    assert _comida_muy_reciente(comidas, ahora, min_horas=2.0) is True


def test_comida_muy_reciente_ignora_suplementos():
    ahora = datetime(2026, 10, 27, 19, 30, tzinfo=timezone.utc)
    # Usuario tomó creatina o vitamina hace 15 minutos (19:15)
    comidas = [{
        "meal_type": "suplemento",
        "meal_name": "Creatina monohidratada",
        "calories": 0,
        "consumed_at": "2026-10-27T19:15:00Z",
    }]
    # Un suplemento no es comida y no debe posponer la cena
    assert _comida_muy_reciente(comidas, ahora, min_horas=2.5) is False


def test_comida_muy_reciente_ignora_registros_menores_sin_tipo():
    ahora = datetime(2026, 10, 27, 19, 30, tzinfo=timezone.utc)
    # Un café solo o agua con 2 kcal sin meal_type
    comidas = [{
        "meal_type": None,
        "meal_name": "Café negro solo",
        "calories": 2,
        "consumed_at": "2026-10-27T19:15:00Z",
    }]
    assert _comida_muy_reciente(comidas, ahora, min_horas=2.5) is False


def test_almuerzo_muy_reciente_filtra_por_almuerzo():
    ahora = datetime(2026, 10, 27, 15, 30, tzinfo=timezone.utc)
    # Solo desayuno registrado hace 1h
    comidas_desayuno = [{
        "meal_type": "desayuno",
        "meal_name": "Huevos revueltos",
        "calories": 300,
        "consumed_at": "2026-10-27T14:30:00Z",
    }]
    assert _almuerzo_muy_reciente(comidas_desayuno, ahora, min_horas=2.0) is False

    # Almuerzo registrado hace 1h
    comidas_almuerzo = [{
        "meal_type": "almuerzo",
        "meal_name": "Pechuga a la plancha con yuca",
        "calories": 500,
        "consumed_at": "2026-10-27T14:30:00Z",
    }]
    assert _almuerzo_muy_reciente(comidas_almuerzo, ahora, min_horas=2.0) is True


def test_seleccion_candidata_pospone_cena_tras_almuerzo_tarde():
    """Simulación del bucle de selección en proactive_agent:
    Candidata es Cena a las 19:30, pero el usuario almorzó a las 18:23.
    Cena debe ser pospuesta."""
    _now_utc = datetime(2026, 10, 27, 19, 30, tzinfo=timezone.utc)
    consumed = [
        {
            "meal_type": "almuerzo",
            "meal_name": "Carne de res",
            "calories": 495,
            "consumed_at": "2026-10-27T18:23:00Z",
        }
    ]
    candidatas = [
        {"meal": "Cena", "hora": "7:30 PM", "tono": "normal", "meal_rate": 1.0, "meal_total": 5, "atraso": 0, "orden": 3}
    ]

    elegida = None
    for cand in candidatas:
        if _comida_ya_registrada(consumed, cand["meal"]):
            continue
        _gap_h = _min_horas_separacion_comida(cand["meal"])
        if cand["meal"] in ("Almuerzo", "Merienda", "Cena") and _comida_muy_reciente(consumed, _now_utc, min_horas=_gap_h):
            continue
        elegida = cand
        break

    # Cena fue pospuesta porque el almuerzo fue hace solo 1.1h
    assert elegida is None


def test_seleccion_candidata_dispara_cena_cuando_paso_umbral():
    """A las 21:00 (2h 37m tras almuerzo de 18:23), el umbral de 2.5h ya se cumplió.
    Cena debe ser elegida normalmente."""
    _now_utc = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)
    consumed = [
        {
            "meal_type": "almuerzo",
            "meal_name": "Carne de res",
            "calories": 495,
            "consumed_at": "2026-10-27T18:23:00Z",
        }
    ]
    candidatas = [
        {"meal": "Cena", "hora": "7:30 PM", "tono": "normal", "meal_rate": 1.0, "meal_total": 5, "atraso": 90, "orden": 3}
    ]

    elegida = None
    for cand in candidatas:
        if _comida_ya_registrada(consumed, cand["meal"]):
            continue
        _gap_h = _min_horas_separacion_comida(cand["meal"])
        if cand["meal"] in ("Almuerzo", "Merienda", "Cena") and _comida_muy_reciente(consumed, _now_utc, min_horas=_gap_h):
            continue
        elegida = cand
        break

    assert elegida is not None
    assert elegida["meal"] == "Cena"


def test_proactive_prompt_variedad_reglas_anti_plantilla():
    from prompts.proactive import PROACTIVE_PROMPT
    assert "PROHIBIDO arrancar repetitivamente con plantillas como «Ya puedes" in PROACTIVE_PROMPT
    assert "PROHIBIDO terminar todos los avisos con fórmulas clónicas" in PROACTIVE_PROMPT
    assert "¡Que lo disfrutes!" in PROACTIVE_PROMPT
    assert "VARIEDAD DE APERTURAS:" in PROACTIVE_PROMPT

