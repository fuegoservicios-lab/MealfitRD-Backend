"""[P1-PLAN-LOTE-1077 · 2026-11-18] «Lo mismo» sin día = el último registro de hoy (≤3 h), buscado por el sistema.

Caso vivo del dueño (18-nov, RD): 00:25 «jugo de toronja con muy poca azúcar»; 00:31 a la duda de la foto «¿Qué bebida
es exactamente?» respondió «Lo mismo» y el coach anotó «bebida con limón»."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import lo_mismo_de_antes as lm

AHORA = datetime(2026, 11, 18, 4, 31, 24, tzinfo=timezone.utc)   # 00:31 en RD
TORONJA = {"meal_name": "Bolita energética con jugo de toronja con muy poca azúcar", "meal_type": "merienda",
           "created_at": AHORA - timedelta(minutes=6), "consumed_at": AHORA - timedelta(minutes=6),
           "ingredients": ["1 bolita energética (≈1 unidad)", "jugo de toronja con muy poca azúcar (≈1 taza)"]}
CENA = {"meal_name": "Arroz con pollo", "meal_type": "cena", "created_at": AHORA - timedelta(hours=2),
        "ingredients": ["arroz blanco (≈1 taza)"]}


def test_caso_del_duenno_la_bebida_es_el_jugo_de_toronja():
    b = lm.bloque_para_chat("u1", "Merendé esto otro\nLo mismo · Un poco de azúcar · Solo queso", 240,
                            ahora_utc=AHORA, filas=[CENA, TORONJA])
    assert "P1-PLAN-LOTE-1077" in b and "«Lo mismo»" in b
    assert "a las 00:25" in b and "jugo de toronja con muy poca azúcar (≈1 taza)" in b
    assert "NO con lo que aparenta la foto" in b


@pytest.mark.parametrize("texto", ["Lo mismo", "la misma", "Igual", "lo de antes", "es lo mismo que antes",
                                   "Merendé esto\nLa misma · 2 bolitas", "me tomé lo mismo que hace rato",
                                   "¡Lo mismo!"])
def test_detecta_lo_mismo(texto):
    assert lm.detectar(texto)


@pytest.mark.parametrize("texto", ["lo mismo de ayer", "las mismas galletas de anoche", "me da igual",
                                   "igual me comí dos", "Merendé esto\n1 bolita · Jugo de toronja",
                                   "almorcé lo mismo el lunes", "", None])
def test_no_es_lo_mismo_de_hoy(texto):
    assert lm.detectar(texto) is None


def test_sin_registro_reciente_o_invitado_no_inventa():
    viejo = {**TORONJA, "created_at": AHORA - timedelta(hours=4)}
    assert lm.bloque_para_chat("u1", "Lo mismo", 240, ahora_utc=AHORA, filas=[viejo]) == ""
    assert lm.bloque_para_chat("u1", "Lo mismo", 240, ahora_utc=AHORA, filas=[]) == ""
    assert lm.bloque_para_chat("guest", "Lo mismo", 240, ahora_utc=AHORA, filas=[TORONJA]) == ""


def test_elige_lo_ultimo_que_dijo_por_created_at():
    # el almuerzo atrasado se guarda a las 13:00 (lote 1075) pero se DIJO hace 1 min: es lo último
    atrasado = {"meal_name": "Pollo guisado", "created_at": AHORA - timedelta(minutes=1),
                "consumed_at": AHORA - timedelta(hours=11)}
    assert lm.elegir([TORONJA, atrasado], AHORA) is atrasado


def test_nunca_lanza():
    assert lm.bloque_para_chat("u1", "Lo mismo", "x", ahora_utc=AHORA, filas=[{"created_at": "basura"}]) == ""


def test_engancha_en_el_prompt_del_chat():
    src = (Path(__file__).resolve().parents[1] / "agent.py").read_text(encoding="utf-8")
    i = src.index('__import__("referencia_a_registro").bloque_para_chat(user_id, prompt, tz_offset)')
    assert '__import__("lo_mismo_de_antes").bloque_para_chat(user_id, prompt, tz_offset)' in src[i:i + 400]
