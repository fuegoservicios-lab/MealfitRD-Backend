# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1161 · 2026-11-22] El coach, el resumen y el aviso del domingo hablan del juego por puntos."""
import aviso_semanal
import coach_progreso as cp
import resumen_del_domingo as rd

SEMANA = {"numero": 9, "dias_en_meta": 2, "dias_registrados": 6, "dias_bajo": 3, "dias_sobre": 1, "estrellas": 2,
          "puntos": 190, "proteina_media": 100}


def _res():
    return {
        "semana": {"numero": 10, "empieza": "2026-11-15", "termina": "2026-11-21", "dia": 3, "dias_en_meta": 1,
                   "estrellas": 1, "puntos": 120, "faltan_puntos_para_otra_estrella": 40, "faltan_para_otra_estrella": 2},
        "hoy_puntos": {"puntos": 42, "mult": 1.5}, "racha": 9, "mejor_racha": 12,
        "racha_tramo": {"mult": 1.5, "clave": "en_llamas"}, "escudos": 2, "escudos_max": 2,
        "semana_pasada": {"numero": 9, "puntos": 190, "estrellas": 2, "dias_en_meta": 2},
        "total": {"semanas": 10, "estrellas": 8, "puntos": 1000},
        "rango": {"clave": "plata", "division": 2, "siguiente": {"clave": "plata", "division": 3},
                  "puntos_para_la_siguiente": 400},
        "eventos_de_hoy": ["division:plata:2", "escudo:2026-11-16", "salvo:2026-11-14", "record:2026-11-17"],
    }


def test_bloque_puntos_racha_escudos_y_division():
    t = cp.texto_del_bloque(_res())
    assert "Hoy: 42 pts (×1,5)." in t
    assert "Esta semana: 120 pts, 1★; faltan 40 pts para la 2.ª estrella." in t
    assert "Racha: 9 días registrando (×1,5; mejor: 12); escudos 2/2." in t
    # [1163] divisiones IV→I: la 2.ª interna de Plata se ve «Plata III» y la 3.ª «Plata II».
    assert "rango Plata III (faltan 400 pts para Plata II)" in t
    assert "subió a Plata III" in t and "ganó un escudo de racha" in t and "un escudo salvó su racha" in t
    assert "récord de puntos en un día (42 pts)" in t
    assert len(t.split()) <= 110


def test_domingo_con_puntos_y_racha_registrando():
    es = rd.mensaje("es-DO", SEMANA, racha=9)
    assert "Sumaste 190 pts." in es and "Llevas 9 días seguidos registrando." in es
    en = rd.mensaje("en-US", SEMANA, racha=9)
    assert "You earned 190 pts." in en and "9-day logging streak" in en


def test_aviso_del_domingo_con_puntos():
    _, cuerpo = aviso_semanal.texto_del_aviso("es-DO", 10, SEMANA)
    assert cuerpo == "Empieza tu Semana 10. La 9 cerró con ★★ · 190 pts."


def test_regla_r5_habla_de_puntos():
    from prompts import chat_agent
    src = __import__("inspect").getsource(chat_agent)
    assert "R5. PROGRESO [P1-PLAN-LOTE-1059]: puntos, estrellas, racha, escudos y rango" in src
