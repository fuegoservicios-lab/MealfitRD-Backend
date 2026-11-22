# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1161 · 2026-11-22] Motor puro del juego 2.0: puntos del día, racha de días registrados que
multiplica, escudos de racha, estrellas por puntos y rangos con divisiones."""
import json
from pathlib import Path

import pytest

import juego_puntos as jp

CASOS = json.loads((Path(__file__).parent / "fixtures" / "juego_puntos_casos.json").read_text(encoding="utf-8"))


def _logros(**kw):
    base = dict(calories=1900, protein=110, meals_count=3, meta_kcal=2000, meta_proteina=120, objetivo="maintenance")
    base.update(kw)
    return jp.logros_del_dia(**base)


def _dias(patron, hoy_vacio=False):
    """patron: cadena de 'x' (registrado, base 10) y '.' (vacío). El último es hoy."""
    dias = [{"iso": f"2026-01-{i + 1:02d}", "registrado": c == "x", "base": 10 if c == "x" else 0}
            for i, c in enumerate(patron)]
    return dias, dias[-1]["iso"]


def test_logros_sin_meta_de_proteina_no_puntua():
    lg = _logros(meta_proteina=None)
    assert lg["proteina"] is False
    assert lg["registro"] is True


def test_agua_apagada_es_none_y_max_40():
    lg = _logros(protein=120, vasos=None, meta_vasos=None)
    assert lg["agua"] is None
    assert jp.puntos_base(lg) == 45


def test_base_maxima_45():
    lg = _logros(protein=120, vasos=8, meta_vasos=8)
    assert lg == {
        "registro": True, "completo": True, "banda": True, "calorias_cerca": False, "calorias_pasado": False,
        "proteina": True, "proteina_parcial": False, "agua": True, "dia_perfecto": True,
    }
    assert jp.puntos_base(lg) == 50


def test_ayuno_2_comidas_y_proteina_parcial():
    lg_ayuno = _logros(calories=1600, protein=88, meals_count=2, meta_kcal=2000, meta_proteina=120)
    assert lg_ayuno["completo"] is True           # 2 comidas ≥ 70 % kcal
    assert lg_ayuno["banda"] is False             # 80 % < 90 %
    assert lg_ayuno["calorias_cerca"] is True     # ±20 % de la meta → +8
    assert lg_ayuno["proteina"] is False
    assert lg_ayuno["proteina_parcial"] is True   # 88/120 ≥ 70 % → +5
    assert jp.puntos_base(lg_ayuno) == 18         # 0 registro + 5 completo + 8 cerca + 5 proteína parcial

    lg_corto = _logros(calories=1000, protein=60, meals_count=2, meta_kcal=2000, meta_proteina=120)
    assert lg_corto["completo"] is False          # 2 comidas < 70 % kcal
    assert lg_corto["calorias_cerca"] is False
    assert lg_corto["proteina_parcial"] is False
    assert jp.puntos_base(lg_corto) == 0          # sin logros en el día ⇒ 0 pts (solo racha)

    lg_pasado_lejos = _logros(calories=2800, protein=60, meals_count=1, meta_kcal=2000, meta_proteina=120)
    assert lg_pasado_lejos["calorias_pasado"] is True
    assert jp.puntos_base(lg_pasado_lejos) == 3

    lg_pasado_cerca = _logros(calories=2300, protein=120, meals_count=3, meta_kcal=2000, meta_proteina=120)
    assert lg_pasado_cerca["calorias_cerca"] is True and lg_pasado_cerca["dia_perfecto"] is False
    assert jp.puntos_base(lg_pasado_cerca) == 23  # 5 completo + 8 cerca + 10 proteína (sin bono perfecto)


def test_dia_vacio_no_da_nada():
    lg = _logros(calories=0, protein=0, meals_count=0, vasos=8, meta_vasos=8)
    assert jp.puntos_base(lg) == 0


@pytest.mark.parametrize("racha,mult", [(0, 1.0), (2, 1.0), (3, 1.15), (6, 1.15), (7, 1.3), (14, 1.45), (29, 1.45), (30, 1.6), (90, 1.6)])
def test_tramos(racha, mult):
    assert jp.tramo_de_racha(racha)["mult"] == mult


def test_tramo_dice_el_siguiente():
    t = jp.tramo_de_racha(9)
    assert t["clave"] == "en_llamas" and t["siguiente_en"] == 14 and t["siguiente_mult"] == 1.45
    assert jp.tramo_de_racha(30)["siguiente_en"] is None


def test_proximo_hito_y_dias_para_escudo():
    assert jp.proximo_hito_racha(0) == 3
    assert jp.proximo_hito_racha(33) == 50
    assert jp.proximo_hito_racha(365) is None
    assert jp.dias_para_escudo(0, 0) == 7
    assert jp.dias_para_escudo(9, 1) == 5
    assert jp.dias_para_escudo(9, 2) is None


def test_rango_con_piso_nunca_baja():
    # [1164] 4 divisiones escaladas: Plata (700..2200) = 700 / 1000 / 1350 / 1750; la 4.ª interna es «Plata I».
    piso_plata_i = jp.nivel_indice("plata", 4)
    bajada = jp.rango_con_piso(1200, piso_plata_i)
    assert bajada["clave"] == "plata" and bajada["division"] == 4 and bajada["romano"] == "I"
    assert bajada["avance"] == 0 and bajada["puntos_para_la_siguiente"] == 1000


def test_hoy_vacio_no_rompe():
    dias, hoy = _dias("xxxx.")
    r = jp.recorrer(dias, hoy)
    assert r["racha"] == 4 and r["escudos_usados"] == []


def test_escudo_salva_un_hueco():
    dias, hoy = _dias("xxxxxxx.x")
    r = jp.recorrer(dias, hoy)
    assert r["racha"] == 8 and r["escudos"] == 0 and r["escudos_usados"] == ["2026-01-08"]


def test_sin_racha_no_gasta_escudo():
    dias, hoy = _dias("xxxxxxx..x")
    r = jp.recorrer(dias, hoy)
    # 7 → 1 escudo; el 1.er hueco lo gasta; el 2.º rompe; luego 1.
    assert r["racha"] == 1 and r["escudos"] == 0 and r["escudos_usados"] == ["2026-01-08"]
    dias, hoy = _dias("..x")
    assert jp.recorrer(dias, hoy)["escudos_usados"] == []


def test_tope_de_escudos_2():
    dias, hoy = _dias("x" * 28)
    assert jp.recorrer(dias, hoy)["escudos"] == 2


def test_puntos_multiplicados_y_mejor_racha():
    dias, hoy = _dias("xxx")
    r = jp.recorrer(dias, hoy)
    assert [r["por_dia"][d["iso"]]["puntos"] for d in dias] == [10, 10, 12]
    assert r["total"] == 32 and r["mejor_racha"] == 3


def test_redondeo_mitades():
    assert jp.redondear(22.5) == 23 and jp.redondear(12.5) == 13 and jp.redondear(12.4) == 12


def test_estrellas_por_puntos():
    u1, u2, u3 = jp.ESTRELLAS_SEMANA
    assert [jp.estrellas_por_puntos(x) for x in (u1 - 1, u1, u2, u3, u3 + 500)] == [0, 1, 2, 3, 3]
    assert jp.faltan_para_otra_estrella_pts(u1 - 5) == 5
    assert jp.faltan_para_otra_estrella_pts(u3) == 0


def test_rango_divisiones():
    r0 = jp.rango_por_puntos(0)
    assert r0["clave"] == "bronce" and r0["division"] == 1 and r0["romano"] == "IV"
    assert r0["siguiente"] == {"clave": "bronce", "division": 2, "romano": "III"}
    plata = dict(jp.RANGOS_PUNTOS)["plata"]
    rp = jp.rango_por_puntos(plata)
    assert rp["clave"] == "plata" and rp["division"] == 1 and rp["romano"] == "IV"
    cima = jp.rango_por_puntos(10 ** 7)
    assert cima["clave"] == "diamante" and cima["division"] == 4 and cima["romano"] == "I" and cima["siguiente"] is None
    assert cima["puntos_para_la_siguiente"] is None and cima["avance"] == 1


def test_rango_nunca_retrocede_al_subir_puntos():
    orden = [(k, d) for k, _ in jp.RANGOS_PUNTOS for d in (1, 2, 3, 4)]
    previo = -1
    for pts in range(0, 20000, 37):
        r = jp.rango_por_puntos(pts)
        i = orden.index((r["clave"], r["division"]))
        assert i >= previo
        previo = i


def test_casos_compartidos():
    for c in CASOS["recorrer"]:
        r = jp.recorrer(c["dias"], c["hoy"])
        for k, v in c["espera"].items():
            assert r[k] == v, (c["nombre"], k)
    for c in CASOS["rango"]:
        r = jp.rango_por_puntos(c["puntos"])
        assert (r["clave"], r["division"], r["romano"], r["puntos_para_la_siguiente"]) == (
            c["clave"], c["division"], c["romano"], c["faltan"]), c
    for c in CASOS["jugador"]:
        r = jp.rango_del_jugador(c["puntos"], c["semanas_perfectas"], c["mejor_racha"], c["piso"])
        m = jp.maestro_progreso(c["puntos"], c["semanas_perfectas"], c["mejor_racha"])
        assert (r["clave"], r["division"], r["romano"], (r["siguiente"] or {}).get("clave"), r["puntos_para_la_siguiente"],
                m["cumplidos"]) == (c["clave"], c["division"], c["romano"], c["siguiente"], c["faltan"], c["cumplidos"]), c
    for c in CASOS["base"]:
        assert jp.puntos_base(c["logros"]) == c["puntos"]
