# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1163 · 2026-11-22] Rangos como en los videojuegos y el rango especial.

El dueño: «en los videojuegos es Bronce IV lo más bajo y va subiendo mientras bajas» y «agrega otro rango especial
que sea muy difícil de llegar». Cuatro divisiones por rango que se ven IV → I (la `division` interna sigue
ascendiendo: 1 = la entrada), +30 pts por semana perfecta, y MAESTRO encima de Diamante con TRES requisitos.
"""
from datetime import date, timedelta

import coach_progreso as cp
import juego_puntos as jp
import progreso_semanas as ps

D = date.fromisoformat
META = {"calories": 2000, "macros": {"protein": 120}, "main_goal": "maintenance"}


def _res(kcal, hoy, desde, prot=120):
    dias = {(D(desde) + timedelta(days=i)).isoformat(): {"calories": k, "protein": prot, "meals_count": 3 if k else 0}
            for i, k in enumerate(kcal)}
    return ps.armar_resumen(hoy=D(hoy), dia_1=D(desde), dias=dias, metas_por_dia={i: META for i in dias},
                            meta_de_hoy=META, objetivo_del_perfil="maintenance")


def test_divisiones_se_ven_de_iv_a_i():
    assert [jp.division_romana(d) for d in (1, 2, 3, 4)] == ["IV", "III", "II", "I"]
    assert jp.division_romana(None) is None and jp.division_romana(5) is None
    vistos = [jp.rango_por_puntos(p)["romano"] for p in (540, 810, 1080, 1350)]
    assert vistos == ["IV", "III", "II", "I"]
    assert cp._rango_txt({"clave": "plata", "division": 1}) == "Plata IV"
    assert cp._rango_txt({"clave": "maestro", "division": None}) == "Maestro"


def test_maestro_pide_los_tres_requisitos():
    assert jp.MAESTRO_PUNTOS == 18900
    for pts, perf, racha in ((18900, 7, 100), (18900, 8, 99), (18899, 8, 100)):
        r = jp.rango_del_jugador(pts, perf, racha)
        assert r["clave"] == "diamante" and r["romano"] == "I" and r["siguiente"]["clave"] == "maestro"
        assert jp.maestro_progreso(pts, perf, racha)["cumplidos"] == 2
    m = jp.rango_del_jugador(18900, 8, 100)
    assert m["clave"] == "maestro" and m["siguiente"] is None and m["avance"] == 1


def test_diamante_i_apunta_a_maestro_con_los_puntos_que_faltan():
    r = jp.rango_del_jugador(17550, 3, 40)
    assert (r["clave"], r["romano"], r["hasta"], r["puntos_para_la_siguiente"]) == ("diamante", "I", 18900, 1350)
    assert r["avance"] == 0.5


def test_maestro_nunca_se_pierde_por_el_piso():
    assert jp.rango_del_jugador(100, 0, 0, jp.indice_maestro())["clave"] == "maestro"
    assert jp.nivel_indice("maestro", None) == jp.indice_maestro() > jp.nivel_indice("diamante", 4)


def test_piso_viejo_de_3_divisiones_no_infla_el_rango():
    """Las claves viejas `division:<rango>:<d>` (3 divisiones) apuntan a un peldaño nuevo que empieza ANTES o igual:
    nadie sube de división por el cambio, solo se mantiene."""
    viejo = {"bronce": (0, 540), "plata": (540, 1620), "oro": (1620, 4050), "platino": (4050, 8100)}
    for clave, (desde, hasta) in viejo.items():
        for d in (1, 2, 3):
            inicio_viejo = desde + jp.redondear((hasta - desde) * (d - 1) / 3)
            nuevo = jp.escalera()[jp.nivel_indice(clave, d)]
            assert nuevo[2] <= inicio_viejo, (clave, d)


def test_semana_perfecta_da_30_de_bono_y_no_toca_las_estrellas():
    # 7 días perfectos (registro+completo+banda+proteína = 40 base, racha que multiplica) → semana con 3★.
    r = _res([2000] * 21 + [0], hoy="2026-11-24", desde="2026-11-03")
    perfectas = [s for s in r["semanas"] if s["estrellas"] == 3]
    assert len(perfectas) >= 2
    assert all(s["bono"] == (30 if s["estrellas"] == 3 else 0) for s in r["semanas"])
    assert all(s["estrellas"] == jp.estrellas_por_puntos(s["puntos"]) for s in r["semanas"])   # el bono no cuenta
    assert r["puntos"]["bono_semanas"] == 30 * len(perfectas)
    assert r["puntos"]["total"] == sum(s["puntos"] for s in r["semanas"]) + 30 * len(perfectas)
    assert r["total"]["puntos"] == r["puntos"]["total"]
    assert r["maestro"]["requisitos"][1] == {"clave": "semanas_perfectas", "actual": len(perfectas), "meta": 8,
                                             "ok": False}


def test_hito_maestro_y_piso_desde_el_kv():
    res = {"semanas": [], "total": {"estrellas": 0}, "racha": 0, "semana": {"numero": 1},
           "rango": {"clave": "maestro", "division": None}}
    assert any(e["codigo"] == "maestro" and e["clave"] == "maestro" for e in ps.hitos_cumplidos(res))
    assert ps._piso_rango_de_kv({"emitidos": {"maestro": "2026-12-01"}}) == jp.indice_maestro()
    assert "llegó a MAESTRO" in cp._linea_de_hoy(["maestro"], {})


def test_eventos_v3_siembran_lo_cumplido_al_desplegar():
    assert ps.EVENTOS_VERSION == 3
    cumplidos = [{"clave": "division:plata:3", "codigo": "division", "datos": {}}]
    pend, nuevo = ps.eventos_nuevos(cumplidos, {"version": 2, "emitidos": {"division:plata:2": "2026-11-20"},
                                                "vistos": ["division:plata:2"]}, "2026-11-22")
    assert pend == [] and nuevo["version"] == 3 and "division:plata:3" in nuevo["vistos"]
