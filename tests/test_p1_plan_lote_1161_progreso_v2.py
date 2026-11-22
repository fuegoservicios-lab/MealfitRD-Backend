# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1161 · 2026-11-22] El resumen del juego habla de puntos: racha de días registrados, estrellas por
puntos, rango con división, puntos de hoy y eventos nuevos sin avalancha al desplegar."""
from datetime import date, timedelta

import juego_puntos as jp
import progreso_semanas as ps

D = date.fromisoformat
META = {"calories": 2000, "macros": {"protein": 120}, "main_goal": "maintenance"}


def _dias(desde, kcal, prot=60, comidas=3):
    return {(desde + timedelta(days=i)).isoformat(): {"calories": k, "protein": prot, "meals_count": comidas if k else 0}
            for i, k in enumerate(kcal)}


def _res(kcal, hoy="2026-11-09", desde="2026-11-03", **kw):
    dias = _dias(D(desde), kcal, **kw)
    metas = {iso: META for iso in dias}
    return ps.armar_resumen(hoy=D(hoy), dia_1=D(desde), dias=dias, metas_por_dia=metas, meta_de_hoy=META,
                            objetivo_del_perfil="maintenance")


def test_resumen_trae_puntos_de_hoy():
    r = _res([2000] * 7, prot=120)
    hp = r["hoy_puntos"]
    assert hp["base"] == 45 and hp["logros"]["agua"] is None
    assert hp["mult"] == 1.3 and hp["puntos"] == 59          # 45 × 1,3 = 58,5 → 59 (7.º día seguido)
    assert r["racha"] == 7 and r["racha_hasta_ayer"] == 6 and r["escudos"] == 1
    assert r["escudos_usados"] == [] and r["proximo_hito_racha"] == 14 and r["dias_para_escudo"] == 7
    assert r["rango"]["clave"] == "bronce" and "division" in r["rango"]
    assert r["puntos"]["total"] == r["total"]["puntos"] > 0


def test_estrellas_de_semana_por_puntos():
    # 1.er día de la semana del 2026-11-08 (domingo): la semana 1 (03-07 nov) quedó cerrada.
    r = _res([1000, 1000, 1000, 1000, 1000, 0, 0], hoy="2026-11-09")
    s1 = r["semanas"][0]
    assert s1["cerrada"] and s1["dias_en_meta"] == 0
    assert s1["puntos"] > 0 and s1["estrellas"] == jp.estrellas_por_puntos(s1["puntos"])


def test_racha_cuenta_registrados_no_meta():
    r = _res([500, 500, 500, 500], hoy="2026-11-06")   # nunca en la banda, pero registró 4 días seguidos
    assert r["racha"] == 4 and r["total"]["dias_en_meta"] == 0


def test_agua_apagada_sin_logro():
    r = _res([2000, 2000])
    assert r["hoy_puntos"]["logros"]["agua"] is None


def test_agua_encendida_puntua():
    dias = _dias(D("2026-11-03"), [2000, 2000])
    for v in dias.values():
        v["vasos"] = 8
    r = ps.armar_resumen(hoy=D("2026-11-04"), dia_1=D("2026-11-03"), dias=dias, metas_por_dia={i: META for i in dias},
                         meta_de_hoy=META, objetivo_del_perfil="maintenance", meta_vasos=8)
    assert r["hoy_puntos"]["logros"]["agua"] is True and r["hoy_puntos"]["base"] == 30


def test_semana_en_curso_trae_7_dias_con_puntos():
    r = _res([2000, 2000, 2000, 2000, 2000, 2000, 2000])
    d = r["semana"]["dias"]
    assert len(d) == 7 and d[0]["iso"] == r["semana"]["empieza"]
    assert r["semana"]["puntos"] == sum(x["puntos"] or 0 for x in d)


def test_primera_evaluacion_v2_siembra_vistos():
    r = _res([2000] * 7, prot=120)
    r = dict(r, hoy_puntos=dict(r["hoy_puntos"], record_hoy=False))
    kv_viejo = {"emitidos": {"semana_cerrada:1": "2026-11-08"}, "vistos": ["semana_cerrada:1"]}
    pendientes, nuevo = ps.eventos_nuevos(ps.hitos_cumplidos(r), kv_viejo, r["hoy"])
    assert pendientes == [] and nuevo["version"] == ps.EVENTOS_VERSION
    assert ps.eventos_de_hoy(nuevo, r["hoy"]) == []           # lo sembrado no se celebra hoy
    # la siguiente evaluación ya emite lo nuevo
    r2 = dict(r, hoy_puntos=dict(r["hoy_puntos"], record_hoy=True, puntos=60))
    pend2, _ = ps.eventos_nuevos(ps.hitos_cumplidos(r2), nuevo, r["hoy"])
    assert [e["codigo"] for e in pend2] == ["record_dia"]


def test_hito_division():
    r = _res([2000] * 7, prot=120)
    r["rango"] = dict(jp.rango_por_puntos(700), estrellas_para_el_siguiente=None)
    claves = {e["clave"] for e in ps.hitos_cumplidos(r)}
    assert "division:plata:1" in claves
    assert not any(c.startswith("rango:") for c in claves)


def test_hito_escudo_ganado_y_salvo():
    r = _res([2000] * 7 + [0] + [2000], hoy="2026-11-11")
    assert r["escudos"] == 0 and r["escudos_usados_semana"] == ["2026-11-10"]
    assert r["escudos_usados"] == ["2026-11-10"]
    codigos = {e["codigo"] for e in ps.hitos_cumplidos(r)}
    assert {"escudo_ganado", "escudo_salvo"} <= codigos


def test_resumen_respeta_piso_de_rango_si_ya_subio(monkeypatch):
    r_base = _res([2000] * 7, prot=120)
    ps.invalidar("u-piso")
    monkeypatch.setattr(ps, "_leer", lambda uid, tz: dict(r_base))
    monkeypatch.setattr(ps, "_sincronizar_semanas", lambda uid, sem: 0)
    monkeypatch.setattr(ps, "_estado_kv", lambda uid: {"version": 2, "emitidos": {"division:plata:2": "2026-11-08"}, "vistos": []})
    monkeypatch.setattr(ps, "_guardar_kv", lambda uid, v: None)
    out = ps.resumen("u-piso")
    assert (out["rango"]["clave"], out["rango"]["division"]) == ("plata", 2)
    ps.invalidar("u-piso")
