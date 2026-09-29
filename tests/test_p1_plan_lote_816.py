# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-816 · 2026-09-29] Los filtros de compra única miden el día del CICLO, no el de la ventana.

`constants.rebase_pending_chunk_offsets` reescribe el `days_offset` de la cola contra el ancla móvil (el shift la lleva a
hoy), y los filtros de durabilidad lo leían como el día desde la compra. Plan vivo 6594aae1 (30 días, sin congelador):
bloque 4 con columna 5 y rebanada 11; los bloques 2 y 3 corrieron con columna 1 (rebanadas 3 y 7 inferidas: su snapshot
está nulo en DB) — con columna 1 la Nevera envejecida no exige nada y el plan lleva pescado fresco los días 7 y 8 del
ciclo (09-29 y 09-30, desde la compra del 09-23).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ai_helpers as ah  # noqa: E402
import compra_unica as cu  # noqa: E402

_KNOB = "MEALFIT_SINGLE_TRIP_CYCLE_DAY_TRUE"
SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"},
          "diet": {"type": "balanced", "allergies": []}}


def _fd(col, rebanada=None, **extra):
    fd = {"_plan_policy_effective": SINGLE, "_days_offset": col, "current_pantry_ingredients": []}
    if rebanada is not None:
        fd["_blueprint_slice"] = {"days_offset": rebanada, "days_count": 4}
    fd.update(extra)
    return fd


def _nevera_virtual(fd, nombres):
    lista = [{"name": n, "_compra_unica": 30} for n in nombres]
    return cu.nevera_virtual(fd, task_id=7, user_id="u1",
                             consultar=lambda sql, params=(), **k: {"activa": lista, "mensual": lista})


# ─────────────────────────────────────────────── el caso medido: 6594aae1, bloque 4 (columna 5, rebanada 11)

def test_6594aae1_bloque_4_los_filtros_ven_el_dia_11():
    import dia_del_ciclo as ddc
    fd = _fd(5, 11)
    assert ddc.dia(fd) == 11
    # sembrador: último día del bloque = 14 ⇒ aguantar 15 días (con la columna: 8 ⇒ 9, y el brócoli de 10 pasaba)
    out = ah._single_trip_durable_filter(["Brócoli", "Yogur griego", "Atún en agua", "Arroz blanco"], fd, 4)
    assert out == ["Atún en agua", "Arroz blanco"], out
    # Nevera envejecida: primer día = 11 ⇒ 12 días (con la columna: 6, y lechuga/tomate de 7 pasaban)
    assert ah._age_pantry_for_block(["Lechuga", "Tomate", "Huevo", "Arroz blanco"], fd, 4) == ["Huevo", "Arroz blanco"]
    # Nevera virtual: lo que LLEGA al día 12
    nv = _nevera_virtual(fd, ["Atún en agua", "Sardinas en lata", "Huevo", "Arroz blanco", "Lechuga", "Brócoli", "Casabe"])
    assert nv["_nevera_virtual"] is True
    assert "Lechuga" not in nv["current_pantry_ingredients"] and "Brócoli" not in nv["current_pantry_ingredients"]
    assert {"Atún en agua", "Huevo", "Arroz blanco", "Casabe"} <= set(nv["current_pantry_ingredients"])
    # cerrador de proteína: el día 9 de la ventana es el 15 del ciclo (índice 14): el yogur (14 días) ya no llega
    assert cu.candidatos_del_dia(["yogur griego", "atun en agua", "huevo"], {"day": 9}, fd) == ["atun en agua", "huevo"]


def test_un_pescado_fresco_sin_congelador_no_pasa_el_dia_11():
    """Columna 1 (así corrieron los bloques 2 y 3 de 6594aae1) y rebanada 11. Con la columna los tres filtros quedan
    dentro de los 3 días libres sin congelador y NO exigen nada: el pescado pasaba. Cada assert es rojo sin el 816
    (verificado contra 349ccf30: con `_fd(5, 11)` la columna 5 ya bastaba para echar al pescado y el test no probaba
    nada — ronda de corrección del revisor)."""
    fd = _fd(1, 11)
    pool = ["Filete de pescado fresco", "Camarones", "Huevo", "Arroz blanco"]
    # sembrador, bloque de 2 días: con la columna el último día es el 2 (sin exigencia); con el ciclo, el 12
    assert ah._single_trip_durable_filter(pool, fd, 2) == ["Huevo", "Arroz blanco"]
    # Nevera envejecida: primer día 1 con la columna (sin exigencia); 11 con el ciclo
    assert ah._age_pantry_for_block(pool, fd, 4) == ["Huevo", "Arroz blanco"]
    # cerrador: el día 2 de la ventana es el índice 1 con la columna (< 3, devuelve todo); 11 con el ciclo
    assert cu.candidatos_del_dia(["filete de pescado", "atun en agua"], {"day": 2}, fd) == ["atun en agua"]


def test_bloque_3_con_columna_1_el_pescado_ya_no_entra_el_dia_8():
    """Así corrió el bloque 3 de 6594aae1 el 28-sep: columna 1 ⇒ `single_trip_requirements(…, 1)` = None. La rebanada 7
    es inferida (el snapshot de ese bloque está nulo en DB); el 8 del nombre es su primer día, 1-based."""
    fd = _fd(1, 7)
    assert ah._age_pantry_for_block(["Filete de pescado fresco", "Arroz blanco", "Huevo"], fd, 4) == [
        "Arroz blanco", "Huevo"]


# ─────────────────────────────────────────────── sin fuente de verdad, o con el knob apagado: la columna

def test_sin_rebanada_ni_ciclo_la_columna():
    import dia_del_ciclo as ddc
    fd = _fd(5)
    assert ddc.dia(fd) == 5 and ddc.desplazamiento(fd) == 0 and ddc.indice_del_chain(fd, 3) == 3
    assert "Brócoli" in ah._single_trip_durable_filter(["Brócoli", "Arroz blanco"], fd, 4)          # último día 8
    assert ah._age_pantry_for_block(["Lechuga", "Huevo"], fd, 4) == ["Lechuga", "Huevo"]           # primer día 5
    assert cu.candidatos_del_dia(["yogur griego", "huevo"], {"day": 9}, fd) == ["yogur griego", "huevo"]


def test_knob_apagado_conducta_vieja(monkeypatch):
    import dia_del_ciclo as ddc
    monkeypatch.setenv(_KNOB, "false")
    fd = _fd(5, 11)
    assert ddc.dia(fd) == 5 and ddc.indice_del_chain(fd, 4) == 4
    assert "Brócoli" in ah._single_trip_durable_filter(["Brócoli", "Arroz blanco"], fd, 4)
    assert ah._age_pantry_for_block(["Lechuga", "Huevo"], fd, 4) == ["Lechuga", "Huevo"]
    assert cu.candidatos_del_dia(["yogur griego", "huevo"], {"day": 9}, fd) == ["yogur griego", "huevo"]
    assert "Lechuga" in _nevera_virtual(fd, ["Atún en agua", "Huevo", "Arroz blanco", "Lechuga", "Casabe"])[
        "current_pantry_ingredients"]
    ddc.sellar(fd, {}, 5)
    assert ddc.CLAVE not in fd, "apagado no sella"


def test_nunca_menos_que_la_columna():
    import dia_del_ciclo as ddc
    assert ddc.dia(_fd(5, 2)) == 5
    assert ddc.dia(_fd(5, None, **{"_single_trip_cycle_day": 3})) == 5


# ─────────────────────────────────────────────── el calendario (el relleno no lleva rebanada)

def _plan(archivados, vivos, **extra):
    pd = {"total_days_requested": 30, "_plan_policy": {"effective": SINGLE},
          "_archived_days": [{"day": 1, "date": d} for d in archivados],
          "days": [{"day": i + 1, "date": d} for i, d in enumerate(vivos)]}
    pd.update(extra)
    return pd


_VIVOS = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]


def test_calendario_del_relleno():
    import dia_del_ciclo as ddc
    limpio = _plan(["2026-09-23", "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-27"], _VIVOS)
    fd = _fd(5, None, _plan_start_date="2026-09-28T16:48:07.348599+00:00")
    assert ddc.por_calendario(fd, limpio) == 10                       # 10-03 − 09-23
    ddc.sellar(fd, limpio, 5)
    assert fd[ddc.CLAVE] == 10 and ddc.dia(fd) == 10 and ddc.desplazamiento(fd) == 5
    # 6594aae1 tal cual: 09-26 archivado dos veces ⇒ `plan_cycle_window` ancla el 09-22 (la estimación más temprana)
    real = _plan(["2026-09-23", "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-26", "2026-09-27"], _VIVOS)
    fd2 = _fd(5, 11, _plan_start_date="2026-09-28T16:48:07.348599+00:00")
    assert ddc.calcular(fd2, real, 5) == 11
    # el ancla del ciclo manda (renovación): el bloque 1 del ciclo nuevo es el día 0
    renovado = _plan(["2026-08-20"] * 30, [], _cycle_started_at="2026-09-28", grocery_start_date="2026-09-28")
    assert ddc.calcular(_fd(0, None, _plan_start_date="2026-09-28"), renovado, 0) == 0
    # sin ancla del ciclo, un plan renovado mide desde el ciclo anterior: fuera del ciclo ⇒ se descarta
    legado = _plan(["2026-08-20"] + ["2026-08-21"] * 29, ["2026-09-28"])
    assert ddc.por_calendario(_fd(1, None, _plan_start_date="2026-09-28"), legado) is None
    assert ddc.calcular(_fd(1, None, _plan_start_date="2026-09-28"), legado, 1) == 1


def test_el_merge_del_bloque_sustituye_con_el_dia_del_ciclo(monkeypatch):
    import dia_del_ciclo as ddc
    import graph_orchestrator as go

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

        def lookup(self, s):
            return None
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    fd = _fd(5, 11)
    off = ddc.indice_del_chain(fd, 5)
    assert off == 11
    bloque = [{"day": 6, "meals": [{"meal": "Cena", "name": "Ensalada", "ingredients": ["2 tazas de lechuga"],
                                    "ingredients_raw": ["2 tazas de lechuga"]}]}]
    assert go._single_trip_fresh_substitute(bloque, db=_NoopDB(), effective=SINGLE, diet="balanced", days_offset=5) == 0
    assert go._single_trip_fresh_substitute(bloque, db=_NoopDB(), effective=SINGLE, diet="balanced", days_offset=off) == 1
    assert bloque[0]["meals"][0]["ingredients"] == ["2 tazas de repollo"]


# ─────────────────────────────────────────────── la Nevera virtual NO está dormida (ronda del revisor)

def test_la_nevera_virtual_filtra_con_el_dia_sellado_en_el_2o_refresco():
    """En la rama LLM del worker el sello va ANTES del 2.º `_refresh_chunk_pantry` (el que termina en `nevera_virtual`
    con la puerta `_days_offset > 0` ya abierta). Entre ambos: sin desindentación por debajo del sello y la única
    reasignación de `form_data` es `_merge_chunk_live_profile`, que conserva las claves `_`. La versión anterior del
    docstring de `dia_del_ciclo` la daba por dormida; lo estaba sólo el 1.er refresco."""
    import re
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    sello = src.index('__import__("dia_del_ciclo").sellar(form_data, prior_plan_data, days_offset)')
    llamada = "form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)"
    refresco = src.index(llamada, sello)
    ini = src.rfind("\n", 0, sello) + 1
    sangria = len(src[ini:sello]) - len(src[ini:sello].lstrip(" "))
    for linea in src[ini:refresco].splitlines():
        if linea.strip() and not linea.lstrip().startswith("#"):
            assert len(linea) - len(linea.lstrip(" ")) >= sangria, linea
    asignaciones = set(re.findall(r"^\s*form_data = (\w+)\(", src[sello:refresco], flags=re.M))
    assert asignaciones == {"_merge_chunk_live_profile"}, asignaciones
    import cron_tasks as ct
    fd = _fd(1, None, **{"_single_trip_cycle_day": 11})
    fd = ct._merge_chunk_live_profile(fd, {"goal": "lose_fat", "_single_trip_cycle_day": 0, "_days_offset": 0})
    assert fd["_single_trip_cycle_day"] == 11 and fd["_days_offset"] == 1
    # Nevera real vacía (o apagada): la compra del ciclo llega filtrada con el día sellado, sin rebanada
    nv = _nevera_virtual(fd, ["Atún en agua", "Huevo", "Arroz blanco", "Lechuga", "Filete de pescado fresco", "Casabe"])
    assert nv["_nevera_virtual"] is True
    assert "Lechuga" not in nv["current_pantry_ingredients"]
    assert "Filete de pescado fresco" not in nv["current_pantry_ingredients"]
    assert {"Atún en agua", "Huevo", "Arroz blanco", "Casabe"} <= set(nv["current_pantry_ingredients"])


# ─────────────────────────────────────────────── anclas

def test_anclas():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index('form_data["_days_offset"] = days_offset')
    assert '__import__("dia_del_ciclo").sellar(form_data, prior_plan_data, days_offset)' in src[i:i + 400]
    assert '"_days_offset": _first_new_idx_ck,' in src
    assert ('_chain_view_ck["_days_offset"] = __import__("dia_del_ciclo").indice_del_chain(form_data, _first_new_idx_ck)'
            in src)
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    j = go.index("_TRUSTED_INTERNAL_FORM_KEYS: frozenset = frozenset({")
    assert '"_single_trip_cycle_day",' in go[j:go.index("})", j)]
    ah_src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert ah_src.count('__import__("dia_del_ciclo").dia(fd)') == 2
    cu_src = (_BACKEND / "compra_unica.py").read_text(encoding="utf-8")
    assert '__import__("dia_del_ciclo").desplazamiento(' in cu_src and '__import__("dia_del_ciclo").dia(form_data)' in cu_src
    mod = (_BACKEND / "dia_del_ciclo.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-816" in mod and _KNOB in mod
    assert _KNOB in (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
