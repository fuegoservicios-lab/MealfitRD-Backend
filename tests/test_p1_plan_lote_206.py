# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-206 · 2026-09-24] Lo que el siguiente bloque aprende del anterior: sólo lo que el usuario dijo.

Producción: los 15 rellenos desde el 4-sep inyectaron «⛔ El usuario NO consumió (N)… simplifica esa franja al máximo»
con `consumed=0` — nadie registra, y cada plato no registrado se contaba como saltado. Y la ventana del bloque previo se
elegía por número de día, que el shift renumera («window=days 2-6 skipped=4»: UN día de un bloque de cinco). Ver
`adherencia_previa.py`.
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import adherencia_previa as ap  # noqa: E402

_TIPOS = ("Desayuno", "Almuerzo", "Merienda", "Cena")


def _dia(fecha: date, numero: int, sufijo: str) -> dict:
    return {"day": numero, "date": fecha.isoformat(),
            "meals": [{"meal": t, "name": f"{t} {sufijo}"} for t in _TIPOS]}


def _reg(nombre: str, cuando: datetime) -> dict:
    return {"meal_name": nombre, "consumed_at": cuando}


# ─────────────── la honestidad: sin registro representativo no hay «no consumió» ───────────────

def test_sin_registros_no_hay_desglose():
    """El caso de producción: 16 platos planificados, 0 registros ⇒ nada (antes: «NO consumió (16)»)."""
    dias = [_dia(date(2026, 9, 20) + timedelta(days=i), 1, f"d{i}") for i in range(4)]
    assert ap.construir(dias, [], 7) is None


def test_registro_esparso_dice_solo_lo_que_comio():
    dias = [_dia(date(2026, 9, 20) + timedelta(days=i), 1, f"d{i}") for i in range(4)]
    r = ap.construir(dias, [_reg("Almuerzo d0", datetime(2026, 9, 20, 17, tzinfo=timezone.utc))], 7)
    assert r["consumed_meals"] == ["Almuerzo d0"] and r["skipped_meals"] == [] and r["evidence"] == "sparse", r


def test_registro_representativo_dice_tambien_lo_que_no():
    dias = [_dia(date(2026, 9, 20) + timedelta(days=i), 1, f"d{i}") for i in range(3)]      # 12 planificadas
    comidos = [_reg(f"{t} d0", datetime(2026, 9, 20, 15, tzinfo=timezone.utc)) for t in _TIPOS[:3]]  # 3 ≥ max(2, 3)
    r = ap.construir(dias, comidos, 7)
    assert r["evidence"] == "representative" and r["consumed_count"] == 3 and r["skipped_count"] == 9, r


def test_el_prompt_ya_no_ordena_huir_sin_evidencia():
    from prompts.plan_generator import build_prev_chunk_adherence_context
    dias = [_dia(date(2026, 9, 20), 1, "x")]
    assert build_prev_chunk_adherence_context(ap.construir(dias, [], 3)) == ""
    esparso = ap.construir(dias, [_reg("Cena x", datetime(2026, 9, 20, 23, tzinfo=timezone.utc))], 3)
    texto = build_prev_chunk_adherence_context(esparso)
    assert "SÍ consumió" in texto and "NO consumió" not in texto, texto


# ─────────────── la ventana: el bloque previo son sus FECHAS ───────────────

def _plan_renumerado(hoy: date) -> dict:
    """Bloque previo de 5 días (hoy-5 … hoy-1): los 4 primeros archivados (todos `day=1`, como los deja el shift) y el
    último vivo como día 1; más 2 días de un bloque anterior."""
    arch = [_dia(hoy - timedelta(days=k), 1, f"a{k}") for k in (7, 6, 5, 4, 3, 2)]
    return {"_archived_days": arch, "days": [_dia(hoy - timedelta(days=1), 1, "v1")]}


def test_dias_del_bloque_por_fechas():
    hoy = date(2026, 9, 24)
    pd = _plan_renumerado(hoy)
    dias = ap.dias_del_bloque(pd, 5, hoy=hoy)
    assert [d["date"] for d in dias] == [(hoy - timedelta(days=k)).isoformat() for k in (5, 4, 3, 2, 1)]
    assert ap.dias_del_bloque({"days": [{"day": 1, "meals": []}]}, 3) is None, "sin fechas: el caller conserva lo suyo"
    futuro = {"days": [_dia(hoy, 1, "h"), _dia(hoy + timedelta(days=1), 2, "m")]}
    assert ap.dias_del_bloque(futuro, 2, hoy=hoy) == [], "lo no vivido no tiene adherencia"


def test_la_seleccion_por_numero_era_un_dia_de_cinco():
    """La selección de siempre (`day` entre offset+1 y offset+count) sobre el plan renumerado: 1 día de 5."""
    hoy = date(2026, 9, 24)
    pd = _plan_renumerado(hoy)
    legado = [d for d in pd["days"] if 1 <= d["day"] <= 5]          # offset 0, count 5
    assert len(legado) == 1, "el shift archivó los otros cuatro y dejó éste renumerado a 1"
    assert len(ap.dias_del_bloque(pd, 5, hoy=hoy)) == 5


def test_desglose_del_worker_lee_la_ventana_y_filtra_por_fecha():
    ahora = datetime(2026, 9, 24, 5, 0, tzinfo=timezone.utc)                    # 01:00 en RD
    hoy = date(2026, 9, 24)
    pd = _plan_renumerado(hoy)
    dentro = [_reg(f"{t} a5", datetime(2026, 9, 19, 16, tzinfo=timezone.utc)) for t in _TIPOS] + \
             [_reg(f"{t} a4", datetime(2026, 9, 20, 16, tzinfo=timezone.utc)) for t in _TIPOS]
    fuera = [_reg("Almuerzo a7", datetime(2026, 9, 17, 16, tzinfo=timezone.utc))]
    leer = MagicMock(return_value=dentro + fuera)
    consultar = MagicMock(return_value={"days_count": 5})
    legado = MagicMock()
    r = ap.desglose(legado=legado, plan_data=pd, meal_plan_id="p", week_number=5, prev_offset=1, prev_count=5,
                    registros=[], leer_registros=leer, user_id="u", tz_min=240, consultar=consultar, ahora=ahora)
    assert not legado.called
    desde = leer.call_args[0][1]
    assert desde.startswith("2026-09-19T04:00"), f"medianoche local (RD) del primer día del bloque: {desde}"
    assert r["planned_meals"] == 20 and r["logged_meals"] == 8 and r["consumed_count"] == 8, r
    assert r["skipped_count"] == 12 and "Almuerzo a7" not in r["consumed_meals"], r


def test_bloque_2_usa_los_dias_del_bloque_1_en_la_cola():
    hoy = date(2026, 9, 24)
    pd = {"days": [_dia(hoy - timedelta(days=k), 4 - k, f"b{k}") for k in (3, 2, 1)]}
    consultar = MagicMock(return_value={"days_count": 3})
    r = ap.desglose(legado=MagicMock(), plan_data=pd, meal_plan_id="p", week_number=2, prev_offset=0, prev_count=0,
                    registros=[], leer_registros=MagicMock(return_value=[]), user_id="u", tz_min=0,
                    consultar=consultar, ahora=datetime(2026, 9, 24, 1, tzinfo=timezone.utc))
    assert r is None, "sin registros no hay desglose"
    assert consultar.call_args[0][1] == ("p", 1), "pregunta por el bloque 1, no por el offset re-anclado"


def test_knob_apagado_vuelve_el_desglose_de_siempre(monkeypatch):
    monkeypatch.setenv("MEALFIT_PREV_ADHERENCE_HONEST", "false")
    legado = MagicMock(return_value={"x": 1})
    assert ap.desglose(legado=legado, plan_data={}, meal_plan_id="p", week_number=3, prev_offset=0, prev_count=3,
                       registros=[]) == {"x": 1}


# ─────────────── el gate: corrige la ventana SÓLO cuando ya iba a evaluar ───────────────

def test_ventana_gate_no_cambia_si_la_seleccion_de_siempre_no_encontro_dias():
    hoy = date(2026, 9, 24)
    pd = _plan_renumerado(hoy)
    assert ap.ventana_gate(pd, "p", 5, 5, [], "2026-09-23T00:00:00+00:00", 0, hoy, MagicMock()) == \
        ([], "2026-09-23T00:00:00+00:00"), "cero días ⇒ el gate sigue su rama de siempre (fail-open)"


def test_ventana_gate_no_usa_un_plan_data_viejo():
    """P0-3/STALE-RECOVERY: los días vienen de la relectura fresca (otro objeto) ⇒ este plan_data es el viejo."""
    hoy = date(2026, 9, 24)
    pd = _plan_renumerado(hoy)
    copia = [dict(pd["days"][0])]
    assert ap.ventana_gate(pd, "p", 5, 5, copia, "2026-09-23T00:00:00+00:00", 0, hoy,
                           MagicMock(return_value={"days_count": 5}))[0] is copia


def test_ventana_gate_nunca_mueve_el_inicio_hacia_adelante():
    hoy = date(2026, 9, 24)
    pd = _plan_renumerado(hoy)
    dias, desde = ap.ventana_gate(pd, "p", 5, 5, pd["days"], "2026-09-01T00:00:00+00:00", 0, hoy,
                                  MagicMock(return_value={"days_count": 5}))
    assert len(dias) == 5 and desde.startswith("2026-09-01"), desde


def _gate_con_registros(monkeypatch=None):
    """Quien SÍ registra: 8 de las 12 comidas del bloque previo (3 días), todas en sus dos primeros días. El shift dejó
    vivo sólo el último día (renumerado a 1) y movió el ancla a ese día."""
    import cron_tasks
    now = datetime.now(timezone.utc).replace(hour=0, minute=30, second=0, microsecond=0)
    hoy = now.date()
    ancla = datetime.combine(hoy - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    pd = {"_shift_days_accumulated": 2,
          "_archived_days": [_dia(hoy - timedelta(days=3), 1, "a3"), _dia(hoy - timedelta(days=2), 1, "a2")],
          "days": [_dia(hoy - timedelta(days=1), 1, "v1")]}
    snapshot = {"form_data": {"_plan_start_date": ancla.isoformat(), "tz_offset_minutes": 0}, "totalDays": 15}
    registros = [_reg(f"{t} a3", ancla - timedelta(days=2) + timedelta(hours=14)) for t in _TIPOS] + \
                [_reg(f"{t} a2", ancla - timedelta(days=1) + timedelta(hours=14)) for t in _TIPOS]

    def _sql(q, *a, **k):
        if "days_offset, days_count" in q:
            return {"days_offset": 0, "days_count": 3}
        if "SELECT days_count FROM plan_chunk_queue" in q:
            return {"days_count": 3}
        return None

    def _desde(uid, since, *a, **k):
        corte = datetime.fromisoformat(str(since).replace("Z", "+00:00"))
        return [r for r in registros if r["consumed_at"] >= corte]

    leer = MagicMock(side_effect=_desde)
    with patch("cron_tasks._dt_p0b_now", return_value=now), \
            patch("cron_tasks.execute_sql_query", side_effect=_sql), \
            patch("cron_tasks.execute_sql_write", MagicMock()), \
            patch("cron_tasks._record_chunk_deferral", MagicMock()), \
            patch("cron_tasks._dispatch_push_notification", MagicMock()), \
            patch("cron_tasks.get_consumed_meals_since", leer), \
            patch("cron_tasks.get_inventory_activity_since", return_value={}), \
            patch("db_facts.get_plan_meal_deviations_since", return_value=[]):
        r = cron_tasks._check_chunk_learning_ready(user_id="u", meal_plan_id="p", week_number=5, days_offset=1,
                                                   plan_data=pd, snapshot=snapshot)
    return r, leer, hoy


def test_el_gate_ve_el_bloque_entero_de_quien_registra():
    r, leer, hoy = _gate_con_registros()
    assert r.get("planned_meals") == 12, r
    assert r.get("explicit_logged_meals") == 8 and r.get("matched_meals") == 8, r
    assert r.get("zero_log_proxy") is False and r.get("ready") is True, r
    assert str(leer.call_args[0][1]).startswith((hoy - timedelta(days=3)).isoformat()), leer.call_args


def test_con_el_knob_apagado_el_gate_lo_tomaba_por_sin_registros(monkeypatch):
    monkeypatch.setenv("MEALFIT_PREV_ADHERENCE_HONEST", "false")
    r, _, _ = _gate_con_registros()
    assert r.get("planned_meals") == 4 and r.get("zero_log_proxy") is True and r.get("ready") is False, r


# ─────────────── anclas ───────────────

def test_anclas_en_el_worker_y_el_gate():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert '__import__("adherencia_previa").desglose(' in src
    assert '__import__("adherencia_previa").ventana_gate(' in src
    assert 'prior_days or (prior_plan_data or {}).get("_archived_days")' in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 206 and m.group(2) >= "2026-09-24"
