"""[P1-CHAT-PAST-DAYS · 2026-07-27] Test ancla del P-fix.

Spec: backend/docs/chat_past_days_memory.md
"""
import os
import re
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chat_history_context import resolve_day_dates, find_plan_day_for_date  # noqa: E402

HOY = date(2026, 7, 27)  # lunes


def _day(name, n, meals=None, fecha=None):
    d = {"day": n, "day_name": name, "meals": meals if meals is not None else []}
    if fecha:
        d["date"] = fecha
    return d


def test_ancla_1_fecha_estampada_gana():
    """Si el día trae `date`, es autoritativa y `inferred` es False."""
    plan = {"days": [_day("Lunes", 1, fecha="2026-07-27"), _day("Martes", 2, fecha="2026-07-28")]}
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28)]
    assert [r["inferred"] for r in rows] == [False, False]


def test_ancla_estampada_en_indice_distinto_de_cero():
    """El ancla estampada puede NO estar en days[0]: la fórmula
    `anchor_date + (i - anchor_idx)` tiene que proyectar hacia ATRÁS y hacia
    ADELANTE desde ella. Sin este guard, un off-by-anchor-index pasaría mudo.
    """
    plan = {"days": [_day("Domingo", 1), _day("Lunes", 2, fecha="2026-07-27"), _day("Martes", 3)]}
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 26), date(2026, 7, 27), date(2026, 7, 28)]
    assert [r["inferred"] for r in rows] == [True, False, True]


def test_ancla_2_weekday_match_para_plan_shifteado():
    """Sin `date`, el ancla es el day_name que coincide con HOY.

    Regresión: `cycle_start_date + i` desplazaría el plan entero, porque tras
    un shift days[0] es HOY y NO el inicio del ciclo.
    """
    plan = {
        "days": [_day("Lunes", 1), _day("Martes", 2), _day("Miércoles", 3)],
        "cycle_start_date": "2026-07-20T10:00:00+00:00",  # una semana antes: trampa
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
    assert all(r["inferred"] for r in rows)


def test_ancla_3_cycle_start_cuando_ningun_day_name_es_hoy():
    """Plan que aún no shifteó y cuyo primer día ya pasó."""
    plan = {
        "days": [_day("Sábado", 1), _day("Domingo", 2)],
        "cycle_start_date": "2026-07-25T10:00:00+00:00",  # sábado
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 25), date(2026, 7, 26)]


def test_archivados_van_antes_del_primer_dia_vivo():
    plan = {
        "days": [_day("Lunes", 1)],
        "_archived_days": [_day("Sábado", 1), _day("Domingo", 2)],
    }
    rows = resolve_day_dates(plan, HOY)
    assert [r["date"] for r in rows] == [date(2026, 7, 25), date(2026, 7, 26), date(2026, 7, 27)]
    assert [r["archived"] for r in rows] == [True, True, False]


def test_find_plan_day_for_date_encuentra_en_el_archivo():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 2)]}
    got = find_plan_day_for_date(plan, date(2026, 7, 26), HOY)
    assert got is not None and got["day"]["day_name"] == "Domingo"
    assert find_plan_day_for_date(plan, date(2026, 1, 1), HOY) is None


@pytest.mark.parametrize("basura", [None, "no soy un dict", 42, [], {"days": "tampoco"}])
def test_fail_open_ante_shapes_raras(basura):
    assert resolve_day_dates(basura, HOY) == []


from chat_history_context import (  # noqa: E402
    build_past_diary_block,
    build_past_plan_days_block,
)

_MEALS = [
    {"meal": "Desayuno", "name": "Revoltillo de Tayota con Atún", "cals": 793},
    {"meal": "Almuerzo", "name": "Pulpo al Horno con Croquetas", "cals": 549},
]


def test_bloque_plan_lista_solo_dias_pasados():
    plan = {
        "days": [_day("Lunes", 1, _MEALS)],
        "_archived_days": [_day("Domingo", 1, _MEALS)],
    }
    out = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "Domingo 26 jul" in out
    assert "Revoltillo de Tayota con Atún" in out
    assert "793 kcal" in out
    # HOY no es un día pasado: no debe aparecer como tal.
    assert "Lunes 27 jul" not in out


def test_bloque_plan_no_filtra_cantidades_ni_recetas():
    """El índice es barato a propósito: los gramos y los pasos van por la tool."""
    meals = [{"meal": "Cena", "name": "Pescado", "cals": 400,
              "ingredients": ["255g de pescado"], "recipe": ["Paso secreto"]}]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, meals)]}
    out = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "255g" not in out
    assert "Paso secreto" not in out
    assert "consultar_dia_del_plan" in out


def test_bloque_plan_marca_fechas_inferidas_con_tilde():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, _MEALS)]}
    out = build_past_plan_days_block(plan, HOY, days_back=7, max_chars=3000)
    assert "~Domingo 26 jul" in out


@pytest.mark.parametrize("cap", [520, 700, 1200, 3000])
def test_bloque_plan_nunca_supera_el_cap(cap):
    """El cap es DURO: el docstring lo promete y la nota de recorte también cuenta."""
    archived = [_day("D%d" % i, i, _MEALS) for i in range(12)]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": archived}
    out = build_past_plan_days_block(plan, HOY, days_back=30, max_chars=cap)
    assert len(out) <= cap, "cap duro violado: %d > %d" % (len(out), cap)


def test_bloque_plan_declara_el_recorte_cuando_recorta():
    """No silent caps: si se cae un día, el prompt lo dice."""
    archived = [_day("D%d" % i, i, _MEALS) for i in range(12)]
    plan = {"days": [_day("Lunes", 1)], "_archived_days": archived}
    apretado = build_past_plan_days_block(plan, HOY, days_back=30, max_chars=520)
    holgado = build_past_plan_days_block(plan, HOY, days_back=30, max_chars=6000)
    assert "omitidos por espacio" in apretado
    assert "omitidos por espacio" not in holgado
    assert apretado.count("\n- ") < holgado.count("\n- ")


def test_bloque_plan_apagado_con_days_back_cero():
    plan = {"days": [_day("Lunes", 1)], "_archived_days": [_day("Domingo", 1, _MEALS)]}
    assert build_past_plan_days_block(plan, HOY, days_back=0, max_chars=3000) == ""


def test_bloque_diario_declara_los_dias_sin_registro():
    """La guarda que impide que el modelo rellene el hueco con el plan."""
    rows = [{"meal_name": "Salami con queso", "meal_type": "desayuno",
             "calories": 820, "consumed_at": "2026-07-25T17:56:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Salami con queso" in out
    assert "820 kcal" in out
    assert "Domingo 26 jul: SIN REGISTRO" in out
    assert "NUNCA respondas con lo que el plan mandaba" in out


def test_bloque_diario_excluye_hoy():
    """DIARIO DE HOY es su propio bloque; este NO debe duplicarlo."""
    rows = [{"meal_name": "Almuerzo de hoy", "meal_type": "almuerzo",
             "calories": 500, "consumed_at": "2026-07-27T15:00:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Almuerzo de hoy" not in out


def test_bloque_diario_vacio_sigue_declarando_ignorancia():
    out = build_past_diary_block([], HOY, days_back=2, max_chars=3000)
    assert "SIN REGISTRO" in out
    assert "Domingo 26 jul" in out and "Sábado 25 jul" in out


def test_diario_atribuye_la_comida_al_dia_LOCAL_no_al_utc():
    """Regresión: 10:30pm RD del 25 se guarda como 02:30 UTC del 26. Tomar
    `.date()` del UTC crudo la movía al 26 Y declaraba el 25 'SIN REGISTRO'."""
    rows = [{"meal_name": "Merienda nocturna", "meal_type": "cena",
             "calories": 300, "consumed_at": "2026-07-26T02:30:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Sábado 25 jul: cena: Merienda nocturna (300 kcal)" in out
    assert "Sábado 25 jul: SIN REGISTRO" not in out
    assert "Domingo 26 jul: SIN REGISTRO" in out


def test_diario_excluye_el_dia_anterior_al_floor():
    """El borde cercano (hoy excluido) ya está fijado; este fija el lejano."""
    rows = [{"meal_name": "Demasiado viejo", "meal_type": "cena",
             "calories": 100, "consumed_at": "2026-07-23T15:00:00+00:00"}]
    out = build_past_diary_block(rows, HOY, days_back=3, max_chars=3000)
    assert "Demasiado viejo" not in out
    assert "Jueves 23 jul" not in out
    assert "Viernes 24 jul: SIN REGISTRO" in out


def test_assemble_respeta_el_cap_aunque_la_nota_de_recorte_ocupe_espacio():
    """Regresión DIRECTA sobre `_assemble`: la nota "(+N ... omitidos)" ocupa
    espacio, y descontarla DESPUÉS de haber gastado el presupuesto desbordaba
    el cap que el docstring promete como duro.

    Se prueba el helper con entradas sintéticas y un BARRIDO de caps, no con
    los bloques reales: así el guard no depende del largo del copy (que cambia)
    y cubre por construcción la franja donde el recorte se activa. Un cap
    concreto tuneado al umbral de hoy caduca a la primera edición del texto.
    """
    from chat_history_context import _assemble
    for cap in range(20, 141):
        out = _assemble("H" * 10, ["L" * 19] * 6, "F" * 10, cap, "test")
        assert len(out) <= cap, "cap duro violado con cap=%d: len=%d" % (cap, len(out))


# --- Estampado de `date`: tests parser-based sobre código de producción ---

_BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _src(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as fh:
        return fh.read()


def test_generacion_estampa_date():
    src = _src("graph_orchestrator.py")
    assert 'day["date"] = target_date.date().isoformat()' in src, (
        "graph_orchestrator debe estampar la fecha del día en la generación "
        "(P1-CHAT-PAST-DAYS); `target_date` ya se calcula ahí para el day_name."
    )


def test_shift_api_estampa_date_en_vivos_y_archivados():
    src = _src("routers/plans.py")
    # [P1-CHAT-PAST-DAYS · 2026-07-27] `today` en este scope es un
    # `datetime` (datetime.now(timezone.utc)), NO un `date` — verificado
    # leyendo el código fuente antes de escribir este test (el brief
    # original asumía `date`; se adaptó). `.date()` evita filtrar hora/TZ
    # al JSON persistido.
    assert "day_obj['date'] = target_date.date().isoformat()" in src, "días vivos del shift sin fecha"
    assert "P1-CHAT-PAST-DAYS" in src, "falta el marker en el bloque de archivado"
    assert "_arch_day['date']" in src, "los días archivados deben nacer fechados"


def test_shift_cron_es_gemelo_del_shift_api():
    """El bloque del cron es un duplicado literal en lógica (no en estilo de
    comillas: el renumber loop de cron_tasks.py ya usaba dobles antes de este
    cambio — `day_obj["day_name"]`/`day_obj["day"]` — vs comillas simples en
    plans.py; se respetó la convención local de cada archivo)."""
    src = _src("cron_tasks.py")
    assert "_arch_day['date']" in src
    assert 'day_obj["date"] = target_date.date().isoformat()' in src
