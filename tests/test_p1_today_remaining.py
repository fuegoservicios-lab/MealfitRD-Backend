"""[P1-TODAY-REMAINING · 2026-07-28] "Comí el desayuno y renové el plan — el
desayuno de ayer no debería reaparecer" (owner). La implementación literal
(recortar el slot del `plan_data`) rompe el piso de proteína por-día
(`graph_orchestrator.py:11780-11819`, hard-gate < 0.90x target) y la lista de
compras (promedio-de-día × 7, un día corto hace comprar de menos toda la
semana). La solución correcta: DERIVAR del diario en cada render, nunca
persistir. Este test cubre la mitad backend (coach): el helper compartido
`agent._build_today_remaining_context`, invocado desde AMBOS paths de chat
(stream ~:4650, non-stream ~:4268).

Dos gaps que cierra:
  (a) tier factual (sin 🚨) arriba del gate viejo de 35% — un desayuno normal
      (~20-25% del presupuesto) dejaba al coach en silencio absoluto,
      exactamente el caso descrito por el owner.
  (b) sentencia explícita de cuántas comidas del plan quedan hoy, por nombre,
      con la REGLA DE AMBIGÜEDAD: si ≥2 slots de hoy canonicalizan a la misma
      key (2-3 meriendas en planes de 5-6 comidas) y el diario solo trae una
      fila de esa key, NO se marca ninguna como comida — atribuir la
      incorrecta es peor que no atribuir ninguna.

tooltip-anchor: P1-TODAY-REMAINING
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

from agent import _build_today_remaining_context, _resolve_today_plan_day_index  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures de plan
# ---------------------------------------------------------------------------

def _plan_4_meals(day_name="Martes"):
    return {
        "days": [
            {"day": 1, "day_name": day_name, "meals": [
                {"meal": "Desayuno", "name": "Mangú con los tres golpes", "cals": 500},
                {"meal": "Almuerzo", "name": "Arroz con pollo guisado", "cals": 700},
                {"meal": "Merienda", "name": "Yogur con fruta", "cals": 250},
                {"meal": "Cena", "name": "Pescado a la plancha", "cals": 550},
            ]},
        ],
    }


def _plan_two_meriendas(day_name="Martes"):
    return {
        "days": [
            {"day": 1, "day_name": day_name, "meals": [
                {"meal": "Desayuno", "name": "Avena con fruta", "cals": 400},
                {"meal": "Almuerzo", "name": "Arroz con habichuela y pollo", "cals": 700},
                {"meal": "Merienda AM", "name": "Yogur", "cals": 150},
                {"meal": "Merienda PM", "name": "Batido de proteína", "cals": 200},
                {"meal": "Cena", "name": "Pescado con vegetales", "cals": 550},
            ]},
        ],
    }


_TODAY = "2026-07-28"  # martes


def _remaining_line(out: str) -> str:
    m = re.search(r"^📋.*$", out, re.MULTILINE)
    assert m, f"No encontré la línea de comidas restantes en:\n{out}"
    return m.group(0)


# ---------------------------------------------------------------------------
# (a) Tier factual — el caso "ya desayuné" que el gate de 35% dejaba mudo
# ---------------------------------------------------------------------------

def test_soft_tier_fires_where_old_35pct_gate_was_silent():
    """~25% consumido → remaining=75% del target. El gate viejo
    (`remaining < 0.35*target`) NO disparaba nada en este caso — exactamente
    el escenario que el owner describió (comió el desayuno, ~20-25% del día).
    El tier nuevo debe aparecer, SIN el emoji de alarma 🚨."""
    out = _build_today_remaining_context(
        _plan_4_meals(), consumed_today=[{"meal_type": "desayuno", "calories": 500}],
        target_cal_int=2000, total_consumed=500, local_date_str=_TODAY,
    )
    assert "📊 ESTADO DEL DÍA" in out, "el tier factual nuevo no disparó"
    assert "🚨" not in out, "el tier factual NO debe usar el emoji de alarma"
    assert "estimad" in out.lower(), "los kcal deben enmarcarse como estimado, no medición"


def test_micro_adaptation_tier_still_fires_under_35pct():
    """Regresión: el gate viejo (ajustado, <35% restante) sigue vivo tal cual."""
    out = _build_today_remaining_context(
        _plan_4_meals(), consumed_today=[{"meal_type": "desayuno", "calories": 1700}],
        target_cal_int=2000, total_consumed=1700, local_date_str=_TODAY,
    )
    assert "🚨 ALERTA DE MICRO-ADAPTACIÓN" in out
    assert "📊 ESTADO DEL DÍA" not in out, "los tiers son mutuamente excluyentes"


def test_critical_tier_still_fires_over_budget():
    """Regresión: exceso de presupuesto → alerta crítica, unificada entre paths."""
    out = _build_today_remaining_context(
        _plan_4_meals(), consumed_today=[{"meal_type": "cena", "calories": 2100}],
        target_cal_int=2000, total_consumed=2100, local_date_str=_TODAY,
    )
    assert "🚨 ALERTA CRÍTICA" in out
    assert "estimad" in out.lower()


# ---------------------------------------------------------------------------
# (b) Comidas restantes — nombra los slots correctos
# ---------------------------------------------------------------------------

def test_remaining_meals_sentence_names_the_right_slots():
    """Desayuno registrado hoy → deben quedar 3: Almuerzo, Merienda, Cena
    (Desayuno removido de la lista, por match inequívoco de key)."""
    out = _build_today_remaining_context(
        _plan_4_meals(),
        consumed_today=[{"meal_type": "desayuno", "meal_name": "Mangú", "calories": 500}],
        target_cal_int=2000, total_consumed=500, local_date_str=_TODAY,
    )
    line = _remaining_line(out)
    assert "3 comida(s)" in line
    assert "Almuerzo" in line and "Merienda" in line and "Cena" in line
    assert "Desayuno" not in line, "el desayuno ya comido no debe listarse como restante"


def test_no_meals_eaten_lists_all_four():
    out = _build_today_remaining_context(
        _plan_4_meals(),
        # calorías consumidas de un ítem que NO canonicaliza a ningún slot del
        # plan (p.ej. un registro con meal_type vacío) — no debe atribuir nada.
        consumed_today=[{"meal_type": "", "meal_name": "agua con limón", "calories": 5}],
        target_cal_int=2000, total_consumed=5, local_date_str=_TODAY,
    )
    line = _remaining_line(out)
    assert "4 comida(s)" in line
    for name in ("Desayuno", "Almuerzo", "Merienda", "Cena"):
        assert name in line


def test_unplanned_extra_snack_does_not_remove_any_plan_meal():
    """Un registro de diario que SÍ canonicaliza (ej. 'merienda') pero el
    plan de hoy no tiene NINGÚN slot con esa key (0 matches, no 1) tampoco
    debe remover nada — groups.get(key) da lista vacía."""
    plan = {
        "days": [{"day": 1, "day_name": "Martes", "meals": [
            {"meal": "Desayuno", "name": "Mangú", "cals": 500},
            {"meal": "Almuerzo", "name": "Arroz con pollo", "cals": 700},
            {"meal": "Cena", "name": "Pescado", "cals": 550},
        ]}],
    }
    out = _build_today_remaining_context(
        plan, consumed_today=[{"meal_type": "merienda", "meal_name": "snack extra", "calories": 200}],
        target_cal_int=2000, total_consumed=200, local_date_str=_TODAY,
    )
    line = _remaining_line(out)
    assert "3 comida(s)" in line
    assert "Desayuno" in line and "Almuerzo" in line and "Cena" in line


# ---------------------------------------------------------------------------
# REGLA DE AMBIGÜEDAD — el corazón del fix
# ---------------------------------------------------------------------------

def test_ambiguity_rule_attributes_nothing_with_two_meriendas():
    """Plan con 2 meriendas hoy + diario con UNA fila 'merienda'. No hay forma
    de saber cuál — NINGUNA se remueve. Las 5 comidas siguen "restantes"
    (las kcal ya están reflejadas en total_consumed/remaining; lo que NO se
    hace es fingir saber cuál merienda específica fue)."""
    out = _build_today_remaining_context(
        _plan_two_meriendas(),
        consumed_today=[{"meal_type": "merienda", "meal_name": "Yogur", "calories": 150}],
        target_cal_int=2000, total_consumed=150, local_date_str=_TODAY,
    )
    line = _remaining_line(out)
    assert "5 comida(s)" in line, (
        "regla de ambigüedad violada: se removió una merienda sin poder "
        "distinguir cuál fue"
    )
    assert "Merienda AM" in line and "Merienda PM" in line


def test_ambiguity_rule_still_attributes_unambiguous_slots_in_same_day():
    """Mismo plan de 2 meriendas, pero HOY también se registró el desayuno
    (key con un solo match → SÍ se atribuye). La ambigüedad de merienda no
    debe contaminar el match inequívoco de desayuno."""
    out = _build_today_remaining_context(
        _plan_two_meriendas(),
        consumed_today=[
            {"meal_type": "desayuno", "meal_name": "Avena", "calories": 400},
            {"meal_type": "merienda", "meal_name": "Yogur", "calories": 150},
        ],
        target_cal_int=2000, total_consumed=550, local_date_str=_TODAY,
    )
    line = _remaining_line(out)
    assert "4 comida(s)" in line, "el desayuno (match único) sí debe removerse"
    assert "Desayuno" not in line
    assert "Merienda AM" in line and "Merienda PM" in line, "ambas meriendas siguen restantes (ambiguo)"


def test_all_matched_when_unambiguous_says_no_more_meals():
    out = _build_today_remaining_context(
        _plan_4_meals(),
        consumed_today=[
            {"meal_type": "desayuno", "calories": 500},
            {"meal_type": "almuerzo", "calories": 700},
            {"meal_type": "merienda", "calories": 250},
            {"meal_type": "cena", "calories": 550},
        ],
        target_cal_int=2000, total_consumed=2000, local_date_str=_TODAY,
    )
    assert "Hoy ya no te quedan más comidas del plan" in out


# ---------------------------------------------------------------------------
# `_resolve_today_plan_day_index` — SSOT del mapeo HOY → día del plan
# ---------------------------------------------------------------------------

def test_resolve_today_index_matches_day_name():
    idx = _resolve_today_plan_day_index(_plan_4_meals(day_name="Martes"), local_date_str=_TODAY)
    assert idx == 0


def test_resolve_today_index_none_when_no_match():
    idx = _resolve_today_plan_day_index(_plan_4_meals(day_name="Lunes"), local_date_str=_TODAY)
    assert idx is None


def test_helper_fail_open_on_garbage_plan():
    out = _build_today_remaining_context(
        "not a dict", consumed_today=[{"meal_type": "desayuno", "calories": 100}],
        target_cal_int=2000, total_consumed=100, local_date_str=_TODAY,
    )
    # No debe reventar; el tier de calorías sigue funcionando aunque el plan
    # sea basura (fail-open SOLO en la parte (b) de comidas restantes).
    assert "📊 ESTADO DEL DÍA" in out


# ---------------------------------------------------------------------------
# Ambos paths de chat invocan el MISMO helper (source-level, sin duplicar)
# ---------------------------------------------------------------------------

def test_both_chat_paths_call_the_shared_helper():
    src = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
    # 1 def + 2 callsites (non-stream ~4268, stream ~4650). Si baja a 2, un
    # path se desconectó del helper y volvió a duplicar/perder el bloque.
    assert len(re.findall(r"_build_today_remaining_context\(", src)) == 3, (
        "esperaba 1 def + 2 callsites (non-stream y stream) de "
        "_build_today_remaining_context — alguno de los dos paths dejó de "
        "usar la SSOT compartida."
    )
    # La copy de la alerta vieja NO debe volver a estar duplicada inline en
    # los dos call-sites (regresión: cada edición futura del copy solo
    # tocaría un path y los dos divergirían de nuevo, como ya pasó una vez
    # con el texto de ALERTA CRÍTICA).
    assert src.count("ALERTA DE MICRO-ADAPTACIÓN (MEJORA 6)") == 1, (
        "el copy de la alerta de micro-adaptación debe vivir UNA sola vez "
        "(dentro del helper compartido), no duplicado por path."
    )


def test_no_print_statements_introduced():
    """[P2-LOGGER-MIGRATION] agent.py es un archivo productivo — print()
    prohibido sin marker de exención."""
    src = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
    # Ventana alrededor del helper nuevo únicamente (no re-litigar el resto
    # del archivo, ya cubierto por test_p2_logger_migration.py).
    start = src.index("def _build_today_remaining_context")
    end = src.index("\ndef _build_pantry_context")
    body = src[start:end]
    assert "print(" not in body


def test_no_naive_utcnow_introduced():
    """[P3-DEPRECATED-UTCNOW] `datetime.utcnow()` prohibido; usar
    `datetime.now(timezone.utc)`."""
    src = (_BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
    start = src.index("def _resolve_today_plan_day_index")
    end = src.index("\ndef _build_pantry_context")
    body = src[start:end]
    assert "utcnow(" not in body


# ---------------------------------------------------------------------------
# Marker + freshness — supersession-proof (mirror test_p2_audit_v5_batch.py)
# ---------------------------------------------------------------------------

def test_marker_bumped():
    """Supersession-proof: este marker o uno posterior (fecha ≥)."""
    src = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-TODAY-REMAINING" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-28", f"marker {m.group(1)!r} anterior a P1-TODAY-REMAINING"
