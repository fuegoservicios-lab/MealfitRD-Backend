"""[P2-BUDGET-CONVERGENCE-FUTURE-ONLY · 2026-08-03] La convergencia de presupuesto
reescribía días que el usuario YA COCINÓ y semanas que YA PAGÓ.

`apply_budget_convergence_for_days` barría `plan_data["days"]` COMPLETO sin filtro
temporal, y el seam T2 del chunk worker corre DÍAS después de la generación (chunks
programados). Daño doble medido en el audit v7:

  (a) **Historial reescrito.** La sustitución quinoa→arroz caía sobre los días 1-4 ya
      cocinados. `consultar_dia_del_plan` (chat) y el quality index leen esos días: el
      sistema le contaba al usuario una comida que nunca hizo.
  (b) **El gasto real SUBE mientras el banner dice "dentro".** Si la despensa de esa
      semana ya se pagó, la quinoa comprada queda huérfana y la lista nueva exige
      comprar arroz ADEMÁS. La convergencia "converge" en el papel y encarece en la
      caja.

Cierre: ventana de futuro (`date` ISO de P1-CHAT-PAST-DAYS, fallback índice relativo a
`grocery_start_date`) sobre la LISTA de días que reciben sustituciones + skip de los
alimentos que el usuario YA TIENE en la Nevera (snapshot `_inv_s` que el seam ya cargó
— cero IO nuevo). El re-costeo posterior sigue viendo el plan COMPLETO: el costo del
ciclo es de todo el plan, solo la ESCRITURA se ventanea.

Knob `MEALFIT_BUDGET_CONVERGENCE_FUTURE_ONLY` (default True) → OFF restaura el barrido
completo byte-idéntico.

tooltip-anchor: P2-BUDGET-CONVERGENCE-FUTURE-ONLY
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parents[1]
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_CRON_SRC = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")

_HOY = date(2026, 8, 2)
# quinoa RD$300/lb → arroz integral RD$60/lb (ahorro 80% ≫ el 30% mínimo de los dos pases).
_PRECIOS = {"quinoa": 300.0, "arroz integral": 60.0}


def _dia(idx: int, fecha: str | None) -> dict:
    d = {
        "day": idx,
        "meals": [{
            "name": "Bowl de quinoa",
            "ingredients": ["100 g de quinoa"],
            "ingredients_raw": ["100 g de quinoa"],
            "recipe": ["Cuece la quinoa 12 min."],
        }],
    }
    if fecha:
        d["date"] = fecha
    return d


def _plan(pasados: int = 6, futuros: int = 9, con_fecha: bool = True) -> dict:
    days = [
        _dia(i + pasados + 1,
             (_HOY + timedelta(days=i)).isoformat() if con_fecha else None)
        for i in range(-pasados, futuros)
    ]
    return {
        "days": days,
        "aggregated_shopping_list_weekly": [
            {"name": "Quinoa", "estimated_cost_rd": 1800.0},
        ],
    }


def _subs_de(plan: dict) -> list[int]:
    """Índices de los días que recibieron al menos una sustitución."""
    return [i for i, d in enumerate(plan["days"])
            if any(m.get("_budget_substitutions") for m in d["meals"])]


@pytest.fixture()
def _go(monkeypatch):
    """Convergencia con el catálogo de precios mockeado y la cola pesada (motor de
    macros, micros, finalize chain) neutralizada: este test mide QUÉ DÍAS se tocan,
    no la cadena de re-cap."""
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: dict(_PRECIOS))
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_MAX_SUBS", 3)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MAX_SUBS", 5)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_TOP_ITEMS", 8)
    monkeypatch.setattr(go, "BUDGET_DRIVER_AWARE_MIN_SAVING_PCT", 0.30)
    monkeypatch.setattr(go, "BUDGET_CONVERGENCE_FUTURE_ONLY", True, raising=False)
    monkeypatch.setattr(go, "_protein_repeat_autofix", lambda days, fd: 0)
    monkeypatch.setattr(go, "apply_update_macro_engine",
                        lambda pd, surface=None, db=None, **kw: None)
    monkeypatch.setattr(go, "recompute_micronutrient_report_for_plan",
                        lambda pd, fd, db=None: None)
    return go


# ---------------------------------------------------------------------------
# 1. La ventana de futuro
# ---------------------------------------------------------------------------

def test_dias_ya_cocinados_no_reciben_sustitucion(_go):
    """El daño (a): reescribir un día pasado corrompe el historial que leen el chat y
    el quality index."""
    plan = _plan(pasados=6, futuros=9)
    n = _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    assert n > 0, "la convergencia debe seguir sustituyendo (si no, el test no prueba nada)"
    for i, d in enumerate(plan["days"][:6]):
        assert not any(m.get("_budget_substitutions") for m in d["meals"]), (
            f"día {d['date']} (ya cocinado) reescrito: historial corrupto")
        assert "quinoa" in d["meals"][0]["ingredients"][0].lower()
        assert "quinoa" in d["meals"][0]["ingredients_raw"][0].lower()
        assert "quinoa" in d["meals"][0]["name"].lower()


def test_dias_futuros_si_reciben_sustitucion(_go):
    plan = _plan(pasados=6, futuros=9)
    _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    tocados = _subs_de(plan)
    assert tocados, "la ventana no puede dejar el plan sin converger cuando hay días futuros"
    assert min(tocados) >= 6, f"un día pasado entró a la ventana: {tocados}"


def test_hoy_cuenta_como_futuro(_go):
    """`date >= today`: el día de HOY todavía no se cocinó entero — es sustituible."""
    plan = _plan(pasados=1, futuros=2)  # índices: ayer, hoy, mañana
    _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    tocados = _subs_de(plan)
    assert 1 in tocados, "el día de hoy debe seguir siendo sustituible"
    assert 0 not in tocados, "ayer no"


def test_today_acepta_iso_string(_go):
    plan = _plan(pasados=6, futuros=9)
    _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today="2026-08-02")
    assert min(_subs_de(plan)) >= 6


# ---------------------------------------------------------------------------
# 2. Fallbacks de fecha (planes viejos sin `date` estampada)
# ---------------------------------------------------------------------------

def test_fallback_grocery_start_date_infiere_indice(_go):
    """Plan viejo sin `date`: el ancla es `grocery_start_date` (el campo que el shift
    reescribe a hoy siguiendo a `days[0]`) + índice."""
    plan = _plan(pasados=6, futuros=9, con_fecha=False)
    plan["grocery_start_date"] = (_HOY - timedelta(days=6)).isoformat()
    _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    tocados = _subs_de(plan)
    assert tocados and min(tocados) >= 6, (
        f"la inferencia por grocery_start_date debe excluir los 6 días pasados: {tocados}")


def test_sin_fecha_ni_ancla_todo_es_ventana_fail_open(_go):
    """Fail-open documentado: sin `date` NI ancla no hay forma honesta de saber qué día
    es cuál — mejor sustituir de más que dejar de converger por un dato ausente."""
    plan = _plan(pasados=6, futuros=9, con_fecha=False)
    n = _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    assert n > 0
    assert _subs_de(plan)[0] == 0, "sin ancla, el barrido arranca en el día 0 (pre-fix)"


def test_dia_suelto_sin_fecha_entra_a_la_ventana(_go):
    """Un día con `date` corrupta dentro de un plan estampado: entra (fail-open) en vez
    de abortar la convergencia entera."""
    plan = _plan(pasados=0, futuros=3)
    plan["days"][1]["date"] = "no-es-una-fecha"
    _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    assert 1 in _subs_de(plan)


# ---------------------------------------------------------------------------
# 3. Lo que el usuario YA COMPRÓ
# ---------------------------------------------------------------------------

def test_alimento_en_la_nevera_no_se_sustituye(_go):
    """El daño (b): sustituir lo YA comprado deja el original huérfano y AÑADE el
    sustituto a la lista — el gasto real sube mientras el banner dice 'dentro'."""
    plan = _plan(pasados=0, futuros=9)
    n = _go.apply_budget_convergence_for_days(
        plan, {"budget": "low"}, today=_HOY,
        inventory_names=[{"ingredient_name": "Quinoa", "quantity": 500}],
    )
    assert n == 0
    assert all("quinoa" in m["ingredients"][0].lower()
               for d in plan["days"] for m in d["meals"]), \
        "sustituir lo YA comprado sube el gasto real del usuario"


def test_inventario_acepta_strings_y_no_afecta_a_otros_alimentos(_go):
    plan = _plan(pasados=0, futuros=9)
    n = _go.apply_budget_convergence_for_days(
        plan, {"budget": "low"}, today=_HOY, inventory_names=["Pollo", "Cebolla"],
    )
    assert n > 0, "un inventario sin el alimento en cuestión no debe frenar la convergencia"


def test_sin_inventario_comportamiento_previo(_go):
    plan = _plan(pasados=0, futuros=9)
    assert _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY) > 0


# ---------------------------------------------------------------------------
# 4. Knob OFF = pre-fix
# ---------------------------------------------------------------------------

def test_knob_off_restaura_el_barrido_completo(_go, monkeypatch):
    monkeypatch.setattr(go, "BUDGET_CONVERGENCE_FUTURE_ONLY", False)
    plan = _plan(pasados=6, futuros=9)
    n = _go.apply_budget_convergence_for_days(plan, {"budget": "low"}, today=_HOY)
    assert n > 0
    assert _subs_de(plan)[0] == 0, (
        "con el knob OFF el barrido arranca en days[0] como antes del fix")


def test_knob_off_ignora_el_inventario(_go, monkeypatch):
    monkeypatch.setattr(go, "BUDGET_CONVERGENCE_FUTURE_ONLY", False)
    plan = _plan(pasados=0, futuros=9)
    n = _go.apply_budget_convergence_for_days(
        plan, {"budget": "low"}, today=_HOY, inventory_names=["Quinoa"])
    assert n > 0, "el knob OFF debe ser byte-idéntico al pre-fix (que no miraba la Nevera)"


# ---------------------------------------------------------------------------
# 5. Fail-open + anclas de source
# ---------------------------------------------------------------------------

def test_fail_open_intacto():
    assert go.apply_budget_convergence_for_days(None, None) == 0
    assert go.apply_budget_convergence_for_days({}, {}) == 0
    assert go.apply_budget_convergence_for_days({"days": []}, {}, today=_HOY) == 0


def test_marker_y_knob_anclados():
    assert "P2-BUDGET-CONVERGENCE-FUTURE-ONLY" in _GO_SRC
    assert '_env_bool("MEALFIT_BUDGET_CONVERGENCE_FUTURE_ONLY", True)' in _GO_SRC, \
        "el knob debe registrarse vía _env_bool (auto-registro en _KNOBS_REGISTRY)"
    assert "def _budget_future_days_window(" in _GO_SRC
    assert "def _budget_owned_food_keys(" in _GO_SRC


def test_hoy_se_deriva_tz_aware():
    """El `utcnow()` naive está prohibido en producción (P3-DEPRECATED-UTCNOW): el
    default de `today` sale de `datetime.now(timezone.utc)` menos el offset RD."""
    i = _GO_SRC.index("def _budget_future_days_window(")
    blk = _GO_SRC[i:_GO_SRC.index("def _apply_budget_cheapen_pass(")]
    assert "datetime.now(timezone.utc)" in blk
    assert "datetime.utcnow" not in blk


def test_seam_t2_pasa_el_inventario_que_ya_tiene_cargado():
    """El seam ya cargó `_inv_s` para las 3 multiplicidades — pasárselo a la
    convergencia es cero IO nuevo. tooltip: P1-BUDGET-T2-CONVERGENCE."""
    i = _CRON_SRC.index("P1-BUDGET-T2-CONVERGENCE")
    blk = _CRON_SRC[i:i + 7000]
    assert "_abc_t2(full_plan_data, form_data, inventory_names=_inv_s)" in blk, \
        "la convergencia T2 debe recibir el snapshot de Nevera ya cargado (sin abrir conexión)"


def test_el_recosteo_sigue_viendo_el_plan_completo():
    """Decisión 2: solo la ESCRITURA se ventanea. El rebuild de listas del seam sigue
    pasando `full_plan_data` entero (el costo del ciclo es de todo el plan)."""
    i = _CRON_SRC.index("P1-BUDGET-T2-CONVERGENCE")
    blk = _CRON_SRC[i:i + 7000]
    assert blk.count("get_shopping_list_delta(") == 3
    assert blk.count("user_id, full_plan_data, is_new_plan=True") == 3, \
        "el re-costeo no puede recibir una lista de días recortada"
