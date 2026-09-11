"""[P1-PLAN-LOTE-10 · 2026-09-11] Décimo lote del plan de pendientes: B7, B9, D6, F3, E3 — medición y documentación.

B7 · El doc del día determinista decía «proteína −18,6 %, el scorer no es la palanca». Re-medido el 2026-09-11 con un CLI
     nuevo (`scripts/measure_deterministic_day_macros.py`): la proteína estaba CERRADA (+2,4 %, 14/14 en banda) desde el
     catálogo de proteína del 09-09; el sesgo real era carbohidrato +18,0 % / grasa −17,7 %. Y el scorer SÍ era la
     palanca, pero por la GRASA: la biblioteca es alta en carbohidrato y baja en grasa de forma sistemática, y un score
     simétrico prefería el plato exacto en proteína aunque se pasara de carbohidrato — los bajos en carbohidrato existían
     y escalaban y no se servían nunca. Los pesos direccionales son knobs (default 1.0/1.0 = anterior). Medido en TRES
     dianas: 2.0/2.0 parecía perfecto en la estándar y hundía la proteína en pérdida (−12 %); 2.0/1.0 es el compromiso y
     encenderlo lo decide el dueño. El resto es composición de la biblioteca (dueño).
B9 · `pipeline_metrics.wizard_funnel`: 121 filas, 5 sesiones, 2 días; el frontend nunca emite `step_done` ⇒ el embudo
     por paso del doc era ciego. SQL proxy documentado.
D6 · El «bug latente» de la rama ventaneada ya estaba cerrado; anotado donde el plan lo citaba.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── B7 · el scorer pesa la dirección en que la biblioteca se equivoca ───────────────────────────

_CAT = {
    "Proteína pura": {"kcal_per_100g": 100.0, "protein_g_per_100g": 25.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 0.0},
    "Carbo puro": {"kcal_per_100g": 100.0, "protein_g_per_100g": 0.0, "carbs_g_per_100g": 25.0, "fats_g_per_100g": 0.0},
    "Grasa pura": {"kcal_per_100g": 90.0, "protein_g_per_100g": 0.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 10.0},
}


def _tpl(tid, p, c, f):
    """Plantilla con exactamente p/c/f gramos de macros (vía ingredientes puros)."""
    return {"template_id": tid, "name": tid, "constituents": [
        {"grams": p * 4, "name": "Proteína pura"}, {"grams": c * 4, "name": "Carbo puro"}, {"grams": f * 10, "name": "Grasa pura"}]}


# objetivo de la franja: 400 kcal · P 30 · C 40 · F 12
_OBJ = {"kcal": 400.0, "protein_g": 30.0, "carbs_g": 40.0, "fats_g": 12.0}
# A: exacto en proteína, pasado de carbohidrato y corto de grasa (el plato típico de la biblioteca)
# B: corto en proteína, casi exacto en carbohidrato y con la grasa cubierta
_POR_ID = {"A_carbo_alto": _tpl("A_carbo_alto", 30, 52, 8), "B_equilibrado": _tpl("B_equilibrado", 22, 40, 15)}


def test_b7_por_defecto_los_pesos_son_1_0_y_el_scorer_es_el_simetrico_anterior(monkeypatch):
    """El default NO cambia la conducta: 2.0/2.0 parecía perfecto en una diana y hundía la proteína en pérdida."""
    monkeypatch.delenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS", raising=False)
    monkeypatch.delenv("MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT", raising=False)
    import deterministic_day as dd
    assert dd._pesos_scorer() == (1.0, 1.0)
    t, _f = dd.elegir_plantilla(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo")
    assert t["template_id"] == "A_carbo_alto", "el scorer simétrico prefiere el exacto en proteína aunque se pase de carbohidrato"


def test_b7_con_el_compromiso_medido_2_1_gana_el_equilibrado(monkeypatch):
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS", "2.0")
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT", "1.0")
    import deterministic_day as dd
    assert dd._pesos_scorer() == (2.0, 1.0)
    t, _f = dd.elegir_plantilla(list(_POR_ID), _OBJ, _CAT, _POR_ID, "almuerzo")
    assert t["template_id"] == "B_equilibrado", "pasarse de carbohidrato cuesta lo que la proteína: el canario del dueño"


def test_b7_el_deficit_de_carbohidrato_y_el_exceso_de_grasa_no_cambian_de_peso(monkeypatch):
    """Los pesos son DIRECCIONALES: castigan la dirección en que la biblioteca se equivoca, no la contraria."""
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS", "2.0")
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT", "2.0")
    import deterministic_day as dd
    por_id = {"C_carbo_bajo_grasa_alta": _tpl("C_carbo_bajo_grasa_alta", 30, 32, 16),   # −8 C, +4 F
              "D_carbo_alto_grasa_baja": _tpl("D_carbo_alto_grasa_baja", 30, 48, 8)}    # +8 C, −4 F
    t, _f = dd.elegir_plantilla(list(por_id), _OBJ, _CAT, por_id, "almuerzo")
    assert t["template_id"] == "C_carbo_bajo_grasa_alta", "misma distancia: gana el que se equivoca en la dirección barata"


def test_b7_los_pesos_se_acotan_a_un_rango_real(monkeypatch):
    import deterministic_day as dd
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS", "99")
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT", "0")
    assert dd._pesos_scorer() == (1.0, 1.0), "fuera de [0.5, 5] cae al default, no revienta ni anula un macro"
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS", "3")
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT", "0.5")
    assert dd._pesos_scorer() == (3.0, 0.5)


def test_b7_la_proteina_sigue_pesando_el_doble_y_simetrica():
    src = _src("deterministic_day.py")
    i = src.find("def elegir_plantillas(")
    cuerpo = src[i:src.find("\ndef ", i + 10)]
    assert '2.0 * abs(base["protein_g"] * f - op) / op' in cuerpo
    assert '(_w_cs if _dc > 0 else 1.0) * abs(_dc) / oc' in cuerpo
    assert '(_w_fd if _df < 0 else 1.0) * abs(_df) / of' in cuerpo
    assert "_w_cs, _w_fd = _pesos_scorer()" in cuerpo, "los knobs se leen UNA vez por franja, no por candidato"


def test_b7_el_cli_de_medicion_existe_y_se_explica_solo():
    p = _BACKEND / "scripts" / "measure_deterministic_day_macros.py"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    assert "[P2-LOGGER-EXEMPT" in src and "def medir(" in src and "--list-low-carb" in src
    out = subprocess.run([sys.executable, str(p), "--help"], capture_output=True, text=True, cwd=str(_BACKEND), timeout=120)
    assert out.returncode == 0 and "--carbs" in out.stdout and "--condition" in out.stdout


def test_b7_el_doc_ya_no_dice_que_el_scorer_no_es_la_palanca():
    doc = _src("docs/deterministic_day.md")
    assert "MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS" in doc and "MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT" in doc
    assert "Re-medición 2026-09-11" in doc and "re-medido el 2026-09-11" in doc
    assert "medido en tres dianas dejó de parecer perfecto" in doc, "la matriz de tres dianas es lo que impide vender 2.0/2.0"
    assert "el default no cambia y el dueño decide" in doc
    assert "Queda abierto: las palancas reales son el **scorer**" not in doc, "el párrafo viejo reabría lo que ya está medido"
    knobs = _src("docs/knobs_reference.md")
    assert "### Día determinista (scorer)" in knobs and "MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT" in knobs


# ─────────────────────────── B9 · el embudo por paso, honesto ───────────────────────────

def test_b9_el_doc_dice_que_step_done_no_se_emite_y_da_el_proxy():
    doc = _src("docs/plan_policy_f4.md")
    assert "nunca `step_done`" in doc and "lo emite al AVANZAR" in doc, "la ceguera medida y su cierre en el frontend"
    assert "WHERE b.session_id = a.session_id AND b.idx > a.idx" in doc, "el proxy: terminado = la sesión vio un paso posterior"
    assert "121 filas de 5 sesiones" in doc


# ─────────────────────────── D6 · lo latente ya cerrado, anotado donde se citaba ───────────────────────────

def test_d6_el_bug_latente_esta_cerrado_en_codigo_y_en_el_doc():
    sc = _src("shopping_calculator.py")
    i = sc.find("_res_window = aggregate_and_deduct_shopping_list(")
    assert i > 0 and "text_demand_g_map=_tdg_para_agg" in sc[i:i + 600], "la 2.ª pasada recibe el mapa de demanda (el bug latente)"
    assert "[P1-PLAN-LOTE-10 · 2026-09-11 · D6]" in _src("docs/knobs_reference.md")


def test_e5_e7_el_diseno_existe_y_no_transfiere_autoridad_en_el_mismo_paso():
    doc = _src("docs/arq30_e5_e7_diseno_canario.md")
    for ancla in ("MEALFIT_CANONICAL_SHOPPING_USERS", "shopping_source_days", "allocator_version", "_portion_repairs",
                  "ValidationReport", "nunca se borra la vía anterior"):
        assert ancla in doc, ancla
    import canonical_recipe as cr
    assert all(hasattr(cr, n) for n in ("IngredientLine", "parse_line", "shopping_view")), "el doc cita una API que existe"


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-10 · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
