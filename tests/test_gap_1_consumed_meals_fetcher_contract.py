"""[GAP-1 · 2026-05-29] Lock-the-contract fetcher↔consumidor de `consumed_meals`.

Bug cerrado (audit de optimización cron_tasks.py, workflow 2026-05-29, finding S09-1/S09-4):

    `cron_tasks.calculate_ingredient_fatigue` y `calculate_day_of_week_adherence`
    obtienen filas vía `db_facts.get_consumed_meals_since`, que SIEMPRE proyectó
    columnas explícitas `meal_name, calories, protein, carbs, healthy_fats,
    consumed_at, meal_type` (P2-SELECT-STAR-CONSUMED-MEALS) — NUNCA `created_at`
    ni `ingredients`. Pero ambas funciones leían `meal.get('created_at')` (→ None
    → days_ago=0 → decay/EMA muertos) y la fatiga además leía `meal.get('ingredients')`
    (→ None → lista de fatiga SIEMPRE vacía). Net en prod:

      * calculate_ingredient_fatigue → siempre {score:0.0, fatigued_ingredients:[]}
      * calculate_day_of_week_adherence → siempre {día:1.0} para los 7 días

    El masking: los fixtures de test usaban las claves ERRÓNEAS (`created_at` +
    `ingredients`) que casaban con lo que la función LEÍA, por eso los tests
    pasaban verde mientras prod estaba roto. Clase key-drift-fixture documentada en
    memoria P1-CHUNK-LEARN-AUDIT.

Fix:
    1. `get_consumed_meals_since(..., include_ingredients=True)` añade la columna
       `ingredients` (default lean preservado para los demás callers).
    2. Ambas funciones leen `consumed_at` vía `_coerce_consumed_at_to_dt`, que
       tolera datetime (psycopg dict_row sobre timestamptz) y string ISO
       (supabase/fixtures).
    3. `calculate_meal_success_scores` deja de escribir el dead-key `meal_type`
       (S09-4).

Este test ancla las DOS mitades del contrato (estructural sobre fuente +
conductual con filas reales-shaped) para que un futuro key-drift falle aquí
ANTES de degradar silenciosamente la personalización.

Tooltip-anchor: GAP-1.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import cron_tasks


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CRON_PY = _REPO_ROOT / "backend" / "cron_tasks.py"
_DBFACTS_PY = _REPO_ROOT / "backend" / "db_facts.py"


def _slice_fn(src: str, header: str) -> str:
    start = src.find(header)
    assert start >= 0, f"no se encontró {header!r}"
    after = src[start + len(header):]
    nxt = re.search(r"\n(?:def |class )\w", after)
    return after[: nxt.start()] if nxt else after


# ---------------------------------------------------------------------------
# Estructural: el fetcher proyecta consumed_at (no created_at) + ingredients opt-in
# ---------------------------------------------------------------------------

def _columns_literal(body: str) -> str:
    """Extrae el literal de la tupla `_COLUMNS = ( ... )` (las columnas SQL reales),
    excluyendo docstrings/comentarios que puedan mencionar nombres de columnas."""
    m = re.search(r"_COLUMNS\s*=\s*\((.*?)\)", body, re.S)
    assert m, "no se encontró el literal _COLUMNS"
    extra = re.search(r"_COLUMNS\s*=\s*_COLUMNS\s*\+\s*([\"'][^\"']*[\"'])", body)
    return m.group(1) + (extra.group(1) if extra else "")


def test_fetcher_projects_consumed_at_not_created_at():
    body = _slice_fn(_DBFACTS_PY.read_text(encoding="utf-8"), "def get_consumed_meals_since(")
    cols = _columns_literal(body)
    assert "consumed_at" in cols, "el fetcher debe proyectar consumed_at"
    # En las COLUMNAS SQL reales (no en docstrings) jamás debe aparecer created_at.
    assert "created_at" not in cols, (
        "get_consumed_meals_since NO proyecta created_at; si esto cambia, los "
        "consumidores que leen consumed_at se romperían (key-drift)."
    )


def test_fetcher_has_include_ingredients_opt_in():
    body = _slice_fn(_DBFACTS_PY.read_text(encoding="utf-8"), "def get_consumed_meals_since(")
    assert "include_ingredients" in body, "debe existir el parámetro include_ingredients"
    assert re.search(r"if include_ingredients", body), "include_ingredients debe gatear ', ingredients'"
    assert "ingredients" in body


# ---------------------------------------------------------------------------
# Estructural: los consumidores leen consumed_at (no created_at) + opt-in ingredients
# ---------------------------------------------------------------------------

def test_fatigue_reads_consumed_at_and_opts_into_ingredients():
    body = _slice_fn(_CRON_PY.read_text(encoding="utf-8"), "def calculate_ingredient_fatigue(")
    assert "include_ingredients=True" in body, "fatiga debe pedir la columna ingredients"
    assert "consumed_at" in body, "fatiga debe leer consumed_at"
    assert "meal.get('created_at')" not in body and 'meal.get("created_at")' not in body, (
        "fatiga NO debe leer created_at (clave inexistente en el fetcher → bug original)"
    )


def test_day_of_week_reads_consumed_at_not_created_at():
    body = _slice_fn(_CRON_PY.read_text(encoding="utf-8"), "def calculate_day_of_week_adherence(")
    assert "consumed_at" in body
    assert "meal.get('created_at')" not in body and 'meal.get("created_at")' not in body


def test_meal_success_scores_drops_dead_meal_type_write():
    body = _slice_fn(_CRON_PY.read_text(encoding="utf-8"), "def calculate_meal_success_scores(")
    assert "'meal_type': meal.get('meal')" not in body, "dead-write meal_type debe estar eliminado (S09-4)"


# ---------------------------------------------------------------------------
# Conductual: con filas REAL-shaped, las funciones producen señal no-vacía
# ---------------------------------------------------------------------------

def _real_shaped_rows(consumed_at_as_datetime: bool):
    """Filas como las que devuelve el fetcher REAL: meal_name + consumed_at + ingredients."""
    now = datetime.now(timezone.utc)
    rows = []
    for _ in range(10):
        ca = now if consumed_at_as_datetime else now.isoformat()
        rows.append({"meal_name": "Pollo guisado", "consumed_at": ca, "ingredients": ["pollo", "arroz"]})
    return rows


@pytest.mark.parametrize("as_dt", [False, True])
def test_fatigue_nonempty_with_real_fetcher_shape(as_dt):
    """consumed_at como string ISO (supabase) Y como datetime (psycopg) deben
    ambos producir fatiga de 'pollo' — prueba el coerce robusto + el contrato real."""
    with patch.object(cron_tasks, "get_consumed_meals_since", return_value=_real_shaped_rows(as_dt)):
        res = cron_tasks.calculate_ingredient_fatigue("u-1", days_back=14)
    fatigued = " ".join(res.get("fatigued_ingredients", [])).lower()
    assert "pollo" in fatigued, f"pollo 10x hoy debe ser fatiga; got {res.get('fatigued_ingredients')}"
    assert res["score"] > 0.0


def test_fatigue_passes_include_ingredients_to_fetcher():
    """El callsite real debe invocar el fetcher con include_ingredients=True."""
    captured = {}

    def _spy(user_id, since_iso_date=None, include_ingredients=False, **kw):
        captured["include_ingredients"] = include_ingredients
        return []

    with patch.object(cron_tasks, "get_consumed_meals_since", side_effect=_spy):
        cron_tasks.calculate_ingredient_fatigue("u-1", days_back=14)
    assert captured.get("include_ingredients") is True


# ---------------------------------------------------------------------------
# S09-2: el param opcional `consumed` filtra a la ventana days_back (equivale al fetch)
# ---------------------------------------------------------------------------

def test_fatigue_consumed_param_filters_to_window():
    """[S09-2] Pasar un superset pre-fetcheado debe filtrarse a la ventana days_back —
    equivalente exacto a fetchear a esa ventana (consumed_meals es append-only)."""
    now = datetime.now(timezone.utc)
    superset = (
        [{"meal_name": "Pollo", "consumed_at": now.isoformat(), "ingredients": ["pollo"]} for _ in range(10)]
        + [{"meal_name": "Pescado", "consumed_at": (now - timedelta(days=40)).isoformat(), "ingredients": ["pescado"]} for _ in range(10)]
    )
    # days_back=14 → el pescado de hace 40d queda FUERA de ventana → no debe aparecer.
    res = cron_tasks.calculate_ingredient_fatigue("u-1", days_back=14, consumed=superset)
    fatigued = " ".join(res.get("fatigued_ingredients", [])).lower()
    assert "pollo" in fatigued
    assert "pescado" not in fatigued, f"pescado a 40d debe filtrarse con days_back=14; got {res.get('fatigued_ingredients')}"


def test_fatigue_consumed_param_does_not_refetch():
    """[S09-2] Si se provee `consumed`, NO debe llamarse al fetcher (evita el round-trip)."""
    now = datetime.now(timezone.utc)
    superset = [{"meal_name": "Res", "consumed_at": now.isoformat(), "ingredients": ["res"]} for _ in range(6)]
    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return []

    with patch.object(cron_tasks, "get_consumed_meals_since", side_effect=_spy):
        cron_tasks.calculate_ingredient_fatigue("u-1", days_back=14, consumed=superset)
    assert called["n"] == 0, "con consumed provisto, calculate_ingredient_fatigue NO debe re-fetchear"


def test_day_of_week_consumed_param_filters_to_window():
    """[S09-2] day_of_week con superset filtra a la ventana y produce señal real."""
    now = datetime.now(timezone.utc)
    superset = (
        [{"meal_name": f"m{i}", "consumed_at": (now - timedelta(days=2)).isoformat()} for i in range(6)]
        + [{"meal_name": f"old{i}", "consumed_at": (now - timedelta(days=120)).isoformat()} for i in range(6)]
    )
    res = cron_tasks.calculate_day_of_week_adherence("u-1", days_back=30, consumed=superset)
    assert isinstance(res, dict) and len(res) == 7
    assert any(v != 1.0 for v in res.values())


@pytest.mark.parametrize("as_dt", [False, True])
def test_day_of_week_detects_real_dates(as_dt):
    """Con consumed_at real, la EMA NO debe colapsar al fallback uniforme {día:1.0}."""
    now = datetime.now(timezone.utc)
    # 6 comidas el mismo día reciente, 0 en el resto → un weekday debe diverger de 1.0.
    rows = []
    for i in range(6):
        ca = (now - timedelta(days=2)) if as_dt else (now - timedelta(days=2)).isoformat()
        rows.append({"meal_name": f"m{i}", "consumed_at": ca})
    with patch.object(cron_tasks, "get_consumed_meals_since", return_value=rows):
        res = cron_tasks.calculate_day_of_week_adherence("u-1", days_back=30)
    assert isinstance(res, dict) and len(res) == 7
    # Al menos un día con datos reales debe diferir del fallback uniforme 1.0.
    assert any(v != 1.0 for v in res.values()), (
        f"con datos reales la EMA debe diverger de todo-1.0; got {res}"
    )


# ===========================================================================
# NG-1 (key-drift round 2, audit 2026-05-29 segunda pasada) — hermanos de GAP-1
# que el primer barrido omitió. Mismo principio: anclar fetcher↔consumidor.
# ===========================================================================

# --- NG-1A: calculate_meal_level_adherence leía 'created_at' (inexistente) → -------
#            unique_dates siempre vacío → days_passed=1 → adherencia por meal-type
#            saturaba a ~1.0 con una sola comida → abandono de meal-type indetectable.

def test_meal_level_adherence_reads_consumed_at_not_created_at():
    body = _slice_fn(_CRON_PY.read_text(encoding="utf-8"), "def calculate_meal_level_adherence(")
    assert "consumed_at" in body, "meal-level adherence debe leer consumed_at"
    assert "'created_at' in record" not in body and '"created_at" in record' not in body, (
        "meal-level adherence NO debe leer created_at (clave inexistente en el fetcher → "
        "days_passed siempre 1 → adherencia por meal-type saturada)"
    )


@pytest.mark.parametrize("as_dt", [False, True])
def test_meal_level_adherence_days_passed_reflects_unique_dates(as_dt):
    """Con 1 desayuno comido pero 5 días distintos registrados, days_passed=5 → la
    adherencia del desayuno cae a ~0.2 (1/(1*5)). Con el bug original (created_at)
    days_passed=1 → saturaba a 1.0. Este test discrimina el fix."""
    now = datetime.now(timezone.utc)
    plan_days = [{"meals": [{"meal": "desayuno"}]}]
    rows = [{"meal_type": "desayuno", "consumed_at": (now if as_dt else now.isoformat())}]
    for i in range(1, 5):  # 4 días distintos adicionales (otro meal-type)
        ca = (now - timedelta(days=i)) if as_dt else (now - timedelta(days=i)).isoformat()
        rows.append({"meal_type": "almuerzo", "consumed_at": ca})
    res = cron_tasks.calculate_meal_level_adherence("u-1", plan_days, rows, household_size=1)
    assert "desayuno" in res
    assert res["desayuno"] < 1.0, (
        f"con 5 días distintos y 1 desayuno, la adherencia debe ser ~0.2 (no saturar a 1.0); "
        f"got {res['desayuno']} — indica days_passed=1 (bug created_at)"
    )


# --- NG-1B: calculate_plan_quality_score lee r.get('ingredients'); 3 callsites no -----
#            proyectaban la columna → diversity_score SIEMPRE 0 (20% del score).

def test_quality_score_callsites_opt_into_ingredients():
    """Los 3 fetches que alimentan calculate_plan_quality_score deben pasar
    include_ingredients=True (chunk worker, persist nightly, trigger incremental)."""
    src = _CRON_PY.read_text(encoding="utf-8")
    # 26753 (chunk worker) + 29549 (persist nightly post-chunk): mismo literal, ≥2 veces.
    n_chunk = src.count("get_consumed_meals_since(user_id, since_time, include_ingredients=True)")
    assert n_chunk >= 2, (
        f"se esperaban ≥2 callsites `since_time, include_ingredients=True`; got {n_chunk}"
    )
    # 30006 (trigger_incremental_learning).
    assert "get_consumed_meals_since(user_id, plan_start_date_str, include_ingredients=True)" in src, (
        "trigger_incremental_learning debe fetchear con include_ingredients=True"
    )


def test_quality_score_diversity_uses_ingredients():
    """Con ingredients presentes, diversity contribuye → el score es estrictamente
    mayor que el mismo cálculo sin ingredients (donde diversity colapsa a 0)."""
    import db_facts
    plan_data = {"days": [{"meals": [{"name": "X"}, {"name": "Y"}]}]}
    base = {"meal_name": "almuerzo", "consumed_at": datetime.now(timezone.utc).isoformat()}
    rec_with = dict(base, ingredients=[
        {"name": "pollo"}, {"name": "arroz"}, {"name": "habichuela"},
        {"name": "platano"}, {"name": "aguacate"},
    ])
    rec_without = dict(base)  # sin la clave ingredients (estado pre-fix)
    with patch.object(cron_tasks, "get_user_likes", return_value=[]), \
         patch.object(cron_tasks, "get_active_rejections", return_value=[]), \
         patch.object(db_facts, "get_consumed_meals_since", return_value=[]):
        q_with = cron_tasks.calculate_plan_quality_score("u-1", plan_data, [rec_with], household_size=1)
        q_without = cron_tasks.calculate_plan_quality_score("u-1", plan_data, [rec_without], household_size=1)
    assert q_with > q_without, (
        f"diversity debe elevar el score cuando hay ingredients; with={q_with} without={q_without}"
    )
