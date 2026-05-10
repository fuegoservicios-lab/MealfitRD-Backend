"""
Tests P1-5: Smart Shuffle empty/insufficient day rescue.

Smart Shuffle puede producir días con `meals` vacío o insuficiente cuando:
  - Todos los candidatos del pool son edge recipes con catálogo restringido.
  - Pantry items no intersectan con el catálogo del edge recipe.
  - Pools degradados (safe_pool, backup_days) están vacíos.

Antes el día insuficiente se persistía silenciosamente y el usuario veía un
día sin comidas en su plan.

Cambio P1-5:
  1. Calcular `expected_meals` = max meals_count en safe_pool/backup_days, o 3.
  2. Detectar días con meals_count < expected.
  3. Para cada día insuficiente, intentar reemplazar con edge_day sin
     restricción de pantry (pantry_items=None).
  4. Si el reemplazo tampoco alcanza, pausar el chunk con
     `_pause_reason='shuffle_empty_day'` en vez de persistir días vacíos.

Tests unitarios validan la lógica core extraída del flow.
"""
import pytest


def _meal_count(d):
    """Replica la lógica de _p15_meal_count en cron_tasks.py:_chunk_worker.

    Cuenta meals con campo `name` no vacío. Meals dict sin name no cuentan.
    """
    if not isinstance(d, dict):
        return 0
    return len([
        m for m in (d.get("meals") or [])
        if isinstance(m, dict) and m.get("name")
    ])


def _compute_expected(pool):
    """Replica el cálculo de _p15_expected_meals."""
    counts = [_meal_count(d) for d in (pool or [])[:10]]
    counts = [c for c in counts if c > 0]
    return max(counts) if counts else 3


def _detect_insufficient(new_days, expected):
    """Returns lista de índices de días con meals_count < expected."""
    return [
        i for i, d in enumerate(new_days)
        if _meal_count(d) < expected
    ]


def _day(day_num, meal_names):
    """Helper: construye día con N meals nombrados."""
    return {
        "day": day_num,
        "meals": [
            {"name": name, "type": "Almuerzo", "ingredients": ["x"]}
            for name in meal_names
        ],
    }


def _empty_day(day_num):
    """Día sin meals (peor caso)."""
    return {"day": day_num, "meals": []}


def _malformed_day(day_num):
    """Día con meals que no son dicts (defensivo)."""
    return {"day": day_num, "meals": [None, "string", 42]}


# ---------------------------------------------------------------------------
# _meal_count: invariante base
# ---------------------------------------------------------------------------
def test_meal_count_zero_when_no_meals():
    assert _meal_count(_empty_day(1)) == 0


def test_meal_count_counts_only_named_meals():
    day = {
        "day": 1,
        "meals": [
            {"name": "Pollo", "type": "Almuerzo"},
            {"name": "", "type": "Cena"},  # name vacío
            {"type": "Snack"},  # sin name
        ],
    }
    assert _meal_count(day) == 1


def test_meal_count_handles_malformed():
    assert _meal_count(_malformed_day(1)) == 0
    assert _meal_count(None) == 0
    assert _meal_count("string") == 0
    assert _meal_count({}) == 0


# ---------------------------------------------------------------------------
# _compute_expected: max heurístico desde pool
# ---------------------------------------------------------------------------
def test_compute_expected_uses_max_in_pool():
    pool = [
        _day(1, ["A", "B"]),  # 2
        _day(2, ["C", "D", "E", "F"]),  # 4
        _day(3, ["G", "H", "I"]),  # 3
    ]
    assert _compute_expected(pool) == 4


def test_compute_expected_default_when_empty_pool():
    """Pool vacío → default a 3."""
    assert _compute_expected([]) == 3
    assert _compute_expected(None) == 3


def test_compute_expected_default_when_pool_all_empty_days():
    """Pool con días sin meals → default a 3 (no zero)."""
    pool = [_empty_day(1), _empty_day(2)]
    assert _compute_expected(pool) == 3


def test_compute_expected_caps_at_first_10():
    """Solo considera los primeros 10 días del pool."""
    # 9 días con 2 meals y 1 día con 5 meals al final (fuera del cap)
    pool = [_day(i, ["A", "B"]) for i in range(1, 11)] + [_day(11, ["A", "B", "C", "D", "E"])]
    assert _compute_expected(pool) == 2  # max de los primeros 10


# ---------------------------------------------------------------------------
# _detect_insufficient: identificar días sub-expected
# ---------------------------------------------------------------------------
def test_detect_insufficient_finds_short_days():
    new_days = [
        _day(4, ["A", "B", "C"]),  # 3 ok
        _day(5, ["D"]),  # 1 < 3
        _empty_day(6),  # 0 < 3
        _day(7, ["E", "F", "G"]),  # 3 ok
    ]
    insufficient = _detect_insufficient(new_days, expected=3)
    assert insufficient == [1, 2]


def test_detect_insufficient_returns_empty_when_all_ok():
    new_days = [_day(i, ["A", "B", "C", "D"]) for i in range(4, 8)]
    assert _detect_insufficient(new_days, expected=3) == []


def test_detect_insufficient_all_short():
    new_days = [_empty_day(i) for i in range(4, 7)]
    assert _detect_insufficient(new_days, expected=3) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Escenarios de rescate
# ---------------------------------------------------------------------------
def test_rescue_replaces_insufficient_day_when_edge_day_meets_expected():
    """Si edge_day rescue tiene >= expected meals, reemplaza el slot insuficiente."""
    new_days = [
        _day(4, ["A", "B", "C"]),
        _day(5, ["D"]),  # insuficiente
    ]
    expected = 3
    edge_rescue = _day(99, ["X", "Y", "Z"])  # 3 meals, suficiente

    insufficient = _detect_insufficient(new_days, expected)
    assert insufficient == [1]

    # Simular rescate: preservar day/day_name del slot original
    rescue_idx = insufficient[0]
    orig_day_num = new_days[rescue_idx].get("day")
    edge_rescue["day"] = orig_day_num
    edge_rescue["day_name"] = new_days[rescue_idx].get("day_name", "")
    edge_rescue["_p1_5_rescued"] = True
    new_days[rescue_idx] = edge_rescue

    # Re-validar
    still_insufficient = _detect_insufficient(new_days, expected)
    assert still_insufficient == []
    assert new_days[1]["day"] == 5  # día preservado
    assert new_days[1]["_p1_5_rescued"] is True


def test_rescue_does_not_touch_sufficient_days():
    """Días con meals_count == expected NO deben tocarse durante el rescate."""
    new_days = [
        _day(4, ["A", "B", "C"]),  # exactamente expected
        _day(5, ["D", "E", "F", "G"]),  # más que expected
    ]
    expected = 3
    insufficient = _detect_insufficient(new_days, expected)
    assert insufficient == []


def test_rescue_pause_when_edge_day_also_insufficient():
    """Si edge_day rescue NO tiene meals suficientes, NO se reemplaza el slot
    y se debe disparar la pausa."""
    new_days = [_day(4, ["A"])]  # 1 meal, expected=3
    expected = 3
    edge_rescue_short = _day(99, ["X"])  # también insuficiente

    insufficient = _detect_insufficient(new_days, expected)
    assert insufficient == [0]

    # Lógica del fix: reemplazar SOLO si edge_rescue >= expected
    if _meal_count(edge_rescue_short) >= expected:
        new_days[0] = edge_rescue_short

    still_insufficient = _detect_insufficient(new_days, expected)
    # El slot sigue insuficiente → debería triggerear pausa
    assert still_insufficient == [0]


def test_rescue_pause_when_edge_day_returns_none():
    """Si _build_filtered_edge_recipe_day retorna None (catálogo agotado),
    no se reemplaza nada → pausa."""
    new_days = [_empty_day(4)]
    expected = 3
    edge_rescue = None  # catálogo no tiene candidatos

    insufficient = _detect_insufficient(new_days, expected)
    assert insufficient == [0]

    # No replacement
    if edge_rescue and _meal_count(edge_rescue) >= expected:
        new_days[0] = edge_rescue

    still_insufficient = _detect_insufficient(new_days, expected)
    assert still_insufficient == [0]


# ---------------------------------------------------------------------------
# Sanity check de imports
# ---------------------------------------------------------------------------
def test_cron_tasks_imports_after_p1_5_changes():
    """El módulo cron_tasks.py debe importar sin errores tras los cambios P1-5."""
    import cron_tasks  # noqa: F401
    # Helpers de fixes anteriores siguen presentes.
    assert hasattr(cron_tasks, "_touch_chunk_heartbeat")
    assert hasattr(cron_tasks, "_validate_merged_days_against_pantry")
    assert hasattr(cron_tasks, "_build_zero_log_push_payload")
