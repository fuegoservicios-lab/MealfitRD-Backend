"""[P3-NEW-B · 2026-05-08] Test unitario directo de `_pantry_refresh_horizon_hours_for_plan`.

Antes solo estaba probada por integración (vía
`_proactive_refresh_pending_pantry_snapshots`), así un refactor que cambiara
el threshold de 30 días o el tipado de la entrada podía romper silenciosamente
el horizonte de refresh proactivo de despensa sin que ningún test fallara.

Función bajo test (cron_tasks.py:1236):

    def _pantry_refresh_horizon_hours_for_plan(total_days_requested: int | None) -> int:
        try:
            total_days = int(total_days_requested or 0)
        except Exception:
            total_days = 0
        if total_days >= 30:
            return CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS
        return min(CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS, 48)

Comportamiento esperado (default `CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS=168`):
  - `total_days_requested >= 30` → 168 (horizonte amplio: planes 30d necesitan
    barrido hasta 7 días para captar chunks lejanos).
  - `total_days_requested < 30` → 48 (planes 7d/15d, ventana ajustada).
  - `None`/falsy/no parseable → tratado como 0 → 48.

NO toca código de producción. Defensa contra refactors futuros.
"""
import pytest

from cron_tasks import (
    _pantry_refresh_horizon_hours_for_plan as horizon_for,
)
from constants import CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS as _MAX_HORIZON


# ---------------------------------------------------------------------------
# Valor "short" canónico: lo que se devuelve para planes <30 días.
# La función calcula `min(_MAX_HORIZON, 48)`. Mientras el knob env esté
# por encima de 48 (default 168, mínimo enforced=48), el resultado es 48.
# ---------------------------------------------------------------------------
_SHORT_HORIZON = min(_MAX_HORIZON, 48)


# ===========================================================================
# 1. Boundary del threshold (30 días)
# ===========================================================================
@pytest.mark.parametrize(
    "days, expected",
    [
        (29, _SHORT_HORIZON),   # justo bajo el threshold
        (30, _MAX_HORIZON),     # threshold exacto: >=30 entra en rama amplia
        (31, _MAX_HORIZON),     # justo encima
    ],
)
def test_threshold_boundary_30_days(days, expected):
    """`>= 30` discrimina entre horizonte corto (48h) y amplio (168h por
    default). Si un refactor cambia el threshold (e.g., a 28 o 45) sin
    actualizar callers, este test falla intencionalmente."""
    assert horizon_for(days) == expected


# ===========================================================================
# 2. Plan lengths comunes del producto
# ===========================================================================
@pytest.mark.parametrize(
    "days, expected, label",
    [
        (1,   _SHORT_HORIZON, "plan 1d (test/manual)"),
        (3,   _SHORT_HORIZON, "plan 3d (preview)"),
        (7,   _SHORT_HORIZON, "plan 7d (canónico semanal)"),
        (14,  _SHORT_HORIZON, "plan 14d (quincenal)"),
        (15,  _SHORT_HORIZON, "plan 15d (canónico quincenal)"),
        (30,  _MAX_HORIZON,   "plan 30d (canónico mensual)"),
        (45,  _MAX_HORIZON,   "plan 45d (long-tenured)"),
        (90,  _MAX_HORIZON,   "plan 90d (enterprise)"),
    ],
)
def test_common_plan_lengths(days, expected, label):
    """Los plan lengths que el producto soporta deben caer en el bucket
    correcto. Documenta el label para que pytest -v diga claramente qué
    semántica está cubriendo cada caso."""
    assert horizon_for(days) == expected, f"plan={label} ({days}d)"


# ===========================================================================
# 3. None y otros falsy → tratados como 0 → horizon corto
# ===========================================================================
@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        False,
        "",
        [],
        {},
    ],
    ids=["None", "int_0", "False", "empty_string", "empty_list", "empty_dict"],
)
def test_falsy_inputs_return_short_horizon(value):
    """`int(falsy or 0) = 0`, que cae en `<30 → short`. Cubre el path donde
    el caller no tiene `total_days_requested` poblado (plan recién creado,
    fixture sin setup)."""
    assert horizon_for(value) == _SHORT_HORIZON


# ===========================================================================
# 4. Valores patológicos: negativos y muy grandes
# ===========================================================================
@pytest.mark.parametrize("days", [-1, -7, -100, -99999])
def test_negative_days_return_short_horizon(days):
    """Días negativos no tienen sentido semántico pero la función los
    procesa: `int(-N or 0) = -N` y `-N < 30` → short. El comportamiento
    actual no escala el bug (si el caller pasa negativos, su modo de fallo
    es el query SQL downstream, no este helper)."""
    assert horizon_for(days) == _SHORT_HORIZON


@pytest.mark.parametrize("days", [99999, 1_000_000, 2**31 - 1])
def test_extremely_large_days_return_max_horizon(days):
    """Días absurdamente grandes (overflow accidental, payload bogus) caen
    igual en la rama amplia. La función no enforca un cap superior — eso
    es responsabilidad de validators upstream (`_BIO_RANGES`/router).
    Aquí solo verificamos que no rompa para inputs grandes."""
    assert horizon_for(days) == _MAX_HORIZON


# ===========================================================================
# 5. No castable a int → exception caught → 0 → short
# ===========================================================================
@pytest.mark.parametrize(
    "value",
    [
        "abc",                  # string no numérico
        "30 days",              # numérico sucio
        {"days": 30},           # dict
        ("not", "numeric"),     # tupla
        object(),               # objeto arbitrario
    ],
    ids=["non_numeric_str", "dirty_string", "dict", "tuple", "object"],
)
def test_non_castable_inputs_default_to_short_horizon(value):
    """`int(<value>)` lanza ValueError/TypeError → caught → `total_days = 0`
    → short horizon. Cubre payloads malformados sin que la función propague
    excepciones al cron caller (que se ejecuta cada N minutos sin retry)."""
    assert horizon_for(value) == _SHORT_HORIZON


# ===========================================================================
# 6. Strings numéricos válidos
# ===========================================================================
@pytest.mark.parametrize(
    "value, expected",
    [
        ("0",   _SHORT_HORIZON),
        ("7",   _SHORT_HORIZON),
        ("29",  _SHORT_HORIZON),
        ("30",  _MAX_HORIZON),
        ("100", _MAX_HORIZON),
    ],
)
def test_numeric_strings_parsed(value, expected):
    """`int("30")` parsea correctamente; el helper acepta strings numéricos.
    Cubre el caso donde el caller leyó `total_days_requested` desde una
    columna VARCHAR sin coerción explícita."""
    assert horizon_for(value) == expected


# ===========================================================================
# 7. Floats (truncan, no redondean)
# ===========================================================================
@pytest.mark.parametrize(
    "value, expected, why",
    [
        (29.9, _SHORT_HORIZON, "29.9 trunca a 29 (<30)"),
        (30.0, _MAX_HORIZON,   "30.0 trunca a 30 (=30, entra en rama amplia)"),
        (30.5, _MAX_HORIZON,   "30.5 trunca a 30 (>=30)"),
        (29.99999, _SHORT_HORIZON, "casi-30 trunca a 29"),
    ],
)
def test_floats_truncate_not_round(value, expected, why):
    """`int(30.99) = 30`, NOT 31. La función honra el truncamiento de Python.
    Documentado para que un futuro refactor que use `round()` en su lugar no
    pase silencioso — los planes de 30.5 días NO existen en producción, pero
    si el caller pasa un float (e.g., promedio de duraciones), la semántica
    debe ser determinística."""
    assert horizon_for(value) == expected, why


# ===========================================================================
# 8. Booleans (Python: `int(True)=1`, `int(False)=0`)
# ===========================================================================
def test_true_treated_as_one_day():
    """`int(True or 0) = int(True) = 1`. Caso patológico improbable pero
    documentado: si el caller pasa `True` (bug en upstream), se trata como
    un plan de 1 día → short."""
    assert horizon_for(True) == _SHORT_HORIZON


# ===========================================================================
# 9. Sanity de la constante
# ===========================================================================
def test_max_horizon_constant_is_at_least_48():
    """`CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS` está clampada a `max(48, env)`
    en constants.py. Si en el futuro alguien baja el knob env por debajo de 48,
    la lógica `min(_MAX_HORIZON, 48)` colapsa: short y max devolverían el mismo
    valor → la función pierde su discriminación. Este invariante asegura
    que la rama amplia siempre sea ≥ que la corta."""
    assert _MAX_HORIZON >= 48, (
        f"CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS={_MAX_HORIZON} < 48. "
        f"La rama amplia colapsa con la corta. Revisar el clamp en constants.py."
    )


def test_function_return_type_is_int():
    """La función retorna `int`, no float ni string. Callers downstream
    pueden hacer aritmética con el valor sin coerción."""
    assert isinstance(horizon_for(30), int)
    assert isinstance(horizon_for(7), int)
    assert isinstance(horizon_for(None), int)


# ===========================================================================
# 10. Determinismo: misma entrada → mismo output
# ===========================================================================
def test_function_is_deterministic():
    """No depende de estado externo (DB, time.time, random). Llamadas
    repetidas con el mismo input devuelven siempre el mismo output —
    requisito para tests por contrato y para el caching upstream del cron."""
    for value in (None, 0, 7, 30, 100):
        results = {horizon_for(value) for _ in range(5)}
        assert len(results) == 1, f"Output no determinístico para {value!r}: {results}"
