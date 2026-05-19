"""[P3-2 · 2026-05-08] Tests del mirror `_user_forced_simplified_weeks` en
plan_data tras regenerate-simplified.

Bug original (audit 2026-05-07):
  El endpoint `/api/plans/{plan_id}/chunks/{chunk_id}/regenerate-simplified`
  setea `_user_forced_simplified=True` en `plan_chunk_queue.pipeline_snapshot`
  (línea ~3828) pero el frontend lee `meal_plans.plan_data` directamente vía
  Supabase (AssessmentContext.jsx:471). El flag nunca llegaba al UI — el
  toggle era write-only desde la perspectiva UX.

Fix:
  1. Mirror del flag a `plan_data._user_forced_simplified_weeks: {week_number_str: iso_ts}`
     en la misma transacción que limpia `_user_action_required` y setea
     `generation_status='partial'`.
  2. Tres `jsonb_set` chained (no comma-separated) para que las 3 mutaciones
     sobrevivan — un side-effect del refactor: el patrón legacy con
     `SET plan_data = jsonb_set(...), plan_data = jsonb_set(...)` colapsaba
     en PostgreSQL (la segunda asignación pisa la primera porque ambas leen
     OLD plan_data en paralelo). Las jsonb_set anidadas chainan en orden.
  3. El frontend ya lee `plan_data` via Supabase direct → el flag aparece
     automáticamente sin endpoint nuevo.

Cobertura:
  - SQL contiene path `_user_forced_simplified_weeks`.
  - SQL chainea jsonb_set (anidado), no comma-separated (que pisaría).
  - Path es ARRAY['_user_forced_simplified_weeks', week_number] — dict
    keyed por week para granularidad per-chunk.
  - Las 3 mutaciones (`_user_action_required=null`,
    `generation_status='partial'`, mirror del flag) coexisten.
  - Backwards-compat: el SQL legacy con dos asignaciones a plan_data NO
    debe re-aparecer (regresión silente).
"""
import pathlib
import re

import pytest


_PLANS_PATH = pathlib.Path(__file__).parent.parent / "routers" / "plans.py"


@pytest.fixture(scope="module")
def plans_source() -> str:
    return _PLANS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def regenerate_block(plans_source) -> str:
    """Aísla el bloque del endpoint regenerate-simplified."""
    # Buscamos por el SET _user_forced_simplified que es el marker único
    # del endpoint en cuestión.
    start = plans_source.find('snap["_user_forced_simplified"] = True')
    assert start != -1, "Marker del endpoint regenerate-simplified no encontrado"
    end = plans_source.find("\n@router.", start)
    if end == -1:
        end = start + 5000
    return plans_source[start:end]


# ---------------------------------------------------------------------------
# 1. Mirror se persiste a plan_data
# ---------------------------------------------------------------------------
def test_mirror_path_present_in_sql(regenerate_block):
    """El SQL del UPDATE meal_plans debe escribir a la ruta
    `_user_forced_simplified_weeks` (dict keyed por week_number)."""
    assert "_user_forced_simplified_weeks" in regenerate_block, (
        "El mirror del flag debe escribirse al path "
        "`_user_forced_simplified_weeks` en plan_data. Sin esto, el frontend "
        "(que lee plan_data via Supabase direct) no tiene cómo saber que "
        "el chunk fue regenerado en modo simplificado."
    )


def test_mirror_path_uses_array_for_dynamic_week_key(regenerate_block):
    """Para que el path sea `_user_forced_simplified_weeks.{week_number}`,
    debe usarse ARRAY['_user_forced_simplified_weeks', %s] como path en
    jsonb_set — no un literal '{...}' (que requeriría escape del int)."""
    assert re.search(
        r"ARRAY\[\s*['\"]_user_forced_simplified_weeks['\"]\s*,\s*%s\s*\]",
        regenerate_block,
    ), (
        "Path del jsonb_set debe usar ARRAY['_user_forced_simplified_weeks', %s] "
        "para parametrizar el week_number. Sin parametrización, sustituir el "
        "valor a mano abre vector de SQL injection si week_number proviene de "
        "fuente no-confiable."
    )


def test_week_number_passed_as_string(regenerate_block):
    """JSONB keys son strings; week_number int debe convertirse a str
    antes de pasarse al jsonb_set."""
    assert re.search(
        r"_wn_str\s*=\s*str\(\s*int\(\s*chunk_row\[['\"]week_number['\"]\]\s*\)\s*\)",
        regenerate_block,
    ), (
        "week_number debe convertirse a str (`_wn_str = str(int(...))`) "
        "antes de pasarlo como path key del jsonb_set. Sin int() previo, "
        "un week_number malformado del DB rompe la conversión."
    )


def test_iso_timestamp_value(regenerate_block):
    """El valor del mirror es un ISO timestamp UTC para que el frontend
    pueda mostrar 'simplificado el [fecha]' si lo desea."""
    # _ts_iso = datetime.now(timezone.utc).isoformat()
    assert re.search(
        r"_ts_iso\s*=\s*datetime\.now\(\s*timezone\.utc\s*\)\.isoformat\(\)",
        regenerate_block,
    ), (
        "Timestamp del mirror debe ser ISO UTC para correlación con logs/telemetry."
    )


# ---------------------------------------------------------------------------
# 2. Las 3 mutaciones coexisten (chained jsonb_set)
# ---------------------------------------------------------------------------
def test_three_mutations_chained_not_comma_separated(regenerate_block):
    """En PostgreSQL UPDATE, dos `plan_data = jsonb_set(...)` separadas por
    coma se evalúan en paralelo (ambas leen OLD plan_data) — la segunda
    pisa la primera. P3-2 chainea las 3 mutaciones anidadas para que
    todas sobrevivan en una sola asignación a plan_data."""
    # Buscar la sección del UPDATE meal_plans relevante.
    update_match = re.search(
        r"UPDATE meal_plans.*?WHERE id = %s",
        regenerate_block,
        re.DOTALL,
    )
    assert update_match is not None, "Bloque UPDATE meal_plans no encontrado"
    update_sql = update_match.group(0)
    # Patrón anti-regresión: NO debe haber 2+ asignaciones independientes a plan_data.
    plan_data_assignments = len(re.findall(r"plan_data\s*=\s*jsonb_set\(", update_sql))
    # Esperamos exactamente 1 asignación (con jsonb_set anidado).
    assert plan_data_assignments == 1, (
        f"Encontradas {plan_data_assignments} asignaciones a `plan_data = jsonb_set(...)`. "
        f"P3-2 requiere UNA sola asignación con jsonb_set anidado. Múltiples "
        f"asignaciones en UPDATE causan que la segunda pise la primera "
        f"(ambas leen OLD plan_data en paralelo, no chainan)."
    )


def test_user_action_required_set_to_null(regenerate_block):
    """La mutación de _user_action_required=null debe sobrevivir."""
    assert "'{_user_action_required}'" in regenerate_block
    assert "'null'::jsonb" in regenerate_block


def test_generation_status_set_to_partial(regenerate_block):
    """La mutación de generation_status='partial' debe sobrevivir."""
    assert "'{generation_status}'" in regenerate_block
    assert '\'"partial"\'::jsonb' in regenerate_block


# ---------------------------------------------------------------------------
# 3. Smoke: el flag interno (snap) sigue intacto
# ---------------------------------------------------------------------------
def test_pipeline_snapshot_flag_unchanged(regenerate_block):
    """El flag interno en pipeline_snapshot sigue siendo seteado — P3-2
    NO reemplaza ese write, solo añade el mirror."""
    assert 'snap["_user_forced_simplified"] = True' in regenerate_block
    assert 'snap["_user_forced_simplified_at"]' in regenerate_block


# ---------------------------------------------------------------------------
# 4. Cap del tamaño del dict (no relevant aún — el endpoint solo añade
#    una entry por invocación, naturalmente cap = total chunks del plan)
# ---------------------------------------------------------------------------
def test_no_legacy_double_assignment_pattern(plans_source):
    """Smoke global: el patrón legacy de doble-asignación a plan_data en
    el mismo UPDATE NO debe re-aparecer en otro endpoint del módulo. Si
    alguien copy-pastea el bloque viejo, este test pita."""
    # Buscar SET plan_data = jsonb_set(...), \n ... plan_data = jsonb_set(...)
    # en cualquier bloque del archivo.
    pattern = re.compile(
        r"SET plan_data = jsonb_set\([^;]*?,\s*plan_data\s*=\s*jsonb_set\(",
        re.DOTALL,
    )
    matches = pattern.findall(plans_source)
    assert not matches, (
        f"Encontradas {len(matches)} ocurrencias del patrón legacy "
        f"`SET plan_data = jsonb_set(...), plan_data = jsonb_set(...)`. "
        f"Causa colapso silencioso (la 2ª asignación pisa la 1ª en PostgreSQL). "
        f"Usar jsonb_set anidado en una sola asignación."
    )
