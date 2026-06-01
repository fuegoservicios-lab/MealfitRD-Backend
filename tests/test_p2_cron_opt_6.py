"""[P2-CRON-OPT-6 · 2026-05-31] Anclas de la SEXTA pasada de optimización de
cron_tasks.py. Tras 5 pasadas (~93-94/100), dos workflows frescos (49 agentes,
~3.5M tok, cobertura total del monolito de 31k incluyendo process_plan_chunk_queue
partido en 6) + verificación adversaria por hallazgo confirmaron que las clases
grepables (sargabilidad, alert-dedup resolved_at, jsonb_set anidado) están limpias.

De 8 hallazgos verificados sobrevivieron 4 reales+seguros como implement_now/boy-scout:

  G6-1 (hoist)   _validate_merged_days_against_pantry: castear new_chunk_day_range
                 UNA vez antes del loop (era per-día). Byte-idéntico, 0-riesgo.
  G6-2 (hoist)   _filter_days_by_fresh_pantry: precomputar pantry_bases_sub (el
                 filtro invariante `len(pb) > 2` se re-evaluaba por par (base, pb)).
                 Byte-idéntico, 0-riesgo.
  G6-3 (class C) _inject_advanced_learning_signals: fusionar los 2 execute_sql_write
                 a health_profile (tuning_metrics + last_fatigued_ingredients, claves
                 disjuntas) en UN jsonb_set anidado en el path común de fatiga.
                 Ahorra 1 round-trip/chunk. Hermano no-tocado del fix EMA de CRON-OPT-5.
  G6-4 (class A) _background_shift_plan_for_user: el loop catchup_chunks hacía un
                 SELECT por iteración (existe-week) → pre-fetch `_live_weeks` en 1
                 round-trip. Ahorra ~7-9 round-trips Y acorta el hold-time del advisory
                 lock 'general' + FOR UPDATE row lock. Gemelo del SELECT MAX(week_number).

Diferidos con razón (NO implementados): _chunk_worker:24644 redundant SELECT (rompe
mocks 2-branch en 4 archivos de tests), nightly N+1 (cambia semántica budget-guard,
cohorte casi-vacía), hot_table_bloat (cadencia 6h/N=7), 2 dead-projection sin ahorro
de round-trip, refresh_chunk_pantry doble-hp (hot-path P0, 3 firmas críticas).

Tooltip-anchor: P2-CRON-OPT-6.
"""

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"
_APP_PY = _BACKEND_ROOT / "app.py"


def _src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _slice_fn(src: str, header: str) -> str:
    """Cuerpo de la función `header` hasta el siguiente def top-level."""
    start = src.find(header)
    assert start >= 0, f"no se encontró {header!r}"
    after = src[start + len(header):]
    nxt = re.search(r"\n(?:def |async def )\w", after)
    return after[: nxt.start()] if nxt else after


def _code_only(src: str) -> str:
    """Quita comentarios Python `#` y SQL `--` hasta fin de línea, para que las
    aserciones NEGATIVAS no matcheen el comentario explicativo del propio fix —
    lección repetida del repo. Ninguna ancla POSITIVA de este archivo lleva `#`/`--`."""
    out = []
    for line in src.splitlines():
        cut = len(line)
        for tok in ("#", "--"):
            h = line.find(tok)
            if 0 <= h < cut:
                cut = h
        out.append(line[:cut])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Cross-link: el marker pertenece a la familia P2-CRON-OPT y no retrocede.
# ---------------------------------------------------------------------------
def test_marker_is_p2_cron_opt_family():
    app = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]*)"', app)
    assert m, "No se encontró _LAST_KNOWN_PFIX."
    marker = m.group(1)
    d = re.search(r"(\d{4})-(\d{2})-(\d{2})", marker)
    assert d, "El marker no tiene fecha parseable."
    assert (int(d.group(1)), int(d.group(2)), int(d.group(3))) >= (2026, 5, 31), (
        "El marker _LAST_KNOWN_PFIX retrocedió antes de 2026-05-31 (P2-CRON-OPT-6)."
    )


# ---------------------------------------------------------------------------
# G6-1: _validate_merged_days_against_pantry castea el rango UNA vez ANTES del loop.
# ---------------------------------------------------------------------------
def test_g6_1_validate_merged_range_cast_hoisted():
    body = _slice_fn(_src(), "def _validate_merged_days_against_pantry(")
    # Positivo: sentinel del hoist (solo existe en la versión hoisteada).
    assert "_range_start = _range_end = None" in body, (
        "G6-1: debe inicializarse `_range_start = _range_end = None` (sentinel del hoist)."
    )
    assert "if _range_start is not None and not (_range_start <= day_num <= _range_end):" in body, (
        "G6-1: el gate del loop debe usar el rango ya casteado (`_range_start is not None`)."
    )
    # Ordering: el cast del rango ocurre ANTES del `for d in merged_days` (hoisteado),
    # no dentro del loop. Un revert lo movería después y este assert fallaría.
    cast_pos = body.find("_range_start, _range_end = int(new_chunk_day_range[0])")
    loop_pos = body.find("for d in merged_days or []:")
    assert cast_pos >= 0, "G6-1: no se encontró el cast hoisteado de new_chunk_day_range."
    assert loop_pos >= 0, "G6-1: no se encontró el loop `for d in merged_days`."
    assert cast_pos < loop_pos, (
        "G6-1: el cast `int(new_chunk_day_range[...])` debe estar ANTES del loop "
        "(hoisteado), no re-ejecutarse por cada día."
    )
    # El cast aparece exactamente una vez (no duplicado por-iteración).
    assert body.count("int(new_chunk_day_range[0])") == 1, (
        "G6-1: el cast de new_chunk_day_range[0] debe aparecer exactamente una vez (hoisteado)."
    )


# ---------------------------------------------------------------------------
# G6-2: _filter_days_by_fresh_pantry precomputa pantry_bases_sub fuera del loop.
# ---------------------------------------------------------------------------
def test_g6_2_filter_days_pantry_bases_sub_precomputed():
    body = _slice_fn(_src(), "def _filter_days_by_fresh_pantry(")
    code = _code_only(body)
    assert "pantry_bases_sub = [pb for pb in pantry_bases if pb and len(pb) > 2]" in code, (
        "G6-2: debe precomputarse `pantry_bases_sub` (subconjunto invariante len>2) "
        "una vez antes del loop."
    )
    assert "any(pb in base or base in pb for pb in pantry_bases_sub)" in code, (
        "G6-2: el inner del matching debe iterar `pantry_bases_sub` (ya filtrado), sin "
        "re-evaluar `len(pb) > 2` por par."
    )
    # Negativo: el patrón viejo (filtro invariante en el any anidado) ya no existe.
    assert "len(pb) > 2 and (pb in base" not in code, (
        "G6-2: el filtro `len(pb) > 2 and (pb in base or ...)` re-evaluado en el loop "
        "anidado debe haberse hoisteado a pantry_bases_sub."
    )


# ---------------------------------------------------------------------------
# G6-3: _inject_advanced_learning_signals fusiona tuning_metrics + last_fatigued
#        en UN jsonb_set anidado (path común de fatiga); NO toca el EMA tuning write.
# ---------------------------------------------------------------------------
def _fatigue_tune_block(body: str) -> str:
    """Sub-slice del bloque FATIGUE-TUNE (desde `_pending_tuning = None` hasta MEJORA 2),
    para no colisionar con el write standalone de tuning_metrics del bloque EMA dual."""
    start = body.find("_pending_tuning = None")
    assert start >= 0, "no se encontró el inicio del bloque FATIGUE-TUNE (_pending_tuning)."
    end = body.find("MEJORA 2", start)
    assert end >= 0, "no se encontró el límite del bloque (MEJORA 2)."
    return body[start:end]


def test_g6_3_fatigue_tune_jsonb_set_fused():
    body = _slice_fn(_src(), "def _inject_advanced_learning_signals(")
    block = _fatigue_tune_block(body)
    # Positivo: el write fusionado anidado (inner=tuning_metrics, outer=last_fatigued).
    assert (
        "jsonb_set(jsonb_set(health_profile, '{tuning_metrics}', %s::jsonb), "
        "'{last_fatigued_ingredients}', %s::jsonb)" in block
    ), (
        "G6-3: las 2 escrituras a health_profile (tuning_metrics + last_fatigued_ingredients) "
        "deben fusionarse en un jsonb_set anidado en el path común de fatiga."
    )
    # Positivo: el mecanismo de diferir (deferral) está presente.
    assert "_pending_tuning = tuning_metrics" in block, (
        "G6-3: la rama de fatiga debe diferir el tuning a `_pending_tuning` en vez de "
        "emitir su propio write."
    )
    assert "if _pending_tuning is not None:" in block, (
        "G6-3: debe gatearse el write fusionado vs el simple según `_pending_tuning`."
    )
    # Negativo (en _code_only): dentro del bloque FATIGUE-TUNE ya NO existe el write
    # standalone de tuning_metrics (se movió al jsonb_set anidado).
    code_block = _code_only(block)
    assert (
        "jsonb_set(health_profile, '{tuning_metrics}', %s::jsonb) WHERE id = %s"
        not in code_block
    ), (
        "G6-3: el write standalone de tuning_metrics del bloque FATIGUE-TUNE debe haberse "
        "fusionado (el del bloque EMA dual, fuera de este slice, se conserva intacto)."
    )
    # Back-compat: el write simple de last_fatigued sigue existiendo (rama else, sin tuning).
    assert (
        "jsonb_set(health_profile, '{last_fatigued_ingredients}', %s::jsonb) WHERE id = %s"
        in block
    ), (
        "G6-3: el write simple de last_fatigued_ingredients debe conservarse para la rama "
        "sin tuning pendiente (back-compat)."
    )


# ---------------------------------------------------------------------------
# G6-4: _background_shift_plan_for_user pre-fetchea _live_weeks (N+1 → 1 round-trip).
# ---------------------------------------------------------------------------
def test_g6_4_background_shift_live_weeks_prefetch():
    body = _slice_fn(_src(), "def _background_shift_plan_for_user(")
    code = _code_only(body)
    # Positivo: el pre-fetch del set de weeks vivos (1 sola query, mismo filtro 4-estados).
    assert "SELECT DISTINCT week_number FROM plan_chunk_queue " in code, (
        "G6-4: debe pre-fetchearse el set de week_numbers vivos en 1 round-trip."
    )
    assert "_live_weeks = {" in code, (
        "G6-4: debe materializarse `_live_weeks` (set en memoria)."
    )
    assert "if next_week in _live_weeks:" in code, (
        "G6-4: el check del loop catchup debe ser membership en memoria (`next_week in "
        "_live_weeks`), no un SELECT por iteración."
    )
    # Negativo: el SELECT por-iteración del loop catchup ya no existe (era el N+1).
    assert "SELECT id FROM plan_chunk_queue " not in code, (
        "G6-4: el `SELECT id FROM plan_chunk_queue WHERE ... week_number = %s LIMIT 1` "
        "por iteración del loop catchup debe haberse reemplazado por el pre-fetch."
    )
    # El filtro de 4 estados del pre-fetch espeja el check por-iteración previo.
    assert "AND status IN ('pending','processing','stale','failed')" in code, (
        "G6-4: el pre-fetch debe usar el MISMO filtro de 4 estados que el check previo."
    )
