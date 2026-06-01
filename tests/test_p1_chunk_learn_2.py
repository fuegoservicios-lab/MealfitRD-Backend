"""[P1-CHUNK-LEARN-2 · 2026-05-29] Lock-the-contract de las 11 gaps cerradas en la
RE-auditoría profunda del sistema de chunks de aprendizaje continuo (sigue a
P1-CHUNK-LEARN-AUDIT · 2026-05-29, que cerró 19 gaps).

Estas 11 son DISTINTAS de las 19 previas — varias son hermanos perdidos de las
mismas clases (4º pause helper sin CAS, 3ª clave huérfana de quality, resolver E2
parcial, dead-writes de variety/creative-freedom). Tema sistémico inalterado:
**drift productor↔consumidor / dead-writes** + **CAS faltante en hermanos** + ownership
de crons de recovery.

Gaps cubiertas (id · descripción · archivo):
  G1  recovery-deadlock-missing-prior-lessons   (cron _recover_pantry_paused_chunks skip-guard)   [P1]
  G2  cold-start-pruning-dead-key               (cron signals_to_check → _cold_start_recommendations)
  G3  quality-history-rotations-orphan-key      (cron _persist_nightly NO escribe _rotations)
  G4  t2-success-commit-no-cas                  (cron T2 commit con AND attempts/status + _T2DisplacedError)
  G5  partial-c1-pause-helpers-no-cas           (stale_inventory + inline missing_prior_lessons via CAS)
  G6  dead-letter-pause-failed-no-notify        (_notify_chunk_pause_failed en ambos handlers)
  G7  synth-source-visibility-dead              (cron cuenta synth sobre lista pre-filtro)
  G8  frontend-lm-catalog-honesty               (History.jsx drop dead rows + Dashboard render hint)
  G9  degraded-rate-resolver-stuck-red          (cron resolver hoisteado + cron standalone)
  G10 force-technique-variety-dead-write        (graph consume _force_technique_variety)
  G11 creative-freedom-dead-write               (graph consume _creative_freedom)

Parser-based: corre bajo `py -3 --noconftest` sin langgraph/DB (no gasta Gemini).
Tooltip-anchor: P1-CHUNK-LEARN-2.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_CRON = _BACKEND / "cron_tasks.py"
_ORCH = _BACKEND / "graph_orchestrator.py"
_APP = _BACKEND / "app.py"
_HISTORY = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_DASHBOARD = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _ORCH.read_text(encoding="utf-8")


def _slice_fn(src: str, def_signature: str, max_lines: int = 6000) -> str:
    """Cuerpo de una función top-level desde `def_signature` hasta el siguiente
    `def ` al mismo nivel de indentación (o EOF)."""
    start = src.find(def_signature)
    assert start != -1, f"No encuentro `{def_signature}`"
    line_start = src.rfind("\n", 0, start) + 1
    indent = start - line_start
    rest = src[start + len(def_signature):]
    lines = rest.splitlines(keepends=True)
    body = [def_signature]
    for ln in lines:
        stripped = ln.lstrip()
        cur_indent = len(ln) - len(stripped)
        if stripped.startswith(("def ", "async def ")) and cur_indent <= indent:
            break
        body.append(ln)
        if len(body) > max_lines:
            break
    return "".join(body)


def _slice_between(src: str, start_marker: str, end_marker: str) -> str:
    start = src.find(start_marker)
    assert start != -1, f"No encuentro start `{start_marker}`"
    end = src.find(end_marker, start)
    assert end != -1, f"No encuentro end `{end_marker}` tras `{start_marker}`"
    return src[start:end]


# ===========================================================================
# G1 — recovery dead-lock: skip-guard de ownership para missing_prior_lessons
# ===========================================================================
def test_g1_recovery_skips_missing_prior_lessons(cron_src):
    body = _slice_fn(cron_src, "def _recover_pantry_paused_chunks() -> None:")
    # El skip-guard debe existir con continue.
    assert 'pause_reason == "missing_prior_lessons"' in body, (
        "G1: falta el skip-guard de missing_prior_lessons en _recover_pantry_paused_chunks."
    )
    assert "G1-RECOVERY-DEADLOCK" in body, "G1: falta el tooltip-anchor."
    # El skip debe ocurrir ANTES de cualquier activación de flexible_mode (el tail genérico).
    skip_idx = body.find('pause_reason == "missing_prior_lessons"')
    flex_idx = body.find("_activate_flexible_mode")
    assert skip_idx != -1
    if flex_idx != -1:
        assert skip_idx < flex_idx, (
            "G1: el skip-guard debe estar ANTES del tail de flexible_mode (si no, el "
            "reset de updated_at cada 12h mata la escalación de 24h)."
        )
    # NO debe saltar synthesis_ratio_exceeded (ese reason no tiene cron de re-enqueue alterno).
    assert 'pause_reason == "synthesis_ratio_exceeded"' not in body, (
        "G1: NO saltar synthesis_ratio_exceeded — crearía un nuevo limbo permanente."
    )


# ===========================================================================
# G2 — cold-start pruning leía la clave muerta _cold_start_recs
# ===========================================================================
def test_g2_cold_start_pruning_uses_live_key(cron_src):
    assert '"cold_start": "_cold_start_recommendations"' in cron_src, (
        "G2: signals_to_check debe mapear cold_start → _cold_start_recommendations (clave viva)."
    )
    assert '"_cold_start_recs"' not in cron_src, (
        "G2: la clave MUERTA _cold_start_recs no debe aparecer en cron_tasks.py."
    )


def test_g2_anti_drift_signals_have_writers(cron_src, orch_src):
    """Anti-drift generalizado: cada value del map signals_to_check DEBE tener un
    writer `pipeline_data['<value>'] =` en cron_tasks o graph_orchestrator. Cierra la
    clase entera (no sólo cold_start) — la lección dominante de este subsistema."""
    block = _slice_between(cron_src, "signals_to_check = {", "}")
    values = re.findall(r':\s*"(_[A-Za-z0-9_]+)"', block)
    assert "_cold_start_recommendations" in values, "G2: el map debe incluir la clave viva."
    combined = cron_src + orch_src
    for v in values:
        writer = re.search(
            r"pipeline_data\[['\"]" + re.escape(v) + r"['\"]\]\s*=", combined
        )
        assert writer, (
            f"G2/anti-drift: signals_to_check lee {v!r} pero NINGÚN writer "
            f"`pipeline_data['{v}'] =` existe (dead-key drift)."
        )


# ===========================================================================
# G3 — quality trend escrita a la clave huérfana quality_history_rotations
# ===========================================================================
def test_g3_no_orphan_quality_history_rotations_write(cron_src):
    # NO debe existir una asignación a fresh_profile['quality_history_rotations'].
    assert not re.search(
        r"fresh_profile\[['\"]quality_history_rotations['\"]\]\s*=", cron_src
    ), "G3: dead-write a quality_history_rotations debe estar eliminado."
    # last_plan_quality (sí consumido) se conserva.
    assert re.search(r"fresh_profile\[['\"]last_plan_quality['\"]\]\s*=", cron_src), (
        "G3: last_plan_quality (consumido en el light-path gate) debe conservarse."
    )
    assert "G3-QUALITY-HISTORY-KEYDRIFT" in cron_src, "G3: falta el tooltip-anchor."


# ===========================================================================
# G4 — T2 success-commit sin CAS (clobber por worker desplazado)
# ===========================================================================
def test_g4_t2_success_commit_has_cas(cron_src):
    assert "class _T2DisplacedError" in cron_src, "G4: falta el sentinel _T2DisplacedError."
    # El commit T2 es el único UPDATE con learning_persisted_at; debe llevar el guard de
    # ownership (AND attempts / AND status) en las líneas que siguen al WHERE id.
    m = re.search(
        r"SET status = 'completed',(?:.|\n){0,400}?learning_persisted_at = NOW\(\),"
        r"(?:.|\n){0,300}?AND status = 'processing'",
        cron_src,
    )
    assert m, (
        "G4: el commit T2 status='completed' (con learning_persisted_at) debe filtrar "
        "AND attempts = %s AND status = 'processing'."
    )
    commit = m.group(0)
    assert "AND attempts = %s" in commit, "G4: el commit T2 debe filtrar AND attempts = %s."
    # El handler distingue displaced de error real de DB.
    assert "except _T2DisplacedError" in cron_src, "G4: falta el handler de displacement."
    assert "if cursor.rowcount == 0:" in cron_src, "G4: falta el check rowcount==0."


# ===========================================================================
# G5 — 4º pause helper + pausa inline sin CAS (resto de C1)
# ===========================================================================
def test_g5_stale_inventory_pause_uses_cas(cron_src):
    body = _slice_fn(cron_src, "def _pause_chunk_for_stale_inventory(")
    assert "_cas_pause_chunk_to_pending_user_action(" in body, (
        "G5: _pause_chunk_for_stale_inventory debe enrutar por el helper CAS."
    )
    # Ya NO debe hacer un UPDATE bare directo a plan_chunk_queue.
    assert "UPDATE plan_chunk_queue" not in body, (
        "G5: _pause_chunk_for_stale_inventory ya no debe hacer UPDATE bare (delega en el CAS helper)."
    )


def test_g5_inline_missing_lessons_pause_uses_cas(cron_src):
    # La región de la pausa inline missing_prior_lessons debe usar el CAS helper.
    idx = cron_src.find('_p11_pause["_pause_reason"] = "missing_prior_lessons"')
    assert idx != -1, "G5: no encuentro la pausa inline missing_prior_lessons."
    region = cron_src[idx: idx + 1200]
    assert "_cas_pause_chunk_to_pending_user_action(" in region, (
        "G5: la pausa inline missing_prior_lessons debe enrutar por el helper CAS."
    )


# ===========================================================================
# G6 — dead-letter de pause-failed sin banner/push
# ===========================================================================
def test_g6_notify_helper_writes_banner_and_push(cron_src):
    body = _slice_fn(cron_src, "def _notify_chunk_pause_failed(")
    assert "_user_action_required" in body, "G6: el helper debe escribir _user_action_required."
    assert "_dispatch_push_notification(" in body, "G6: el helper debe despachar push."


def test_g6_both_pause_failed_handlers_notify(cron_src):
    # 1 def + >=2 callsites → la cadena de llamada aparece >=2 veces como invocación.
    callsites = len(re.findall(r"_notify_chunk_pause_failed\(\s*\n?\s*user_id=", cron_src))
    assert callsites >= 2, (
        f"G6: se esperaban >=2 callsites de _notify_chunk_pause_failed (guard_pause_failed + "
        f"shuffle_empty_day_pause_failed), encontrados {callsites}."
    )
    # El branch de shuffle ahora también registra telemetría de la doble-falla.
    assert 'error_message="shuffle_empty_day_pause_failed"' in cron_src, (
        "G6: el branch shuffle debe registrar _record_chunk_metric de la doble-falla."
    )


# ===========================================================================
# G7 — visibilidad de fuentes sintetizadas muerta (pre-filtro metrics_unavailable)
# ===========================================================================
def test_g7_synth_count_over_prefilter_list(cron_src):
    assert "_all_lessons_prefilter = list(_all_lessons)" in cron_src, (
        "G7: debe capturarse la lista pre-filtro de lecciones."
    )
    # El re-conteo de synth debe operar sobre la lista pre-filtro.
    m = re.search(
        r"_agg_synth_source_count \+=\s*sum\((.|\n){0,300}?_all_lessons_prefilter",
        cron_src,
    )
    assert m, (
        "G7: _agg_synth_source_count debe re-sumarse desde _all_lessons_prefilter "
        "(las synth llevan metrics_unavailable=True y el filtro las eliminaba antes del agregador)."
    )


# ===========================================================================
# G8 — superficies de learning del frontend deshonestas
# ===========================================================================
def test_g8_lm_catalog_drops_producerless_keys():
    history = _HISTORY.read_text(encoding="utf-8")
    catalog = _slice_between(history, "const _LM_DISPLAY_GROUPS = [", "];")
    for dead in ("synth_quality_score", "synthesized_count", "queue_count"):
        assert f"'{dead}'" not in catalog, (
            f"G8: {dead} no tiene productor en learning_metrics → no debe estar en _LM_DISPLAY_GROUPS."
        )
    # Las que SÍ tienen productor permanecen.
    assert "'recovery_attempts'" in catalog and "'escalation_reason'" in catalog, (
        "G8: recovery_attempts/escalation_reason (con productor) deben permanecer en el catálogo."
    )


def test_g8_dashboard_no_longer_renders_learning_hint():
    # [silent-bg · 2026-05-29] Decisión de producto: la píldora "Analizando tus
    # preferencias…" (G8) se removió del Dashboard. La generación en background es
    # silenciosa y el texto genérico confundía (sonaba a que tocaba el menú visible).
    # El backend sigue exponiendo last_learning_hint en /chunk-status (campo inerte,
    # SELECT por PK trivial), pero el frontend NO debe volver a renderizarlo sin copy
    # claro + estilos de modo oscuro reales. Invierte el assert original de G8.
    dash = _DASHBOARD.read_text(encoding="utf-8")
    assert "last_learning_hint" not in dash, (
        "El Dashboard no debe renderizar last_learning_hint (removido por decisión de "
        "producto; ver comentario [silent-bg] en Dashboard.jsx)."
    )


# ===========================================================================
# G9 — degraded_rate_high:* puede quedar rojo para siempre (E2 parcial)
# ===========================================================================
def test_g9_resolver_hoisted_out_of_loop(cron_src):
    body = _slice_fn(cron_src, "def _alert_if_degraded_rate_high():")
    assert "_fired_tiers" in body, "G9: debe rastrearse _fired_tiers para resolver fuera del loop."
    # El resolve por-tier debe estar tras el loop `for row in rows:`.
    loop_idx = body.find("for row in rows:")
    post_idx = body.find("for _tipo in ('initial', 'refill'):")
    assert loop_idx != -1 and post_idx != -1, "G9: falta el resolve hoisteado por-tier."
    assert post_idx > loop_idx, "G9: el resolve por-tier debe estar DESPUÉS del per-row loop."
    # Conserva el anchor histórico E2.
    assert "E2-DEGRADED-RESOLVE" in body, "G9: debe conservar el anchor E2-DEGRADED-RESOLVE."


def test_g9_degraded_alert_registered_as_cron(cron_src):
    reg = _slice_fn(cron_src, "def register_plan_chunk_scheduler(scheduler) -> None:")
    assert 'scheduler.get_job("alert_if_degraded_rate_high")' in reg, (
        "G9: _alert_if_degraded_rate_high debe registrarse como cron standalone (cubre el "
        "dead-end 'sin tráfico de éxito')."
    )
    assert "_alert_if_degraded_rate_high," in reg, "G9: falta el _add_job_jittered del resolver."


# ===========================================================================
# G10 / G11 — flags de learning que eran dead-writes ahora se consumen
# ===========================================================================
def test_g10_force_technique_variety_consumed(orch_src):
    assert 'if form_data.get("_force_technique_variety"):' in orch_src, (
        "G10: el prompt builder debe consumir _force_technique_variety (era dead-write)."
    )
    assert "G10-FORCE-TECHNIQUE-VARIETY" in orch_src, "G10: falta el tooltip-anchor."


def test_g11_creative_freedom_consumed(orch_src):
    assert 'if form_data.get("_creative_freedom"):' in orch_src, (
        "G11: el prompt builder debe consumir _creative_freedom (era dead-write)."
    )
    assert "G11-CREATIVE-FREEDOM" in orch_src, "G11: falta el tooltip-anchor."


# ===========================================================================
# Marker — _LAST_KNOWN_PFIX bumpeado
# ===========================================================================
def test_marker_bumped():
    """[Relajado NG-CRON-OPT-2 · 2026-05-30] La aserción de FAMILIA `P1-CHUNK-LEARN`
    se relajó a un FLOOR DE FECHA: el marker ya migró a sucesores fuera de familia
    (P2-CRON-OPT, NG-CRON-OPT-2, …) y la aserción de familia rompía cada bump
    posterior legítimo. La freshness real la enforza `test_p3_1_last_known_pfix_freshness`;
    aquí solo exigimos que el marker no sea anterior a esta sesión (no-stale)."""
    import re as _re
    from datetime import date as _date
    app_src = _APP.read_text(encoding="utf-8")
    m = _re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*·\s*(\d{4})-(\d{2})-(\d{2})"', app_src)
    assert m, 'No se encontró marker `_LAST_KNOWN_PFIX = "... · YYYY-MM-DD"` válido.'
    marker_date = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    assert marker_date >= _date(2026, 5, 29), (
        f"Marker stale: {marker_date} < 2026-05-29 (fecha de P1-CHUNK-LEARN-2)."
    )
