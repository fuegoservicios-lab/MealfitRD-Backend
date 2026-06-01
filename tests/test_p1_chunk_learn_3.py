"""[P1-CHUNK-LEARN-3 · 2026-05-29] Lock-the-contract de los gaps cerrados en la
TERCERA auditoría profunda del sistema de chunks de aprendizaje continuo (sigue a
P1-CHUNK-LEARN-AUDIT (19) y P1-CHUNK-LEARN-2 (11), ambas 2026-05-29).

Combina hallazgo propio (analizador determinístico de flujo de claves) + workflow
multi-agente con verificación adversaria. 13 gaps implementados (1 P1 · 5 P2 · 7 P3);
GAP-11 aceptado-no-fix (costo negligible + test-coupled, documentado en el resumen).

Gaps cubiertos (id · descripción · severidad):
  G1  pantry-drift-warning-dead       wire P1-D _pantry_drift_warning → prompt        [P2]
  G2  failed-window-expired-limbo     escalation pass para failed huérfanos           [P1]
  G3  shopping-fail-reenqueue-no-cas  CAS en el re-enqueue de shopping-list-failed     [P2]
  G4  pantry-post-merge-pause-no-cas  CAS en la pausa pantry_violation_post_merge      [P2]
  G5  gate-pause-cluster-no-cas       6 gate-pauses ruteados por el helper CAS         [P2]
  G6  synth-ratio-high-no-resolver    resolver de chunk_lesson_synth_ratio_high        [P2]
  G7  chunk-lag-excessive-no-resolver resolver de chunk_lag_excessive                  [P3]
  G8  i2-cron-meal-plans-writes       AND user_id=%s en 5 writes de meal_plans         [P3]
  G9  blocked-techniques-dead-write   consumir _blocked_techniques en el selector      [P3]
  G10 quality-degraded-reason-unread  surface _quality_degraded_reason/_severity (UI)  [P3]
  G12 skip-merge-write-dead           eliminar dead flag _skip_merge_write             [P3]
  G13 is-emergency-generation-dead    eliminar dead flag _is_emergency_generation      [P3]
  G14 lm-catalog-stale-synth-comments corregir comentarios stale (G8 doc-drift)        [P3]

Parser-based: corre bajo `py -3 --noconftest` sin langgraph/DB (no gasta Gemini).
Tooltip-anchor: P1-CHUNK-LEARN-3.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_CRON = _BACKEND / "cron_tasks.py"
_ORCH = _BACKEND / "graph_orchestrator.py"
_PLANS = _BACKEND / "routers" / "plans.py"
_PLAN_GEN = _BACKEND / "prompts" / "plan_generator.py"
_APP = _BACKEND / "app.py"
_ALERT_DOC = _BACKEND / "docs" / "system_alerts_resolution_table.md"
_HISTORY = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_DASHBOARD = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _ORCH.read_text(encoding="utf-8")


def _slice_fn(src: str, def_signature: str, max_lines: int = 8000) -> str:
    """Cuerpo de una función top-level desde `def_signature` hasta el siguiente
    `def`/`async def` al mismo nivel de indentación (o EOF)."""
    start = src.find(def_signature)
    assert start != -1, f"No encuentro `{def_signature}`"
    line_start = src.rfind("\n", 0, start) + 1
    indent = start - line_start
    rest = src[start + len(def_signature):]
    body = [def_signature]
    for ln in rest.splitlines(keepends=True):
        stripped = ln.lstrip()
        cur_indent = len(ln) - len(stripped)
        if stripped.startswith(("def ", "async def ")) and cur_indent <= indent:
            break
        body.append(ln)
        if len(body) > max_lines:
            break
    return "".join(body)


# ===========================================================================
# G1 (P2) — _pantry_drift_warning (feature P1-D) cableado al prompt
# ===========================================================================
def test_g1_build_pantry_drift_context_exists():
    src = _PLAN_GEN.read_text(encoding="utf-8")
    assert "def build_pantry_drift_context(" in src, (
        "G1: falta el builder build_pantry_drift_context en plan_generator.py"
    )
    assert "critical_drops" in src and "notable_increases" in src and "new_items" in src, (
        "G1: el builder debe consumir las 3 categorías del dict _pantry_drift_warning"
    )


def test_g1_drift_context_wired_into_shared_context(orch_src):
    body = _slice_fn(orch_src, "def _build_shared_context(")
    assert 'build_pantry_drift_context(form_data.get("_pantry_drift_warning"))' in body, (
        "G1: _build_shared_context debe LEER _pantry_drift_warning y pasarlo al builder "
        "(pre-fix era dead-write: ningún consumer lo leía)."
    )
    assert '"pantry_drift_context"' in body, "G1: falta la key pantry_drift_context en el dict de contexto"


def test_g1_drift_context_interpolated_into_prompts(orch_src):
    # Debe aparecer en al menos 2 f-strings de prompt (skeleton + day generator).
    n = orch_src.count("ctx['pantry_drift_context']")
    assert n >= 2, f"G1: pantry_drift_context debe interpolarse en ≥2 prompts; encontradas {n}"


def test_g1_build_pantry_drift_context_behavioral():
    """Conductual ligero: el builder convierte el dict en instrucción de prompt."""
    try:
        import sys
        sys.path.insert(0, str(_BACKEND))
        from prompts.plan_generator import build_pantry_drift_context
    except Exception as e:  # pragma: no cover - depende del entorno
        pytest.skip(f"No se pudo importar build_pantry_drift_context: {e}")
    assert build_pantry_drift_context(None) == ""
    assert build_pantry_drift_context({}) == ""
    out = build_pantry_drift_context({"critical_drops": [{"name": "pollo"}], "notable_increases": [], "new_items": []})
    assert "pollo" in out and "BAJARON" in out, "G1: el warning de drops debe llegar al texto del prompt"


# ===========================================================================
# G2 (P1) — failed-window-expired-limbo: escalation pass window-free
# ===========================================================================
def test_g2_window_expired_escalation_helper_exists(cron_src):
    assert "def _escalate_failed_window_expired_chunks(" in cron_src, (
        "G2: falta el helper _escalate_failed_window_expired_chunks"
    )
    body = _slice_fn(cron_src, "def _escalate_failed_window_expired_chunks(")
    # Espejo INVERTIDO del window guard: recoge los que el otro cron excluye.
    assert "<= NOW()" in body, "G2: el SELECT debe usar el window invertido (<= NOW())"
    assert "dead_lettered_at IS NULL" in body and "status = 'failed'" in body, (
        "G2: debe filtrar failed-not-dead-lettered"
    )
    assert "_escalate_unrecoverable_chunk(" in body, "G2: debe escalar vía el helper canónico"
    assert 'MEALFIT_CHUNK_FAILED_WINDOW_EXPIRED_ESCALATE' in body, "G2: falta el knob kill-switch"


def test_g2_helper_called_from_recover_failed(cron_src):
    body = _slice_fn(cron_src, "def _recover_failed_chunks_for_long_plans() -> None:")
    assert "_escalate_failed_window_expired_chunks()" in body, (
        "G2: _recover_failed_chunks_for_long_plans debe invocar el escalation pass"
    )
    # Debe correr ANTES del early-return de 'sin candidatos' (el deadlock vive ahí).
    call_idx = body.find("_escalate_failed_window_expired_chunks()")
    early_return_idx = body.find("if not failed_candidates:")
    assert call_idx != -1 and early_return_idx != -1 and call_idx < early_return_idx, (
        "G2: el escalation pass debe invocarse ANTES del early-return de sin-candidatos"
    )


# ===========================================================================
# G3 (P2) — CAS en el re-enqueue de shopping-list-failed
# ===========================================================================
def test_g3_shopping_reenqueue_has_cas(cron_src):
    # Región del re-enqueue (anclada por el dead_letter_reason único).
    idx = cron_src.find('"shopping_list_retry_exhausted"')
    assert idx != -1, "G3: no encuentro el re-enqueue de shopping-list-failed"
    region = cron_src[idx - 900:idx + 400]
    assert "G3-SHOPPING-CAS" in region, "G3: falta el tooltip-anchor G3-SHOPPING-CAS"
    assert "AND attempts = %s" in region, "G3: falta el guard CAS (AND attempts = %s)"
    assert "RETURNING status" in region, "G3: el UPDATE debe usar RETURNING para detectar displacement"


# ===========================================================================
# G4 (P2) — CAS en la pausa pantry_violation_post_merge
# ===========================================================================
def test_g4_post_merge_pause_routes_through_cas(cron_src):
    idx = cron_src.find("G4-POSTMERGE-CAS")
    assert idx != -1, "G4: falta el tooltip-anchor G4-POSTMERGE-CAS"
    region = cron_src[idx:idx + 1100]
    assert "_cas_pause_chunk_to_pending_user_action(" in region, (
        "G4: la pausa post-merge debe rutearse por el helper CAS"
    )
    assert '"pantry_violation_post_merge"' in region


# ===========================================================================
# G5 (P2) — 6 gate-pauses ruteados por el helper CAS
# ===========================================================================
def test_g5_six_gate_pauses_routed_through_cas(cron_src):
    n = cron_src.count("G5-PAUSE-CAS-SIBLINGS")
    # 6 anchors-de-código (uno por sitio); el comentario del docstring del helper
    # no usa este literal exacto, así que el conteo refleja los 6 sitios.
    assert n >= 6, f"G5: deben rutearse 6 gate-pauses por el CAS helper; encontrados {n} anchors"


def test_g5_no_bare_pending_user_action_update_in_worker(cron_src):
    """En el rango del worker (donde pickup_attempts está activo) NO debe quedar
    ningún UPDATE bare a 'pending_user_action' (todos via el helper CAS)."""
    start = cron_src.find("_CHUNK_WORKER_CTX.pickup_attempts = _pickup_attempts")
    end = cron_src.find("_CHUNK_WORKER_CTX.pickup_attempts = None")
    assert start != -1 and end != -1 and start < end
    worker_region = cron_src[start:end]
    assert "SET status = 'pending_user_action'" not in worker_region, (
        "G5: queda un UPDATE bare a pending_user_action en el worker (debe ir por el CAS helper)"
    )


# ===========================================================================
# G6 (P2) — resolver de chunk_lesson_synth_ratio_high + comentario G9 corregido
# ===========================================================================
def test_g6_synth_ratio_resolver(cron_src):
    body = _slice_fn(cron_src, "def _alert_high_synthesized_lesson_ratio() -> None:")
    assert "G6-SYNTH-RATIO-RESOLVE" in body, "G6: falta el tooltip-anchor"
    # Debe resolver SU PROPIA clave (no la de E1 chunk_synthesis_overload).
    assert re.search(
        r"UPDATE system_alerts\s+SET resolved_at = NOW\(\)\s+WHERE alert_key = 'chunk_lesson_synth_ratio_high'",
        body,
    ), "G6: falta el UPDATE resolved_at para chunk_lesson_synth_ratio_high"


def test_g6_false_g9_comment_corrected(cron_src):
    # El comentario G9 que afirmaba que la función ya auto-resolvía debe aclararse.
    # Buscamos la frase de corrección directamente (robusto a múltiples "espeja E1").
    assert "ANTES no se" in cron_src and "era falso" in cron_src, (
        "G6: el comentario G9 debe aclarar que la agregada chunk_lesson_synth_ratio_high "
        "NO se resolvía antes (era falso)"
    )
    # Y debe estar adyacente al anchor G6 de la corrección.
    idx = cron_src.find("era falso")
    region = cron_src[max(0, idx - 400):idx + 200]
    assert "G6" in region, "G6: la aclaración debe referenciar el anchor G6"


def test_g6_g7_alert_doc_updated():
    doc = _ALERT_DOC.read_text(encoding="utf-8")
    # Ambas filas deben estar marcadas Auto (explicit) tras el fix.
    for key in ("chunk_lesson_synth_ratio_high", "chunk_lag_excessive"):
        row = next((l for l in doc.splitlines() if l.startswith(f"| `{key}`")), None)
        assert row is not None, f"doc: falta la fila de {key}"
        assert "Auto (explicit)" in row, f"doc: {key} debe ser Auto (explicit) tras el resolver"


# ===========================================================================
# G7 (P3) — resolver de chunk_lag_excessive
# ===========================================================================
def test_g7_lag_resolver(cron_src):
    body = _slice_fn(cron_src, "def _alert_chunk_lag_excessive() -> None:")
    assert "G7-LAG-RESOLVE" in body, "G7: falta el tooltip-anchor"
    assert re.search(
        r"UPDATE system_alerts\s+SET resolved_at = NOW\(\)\s+WHERE alert_key = 'chunk_lag_excessive'",
        body,
    ), "G7: falta el UPDATE resolved_at para chunk_lag_excessive"


# ===========================================================================
# G8 (P3) — I2 user_id en 5 writes de meal_plans
# ===========================================================================
def test_g8_notify_helpers_have_user_id_filter(cron_src):
    for fn in ("def _notify_zombie_plan_generation_failed(", "def _notify_chunk_pause_failed("):
        body = _slice_fn(cron_src, fn)
        assert "UPDATE meal_plans" in body
        assert "AND user_id = %s" in body, f"G8(I2): {fn} debe filtrar AND user_id = %s"


def test_g8_worker_status_flips_have_user_id_filter(cron_src):
    for status in ('"failed"', '"complete_partial"'):
        # Las dos transiciones de generation_status del worker zombie-fallback.
        m = re.search(
            r"jsonb_set\(plan_data, '\{generation_status\}', '" + re.escape(status) + r"'\)\s*\n\s*WHERE id = %s AND user_id = %s",
            cron_src,
        )
        assert m is not None, f"G8(I2): el flip de generation_status a {status} debe llevar AND user_id = %s"


def test_g8_i2_anchor_count(cron_src):
    assert cron_src.count("I2 · P1-CHUNK-LEARN-3") >= 4, "G8: faltan anchors I2 en los writes de meal_plans"


# ===========================================================================
# G9 (P3) — consumir _blocked_techniques
# ===========================================================================
def test_g9_blocked_techniques_consumed(orch_src):
    assert "G9-BLOCKED-TECHNIQUES" in orch_src, "G9: falta el tooltip-anchor"
    idx = orch_src.find("G9-BLOCKED-TECHNIQUES")
    region = orch_src[idx:idx + 1300]
    assert 'form_data.get("_blocked_techniques")' in region, (
        "G9: el selector debe LEER _blocked_techniques (pre-fix era dead-write)"
    )
    assert "aban_techs" in region, "G9: _blocked_techniques debe mergearse en aban_techs"


# ===========================================================================
# G10 (P3) — surface _quality_degraded_reason/_severity en el banner
# ===========================================================================
def test_g10_dashboard_surfaces_quality_reason():
    src = _DASHBOARD.read_text(encoding="utf-8")
    assert "G10-QUALITY-DEGRADED-SURFACE" in src, "G10: falta el tooltip-anchor en Dashboard.jsx"
    assert "_quality_degraded_reason" in src, "G10: el banner debe leer _quality_degraded_reason"
    assert "_quality_degraded_severity" in src, "G10: el banner debe reflejar _quality_degraded_severity"


# ===========================================================================
# G12 (P3) — _skip_merge_write eliminado
# ===========================================================================
def test_g12_skip_merge_write_removed(cron_src):
    assert '"_skip_merge_write": True' not in cron_src, (
        "G12: _skip_merge_write seguía escribiéndose (dead-write sin consumer)"
    )
    assert "G12-SKIP-MERGE-WRITE-DEAD" in cron_src, "G12: falta el tooltip-anchor de la remoción"


# ===========================================================================
# G13 (P3) — _is_emergency_generation eliminado (write + whitelist)
# ===========================================================================
def test_g13_emergency_flag_removed(cron_src, orch_src):
    assert 'pipeline_data["_is_emergency_generation"] = True' not in cron_src, (
        "G13: el dead-write _is_emergency_generation seguía presente"
    )
    # La entrada quoted de la whitelist debe estar removida (queda solo en comentario).
    assert '"_is_emergency_generation",' not in orch_src, (
        "G13: la entrada de _TRUSTED_INTERNAL_FORM_KEYS debe eliminarse"
    )
    assert "G13-EMERGENCY-FLAG-DEAD" in cron_src and "G13-EMERGENCY-FLAG-DEAD" in orch_src


# ===========================================================================
# G14 (P3) — comentarios stale de synth corregidos
# ===========================================================================
def test_g14_doc_drift_fixed():
    hist = _HISTORY.read_text(encoding="utf-8")
    plans = _PLANS.read_text(encoding="utf-8")
    assert "G14-DOC-DRIFT" in hist, "G14: falta la corrección del comentario en History.jsx"
    assert "G14-DOC-DRIFT" in plans, "G14: falta la corrección del docstring en plans.py"


# ===========================================================================
# Marker freshness
# ===========================================================================
def test_marker_bumped():
    """[Relajado NG-CRON-OPT-2 · 2026-05-30] El exact-match `P1-CHUNK-LEARN-3 ·
    2026-05-29` se relajó a un FLOOR DE FECHA: auditorías sucesoras (P2-CRON-OPT,
    NG-CRON-OPT-2, …) bumpean legítimamente el marker fuera de esta familia, y
    pin-earlo rompía cada bump posterior. La freshness real (formato + floor) la
    enforza `test_p3_1_last_known_pfix_freshness`; aquí solo exigimos no-stale."""
    import re as _re
    from datetime import date as _date
    app = _APP.read_text(encoding="utf-8")
    m = _re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*·\s*(\d{4})-(\d{2})-(\d{2})"', app)
    assert m, 'No se encontró marker `_LAST_KNOWN_PFIX = "... · YYYY-MM-DD"` válido.'
    marker_date = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    assert marker_date >= _date(2026, 5, 29), (
        f"Marker stale: {marker_date} < 2026-05-29 (fecha de P1-CHUNK-LEARN-3). "
        f"Un fix posterior debe bumpear el marker, nunca retrocederlo."
    )
