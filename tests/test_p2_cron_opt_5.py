"""[P2-CRON-OPT-5 · 2026-05-31] Anclas de la QUINTA pasada de optimización de
cron_tasks.py — la cola larga de "hermanos no tocados" de gaps ya arreglados en otras
funciones (las 4 pasadas previas sliceaban per-función). Workflow fresco de 19 finders +
verificación adversaria + RE-verificación MANUAL contra el código de prod (lección repetida:
"el veredicto del workflow sobreestima"). De 9 candidatos implement_now, 6 implementados
(reales+seguros+test-verificados) y 3 diferidos por churn>valor / riesgo en hot-path P0.

La re-verificación manual CORRIGIÓ al workflow en 2 puntos:
  - G5-2: el finder propuso castear `(split_part(...))::uuid` como AND hermano del regex
    guard. PERO SQL no garantiza orden de evaluación de predicados → un `:unknown:` /
    session_id de invitado abortaría el sweep con `invalid input syntax for type uuid`.
    Fix robusto: `CASE WHEN <regex-uuid-canónico> THEN (...)::uuid ELSE NULL END` (el cast
    SÓLO se evalúa cuando el guard pasa) + regex ESTRICTO 8-4-4-4-12 (no `[hex-]{36}`, que
    dejaría pasar 36 hex sin hyphens = 144 bits ≠ uuid → cast-error igual).
  - G5-2: el finder dijo que test_p2_new_10 sólo chequea el token `split_part`. FALSO: su
    línea 99 también exige `alert_key LIKE 'plan_quality_degraded:%%'`. Por eso el fix
    AÑADE el guard sin REEMPLAZAR el LIKE (ambos tokens sobreviven).

Tooltip-anchor: P2-CRON-OPT-5.
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
    """Quita el contenido de comentarios (Python `#` Y SQL `--` dentro de strings) hasta
    fin de línea, para que las aserciones NEGATIVAS no matcheen el propio comentario
    explicativo del fix — lección repetida del repo ("mis comentarios rompieron mis asserts
    negativos"). Strip de AMBOS porque varios fixes de esta pasada documentan el patrón
    viejo dentro de comentarios SQL `--` (cron_tasks tiene SQL embebido). Ninguna ancla
    POSITIVA de este archivo contiene `#` ni `--`, así que sobreviven intactas."""
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
    # Floor-de-fecha (igual que cron_opt_3/4): otras sesiones pueden bumpear el marker hacia
    # adelante. La freshness real la enforza test_p3_1_last_known_pfix_freshness; aquí: no
    # retroceder antes de 2026-05-31 (esta pasada).
    app = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*(\d{4})-(\d{2})-(\d{2})"', app)
    assert m, "No se encontró _LAST_KNOWN_PFIX con fecha parseable."
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= (2026, 5, 31), (
        "El marker _LAST_KNOWN_PFIX retrocedió antes de 2026-05-31 (P2-CRON-OPT-5)."
    )


# ---------------------------------------------------------------------------
# G5-1 (#5): _emit_plan_data_corruption_alert push-dedupe SELECT lleva resolved_at IS NULL
# ---------------------------------------------------------------------------
def test_g5_1_corruption_dedupe_resolved_at_null():
    body = _slice_fn(_src(), "def _emit_plan_data_corruption_alert(")
    idx = body.find("SELECT triggered_at FROM system_alerts WHERE alert_key = %s")
    assert idx >= 0, "G5-1: no se encontró el dedupe SELECT del push de corrupción."
    window = body[idx: idx + 260]
    assert "resolved_at IS NULL" in window, (
        "G5-1: el push-dedupe SELECT de _emit_plan_data_corruption_alert debe filtrar "
        "`AND resolved_at IS NULL` — si no, una recurrencia tras resolve manual (SRE) "
        "queda con el push suprimido por la fila resolved-but-recent dentro de la ventana 24h."
    )


# ---------------------------------------------------------------------------
# G5-2 (#1): plan_quality sweep compara la PK uuid SIN castearla a text (sargable),
# con guard CASE + regex uuid canónico para no abortar el cast en keys no-uuid.
# ---------------------------------------------------------------------------
def test_g5_2_plan_quality_uuid_cast_sargable_and_guarded():
    body = _slice_fn(_src(), "def _resolve_stale_plan_quality_alerts(")
    code = _code_only(body)
    # Negativo: ya NO se castea la columna PK a text dentro del EXISTS.
    assert "mp.user_id::text = split_part" not in code, (
        "G5-2: `mp.user_id::text = split_part(...)` (no-sargable, seq-scan) debe haberse "
        "reemplazado por la comparación contra la columna uuid sin cast."
    )
    # Positivo: el cast del RHS va envuelto en CASE (short-circuit garantizado).
    assert "WHERE mp.user_id = CASE" in code, (
        "G5-2: la comparación debe ser `mp.user_id = CASE WHEN <guard> THEN (...)::uuid ELSE NULL END`."
    )
    assert "(split_part(a.alert_key, ':', 2))::uuid" in code, (
        "G5-2: el segmento-2 debe castearse a uuid (RHS) para usar idx_meal_plans_user_id."
    )
    # El guard es el regex uuid CANÓNICO (8-4-4-4-12), no un `[hex-]{36}` laxo.
    assert "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}" in code, (
        "G5-2: el guard del CASE debe ser el patrón uuid canónico estricto (evita que un "
        "string 36-hex-sin-hyphens pase el guard y aborte el `::uuid`)."
    )
    # El token que el test P2-NEW-10 ancla DEBE seguir presente (fix aditivo, no reemplazo).
    assert "alert_key LIKE 'plan_quality_degraded:%%'" in code, (
        "G5-2: el LIKE original debe conservarse (ancla de test_p2_new_10); el guard se AÑADE."
    )


# ---------------------------------------------------------------------------
# G5-3 (#6): _recover_orphan_chunk_reservations compara la PK uuid sin cast (usa el btree).
# ---------------------------------------------------------------------------
def test_g5_3_orphan_reservation_uuid_array_no_text_cast():
    body = _slice_fn(_src(), "def _recover_orphan_chunk_reservations(")
    code = _code_only(body)
    assert "id::text = ANY(%s)" not in code, (
        "G5-3: `id::text = ANY(%s)` casteaba la PK uuid a text → seq-scan. Debe ser "
        "`id = ANY(%s::uuid[])`."
    )
    assert "WHERE id = ANY(%s::uuid[])" in code, (
        "G5-3: la comparación debe usar `id = ANY(%s::uuid[])` para hit del PK btree."
    )
    assert "_uuid_chunk_ids = [c for c in chunk_ids if _is_valid_uuid(c)]" in code, (
        "G5-3: el bind debe filtrar a uuids válidos (`_uuid_chunk_ids`) — sin reasignar "
        "`chunk_ids`, que sigue alimentando el loop de limpieza con ids no-uuid (cleanable)."
    )


# ---------------------------------------------------------------------------
# G5-4 (#7): EMA short+long se escriben en UN solo jsonb_set anidado (1 round-trip).
# ---------------------------------------------------------------------------
def test_g5_4_ema_writes_merged_nested_jsonb_set():
    body = _slice_fn(_src(), "def _inject_advanced_learning_signals(")
    code = _code_only(body)
    assert (
        "jsonb_set(jsonb_set(health_profile, '{{{profile_key}}}', %s::jsonb), "
        "'{{{long_profile_key}}}', %s::jsonb)" in code
    ), (
        "G5-4: las dos escrituras EMA (corto/largo) deben colapsarse en un jsonb_set anidado "
        "sobre la misma fila/columna (espeja el SSOT last_plan_quality+quality_history_chunks)."
    )
    # Negativo: la escritura standalone del EMA largo (sobre `health_profile` directo) ya no existe.
    assert "jsonb_set(health_profile, '{{{long_profile_key}}}'" not in code, (
        "G5-4: la escritura separada del EMA largo debe haberse fusionado en el jsonb_set anidado."
    )


# ---------------------------------------------------------------------------
# G5-5 (#9): el failure-metric block reusa _pickup_attempts (sin SELECT attempts redundante).
# ---------------------------------------------------------------------------
def test_g5_5_failure_metric_reuses_pickup_attempts():
    code = _code_only(_src())
    # El último `SELECT attempts FROM plan_chunk_queue` de producción se eliminó (el
    # success-path ya usaba _pickup_attempts vía S18-1; este era el gemelo failure-path).
    assert "SELECT attempts FROM plan_chunk_queue WHERE id = %s" not in code, (
        "G5-5: el `SELECT attempts FROM plan_chunk_queue` del failure-metric era un round-trip "
        "redundante (attempts == _pickup_attempts en ese punto). Debe usar _pickup_attempts."
    )
    body = _slice_fn(_src(), "def process_plan_chunk_queue(")
    # Ancla el comentario del fix para que un renombre de la variable falle el test.
    assert body.count("retries == _pickup_attempts en este punto") >= 1, (
        "G5-5: debe documentarse por qué retries == _pickup_attempts en el failure-metric block."
    )


# ---------------------------------------------------------------------------
# G5-6 (#8): _alert_stranded_partial_plans resuelve PER-PLAN (ambas ramas) aunque
# queden otros planes stranded/abandoned — no sólo en la rama "todos recuperados".
# ---------------------------------------------------------------------------
def test_g5_6_stranded_partial_per_plan_auto_resolve():
    body = _slice_fn(_src(), "def _alert_stranded_partial_plans(")
    # Rama stranded: sweep per-plan guardado por truncación de batch.
    assert "if len(rows) < _batch_limit:" in body, (
        "G5-6: la rama stranded debe barrer per-plan sólo si el batch NO se truncó "
        "(`len(rows) < _batch_limit`) para no false-resolver stranded fuera del batch."
    )
    assert body.count("alert_key <> ALL(%s::text[])") >= 2, (
        "G5-6: ambas ramas (stranded + rolling_abandoned) deben incluir el sweep per-plan "
        "`alert_key <> ALL(%s::text[])` (mirror del helper G11). Sin él, un plan recuperado "
        "fuga `resolved_at IS NULL` forever cuando otros planes siguen stranded."
    )
    # Rama rolling-abandoned: guard análogo.
    assert "if len(abandoned_rows) < _abandoned_batch_limit:" in body, (
        "G5-6: la rama rolling-abandoned debe llevar el mismo guard de truncación de batch."
    )
    # Las keys del sweep son las CANÓNICAS de cada rama.
    assert "f\"plan_stranded_partial:{r.get('plan_id')}\"" in body, (
        "G5-6: el sweep stranded debe excluir las keys vivas `plan_stranded_partial:<plan_id>`."
    )
    assert "f\"plan_rolling_abandoned:{r.get('plan_id')}\"" in body, (
        "G5-6: el sweep rolling debe excluir las keys vivas `plan_rolling_abandoned:<plan_id>`."
    )
