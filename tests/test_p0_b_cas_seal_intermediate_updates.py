"""[P0-B] Tests del sellado CAS `_plan_modified_at` en UPDATEs intermedios al plan_data.

Contexto: el CAS sello capturado en `process_chunk_task` (cron_tasks.py:9642 con
`pre_read_modified_at = prior_plan_data.get('_plan_modified_at')`) se valida después
en línea 12586 dentro del FOR UPDATE final. Si el chunk hace UPDATEs intermedios al
plan_data (recovery de _last_chunk_learning, regeneración de _recent_chunk_lessons,
backfill de lifetime lessons) sin actualizar `_plan_modified_at`, el CAS no detecta
esos cambios — un proceso concurrente que leyó el timestamp anterior verá el mismo
y declarará erróneamente "no hubo cambios externos".

Este test bloquea la invariante a nivel de código fuente: cualquier nuevo UPDATE a
meal_plans.plan_data debe incluir el sellado del timestamp. Si en el futuro alguien
añade un UPDATE intermedio sin sellar, el test falla con un mensaje accionable.

Ejecutar:
    cd backend && python -m pytest tests/test_p0_b_cas_seal_intermediate_updates.py -v
"""
import os
import re


_CRON_TASKS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cron_tasks.py",
)


def _read_source() -> str:
    with open(_CRON_TASKS_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers para localizar UPDATEs intermedios por contexto cercano
# ---------------------------------------------------------------------------
def _extract_update_blocks_setting_jsonb_key(source: str, target_key: str) -> list[str]:
    """Devuelve los bloques SQL `UPDATE meal_plans ... jsonb_set(..., '{TARGET}', ...)`.

    Cada bloque es la cadena entre `UPDATE meal_plans` y el siguiente `WHERE id = %s` o
    `WHERE id = %(...)s`, suficiente para inspeccionar si incluye `_plan_modified_at`.
    """
    # Pattern: UPDATE meal_plans ... '{<target_key>}' ... WHERE id = %s
    # Usamos DOTALL para que . abarque newlines.
    pattern = re.compile(
        r"UPDATE meal_plans\b.*?'\{" + re.escape(target_key) + r"\}'.*?WHERE id = %s",
        re.DOTALL,
    )
    return pattern.findall(source)


# ---------------------------------------------------------------------------
# 1. P0-3 rebuild de _last_chunk_learning sella _plan_modified_at
# ---------------------------------------------------------------------------
def test_p03_last_chunk_learning_rebuild_seals_modified_at():
    """El UPDATE que escribe `_last_chunk_learning` desde el rebuild debe sellar timestamp.

    Aplica a `_last_chunk_learning` y a su variante synth (P0-4). Ambos rewriting paths
    deben incluir un `jsonb_set` anidado para `_plan_modified_at`.
    """
    source = _read_source()
    blocks = _extract_update_blocks_setting_jsonb_key(source, "_last_chunk_learning")
    assert blocks, "no se encontró ningún UPDATE intermedio a `_last_chunk_learning`"
    for i, block in enumerate(blocks):
        assert "_plan_modified_at" in block, (
            f"UPDATE #{i+1} a `_last_chunk_learning` NO sella `_plan_modified_at`. "
            f"Sin el sello, un proceso concurrente que leyó el timestamp anterior no "
            f"detectará este cambio en el CAS. Bloque:\n{block[:500]}"
        )
        # Validación adicional: el sello usa NOW() para garantizar monotonicidad.
        assert "NOW()" in block, (
            f"UPDATE #{i+1} sella `_plan_modified_at` pero no con `NOW()`. "
            f"Cualquier valor que no sea NOW() abre ventana a colisiones de timestamp."
        )


# ---------------------------------------------------------------------------
# 2. P1-1 / P1-2 rebuild de _recent_chunk_lessons sella _plan_modified_at
# ---------------------------------------------------------------------------
def test_p11_recent_chunk_lessons_rebuild_seals_modified_at():
    """El UPDATE que reescribe `_recent_chunk_lessons` debe sellar el timestamp CAS."""
    source = _read_source()
    blocks = _extract_update_blocks_setting_jsonb_key(source, "_recent_chunk_lessons")
    assert blocks, "no se encontró ningún UPDATE intermedio a `_recent_chunk_lessons`"
    for i, block in enumerate(blocks):
        assert "_plan_modified_at" in block, (
            f"UPDATE #{i+1} a `_recent_chunk_lessons` NO sella `_plan_modified_at`. "
            f"Bloque:\n{block[:500]}"
        )
        assert "NOW()" in block


# ---------------------------------------------------------------------------
# 3. Path chunk_already_merged: sello in-memory antes del UPDATE
# ---------------------------------------------------------------------------
def test_chunk_already_merged_seals_modified_at_in_memory():
    """En el path donde re-escribimos plan_data completo dentro del FOR UPDATE final,
    el dict in-memory debe llevar `_plan_modified_at` actualizado ANTES del UPDATE.

    Verificamos que la línea de sellado (`plan_data['_plan_modified_at'] = ...now()...`)
    aparece justo antes del comentario "Re-escribir plan_data dentro del mismo FOR UPDATE".
    """
    source = _read_source()
    # Buscar el comentario marker y verificar que justo arriba hay un
    # asignamiento a `plan_data['_plan_modified_at']`.
    marker_idx = source.find("Re-escribir plan_data dentro del mismo FOR UPDATE")
    assert marker_idx > -1, "no se encontró el comentario marker del path chunk_already_merged"
    # [test fix · P1-21] El sello in-memory (plan_data['_plan_modified_at'] = now)
    # SIGUE presente justo en este path, pero un comentario [P1-21] añadido entre
    # el sello y el marker (re-adquisición explícita del advisory lock 'general')
    # empujó el sello a ~1473 chars del marker. La ventana de 600 chars ya no lo
    # alcanzaba aunque prod sí sella. Ampliada a 1600 — sigue acotada al mismo
    # bloque chunk_already_merged (no cruza a otra función/path).
    preceding = source[max(0, marker_idx - 1600):marker_idx]
    assert "plan_data['_plan_modified_at']" in preceding or 'plan_data["_plan_modified_at"]' in preceding, (
        "El path chunk_already_merged reescribe plan_data completo SIN sellar "
        "`_plan_modified_at` en memoria primero. Esto deja al CAS ciego ante este "
        "UPDATE: un chunk concurrente que leyó el timestamp anterior no detectará "
        "el cambio porque el sello quedó stale. Sellar plan_data['_plan_modified_at'] "
        "= datetime.now(timezone.utc).isoformat() justo antes del cursor.execute."
    )


# ---------------------------------------------------------------------------
# 4. Invariante general: todos los UPDATEs intermedios identificados están protegidos
# ---------------------------------------------------------------------------
def test_no_intermediate_jsonb_set_update_to_plan_data_without_seal():
    """Captura regresiones: ningún UPDATE intermedio a meal_plans.plan_data vía
    jsonb_set debe escribir solo una key sin actualizar `_plan_modified_at` también.

    Heurística: localizamos cada `UPDATE meal_plans ... jsonb_set(...) ... WHERE id = %s`
    y verificamos que mencione `_plan_modified_at`. Excepciones permitidas:
      - UPDATEs en _escalate_unrecoverable_chunk (recovery) que escriben
        _recovery_exhausted_chunks / _user_action_required (paths de cron_tasks
        que NO están dentro de process_chunk_task — el CAS no los protege porque
        no son parte del flujo del worker, son crons separados).
      - UPDATE específico de grocery_start_date backfill (idempotente, solo
        escribe si actualmente NULL).
      - UPDATE de generation_status='failed'/'complete_partial' en el path zombie
        fatal: ya no hay merge posterior, no necesita sello CAS.
    """
    source = _read_source()
    pattern = re.compile(
        r"(UPDATE meal_plans\b[^;]*?jsonb_set\([^;]*?WHERE id = %s)",
        re.DOTALL,
    )
    blocks = pattern.findall(source)
    assert blocks, "no se encontraron UPDATEs `UPDATE meal_plans ... jsonb_set ... WHERE id = %s`"

    # Allowlist: keys de paths que NO requieren sello CAS (justificación arriba).
    allowlist_markers = (
        "_recovery_exhausted_chunks",
        "_user_action_required",
        "grocery_start_date",
        "generation_status",
        "aggregated_shopping_list_weekly",
        "_partial_finalized_at",  # P0-A zombie partial finalize: cron separado.
        # `_anchor_recovery_attempts` se incrementa en dos sitios — `_recover_pantry_paused_chunks`
        # (cron de recovery, fuera de process_chunk_task) y el path missing_start_date_no_anchor
        # (process_chunk_task pero con `return` inmediato sin llegar al merge final). En ambos
        # casos no hay competencia con un merge concurrente: el primero solo bumpea un contador
        # benigno; el segundo aborta el chunk antes del FOR UPDATE protegido por CAS.
        "_anchor_recovery_attempts",
        # [test fix · P0-A] `_pantry_supplement_required` se escribe SOLO desde el helper
        # dedicado `_persist_pantry_supplement_to_plan_data` (y se limpia en
        # `_clear_pantry_supplement_from_plan_data`). Ambos hacen su PROPIO
        # read-modify-write (SELECT plan_data ... AND user_id=%s → jsonb_set) y corren
        # en el path de PAUSA (`pending_user_action`) ANTES de mergear los días del
        # chunk — el merge nunca ocurre en ese tick, así que no hay comparación CAS
        # downstream que pudieran cegar. Fuera del flujo de merge de process_chunk_task.
        "_pantry_supplement_required",
        # [test fix · P0-A] `_plan_start_date` se cura en `_check_chunk_learning_ready`
        # (el GATE, antes del pickup/merge). Misma razón que `_anchor_recovery_attempts`,
        # con el que comparte función: el gate corre antes del FOR UPDATE protegido por CAS.
        "_plan_start_date",
        # [test fix] `restocked_items` se reescribe en el sweep cron
        # `_reactivate_shopping_list_after_perishable_cycle` (I2-EXEMPT cross-user
        # system-wide), un cron separado fuera de process_chunk_task — el CAS del worker
        # no lo cubre porque no es parte del flujo del merge.
        "restocked_items",
    )
    violations = []
    for block in blocks:
        if "_plan_modified_at" in block:
            continue
        # Si toca SOLO keys allowlisted, OK.
        if any(marker in block for marker in allowlist_markers):
            continue
        violations.append(block[:500])

    assert not violations, (
        "Se detectaron UPDATEs intermedios a meal_plans.plan_data que NO sellan "
        "`_plan_modified_at` y NO están en el allowlist conocido. Cada uno deja al "
        "CAS ciego ante el cambio. Sellar añadiendo un `jsonb_set` anidado para "
        "`_plan_modified_at` con `to_jsonb(NOW()::text)`. Bloques infractores:\n\n"
        + "\n---\n".join(violations)
    )


# ---------------------------------------------------------------------------
# 5. P0-6: full-overwrite UPDATEs (no jsonb_set) también deben sellar
# ---------------------------------------------------------------------------
def test_full_overwrite_plan_data_updates_seal_modified_at_in_memory():
    """[P0-6] Captura el bug del path BG-REFILL pantry-empty (cron_tasks.py
    ~16724) donde `UPDATE meal_plans SET plan_data = %s::jsonb` con full overwrite
    estructural (generation_status, pending_user_action) NO sellaba `_plan_modified_at`.

    Heurística: para cada `UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s`
    encontrado, verificar que en las ~40 líneas previas haya una asignación a
    `<dict>['_plan_modified_at']` ANTES del cursor.execute. Esto cubre:
        - shifted_data['_plan_modified_at'] = ... (BG/HTTP shift)
        - plan_data['_plan_modified_at'] = ... (T1, chunk_already_merged path)
        - _p04_fresh_plan_data['_plan_modified_at'] = ... (T2, si llegase a sellarse)

    EXCEPCIÓN explícita: T2 del worker (búsqueda por marker
    "[P0-1 FIX] Transacción atómica final") NO debe sellar — escribe SOLO
    P0_4_T2_INCREMENTAL_KEYS (learning + shopping + quality), que son
    no-estructurales y no compiten con `/shift-plan`. Sellar aquí causaría
    falsos positivos en el CAS de chunks subsiguientes que verían el
    timestamp cambiado y abortarían innecesariamente. Decisión documentada
    en cron_tasks.py:16014-16018.
    """
    source = _read_source()
    pattern = re.compile(
        r'UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s',
    )
    violations: list[str] = []

    # Marker que identifica el bloque T2 del worker, donde NO debe sellarse.
    T2_INCREMENTAL_MARKER = "Transacción atómica final"

    # Ventana amplia: en cron_tasks.py el sello in-memory puede aparecer hasta
    # ~80 líneas antes del cursor.execute (path T1 chunk_already_merged sella
    # ~líneas 15600 y el execute ocurre ~línea 15677, ej.).
    # [test fix · P1-1/P0-5] El sello del T1 "Merge normal" (plan_data
    # ['_plan_modified_at'] = now, ~línea 29867) SIGUE presente, pero el bloque
    # de comentarios P0-1/P1-1 + el runtime defense-check P0-5 que se intercalan
    # entre el sello y el `UPDATE ... _t1_persist_view` empujaron el sello a
    # ~8260 chars del execute, justo por encima de la ventana 8000. Ampliada a
    # 9000: cubre el gap real y sigue MUY por debajo de la distancia al T2
    # (~40k chars, que de todos modos está exento por su marker propio), así que
    # no introduce falsos negativos cruzando a un seal ajeno.
    CTX_WINDOW = 9000

    for match in pattern.finditer(source):
        ctx_start = max(0, match.start() - CTX_WINDOW)
        ctx = source[ctx_start:match.start()]

        # Excepción T2: si el contexto contiene el marker de T2 incremental,
        # explícitamente permitido sin sello (decisión documentada).
        if T2_INCREMENTAL_MARKER in ctx:
            continue

        # Verificar que el contexto previo sella `_plan_modified_at` en memoria.
        # Acepta single o double quotes; cualquier nombre de variable seguido por
        # ['_plan_modified_at'] o ["_plan_modified_at"].
        seal_pattern = re.compile(
            r"\w+\[(['\"])_plan_modified_at\1\]\s*="
        )
        if seal_pattern.search(ctx):
            continue

        # Snippet para reportar al desarrollador.
        snippet_start = max(0, match.start() - 600)
        violations.append(source[snippet_start:match.end() + 200])

    assert not violations, (
        "[P0-6] Se detectaron UPDATEs `UPDATE meal_plans SET plan_data = %s::jsonb` "
        "con full overwrite que NO sellan `_plan_modified_at` en memoria antes del "
        "execute. Cada uno deja al CAS ciego ante el cambio estructural. Sellar "
        "añadiendo `<dict>['_plan_modified_at'] = datetime.now(timezone.utc).isoformat()` "
        "ANTES del cursor.execute. Si el path es legítimamente non-structural "
        "(análogo al T2 incremental del worker), añadir un comentario que "
        "incluya 'Transacción atómica final' o documentar la nueva excepción "
        "en este test. Bloques infractores:\n\n"
        + "\n===\n".join(violations)
    )


def test_t2_incremental_does_not_seal_modified_at():
    """[P0-6] Regression guard inverso: el T2 del worker (incremental merge de
    learning + shopping + quality) NO debe sellar `_plan_modified_at`.

    Razón: T2 escribe SOLO `P0_4_T2_INCREMENTAL_KEYS`, que son non-structural y
    no compiten con `/shift-plan`. Si T2 sellara, cualquier chunk subsiguiente
    cuyo `pre_read_modified_at` fuera capturado antes del T2 vería un mismatch
    spurious en el CAS check de su propio T1 (cron_tasks.py:14760), generando
    warnings ruidosos sin race real. La decisión de NO sellar es deliberada.

    Si en el futuro alguien añade un `_p04_fresh_plan_data['_plan_modified_at']`
    al bloque T2, este test cae con un mensaje accionable.
    """
    source = _read_source()
    # Localizar el bloque T2 por su marker único.
    marker = "Transacción atómica final"
    marker_idx = source.find(marker)
    assert marker_idx > -1, (
        "No se encontró el marker 'Transacción atómica final' que identifica "
        "el T2 del worker. ¿Se renombró o eliminó el bloque?"
    )

    # Buscar el cursor.execute más cercano DESPUÉS del marker (el UPDATE de T2).
    update_idx = source.find(
        "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
        marker_idx,
    )
    assert update_idx > -1, "No se encontró el UPDATE de T2 después del marker"

    # Slice entre el marker y el UPDATE: NO debe haber sello en memoria.
    t2_block = source[marker_idx:update_idx]
    forbidden = re.search(r"_p04_fresh_plan_data\[['\"]_plan_modified_at['\"]\]\s*=", t2_block)
    assert forbidden is None, (
        "[P0-6] El bloque T2 del worker está sellando `_plan_modified_at`. "
        "Esto es incorrecto: T2 escribe sólo P0_4_T2_INCREMENTAL_KEYS "
        "(non-structural) y sellar causa falsos positivos en el CAS de "
        "chunks subsiguientes. Remover el sello o, si hay una nueva razón "
        "para sellarlo, actualizar este test y documentar en cron_tasks.py."
    )
