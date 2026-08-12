# -*- coding: utf-8 -*-
"""[P1-RESUME-REVIVES-QUEUE · 2026-08-12] Reanudar REENCOLA lo que la pausa apago.
Extraido de test_p1_plan_mode.py para el cross-link marker↔test."""
from __future__ import annotations

from pathlib import Path

import plan_mode as pm

# ═══════════════════════════════════════════════════════════════════════════
# [P1-RESUME-REVIVES-QUEUE · 2026-08-12] Reanudar REENCOLA lo que la pausa apagó.
# El primer ciclo real pausa→reanuda dejó el plan del owner en 3/30 PARA
# SIEMPRE: catch-up del shift (solo no-partial), bg-refill (solo inactivos
# ≥3 días) y recovery (ignora cancelled a propósito) — nadie cubría el caso.
# ═══════════════════════════════════════════════════════════════════════════

def test_pausa_firma_sus_cancelados():
    """El UPDATE de cancelación estampa PAUSE_CANCEL_REASON en dead_letter_reason
    (con dead_lettered_at intacto): es EL contrato que reanudar usa para saber
    cuáles filas son suyas."""
    src = Path(pm.__file__).read_text(encoding="utf-8")
    i = src.index("SET status = 'cancelled', dead_letter_reason = %s")
    bloque = src[i:i + 400]
    assert "dead_lettered_at IS NULL" in bloque
    assert 'PAUSE_CANCEL_REASON = "[P1-PLAN-MODE] paused_by_user"' in src


def test_revive_solo_las_filas_firmadas_y_rebasea(monkeypatch):
    """Funcional: el revive filtra por firma EXACTA + cancelled + no-dead-letter,
    y por cada plan revivido corre el rebase del SSOT con los días vivos.
    [P1-PAUSE-GC-SURVIVAL] El revive ahora lee el perfil PRIMERO (materia prima
    del snapshot reconstruido) — el mock refleja esa secuencia."""
    from unittest.mock import MagicMock

    ejecutadas = []
    cursor = MagicMock()

    def _exec(sql, params=None):
        ejecutadas.append((" ".join(str(sql).split()), params))
    cursor.execute.side_effect = _exec
    cursor.fetchone.side_effect = [
        {"health_profile": {"age": 30}},
        {"d": 3, "d0": "2026-08-12", "days": []},
    ]
    cursor.fetchall.return_value = [
        {"id": "c1", "meal_plan_id": "plan-aa", "days_count": 4},
        {"id": "c2", "meal_plan_id": "plan-aa", "days_count": 4},
    ]

    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value.transaction.return_value.__enter__.return_value = MagicMock()
    pool.connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cursor

    import db_core
    monkeypatch.setattr(db_core, "connection_pool", pool)

    rebases = []
    import cron_tasks as ct
    monkeypatch.setattr(ct, "_rebase_pending_chunk_offsets_sql", lambda cur, pid, dias: rebases.append((pid, dias)) or 2)

    out = pm._revive_paused_chunks("u-revive")
    assert out == {"revived": 2, "plans": 1}
    revives = [(s, p) for s, p in ejecutadas if "SET status = 'pending'" in s]
    assert len(revives) == 1
    _sql_revive, _params = revives[0]
    assert "dead_letter_reason = NULL" in _sql_revive
    assert "AND status = 'cancelled'" in _sql_revive
    assert "AND dead_letter_reason = %s" in _sql_revive
    assert "AND dead_lettered_at IS NULL" in _sql_revive
    assert pm.PAUSE_CANCEL_REASON in _params
    assert rebases == [("plan-aa", 3)], "el rebase debe correr con los días VIVOS del plan"


def test_revive_sin_pool_es_best_effort(monkeypatch):
    import db_core
    monkeypatch.setattr(db_core, "connection_pool", None)
    assert pm._revive_paused_chunks("u-x") == {"revived": 0, "plans": 0}


def test_resume_revive_despues_de_la_bandera_y_lo_reporta():
    """Orden en el fuente: bandera plan (paso 1) ANTES del revive — revivir con el
    gate puesto deja chunks pending que el pickup ignora. Y el resultado viaja
    en el return (chunks_revived) para que el caller/toast pueda decirlo."""
    src = Path(pm.__file__).read_text(encoding="utf-8")
    cuerpo = src[src.index("def resume_plan_generation"):]
    i_bandera = cuerpo.index("SET plan_mode = 'plan'")
    i_revive = cuerpo.index("_revive_paused_chunks(user_id)")
    assert i_bandera < i_revive
    assert '"chunks_revived": _revive["revived"]' in cuerpo


def test_exec_resincronizado_contra_d0_mas_offset():
    """El rebase mueve exec POR DELTA de offset; una fila que pasó la pausa
    cancelada quedó con offset casualmente cuadrado y exec del ancla VIEJA
    (delta 0 ⇒ relleno tarde, medido: w4 exec 08-17 con ancla 08-12+3=08-15).
    El revive re-deriva exec contra `d0 + offset` conservando la hora local."""
    src = Path(pm.__file__).read_text(encoding="utf-8")
    cuerpo = src[src.index("def _revive_paused_chunks"):src.index("def resume_plan_generation")]
    assert "q.execute_after::date" in cuerpo
    assert "- (p.plan_data->'days'->0->>'date')::date" in cuerpo
    assert "- q.days_offset" in cuerpo
    # con d0 ausente (plan legacy) NO se toca la fila
    assert "IS NOT NULL" in cuerpo
    # y el suelo es NOW(): jamás programar al pasado
    assert "NOW()" in cuerpo
