"""[P1-PLAN-LOTE-1114 · 2026-11-20] El gate del 19-nov se reparó sacando el auto-resolve de la alerta de lecciones
sintetizadas de `cron_tasks.py` (19 líneas sobre su tope) a `admin_avisos.resolver_synth_ratio_quieto`."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import admin_avisos  # noqa: E402


def test_con_cero_sintetizados_cierra_la_alerta_vieja():
    escritas = []
    admin_avisos.resolver_synth_ratio_quieto(lambda sql, params: escritas.append((sql, params)), 0, 48)
    assert len(escritas) == 1
    sql, params = escritas[0]
    assert "chunk_lesson_synth_ratio_high" in sql and "quiet_window_zero_synthesized" in sql and params == (48,)


def test_con_sintetizados_no_toca_nada_y_un_fallo_no_rompe():
    escritas = []
    admin_avisos.resolver_synth_ratio_quieto(lambda sql, params: escritas.append(1), 3, 48)
    assert escritas == []

    def roto(sql, params):
        raise RuntimeError("base caída")
    admin_avisos.resolver_synth_ratio_quieto(roto, 0, 48)   # best-effort: no lanza


def test_cron_tasks_lo_llama_y_sigue_en_su_tope():
    cron = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert '__import__("admin_avisos").resolver_synth_ratio_quieto(execute_sql_write, synthesized, int(CHUNK_LESSON_SYNTH_WINDOW_HOURS))' in cron
    assert "quiet_window_zero_synthesized" not in cron
    assert cron.count("\n") <= 36550   # = wc -l
