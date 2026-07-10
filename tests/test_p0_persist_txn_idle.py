"""[P0-PERSIST-TXN-IDLE · 2026-07-10] El INSERT de meal_plans moría con
`terminating connection due to idle-in-transaction timeout` y el plan generado
se PERDÍA (usuario ve `plan_persist_failed` tras 13 min de pipeline).

Forensics (journalctl 2026-07-03..10, Neon): 3 víctimas en 4 días —
  - Jul 06 13:00: chunk 4 del plan 72c8b965 (path T1 FOR UPDATE, cron_tasks).
  - Jul 09 02:36: 2 INSERTs sync del user 99a02318 (services._save_plan_and_track_background).
  - Jul 10 05:35: plan renovado de bb1ffe42 (corr=d2bc0bcc) perdido tras 800s de
    pipeline con banda entregada 0.92 — el INSERT murió, no la generación.

Causa raíz: `save_new_meal_plan_atomic` abría la transacción y DESPUÉS llamaba
`_build_meal_plan_insert_sql`, cuyos pases finalize pre-INSERT (coherence stack
~12s + protein band ~1s + all-4 band closer ~8s; CPU-bound con fuzzy matching)
dejaban la conexión idle-in-transaction ~21s. El pool setea
`SET idle_in_transaction_session_timeout = 15000` por sesión
(MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS, P1-DB-STMT-TIMEOUT en db_core.py) → el
server mataba la sesión antes del INSERT. El propio diseño P1-DB-STMT-TIMEOUT
declara "idle dentro de transacción = siempre un bug/leak" y bendice los
`SET LOCAL` per-transacción como override para el trabajo legítimo.

Fix de dos capas:
  1) INSERT (root fix): `_finalize_plan_data_for_insert` extraído del builder;
     `save_new_meal_plan_atomic` lo ejecuta ANTES de abrir la transacción y pasa
     `skip_plan_data_finalize=True` al builder. Default False conserva el escudo
     central para `save_new_meal_plan_robust` y call sites futuros (que ya
     construyen FUERA de transacción).
  2) FOR UPDATE (T1 merge y demás mutators row-locked, que NO pueden hoistearse
     sin romper la invariante I7): `set_meal_plan_for_update_timeouts` emite
     además `SET LOCAL idle_in_transaction_session_timeout` (knob
     `MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS`, default 60000). 0 ⇒ NO se
     emite el SET (en Postgres 0 = deshabilitado/infinito — nunca queremos eso).
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


def _fn_body(src: str, defline: str) -> str:
    """Cuerpo de una función top-level: desde su `def` hasta el próximo `def` col-0."""
    m = re.search(rf"def {re.escape(defline)}\([\s\S]+?(?=\ndef |\Z)", src)
    assert m, f"def {defline}( no encontrado en db_plans.py"
    return m.group(0)


# ───────────────────────── estructural: extracción del finalize ─────────────────────────

def test_marker_present():
    assert "P0-PERSIST-TXN-IDLE" in _DBP


def test_finalize_helper_defined_with_all_passes():
    """El helper extraído contiene TODOS los pases que antes vivían inline en el
    builder, en el mismo orden (los tests de orden por-pase siguen anclando via
    .index() sobre el archivo completo)."""
    body = _fn_body(_DBP, "_finalize_plan_data_for_insert")
    assert "_ensure_grocery_start_date" in body
    assert "from graph_orchestrator import finalize_plan_data_coherence" in body
    assert "reconcile_protein_band_post_finalize" in body
    assert "reconcile_all_macros_band_post_finalize" in body
    assert "clear_stale_low_band_degraded" in body
    assert "refresh_clinical_band_score_post_finalize" in body


def test_builder_has_skip_param_and_delegates():
    """El builder gana `skip_plan_data_finalize` (default False = escudo central
    intacto) y delega los pases al helper — el import lazy pesado YA NO vive en
    el cuerpo del builder."""
    body = _fn_body(_DBP, "_build_meal_plan_insert_sql")
    assert "skip_plan_data_finalize: bool = False" in body.split("\n")[0] + body.split(")")[0], (
        "la firma del builder debe declarar skip_plan_data_finalize: bool = False"
    )
    assert "_finalize_plan_data_for_insert(data)" in body
    assert "if not skip_plan_data_finalize" in body
    assert "from graph_orchestrator import finalize_plan_data_coherence" not in body, (
        "los pases finalize deben vivir en _finalize_plan_data_for_insert, no inline "
        "en el builder (volverían a correr dentro de la transacción del path atomic)"
    )


def test_atomic_prefinalizes_before_transaction():
    """save_new_meal_plan_atomic ejecuta el finalize ANTES de _run (que abre la
    transacción) y el builder dentro de _run corre con skip=True."""
    body = _fn_body(_DBP, "save_new_meal_plan_atomic")
    i_deepcopy = body.index("copy.deepcopy(insert_data)")
    i_prefin = body.index("_finalize_plan_data_for_insert(safe_data)")
    i_first_run_call = body.index("plan_id, n_cancelled = _run(safe_data)")
    assert i_deepcopy < i_prefin < i_first_run_call, (
        "el finalize debe correr sobre safe_data DESPUÉS del deepcopy y ANTES de "
        "la primera llamada a _run (que abre la transacción)"
    )
    assert "skip_plan_data_finalize=True" in body, (
        "el builder dentro de _run debe saltarse los pases (ya corrieron fuera "
        "de la transacción)"
    )


def test_builder_skip_true_skips_passes_functional(monkeypatch):
    """Funcional: con skip=True el builder NO invoca el helper; con default sí."""
    import db_plans as dp

    calls: list[dict] = []
    monkeypatch.setattr(dp, "_finalize_plan_data_for_insert", lambda data: calls.append(data))

    data = {"plan_data": {"days": [], "generation_status": "partial"}}
    sql, vals = dp._build_meal_plan_insert_sql(dict(data), skip_plan_data_finalize=True)
    assert calls == [], "skip=True no debe invocar el finalize"
    assert "INSERT INTO meal_plans" in sql and len(vals) == 1

    dp._build_meal_plan_insert_sql(dict(data))
    assert len(calls) == 1, "default (skip=False) debe conservar el escudo central"


# ───────────────────────── FOR UPDATE: SET LOCAL idle-in-txn ─────────────────────────

class _StubCursor:
    def __init__(self):
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(sql)


def test_for_update_helper_emits_idle_txn_set(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", "43210")
    from db_plans import set_meal_plan_for_update_timeouts

    cur = _StubCursor()
    set_meal_plan_for_update_timeouts(cur)
    joined = "\n".join(cur.executed).lower()
    assert "set local idle_in_transaction_session_timeout" in joined, (
        "sin el SET LOCAL, un mutator CPU-only legítimo bajo FOR UPDATE (T1 merge "
        "+ finalize parity) muere a los 15s de sesión — chunk 4 de 72c8b965, Jul 06"
    )
    assert "43210ms" in joined, "el valor debe venir del knob, no hardcoded"


def test_for_update_helper_idle_zero_skips_set(monkeypatch):
    """knob=0 ⇒ NO emitir el SET (0 en Postgres = deshabilitado/infinito; el
    fallback correcto es dejar vivo el default de sesión de 15s)."""
    monkeypatch.setenv("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", "0")
    from db_plans import set_meal_plan_for_update_timeouts

    cur = _StubCursor()
    set_meal_plan_for_update_timeouts(cur)
    joined = "\n".join(cur.executed).lower()
    assert "idle_in_transaction_session_timeout" not in joined
    assert "set local lock_timeout" in joined, "los otros dos SET siguen emitiéndose"
