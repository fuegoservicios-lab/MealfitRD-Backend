"""[P0-PERSIST-TXN-IDLE-ATOMIC · 2026-07-10] Red de idle-in-transaction para los mutators
de `update_plan_data_atomic` + hoist del sodium autofix fuera de la transacción del regen-day.

Incidente en vivo (2026-07-10 15:19, /regenerate-day, plan 9bce8fff): los pases añadidos al
`_day_mutator` (sodium autofix + panel re-eval + chip clear) sumados a los pre-existentes
(micros recompute + rebuild inline de listas) dejaron la conexión idle-in-transaction >15s
(SET de sesión del pool, `MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS` en db_core) →
`terminating connection due to idle-in-transaction timeout` → HTTP 500 con ~3.5 min de
swaps del usuario PERDIDOS (rollback). Misma clase que P0-PERSIST-TXN-IDLE (INSERT);
mismo remedio que su T1 (SET LOCAL con presupuesto amplio por-transacción).

El contrato P2-MUTATOR-PURITY del docstring ya advertía "CPU-only… resuélvelos ANTES";
este test ancla la red (defensa) y el hoist (cumplimiento) para que no se re-introduzca.

tooltip-anchor: P0-PERSIST-TXN-IDLE-ATOMIC
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_DBP = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _fn_body(src: str, name: str) -> str:
    m = re.search(rf"def {re.escape(name)}\([\s\S]+?(?=\ndef |\Z)", src)
    assert m, f"def {name}( no encontrado"
    return m.group(0)


def test_atomic_sets_local_idle_txn_timeout():
    body = _fn_body(_DBP, "update_plan_data_atomic")
    assert "P0-PERSIST-TXN-IDLE-ATOMIC" in body, (
        "P0-PERSIST-TXN-IDLE-ATOMIC: update_plan_data_atomic perdió la red de idle-in-txn — "
        "cualquier mutator con tramos CPU >15s (regen-day: micros+listas) vuelve a morir con "
        "`terminating connection due to idle-in-transaction timeout` y el usuario pierde el update."
    )
    assert "SET LOCAL idle_in_transaction_session_timeout" in body
    assert '_env_int("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", 60000)' in body, (
        "el presupuesto debe compartir el knob del T1 del chunk (un solo dial operacional)"
    )
    # la red se instala ANTES de ejecutar el mutator (que es donde vive el riesgo)
    assert body.index("idle_in_transaction_session_timeout") < body.index("result = mutator(current)")


def test_regen_day_sodium_autofix_hoisted_out_of_txn():
    i_sod = _PLANS.find("P2-REGEN-DAY-SODIUM-AUTOFIX · 2026-07-10] El regen-day")
    assert i_sod > 0, "bloque del sodium autofix del regen-day desapareció"
    i_mutator = _PLANS.find("def _day_mutator(", i_sod)
    assert i_mutator > 0, "def _day_mutator no encontrado tras el bloque de sodio"
    blk = _PLANS[i_sod:i_mutator]
    assert "_day_sodium_autofix" in blk, (
        "P0-PERSIST-TXN-IDLE-ATOMIC: el sodium autofix debe correr ANTES de def _day_mutator "
        "(fuera del FOR UPDATE — P2-MUTATOR-PURITY); dentro contribuyó al idle-kill de las 15:19."
    )
    # y NO debe haber una segunda invocación dentro del mutator
    i_mut_end = _PLANS.find("result = update_plan_data_atomic(plan_id, _day_mutator", i_mutator)
    assert "_day_sodium_autofix" not in _PLANS[i_mutator:i_mut_end], (
        "no re-introducir el autofix DENTRO del mutator (corre en la transacción)"
    )
