"""[P1-HIST-RESTORE-MOVE · 2026-07-12] "Reactivar" es MOVER, no copiar.

Vivo (owner, 09:05Z, tras 2 ping-pongs de restore): "siento que se están duplicando cada
vez que reactivo un plan" — correcto: el swap P1-HIST-RESTORE-PRESERVE archivaba el plan
activo (3b-bis ✓) pero dejaba VIVA la fila source → el plan restaurado quedaba duplicado
(activo + su gemelo archivado) y el Historial crecía +1 por swap sin límite.

Fix: paso 3d en la misma txn — DELETE de la fila source tras el overwrite. Su contenido ya
vive en el target (3c); el estado anterior del target ya vive en la copia (3b-bis) → nada
se pierde y el conteo queda constante. FKs verificadas: plan_chunk_queue CASCADE (chunks ya
cancelados en 3a-bis), chunk_deferrals/chunk_lesson_telemetry/plan_chunk_metrics SET NULL,
meal_plans_audit sin FK. tooltip-anchor: P1-HIST-RESTORE-MOVE
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PL = f.read()


def _restore_body():
    i = _PL.find("def api_restore_plan(")
    assert i != -1, "endpoint /restore desapareció"
    j = _PL.find("\n@router.", i)
    return _PL[i:j if j != -1 else i + 40000]


def test_source_row_deleted_after_overwrite():
    body = _restore_body()
    i = body.find("P1-HIST-RESTORE-MOVE")
    assert i != -1, "el paso 3d (consumir la fila source) desapareció"
    win = body[i:i + 1600]
    assert "DELETE FROM meal_plans WHERE id = %s AND user_id = %s" in win, \
        "el DELETE del source con filtro de ownership (I2)"


def test_order_archive_then_overwrite_then_delete():
    """El orden es load-bearing: copia (3b-bis) → overwrite (3c) → delete source (3d),
    todo dentro de la txn — si algo falla, ROLLBACK deja las 2 filas intactas."""
    body = _restore_body()
    i_copy = body.find("P1-HIST-RESTORE-PRESERVE")
    i_upd = body.find("UPDATE meal_plans\n                            SET plan_data")
    i_del = body.find("P1-HIST-RESTORE-MOVE")
    i_txn = body.find("with conn.transaction():")
    assert -1 not in (i_copy, i_upd, i_del, i_txn)
    assert i_txn < i_copy < i_upd < i_del, \
        "orden 3b-bis → 3c → 3d dentro de la transacción"


def test_noop_branch_does_not_delete():
    """Reactivar el plan YA activo (no-op idempotente) no debe borrar nada."""
    body = _restore_body()
    i_noop = body.find("is_noop = True")
    i_del = body.find("P1-HIST-RESTORE-MOVE")
    assert i_noop != -1 and i_del != -1 and i_noop < i_del, \
        "el DELETE vive en la rama else (el no-op sale antes)"
