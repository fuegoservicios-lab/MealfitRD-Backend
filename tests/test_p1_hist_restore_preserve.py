"""[P1-HIST-RESTORE-PRESERVE · 2026-07-12] "Reactivar" ya no destruye el plan activo.

Vivo (owner, 08:43Z): reactivó "Sabor y Fuerza Criolla" desde el Historial y su plan
activo "activo 1" (rename manual + todos los regens del día) desapareció sin rastro —
el restore (P0-HIST-1) sobrescribe el target (nombre + plan_data + 6 columnas) y NO
había backup en ninguna parte (meal_plans_audit: 0 filas para ese plan).

Fix: paso 3b-bis dentro de la MISMA transacción — antes del overwrite, el estado actual
del target se INSERTA como fila nueva del Historial (copia fiel: 6 columnas + plan_data +
profile_embedding + created_at ORIGINAL para que se ordene en su fecha real) con markers
forenses `_archived_by_restore_at`/`_archived_by_restore_to`. "Reactivar" pasa de
overwrite destructivo a SWAP seguro. Contrato parser sobre api_restore_plan.
tooltip-anchor: P1-HIST-RESTORE-PRESERVE
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PL = f.read()


def _restore_body():
    i = _PL.find("def api_restore_plan(")
    assert i != -1, "endpoint /restore desapareció"
    j = _PL.find("\n@router.", i)
    return _PL[i:j if j != -1 else i + 40000]


def test_archive_insert_present_with_forensic_markers():
    body = _restore_body()
    i = body.find("P1-HIST-RESTORE-PRESERVE")
    assert i != -1, "el paso 3b-bis (archivar antes de sobrescribir) desapareció"
    win = body[i:i + 2200]
    assert "INSERT INTO meal_plans" in win
    assert "_archived_by_restore_at" in win and "_archived_by_restore_to" in win, \
        "markers forenses del swap (quién archivó a quién)"
    assert re.search(r"profile_embedding,\s*created_at", win), \
        "copia FIEL: embedding + created_at ORIGINAL (se ordena en su fecha real)"


def test_archive_happens_before_overwrite_inside_txn():
    body = _restore_body()
    i_ins = body.find("P1-HIST-RESTORE-PRESERVE")
    i_upd = body.find("UPDATE meal_plans\n                            SET plan_data")
    i_txn = body.find("with conn.transaction():")
    assert -1 not in (i_ins, i_upd, i_txn)
    assert i_txn < i_ins < i_upd, \
        "el INSERT-copia vive DENTRO de la txn y ANTES del overwrite (atómico: o swap completo o nada)"


def test_archive_skipped_on_noop():
    """El no-op idempotente (target == source) no debe duplicar filas."""
    body = _restore_body()
    i_noop = body.find("is_noop = True")
    i_ins = body.find("P1-HIST-RESTORE-PRESERVE")
    assert i_noop != -1 and i_ins != -1
    assert i_noop < i_ins, \
        "el INSERT vive en la rama else (post no-op check) — reactivar el plan ya activo no clona filas"


def test_insert_ownership_filtered():
    body = _restore_body()
    i = body.find("P1-HIST-RESTORE-PRESERVE")
    win = body[i:i + 2200]
    assert "WHERE id = %s AND user_id = %s" in win, "I2: la copia también filtra por user_id"
