"""[P2-HIST-RESTORE-STALE-STATE · 2026-07-12] Reactivar un plan no arrastra estado efímero.

Audit del owner ("¿Reactivar plan está al 100%?"): el endpoint /restore (P0-HIST-1) limpiaba
los flags de fallo del source (`_user_action_required`, `_recovery_exhausted_chunks`) pero
features POSTERIORES a ese hardening añadieron estado efímero que el snapshot arrastraba:

  1. `_day_regen_inflight` / `_meal_regen_inflight` (P1-SWAP-REGEN-RESUME, 2026-07-12):
     señales de un regen que murió hace días — el cliente las ignora por edad, pero no
     deben viajar con el restore.
  2. `is_restocked` / `restocked_at_iso` / `restocked_items`: el ciclo de compra del plan
     ORIGINAL. Stale por definición al reactivar — arrastrarlo deja las listas quincenal/
     mensual VACÍAS ("ya compraste todo el ciclo") aunque la Nevera actual no tenga esos
     items (clase del incidente vivo 2026-07-12: 3 de 4 listas en 0 + costo RD$0).

Contrato (parser sobre api_restore_plan): los 5 keys se pop-ean del plan_data enriquecido
ANTES del UPDATE, junto a los 2 originales. tooltip-anchor: P2-HIST-RESTORE-STALE-STATE
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
    return _PL[i:j if j != -1 else i + 20000]


def test_stale_keys_popped_before_update():
    body = _restore_body()
    anchor = body.find("P2-HIST-RESTORE-STALE-STATE")
    assert anchor != -1, "bloque de limpieza de estado efímero desapareció del restore"
    win = body[anchor:anchor + 1400]
    for k in ("_day_regen_inflight", "_meal_regen_inflight",
              "is_restocked", "restocked_at_iso", "restocked_items"):
        assert f'"{k}"' in win, f"key efímera '{k}' ya no se limpia en el restore"
    assert re.search(r"enriched_pd\.pop\(_stale_k, None\)", win), \
        "el pop debe aplicarse sobre enriched_pd (el dict que viaja al UPDATE)"


def test_original_failure_flags_still_cleaned():
    body = _restore_body()
    assert 'enriched_pd.pop("_user_action_required", None)' in body
    assert 'enriched_pd.pop("_recovery_exhausted_chunks", None)' in body, \
        "la limpieza original P1-HIST-AUDIT-3 debe seguir intacta"


def test_cleanup_happens_before_transaction():
    body = _restore_body()
    i_clean = body.find("P2-HIST-RESTORE-STALE-STATE")
    i_txn = body.find("with conn.transaction():")
    assert i_clean != -1 and i_txn != -1 and i_clean < i_txn, \
        "la limpieza vive en el enriquecimiento pre-UPDATE (no dentro del txn, no post-commit)"
