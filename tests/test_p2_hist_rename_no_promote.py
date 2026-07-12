"""[P2-HIST-RENAME-NO-PROMOTE · 2026-07-12] Renombrar un plan archivado no lo promueve a activo.

Vivo (owner, 09:1xZ): renombró 2 planes ARCHIVADOS ("amigo", "amo de la tierra 2") para
trackear sus pruebas de restore, y el sello de `_plan_modified_at` del rename (P1-HIST-
AUDIT-2) los subió por encima del plan activo real en el resolver GREATEST — "amigo" se
volvió el "latest" (= target de restores/recalcs + card "PLAN ACTIVO" del Historial)
mientras el dashboard mostraba otro plan.

Fix: el rename sella `_plan_modified_at` SOLO cuando el plan renombrado YA es el latest
(intención original de AUDIT-2: que el activo renombrado no caiga en el sort). Renombrar
un archivado = name-only (top-level + plan_data.name), sin mover su posición temporal.
tooltip-anchor: P2-HIST-RENAME-NO-PROMOTE
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PL = f.read()


def _rename_body():
    i = _PL.find("def api_rename_plan(")
    assert i != -1, "endpoint rename desapareció"
    j = _PL.find("\n@router.", i)
    return _PL[i:j if j != -1 else i + 20000]


def test_latest_check_before_update():
    body = _rename_body()
    i = body.find("P2-HIST-RENAME-NO-PROMOTE")
    assert i != -1, "el guard de promoción desapareció del rename"
    win = body[i:]
    assert "_renaming_active" in win
    assert re.search(r"ORDER BY GREATEST\(\s*created_at", win), \
        "el check de latest usa el MISMO sort SSOT que restore/History (GREATEST)"


def test_two_branches_seal_vs_name_only():
    body = _rename_body()
    win = body[body.find("P2-HIST-RENAME-NO-PROMOTE"):]
    # Rama activa: doble jsonb_set (name + sello) — contrato AUDIT-2 intacto.
    assert win.count("'{_plan_modified_at}', to_jsonb(%s::text), true") == 1, \
        "el sello existe SOLO en la rama del plan activo"
    # Rama archivada: jsonb_set simple de name, sin sello.
    _else = win[win.find("else:"):]
    assert "'{name}', to_jsonb(%s::text), true" in _else
    assert "_plan_modified_at" not in _else.split("RETURNING id")[0], \
        "renombrar un archivado NO estampa _plan_modified_at (no se promueve)"


def test_both_branches_ownership_filtered():
    body = _rename_body()
    win = body[body.find("P2-HIST-RENAME-NO-PROMOTE"):]
    assert win.count("WHERE id = %s AND user_id = %s") >= 2, \
        "I2 en ambas ramas del UPDATE"
