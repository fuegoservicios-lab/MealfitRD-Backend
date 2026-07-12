"""[P2-HIST-RESTORE-ROW-UID · 2026-07-12] "Reactivar este Plan" estaba roto: projection sin user_id.

Vivo (owner, 08:35Z): toast rojo "No se pudo restaurar el plan (sesión inválida)" + toast verde
"¡Plan reactivado!" SIMULTÁNEOS, y nginx sin ningún POST /api/plans/restore. Cadena:

  1. El guard de ownership del frontend (P1-NEW-8, `restorePlanFromHistory`) rechaza rows sin
     `user_id` (defensa contra filas corruptas) → pero la projection slim de `/history-list`
     (P1-HIST-AUDIT-4) NO incluía `user_id` → el guard abortaba SIEMPRE, para todo usuario,
     sin disparar el request.
  2. `History.jsx::handleRestoreConfirm` mostraba éxito + navigate incondicionalmente porque
     el guard RETORNA `{success:false}` (no lanza) — toasts contradictorios.

Fix bilateral: (a) `/history-list` incluye `mp.user_id` (cero riesgo IDOR: el WHERE ya filtra
por verified_user_id — el valor es siempre el del dueño); (b) el handler respeta el resultado.
tooltip-anchor: P2-HIST-RESTORE-ROW-UID
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PL = f.read()
with open(os.path.join(_ROOT, "frontend", "src", "pages", "History.jsx"), encoding="utf-8") as f:
    _HIST = f.read()
with open(os.path.join(_ROOT, "frontend", "src", "context", "AssessmentContext.jsx"),
          encoding="utf-8") as f:
    _CTX = f.read()


def _history_list_body():
    i = _PL.find("def api_plans_history_list(")
    assert i != -1, "endpoint /history-list desapareció"
    j = _PL.find("\n@router.", i)
    return _PL[i:j if j != -1 else i + 40000]


def test_projection_includes_user_id():
    body = _history_list_body()
    assert re.search(r"mp\.user_id::text AS user_id", body), \
        "el SELECT del history-list debe proyectar user_id (contrato del guard P1-NEW-8)"
    assert '"user_id": row.get("user_id")' in body, \
        "el dict de salida debe incluir user_id (el SELECT solo no basta)"


def test_frontend_handler_respects_result():
    i = _HIST.find("P2-HIST-RESTORE-ROW-UID")
    assert i != -1, "el handler de restore ya no chequea el resultado"
    win = _HIST[i:i + 1200]
    assert "result.success === false" in win, "fallo controlado → toast de error, no éxito"
    assert "return;" in win, "en fallo NO se navega ni se muestra '¡Plan reactivado!'"
    assert "ownership_mismatch" in win, "el caso del guard se explica al usuario"


def test_guard_contract_unchanged():
    """El guard P1-NEW-8 sigue rechazando rows sin user_id — el fix fue DARLE el dato,
    no debilitar la defensa."""
    assert "!pastPlanRow.user_id || pastPlanRow.user_id !== _currentUid" in _CTX, \
        "el guard de ownership del restore no debe debilitarse (anti fila corrupta/ajena)"
