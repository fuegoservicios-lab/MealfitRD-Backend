"""[P2-LOADING-ETA-57 · 2026-07-06] Página de carga: ETA 5-7 min + auditoría producción.

Pedidos del owner sobre /plan ("Diseñando tu plan"):
1. "Puedes salir si quieres" debe ser VERDAD: cerrar y volver → si sigue
   generando, de vuelta a la pantalla de carga; si terminó, al dashboard.
   VEREDICTO AUDIT: ya implementado end-to-end (PendingPipelineRecovery en
   App.jsx: boot-check contra el KV backend aun sin flag local — cross-device;
   polling 10s consciente de visibilidad; ack idempotente; exit tras 6 fallos).
2. ETA "4 y 5 minutos" → "5 y 7" (consistente con prod: la renovación
   monitoreada de hoy tomó 5m49s con un retry quirúrgico — normal, no excepción).
   Umbral "ya casi" 6→8 min (no contradecir el estimado nuevo).
3. Gap de producción encontrado: en modo recovery el contador "Transcurrido"
   reiniciaba en 0:00 (mentía). Ahora arranca del started_at real del flag.
"""
import os
import re

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")


def _read(*parts) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


_PLAN = _read(_FRONTEND, "src", "pages", "Plan.jsx")
_APP = _read(_FRONTEND, "src", "App.jsx")
_REC = _read(_FRONTEND, "src", "components", "PendingPipelineRecovery.jsx")


def test_eta_copy_is_5_7():
    assert "entre 5 y 7 minutos" in _PLAN, "ETA inicial = 5-7 (pedido del owner)"
    assert "estimado 5-7 minutos" in _PLAN, "tracking = 5-7"
    assert "4 y 5 minutos" not in _PLAN and "4-5 minutos" not in _PLAN, "copy viejo fuera"


def test_ya_casi_threshold_beyond_estimate():
    i = _PLAN.index("const timeMessage")
    win = _PLAN[i:i + 800]
    assert "8 * 60" in win, "'ya casi terminamos' arranca DESPUÉS del estimado (8 min), no dentro (6)"
    assert "12 * 60" in win


def test_elapsed_continuity_across_reentry():
    assert "P2-LOADING-ETA-57" in _PLAN
    i = _PLAN.index("const startTimeRef")
    win = _PLAN[i:i + 600]
    assert "mealfit_plan_in_progress" in win and "started_at" in win, (
        "en modo recovery el contador arranca del inicio REAL del pipeline (no 0:00)"
    )
    assert "6 * 3600 * 1000" in win, "sanity: started_at más viejo que el cap del recovery → hoy"


def test_reentry_redirects_already_wired():
    """El 'Puedes salir si quieres' está respaldado end-to-end (audit)."""
    assert "<PendingPipelineRecovery />" in _APP, "recovery montado global"
    assert "status.status === 'generating'" in _REC and "navigate('/plan'" in _REC, (
        "generando + fuera de /plan → de vuelta a la pantalla de carga"
    )
    assert "status.status === 'complete'" in _REC and "navigate('/dashboard'" in _REC, (
        "completo → dashboard con el plan nuevo"
    )
    assert re.search(r"pending-status/ack", _REC), "ack idempotente (sin loops de redirect)"
    assert "visibilitychange" in _REC, "resume inmediato al volver de suspend"
