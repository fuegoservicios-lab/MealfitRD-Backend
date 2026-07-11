"""[P1-RENEWAL-CHECKIN-FRESH · 2026-07-11] El check-in "Antes de tu nuevo ciclo" NO aparece
al renovar un plan RECIENTE (<4 días).

Pedido del owner (2026-07-11): "si estoy renovando el plan el mismo día que creé un plan
nuevo tengo las mismas características físicas y solo quiero renovar para testear o
variedad" — el calibrador de metabolismo no tiene delta de peso que medir en <4 días.

Contrato (parser-based sobre el frontend, patrón test_p1_new_a):
1. useRegeneratePlan.js pasa `plan_created_at` en el state del navigate('/plan').
2. Plan.jsx gatea checkinPending con la edad del plan (umbral 4 días).
3. Sin fecha en el state (entry-points legacy) → comportamiento previo (mostrar check-in).

tooltip-anchor: P1-RENEWAL-CHECKIN-FRESH
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PLAN = (_ROOT / "frontend" / "src" / "pages" / "Plan.jsx").read_text(encoding="utf-8")
_HOOK = (_ROOT / "frontend" / "src" / "hooks" / "useRegeneratePlan.js").read_text(encoding="utf-8")


def test_hook_passes_plan_created_at():
    i = _HOOK.find("navigate('/plan'")
    assert i != -1, "navigate('/plan') desapareció de useRegeneratePlan"
    assert "plan_created_at:" in _HOOK[i:i + 800], (
        "la renovación debe pasar la fecha del plan renovado para que Plan.jsx "
        "pueda saltar el check-in en planes recientes"
    )


def test_plan_gates_checkin_by_age():
    i = _PLAN.find("const [checkinPending, setCheckinPending] = useState(")
    assert i != -1, "inicializador de checkinPending desapareció de Plan.jsx"
    win = _PLAN[i:i + 2500]
    assert "plan_created_at" in win and "_freshPlan" in win, (
        "el check-in debe saltarse cuando el plan renovado es reciente "
        "(mismo día = mismas características físicas)"
    )
    assert "_ageDays < 4" in win, "umbral documentado: 4 días"
    assert "&& !_freshPlan" in win


def test_legacy_without_date_keeps_checkin():
    i = _PLAN.find("const [checkinPending, setCheckinPending] = useState(")
    win = _PLAN[i:i + 2500]
    assert "if (_pc)" in win, (
        "sin plan_created_at en el state el check-in se conserva (fail-open al "
        "comportamiento previo, no al skip)"
    )


def test_marker_anchored():
    assert _PLAN.count("P1-RENEWAL-CHECKIN-FRESH") >= 1
    assert _HOOK.count("P1-RENEWAL-CHECKIN-FRESH") >= 1
