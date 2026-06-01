"""[P2-HYDRATION-GOAL-SSOT · 2026-05-31] El contexto de hidratación que el
chat-agent inyecta al system prompt DEBE reusar la fórmula CANÓNICA de la
meta diaria (`routers/plans.py:_compute_water_goal`), NO reimplementarla.

Bug cerrado (audit del sistema de hidratación 2026-05-31):
    `agent.py:_build_hydration_context` calculaba la meta diaria de vasos
    inline con una fórmula DIVERGENTE de la canónica:
      - 250 ml/vaso  (canónico `_WATER_ML_PER_GLASS = 240`)
      - mapeo de actividad distinto:
          active        → +250 ml   (canónico +500)
          very_active   → +500 ml   (canónico +750)
          athlete/very_high → +0 ml (canónico +750)
          activityLevel ausente/null → +0 ml (canónico → default moderate +250)
    Resultado: el agente afirmaba en su contexto una meta 1-2 vasos
    DISTINTA a la que el usuario ve en el card del Dashboard (que sí usa
    `_compute_water_goal`) y a la que reportan las tools
    `check_hydration_today` / `log_water_glass` (que también la reusan).
    Validado contra data real de prod: 3/8 usuarios con activityLevel=null
    → divergían por +250 ml, y el divisor 240-vs-250 afecta a TODOS.

Fix:
    `_build_hydration_context` ahora delega a `_compute_water_goal(user_id)`
    (import lazy, mismo patrón seguro que `tools.check_hydration_today` —
    cadena de carga routers.plans→agent→tools resuelta en runtime).

Co-located [P3-HYDRATION-CTX-TZ · 2026-05-31]:
    El fallback de fecha cuando el caller no pasa `local_date_str` (path
    non-stream `/api/chat`) caía a UTC del servidor. Para un usuario RD
    (UTC-4) entre las 8 PM y medianoche AST, la fecha UTC ya es "mañana"
    → el agente leía el bucket de mañana (0 vasos). Ahora cae a la fecha
    LOCAL dominicana vía `tools._local_date_str_for_user()` (el MISMO helper
    que usan las tools de hidratación). Misma clase de bug que P1-PROACTIVE-TZ.

Estos son tests parser-based (no ejecutan el grafo del agente ni tocan DB):
anclan el contrato de SSOT en el source. Si alguien re-introduce la fórmula
inline o el fallback a UTC, el test falla loud ANTES de llegar a prod.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"
_TOOLS_PY = _BACKEND_ROOT / "tools.py"


def _hydration_context_body() -> str:
    """Aísla el cuerpo de `_build_hydration_context` en agent.py (hasta el
    siguiente `def`/`async def` de nivel superior)."""
    src = _AGENT_PY.read_text(encoding="utf-8")
    m = re.search(
        r"def _build_hydration_context\(.*?(?=\n(?:async\s+)?def\s)",
        src,
        re.DOTALL,
    )
    assert m, (
        "No se pudo aislar `_build_hydration_context` en agent.py — "
        "¿fue renombrada o movida?"
    )
    return m.group(0)


# ---------------------------------------------------------------------------
# 1. La meta diaria delega al SSOT canónico (no reimplementación inline)
# ---------------------------------------------------------------------------
def test_hydration_context_delegates_goal_to_canonical():
    """`_build_hydration_context` debe invocar `_compute_water_goal` — la
    MISMA función que usa el endpoint /water-intake (card del Dashboard) y
    las tools del agente."""
    body = _hydration_context_body()
    assert "_compute_water_goal" in body, (
        "`_build_hydration_context` no usa `_compute_water_goal` — está "
        "reimplementando la meta diaria inline y diverge del card del "
        "Dashboard (bug P2-HYDRATION-GOAL-SSOT)."
    )
    # Anchor del marker en el source (P3-CLAUDEMD tooltip-anchor convention):
    # un rename de la función canónica debe romper este test antes que prod.
    assert "P2-HYDRATION-GOAL-SSOT" in body, (
        "Falta el anchor `P2-HYDRATION-GOAL-SSOT` en el cuerpo de la función "
        "— necesario para que un refactor del SSOT falle el test."
    )


def test_hydration_context_does_not_reimplement_goal_formula():
    """Negative guard: el cuerpo NO debe contener la fórmula inline
    divergente (divisor 250, factor 35 ml/kg, bonus de actividad inline)."""
    body = _hydration_context_body()
    # Divisor divergente (canónico = 240, vía _WATER_ML_PER_GLASS).
    assert "round(ml / 250)" not in body and "/ 250)" not in body, (
        "`_build_hydration_context` reimplementa la meta con 250 ml/vaso "
        "(canónico = 240). Debe delegar a `_compute_water_goal`."
    )
    # Factor ml/kg inline.
    assert "* 35.0" not in body and "weight_kg * 35" not in body, (
        "`_build_hydration_context` reimplementa la fórmula 35 ml/kg inline. "
        "Debe delegar a `_compute_water_goal`."
    )
    # Bonus de actividad inline (los números mágicos de los dos branches viejos).
    assert not re.search(r"ml\s*\+=\s*(?:250|500|750)", body), (
        "`_build_hydration_context` aún aplica el bonus de actividad inline "
        "(ml += 250/500/750). Debe venir de `_compute_water_goal`."
    )


# ---------------------------------------------------------------------------
# 2. Fallback de fecha: local dominicana (UTC-4), NO UTC  [P3-HYDRATION-CTX-TZ]
# ---------------------------------------------------------------------------
def test_hydration_context_fallback_date_is_do_local_not_utc():
    """Cuando el caller no pasa `local_date_str`, el fallback debe usar la
    fecha LOCAL dominicana (`_local_date_str_for_user`, UTC-4), NO UTC."""
    body = _hydration_context_body()
    assert "_local_date_str_for_user" in body, (
        "El fallback de fecha no usa `_local_date_str_for_user` (UTC-4) — "
        "si cae a UTC, un usuario RD de noche lee el bucket de mañana "
        "(bug P3-HYDRATION-CTX-TZ)."
    )
    # Negative: el fallback ya no debe derivar la fecha desde UTC directo.
    assert "_dt.now(_tz.utc).strftime" not in body, (
        "El fallback aún deriva la fecha desde UTC (`_dt.now(_tz.utc)`). "
        "Debe usar la fecha local dominicana via `_local_date_str_for_user`."
    )
    assert "P3-HYDRATION-CTX-TZ" in body, (
        "Falta el anchor `P3-HYDRATION-CTX-TZ` en el cuerpo de la función."
    )


def test_local_date_helper_is_do_offset():
    """Sanity del SSOT que reusamos: `_local_date_str_for_user` en tools.py
    debe calcular la fecha en UTC-4 (Atlantic Standard Time, RD)."""
    src = _TOOLS_PY.read_text(encoding="utf-8")
    m = re.search(
        r"def _local_date_str_for_user\(.*?(?=\n(?:async\s+)?def\s|\n@tool)",
        src,
        re.DOTALL,
    )
    assert m, "No se encontró `_local_date_str_for_user` en tools.py."
    helper = m.group(0)
    assert "timedelta(hours=4)" in helper, (
        "`_local_date_str_for_user` debe restar 4 horas (UTC-4 = RD). Si el "
        "offset cambia, revisar la consistencia con el resto del sistema."
    )


# ---------------------------------------------------------------------------
# 3. Sanity: la fuente canónica sigue divergiendo del valor viejo (240≠250)
# ---------------------------------------------------------------------------
def test_canonical_goal_uses_240_ml_per_glass():
    """Confirma que la divergencia que cerramos es real: el canónico usa
    240 ml/vaso, NO los 250 que tenía la reimplementación inline. Si alguien
    alinea el canónico a 250, este test obliga a re-evaluar la decisión."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    assert "_WATER_ML_PER_GLASS = 240" in src, (
        "El canónico `_WATER_ML_PER_GLASS` ya no es 240 — revisar si la "
        "reimplementación inline que removimos seguía siendo divergente."
    )
    # La función canónica existe y es la que reusamos.
    assert "def _compute_water_goal(user_id: str) -> dict:" in src
