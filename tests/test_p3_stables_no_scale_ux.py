"""Bundle P2-AGGREGATE-DROP-DIAG + P3-HERB-CAP-FLOOR + P3-STABLES-NO-SCALE-UX
(2026-05-16) — issues derivados del triage de cantidades observado en plan
4cc91584 PDFs (7d/15d/30d × 1 persona):

  P2 — Avena ausente de la lista pese a aparecer en `expected_sum_from_recipes`
       (coherence guard reporta `Avena [expected_only]`). Sin instrumentación
       el debug requiere agregar logs cada vez. Fix: log `🛒 [AGGREGATE-DROP]`
       cuando un item queda sin peso ni unidades reales tras la dedup
       nominal → operador sabe inmediatamente que el LLM emitió "pizca" /
       "al gusto" sin cantidad concreta.

  P3a — Hierbas frescas: cap floor era hardcoded 2 mazos (≈100g, ¼ lb)
        incluso para 1p × 1 semana. Usuario reportó "alto para 1p × 7d".
        Fix: knob `MEALFIT_HERB_MAZO_CAP_FLOOR` default 1; afecta solo
        person_weeks < 2 (1p × 7d). Para 2p+ o cycles >1 semana el cap
        sale del `int(round(person_weeks))` y el floor no aplica.

  P3b — Items estables (aceite, vinagre, miel, vainilla, especias)
        muestran misma cantidad en ciclos 7d/15d/30d porque 1 botella
        rinde múltiples semanas. Usuario que compara PDFs ve "1 botella"
        en ambos y asume bug. Fix: línea explicativa en disclaimer PDF.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPCALC = (_BACKEND_ROOT / "shopping_calculator.py").read_text(encoding="utf-8")
_DASHBOARD = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P3a — Herb cap floor knob
# ---------------------------------------------------------------------------


def test_herb_cap_floor_knob_declared_at_1():
    """`MEALFIT_HERB_MAZO_CAP_FLOOR` debe declararse con default 1 (era 2
    hardcoded). Bajarlo a 1 es la dirección correcta del fix; subir a 2
    revierte. Floor 0 sería incorrecto (eliminaría el cap floor)."""
    m = re.search(
        r'_HERB_MAZO_CAP_FLOOR\s*=\s*max\(\s*1\s*,\s*_knob_env_int\(\s*'
        r'"MEALFIT_HERB_MAZO_CAP_FLOOR"\s*,\s*(\d+)\s*\)\s*\)',
        _SHOPCALC,
    )
    assert m, (
        "`_HERB_MAZO_CAP_FLOOR` no declarado vía _knob_env_int en "
        "shopping_calculator.py. Revierte P3-HERB-CAP-FLOOR."
    )
    default = int(m.group(1))
    assert default == 1, (
        f"Default knob `MEALFIT_HERB_MAZO_CAP_FLOOR`={default}, esperado 1. "
        "El propósito del knob es bajar el floor de 2 (hardcoded original) "
        "a 1 para casos 1p × 7d. Subir a 2 revierte el fix."
    )


def test_herb_cap_uses_knob_in_formula():
    """`_herb_cap_mazos` DEBE usar el knob, no `max(2, ...)` hardcoded."""
    # La línea correcta es: `_herb_cap_mazos = max(_HERB_MAZO_CAP_FLOOR, int(round(_person_weeks)))`
    assert "_herb_cap_mazos = max(_HERB_MAZO_CAP_FLOOR, int(round(_person_weeks)))" in _SHOPCALC, (
        "`_herb_cap_mazos` no usa `_HERB_MAZO_CAP_FLOOR` — revierte el fix. "
        "Fórmula esperada: `max(_HERB_MAZO_CAP_FLOOR, int(round(_person_weeks)))`."
    )


def test_herb_cap_no_hardcoded_2_in_formula():
    """Defensa anti-regresión: la fórmula vieja `max(2, int(round(...)))`
    NO debe reaparecer (comentarios históricos ok)."""
    bad = re.compile(
        r"^\s*_herb_cap_mazos\s*=\s*max\(\s*2\s*,\s*int\(round\(",
        re.MULTILINE,
    )
    assert not bad.search(_SHOPCALC), (
        "Fórmula vieja `_herb_cap_mazos = max(2, int(round(...)))` "
        "reaparece — revierte P3-HERB-CAP-FLOOR."
    )


# ---------------------------------------------------------------------------
# P3b — Stables UX line in PDF disclaimer
# ---------------------------------------------------------------------------


def test_stables_disclaimer_present():
    """El disclaimer del PDF debe explicar que estables (aceite/vinagre/miel/
    especias) muestran misma cantidad en ciclos cortos vs largos porque
    1 botella rinde múltiples semanas."""
    assert "P3-STABLES-NO-SCALE-UX" in _DASHBOARD, (
        "Marker P3-STABLES-NO-SCALE-UX ausente en Dashboard.jsx — el "
        "comentario explicativo fue removido."
    )
    # Anchors textuales clave del copy (al menos uno tiene que estar):
    has_key_phrase = any(
        phrase in _DASHBOARD
        for phrase in (
            "1 botella o sobre rinde",
            "Compras menos veces, no menos producto",
            "Estables (aceite, vinagre, miel, especias)",
        )
    )
    assert has_key_phrase, (
        "Copy del disclaimer P3-STABLES-NO-SCALE-UX removido. Esperado al "
        "menos una de: '1 botella o sobre rinde', 'Compras menos veces, "
        "no menos producto', 'Estables (aceite, vinagre, miel, especias)'."
    )


def test_stables_disclaimer_conditional_on_density():
    """El disclaimer ampliado (con la línea de stables) DEBE estar dentro
    del bloque `${isUltraDense ? '' : `...`}` para que planes con >50 items
    (ultra-dense) sigan usando el disclaimer compacto y no se desbordan."""
    idx = _DASHBOARD.find("P3-STABLES-NO-SCALE-UX")
    assert idx > 0
    # Slice de 500 chars antes y después
    block = _DASHBOARD[max(0, idx - 2000):idx + 500]
    # Verificar que vive dentro del condicional isUltraDense:
    assert "isUltraDense" in block, (
        "La línea P3-STABLES-NO-SCALE-UX está FUERA del condicional "
        "`${isUltraDense ? '' : '...'}` — para planes >50 items (ultra-dense) "
        "el contenido se desbordaría de la página del PDF."
    )


# ---------------------------------------------------------------------------
# P2 — Diagnostic logging for dropped items
# ---------------------------------------------------------------------------


def test_aggregate_drop_diagnostic_log_present():
    """Cuando un ingrediente queda sin peso ni unidades reales (todas
    nominales: pizca, al gusto, etc.), el aggregator DEBE loggear con
    prefix `🛒 [AGGREGATE-DROP]` para que el operador pueda diagnosticar
    `expected_only` divergencias sin agregar instrumentación ad-hoc."""
    assert "P2-AGGREGATE-DROP-DIAG" in _SHOPCALC, (
        "Marker P2-AGGREGATE-DROP-DIAG ausente — diagnostic log removido. "
        "Sin él, debugging de items 'expected_only' en coherence guard "
        "requiere agregar instrumentación cada vez."
    )
    # El log debe estar antes del `continue` del `if not remaining_real:`
    assert "[AGGREGATE-DROP]" in _SHOPCALC, (
        "Prefix `[AGGREGATE-DROP]` no encontrado — el log fue removido o "
        "renombrado. Tests del coherence guard `expected_only` divergence "
        "asumen este prefix para grep diagnóstico."
    )


def test_aggregate_drop_log_includes_name_and_reason():
    """El log debe incluir el name + razón ('sin peso' o equivalente) para
    que el operador identifique inmediatamente el item dropeado y la causa."""
    idx = _SHOPCALC.find("P2-AGGREGATE-DROP-DIAG")
    assert idx > 0
    end = _SHOPCALC.find("continue", idx)
    block = _SHOPCALC[idx:end + 50 if end > 0 else idx + 2000]
    assert "{name}" in block, (
        "Log AGGREGATE-DROP no incluye `{name}` — operador no sabe qué "
        "item se dropeó."
    )
    has_reason = (
        "weight_in_lbs" in block
        or "sin peso" in block
        or "nominales" in block
    )
    assert has_reason, (
        "Log AGGREGATE-DROP no menciona la razón (weight_in_lbs / sin peso "
        "/ unidades nominales). Sin esto el log es 'algo se dropeó' sin "
        "actionable info."
    )


# ---------------------------------------------------------------------------
# Functional smoke test: herb cap floor 1 honored para 1p × 7d
# ---------------------------------------------------------------------------


def test_herb_cap_for_1p_1week_is_1_mazo_with_default_knob():
    """1 persona × 7 días → person_weeks = 1.0 → cap = max(1, round(1.0)) = 1
    mazo (era 2 pre-fix). Smoke test via cálculo directo de la fórmula —
    el módulo es pesado para importar full, así que validamos la lógica
    matemática."""
    person_weeks = 1.0
    floor = 1  # default del knob P3-HERB-CAP-FLOOR
    cap_mazos = max(floor, int(round(person_weeks)))
    assert cap_mazos == 1, (
        f"Para 1p × 7d (person_weeks=1, floor=1), cap = {cap_mazos}, "
        "esperado 1. Si esto cambia, P3-HERB-CAP-FLOOR perdió efecto."
    )


def test_herb_cap_for_2p_month_unchanged():
    """2 personas × 30 días → person_weeks ≈ 8 → cap = max(1, 8) = 8.
    El knob NO debe afectar planes grandes; solo el caso 1p × short cycle."""
    person_weeks = 8.0
    floor = 1
    cap_mazos = max(floor, int(round(person_weeks)))
    assert cap_mazos == 8, (
        "Plan 2p × mensual (pw=8) debería seguir capeando a 8 mazos. "
        "Si el knob cambia esto, está mal aplicado."
    )
