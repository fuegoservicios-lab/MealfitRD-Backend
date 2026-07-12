"""[P2-REGEN-DAY-FATS-RELEVEL · 2026-07-12] + [P2-NAME-STYLE-DESCRIPTOR · 2026-07-12]
Los dos chips ámbar del regen-día del owner (04:47Z, plan 1bfda745):

1. "Macros algo fuera de la banda objetivo": el día regenerado entregó band fats=0.0/
   kcal=0.0 TRAS el rebalance ("re-apuntadas al target") — los clamps por-línea saturaron
   en 'quería-menos' y el recortador dedicado de grasas (`_relevel_fats_universal`, SSOT
   de S1) JAMÁS corría en regen-day (callsites: assemble + finalize-con-target). Fix:
   callsite post-rebalance/pre-refine en regenerate-day (shrink-only ⇒ pantry-safe).

2. "El nombre puede no reflejar la proteína real" sobre 'Soya Guisada al Estilo Bistec
   Encebollado': 'al estilo <carne>' es descriptor de ESTILO, no claim de proteína — el
   detector phantom flaggeaba un nombre honesto. Menciones precedidas por 'estilo' no
   cuentan como proteína del título; el fantasma REAL (sin 'estilo') sigue detectándose.

tooltip-anchor: P2-REGEN-DAY-FATS-RELEVEL
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_relevel_wired_after_rebalance_before_refine():
    i_rb = _PLANS.find("macros del día re-apuntadas al target")
    i_frl = _PLANS.find("MEALFIT_REGEN_DAY_FATS_RELEVEL")
    i_ref = _PLANS.find("[P1-UPDATE-MACRO-PARITY · 2026-07-03] (audit v6 · P1-1) Refinador GLOBAL")
    assert -1 not in (i_rb, i_frl, i_ref)
    assert i_rb < i_frl < i_ref, (
        "orden load-bearing: rebalance (re-apunta) → relevel (recorta el residuo de grasa "
        "que los clamps dejaron) → refine (pulido 5g sobre el estado recortado)"
    )
    win = _PLANS[i_frl:i_frl + 1500]
    assert "_relevel_fats_universal" in win, "reusa el SSOT de S1 (no lógica nueva de trim)"
    assert 'day_target.get("fats_g")' in win


def test_style_descriptor_not_a_protein_claim():
    from graph_orchestrator import _fix_phantom_protein_in_name
    from constants import strip_accents
    m = {"name": "Moro de Frijoles Pintos con Soya Guisada al Estilo Bistec Encebollado",
         "ingredients": ["1/2 taza de frijoles pintos", "53g de soya texturizada", "arroz blanco"]}
    _fix_phantom_protein_in_name(m, strip_accents)
    assert not m.get("_name_honesty_degraded"), (
        "'al estilo bistec' describe la preparación — flaggearlo era un falso positivo "
        "sobre un nombre honesto (vivo: chip ámbar en el moro de soya del owner)"
    )
    assert m["name"].startswith("Moro de Frijoles"), "el nombre no se toca"


def test_stale_honesty_flag_self_heals():
    """El flag inocente arrastrado de una corrida vieja se retira solo al re-evaluar
    (vivo: el moro conservó su slot en el regen y el flag pre-fix requirió limpieza
    manual). Clear-only: jamás inventa flags."""
    from graph_orchestrator import _fix_phantom_protein_in_name
    from constants import strip_accents
    m = {"name": "Moro de Frijoles Pintos con Soya Guisada al Estilo Bistec Encebollado",
         "ingredients": ["1/2 taza de frijoles pintos", "53g de soya texturizada"],
         "_name_honesty_degraded": True}
    _fix_phantom_protein_in_name(m, strip_accents)
    assert "_name_honesty_degraded" not in m, "flag stale sobre nombre inocente debe auto-limpiarse"


def test_real_phantom_still_detected():
    from graph_orchestrator import _fix_phantom_protein_in_name
    from constants import strip_accents
    m = {"name": "Bistec Encebollado con Arroz",
         "ingredients": ["1/2 taza de arroz", "100g de queso"]}
    _fix_phantom_protein_in_name(m, strip_accents)
    assert m.get("_name_honesty_degraded") is True, (
        "sin 'estilo', un título que lidera con carne ausente sigue siendo fantasma"
    )


def test_markers_anchored():
    assert _PLANS.count("P2-REGEN-DAY-FATS-RELEVEL") >= 2
    assert _GO.count("P2-NAME-STYLE-DESCRIPTOR") >= 1
