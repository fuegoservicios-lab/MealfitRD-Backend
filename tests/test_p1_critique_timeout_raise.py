"""[P1-CRITIQUE-TIMEOUT-RAISE · 2026-05-15] Regression guard del knob
`MEALFIT_CRITIQUE_FIX_TIMEOUT_S=150` en `.env`.

Pre-fix (120s): test E2E del plan `1f667f16-a143-4bc6-b2d3-aa0f41d18fca`
mostró 2/3 días timeoutearon Flash → escalaron a Pro fallback. `llm_usage_events`
registró $0.2873 de Pro de un total de $0.44 (66% del costo del plan).

Hipótesis: subir el timeout a 150s permite a Flash absorber los timeouts
sin recurrir a Pro. Trade-off: +30s latencia adicional en happy path; Pro
fallback ahorra ese tiempo en el peor caso (Pro también tarda ~60s).

Test: parser-based sobre `.env`. NO valida comportamiento runtime (eso
requiere generar plan y observar `llm_usage_events.model` distribution).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"


def test_critique_timeout_raised_to_150():
    text = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"^MEALFIT_CRITIQUE_FIX_TIMEOUT_S\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, (
        "Falta `MEALFIT_CRITIQUE_FIX_TIMEOUT_S` en .env."
    )
    val = int(m.group(1))
    assert val >= 150, (
        f"P1-CRITIQUE-TIMEOUT-RAISE: timeout debe ser ≥150s (era 120s pre-fix). "
        f"Encontré {val}. Sin esto, los días que tardan 120-150s en self_critique "
        f"escalan a Pro fallback ($0.20+/plan extra). Si subes >180s, considera "
        f"impacto en latencia P95 (timeout total del pipeline `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S=720s`)."
    )


def test_pro_fallback_kept_enabled_as_safety_net():
    """Aunque subimos el Flash timeout, el Pro fallback debe permanecer habilitado
    como red de seguridad para casos extremos (días que genuinamente requieren
    Pro porque Flash no puede corregir). Solo si Pro fallback escala >0% post-fix,
    revisitar."""
    text = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"^MEALFIT_CRITIQUE_PRO_FALLBACK_ENABLED\s*=\s*(\w+)",
        text,
        re.MULTILINE,
    )
    assert m, "Falta `MEALFIT_CRITIQUE_PRO_FALLBACK_ENABLED` en .env."
    val = m.group(1).strip().lower()
    assert val == "true", (
        f"Pro fallback debe seguir HABILITADO como red de seguridad. "
        f"Encontré: {val!r}. Para deshabilitarlo (palanca B), usar otro P-fix dedicado."
    )
