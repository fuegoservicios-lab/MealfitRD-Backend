"""[P2-AUDIT-5 · 2026-05-15] Test parser-based: `frontend/src/main.jsx`
lee `replaysSessionSampleRate` y `replaysOnErrorSampleRate` desde knobs
(`VITE_SENTRY_REPLAYS_SESSION_RATE` y `VITE_SENTRY_REPLAYS_ON_ERROR_RATE`),
NO desde literales hardcoded.

Por qué este test:
    P1-SENTRY-SAMPLE-COST (2026-05-12) extrajo `tracesSampleRate` a knob
    `VITE_SENTRY_TRACES_SAMPLE_RATE` pero olvidó los 2 sample rates de
    replays. Replays son el output MÁS CARO de Sentry (vídeo de sesión
    completo) — un default hardcoded sin escape hatch impide a SRE bajar
    el sample rate sin redeploy si la cuota satura.

Fix esperado:
    - `_parseSentrySampleRate(raw, fallback)` (helper preexistente) reusado
      para los 2 nuevos knobs.
    - `replaysSessionSampleRate: SENTRY_REPLAYS_SESSION_RATE` (default 0.1).
    - `replaysOnErrorSampleRate: SENTRY_REPLAYS_ON_ERROR_RATE` (default 1.0).
    - Clamp `[0.0, 1.0]` heredado del helper.

Drift detection:
    - Las 2 constantes `SENTRY_REPLAYS_*_RATE` se definen llamando
      `_parseSentrySampleRate(import.meta.env.VITE_SENTRY_REPLAYS_*, default)`.
    - `Sentry.init({ ... })` usa esas constantes (no literales).

Cross-link convention (P2-HIST-AUDIT-14): slug `p2_audit_5`.

Tooltip-anchor: P2-AUDIT-5-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MAIN_JSX = _REPO_ROOT / "frontend" / "src" / "main.jsx"


@pytest.fixture(scope="module")
def main_src() -> str:
    return _MAIN_JSX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Las 2 knobs ENV vars aparecen
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("knob_name", [
    "VITE_SENTRY_REPLAYS_SESSION_RATE",
    "VITE_SENTRY_REPLAYS_ON_ERROR_RATE",
])
def test_knob_env_var_referenced(main_src: str, knob_name: str):
    assert knob_name in main_src, (
        f"P2-AUDIT-5 regresión: knob `{knob_name}` no aparece en main.jsx. "
        f"SRE no puede bajar el sample rate sin redeploy si la cuota Sentry "
        f"satura — replays son el output más caro."
    )


# ---------------------------------------------------------------------------
# 2. Las 2 constantes parsean via _parseSentrySampleRate
# ---------------------------------------------------------------------------
def test_session_rate_uses_helper(main_src: str):
    assert re.search(
        r"SENTRY_REPLAYS_SESSION_RATE\s*=\s*_parseSentrySampleRate\s*\([^)]*"
        r"VITE_SENTRY_REPLAYS_SESSION_RATE",
        main_src,
        re.DOTALL,
    ), (
        "P2-AUDIT-5 regresión: `SENTRY_REPLAYS_SESSION_RATE` no se inicializa "
        "via `_parseSentrySampleRate(import.meta.env.VITE_SENTRY_REPLAYS_SESSION_RATE, "
        "<default>)`. Sin el helper, no hay clamp [0.0, 1.0] y valores "
        "inválidos rompen Sentry.init."
    )


def test_on_error_rate_uses_helper(main_src: str):
    assert re.search(
        r"SENTRY_REPLAYS_ON_ERROR_RATE\s*=\s*_parseSentrySampleRate\s*\([^)]*"
        r"VITE_SENTRY_REPLAYS_ON_ERROR_RATE",
        main_src,
        re.DOTALL,
    ), (
        "P2-AUDIT-5 regresión: `SENTRY_REPLAYS_ON_ERROR_RATE` no se inicializa "
        "via `_parseSentrySampleRate(import.meta.env.VITE_SENTRY_REPLAYS_ON_ERROR_RATE, "
        "<default>)`."
    )


# ---------------------------------------------------------------------------
# 3. Sentry.init usa las constantes (no literales)
# ---------------------------------------------------------------------------
def test_sentry_init_uses_constants_not_literals(main_src: str):
    # Buscar `replaysSessionSampleRate: X` y `replaysOnErrorSampleRate: X`.
    # X NO debe ser un literal numérico — debe ser una constante (identifier).
    session_re = re.compile(
        r"replaysSessionSampleRate\s*:\s*([^,}]+)",
    )
    error_re = re.compile(
        r"replaysOnErrorSampleRate\s*:\s*([^,}]+)",
    )
    session_m = session_re.search(main_src)
    error_m = error_re.search(main_src)
    assert session_m, "P2-AUDIT-5: `replaysSessionSampleRate` no encontrado en Sentry.init"
    assert error_m, "P2-AUDIT-5: `replaysOnErrorSampleRate` no encontrado en Sentry.init"
    # El valor NO debe ser un literal float/int (e.g., `0.1`, `1.0`).
    for label, val in (("session", session_m.group(1).strip()), ("error", error_m.group(1).strip())):
        assert not re.match(r"^[\d.]+$", val), (
            f"P2-AUDIT-5 regresión: `replays{label.title()}SampleRate` usa "
            f"literal numérico `{val}` en Sentry.init en lugar de constante. "
            f"Reemplazar por `SENTRY_REPLAYS_{label.upper()}_RATE`."
        )


# ---------------------------------------------------------------------------
# 4. Defaults documentados (0.1 session, 1.0 on error)
# ---------------------------------------------------------------------------
def test_defaults_preserve_pre_fix_behavior(main_src: str):
    """Sin env var setteada, el comportamiento debe coincidir con el
    pre-fix hardcoded: 0.1 session + 1.0 on error."""
    # Aceptamos saltos de línea entre args del helper:
    #   _parseSentrySampleRate(
    #     import.meta.env.VITE_SENTRY_REPLAYS_SESSION_RATE,
    #     0.1,
    #   );
    # default 0.1 para session.
    assert re.search(
        r"SENTRY_REPLAYS_SESSION_RATE\s*=\s*_parseSentrySampleRate\s*\("
        r"[^)]*VITE_SENTRY_REPLAYS_SESSION_RATE[^)]*,\s*0\.1\s*,?\s*\)",
        main_src,
        re.DOTALL,
    ), (
        "P2-AUDIT-5 regresión: default de `SENTRY_REPLAYS_SESSION_RATE` no "
        "es 0.1. Sin env var, debe coincidir con el comportamiento pre-fix "
        "hardcoded para evitar cambio implícito de costo."
    )
    # default 1.0 para on error.
    assert re.search(
        r"SENTRY_REPLAYS_ON_ERROR_RATE\s*=\s*_parseSentrySampleRate\s*\("
        r"[^)]*VITE_SENTRY_REPLAYS_ON_ERROR_RATE[^)]*,\s*1\.0\s*,?\s*\)",
        main_src,
        re.DOTALL,
    ), (
        "P2-AUDIT-5 regresión: default de `SENTRY_REPLAYS_ON_ERROR_RATE` no "
        "es 1.0. Sin env var, debe coincidir con el comportamiento pre-fix."
    )


# ---------------------------------------------------------------------------
# 5. Anchor textual P2-AUDIT-5 presente
# ---------------------------------------------------------------------------
def test_anchor_present(main_src: str):
    assert "P2-AUDIT-5" in main_src, (
        "P2-AUDIT-5 regresión: anchor textual `P2-AUDIT-5` perdido en "
        "main.jsx. Restaurar para grep cross-incidente."
    )
