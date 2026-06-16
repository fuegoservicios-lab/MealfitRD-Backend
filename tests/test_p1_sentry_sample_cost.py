"""[P1-SENTRY-SAMPLE-COST · 2026-05-12] Anchor + regression guard.

Backend (`app.py`) y frontend (`main.jsx`) NO deben volver a hardcodear
`traces_sample_rate=1.0` / `profiles_sample_rate=1.0` / `tracesSampleRate: 1.0`
en `Sentry.init(...)`. Pre-fix capturaban el 100% de transacciones más
profiling continuo, costo escala lineal con tráfico y arriesga throttle de
la cuota Sentry (los errores que necesitas son los primeros en ser dropeados).

Defensas que el test enforza:
  1. Anchor `P1-SENTRY-SAMPLE-COST` presente en ambos archivos.
  2. Backend resuelve via `_env_float("MEALFIT_SENTRY_TRACES_SAMPLE_RATE", ...)`
     y `_env_float("MEALFIT_SENTRY_PROFILES_SAMPLE_RATE", ...)` con default 0.1
     y validator lambda v: 0.0 <= v <= 1.0.
  3. Backend `sentry_sdk.init(...)` referencia las variables, NO literales 1.0.
  4. Frontend usa `import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE` con parse
     que clamp [0.0, 1.0] y default 0.1.
  5. Frontend `Sentry.init(...)` referencia la variable, NO literal 1.0.

Test parser-based — no levanta el server, solo escanea source con regex.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_APP = _REPO_ROOT / "backend" / "app.py"
_FRONTEND_MAIN = _REPO_ROOT / "frontend" / "src" / "main.jsx"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present_in_backend_app():
    src = _read(_BACKEND_APP)
    assert "P1-SENTRY-SAMPLE-COST" in src, (
        "Falta anchor `P1-SENTRY-SAMPLE-COST` en backend/app.py. "
        "Sin anchor, un futuro reader que vea env vars `MEALFIT_SENTRY_*` "
        "no sabrá el modo de fallo que cierran (100% sample → throttle Sentry)."
    )


def test_anchor_present_in_frontend_main():
    src = _read(_FRONTEND_MAIN)
    assert "P1-SENTRY-SAMPLE-COST" in src, (
        "Falta anchor `P1-SENTRY-SAMPLE-COST` en frontend/src/main.jsx."
    )


def test_backend_uses_env_var_for_traces_sample_rate():
    src = _read(_BACKEND_APP)
    # Pattern: _env_float("MEALFIT_SENTRY_TRACES_SAMPLE_RATE", 0.1, ...)
    pat = re.compile(
        r"_env_float\(\s*[\"']MEALFIT_SENTRY_TRACES_SAMPLE_RATE[\"']\s*,\s*0\.1\s*,",
        re.DOTALL,
    )
    assert pat.search(src), (
        "Backend debe resolver `traces_sample_rate` via "
        "`_env_float(\"MEALFIT_SENTRY_TRACES_SAMPLE_RATE\", 0.1, ...)`. "
        "Default 0.1 obligatorio (90% reducción vs hardcoded 1.0 pre-fix)."
    )


def test_backend_uses_env_var_for_profiles_sample_rate():
    src = _read(_BACKEND_APP)
    pat = re.compile(
        r"_env_float\(\s*[\"']MEALFIT_SENTRY_PROFILES_SAMPLE_RATE[\"']\s*,\s*0\.1\s*,",
        re.DOTALL,
    )
    assert pat.search(src), (
        "Backend debe resolver `profiles_sample_rate` via "
        "`_env_float(\"MEALFIT_SENTRY_PROFILES_SAMPLE_RATE\", 0.1, ...)`."
    )


def test_backend_clamps_via_validator():
    """El _env_float debe pasar un validator que clamp [0.0, 1.0]. Sin esto
    un override accidental a MEALFIT_SENTRY_TRACES_SAMPLE_RATE=42 pasaría
    raw a sentry_sdk causando comportamiento indefinido."""
    src = _read(_BACKEND_APP)
    # Buscar al menos un validator que verifique <= 1.0 y >= 0.0 cerca de
    # MEALFIT_SENTRY_*. La regex acepta variaciones de espaciado y orden.
    pat = re.compile(
        r"MEALFIT_SENTRY_(?:TRACES|PROFILES)_SAMPLE_RATE.*?validator\s*=\s*lambda\s+\w+\s*:\s*0\.0\s*<=\s*\w+\s*<=\s*1\.0",
        re.DOTALL,
    )
    matches = pat.findall(src)
    assert len(matches) >= 2, (
        "Ambos knobs MEALFIT_SENTRY_*_SAMPLE_RATE deben tener "
        "`validator=lambda v: 0.0 <= v <= 1.0`. "
        f"Encontrados: {len(matches)}/2."
    )


def test_backend_sentry_init_uses_variables_not_literals():
    """`sentry_sdk.init(...)` NO debe contener literales 1.0 (ni 0.5/etc).
    Solo referencias a `_SENTRY_TRACES_SAMPLE_RATE` y
    `_SENTRY_PROFILES_SAMPLE_RATE`."""
    src = _read(_BACKEND_APP)
    # Aislar el bloque sentry_sdk.init(...) — desde la línea con `sentry_sdk.init(`
    # hasta el primer `)` solitario en columna 0.
    m = re.search(r"sentry_sdk\.init\(\s*(.*?)\n\)", src, re.DOTALL)
    assert m is not None, "No se encontró bloque `sentry_sdk.init(...)` en app.py"
    block = m.group(1)
    # Debe referenciar las dos variables
    assert "_SENTRY_TRACES_SAMPLE_RATE" in block, (
        "sentry_sdk.init debe usar `traces_sample_rate=_SENTRY_TRACES_SAMPLE_RATE`"
    )
    assert "_SENTRY_PROFILES_SAMPLE_RATE" in block, (
        "sentry_sdk.init debe usar `profiles_sample_rate=_SENTRY_PROFILES_SAMPLE_RATE`"
    )
    # No literales numéricos sospechosos en las líneas de sample_rate
    bad = re.search(r"(traces_sample_rate|profiles_sample_rate)\s*=\s*[01]\.\d+", block)
    assert bad is None, (
        f"sentry_sdk.init contiene literal numérico hardcoded: {bad.group(0)!r}. "
        "Usar `_SENTRY_*_SAMPLE_RATE` resuelto via knobs."
    )


def test_frontend_uses_env_var_for_traces_sample_rate():
    src = _read(_FRONTEND_MAIN)
    # Pattern: import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE
    assert "VITE_SENTRY_TRACES_SAMPLE_RATE" in src, (
        "Frontend debe leer `VITE_SENTRY_TRACES_SAMPLE_RATE` desde "
        "`import.meta.env` (Vite-canonical). Vite inyecta solo vars con "
        "prefijo `VITE_` al bundle del cliente — `process.env` no funciona."
    )


def test_frontend_clamps_and_defaults_to_0_1():
    """El parse del env var debe (a) clamp [0.0, 1.0] y (b) default a 0.1
    cuando el raw es undefined / NaN / fuera de rango."""
    src = _read(_FRONTEND_MAIN)
    # Buscar `_parseSentrySampleRate(... , 0.1)` — el fallback debe ser 0.1
    pat = re.compile(
        r"_parseSentrySampleRate\(\s*import\.meta\.env\.VITE_SENTRY_TRACES_SAMPLE_RATE\s*,\s*0\.1",
        re.DOTALL,
    )
    assert pat.search(src), (
        "Frontend debe invocar `_parseSentrySampleRate(import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE, 0.1)` "
        "con fallback 0.1 explícito."
    )
    # El cuerpo del parse debe verificar v >= 0.0 && v <= 1.0
    body_pat = re.compile(
        r"_parseSentrySampleRate\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*?v\s*>=\s*0\.0\s*&&\s*v\s*<=\s*1\.0",
        re.DOTALL,
    )
    assert body_pat.search(src), (
        "El parser debe clamp `v >= 0.0 && v <= 1.0`. Sin esto, "
        "`VITE_SENTRY_TRACES_SAMPLE_RATE=42` pasaría 42 a Sentry causando "
        "comportamiento indefinido."
    )


def test_frontend_sentry_init_uses_variable_not_literal():
    src = _read(_FRONTEND_MAIN)
    # Aislar el bloque del init de Sentry.
    # [P2-SENTRY-TREESHAKE · 2026-05-23] main.jsx pasó de `import * as Sentry`
    # (`Sentry.init({...})`) a un named import `import { init as sentryInit }`
    # (`sentryInit({...})`) para habilitar tree-shaking. El regex acepta ambas
    # formas para reflejar el callsite real sin re-acoplar al star-import viejo.
    m = re.search(r"(?:Sentry\.init|sentryInit)\(\{\s*(.*?)\n\}\)", src, re.DOTALL)
    assert m is not None, (
        "No se encontró bloque `Sentry.init({...})` ni `sentryInit({...})` en main.jsx"
    )
    block = m.group(1)
    assert "SENTRY_TRACES_SAMPLE_RATE" in block, (
        "El init de Sentry debe usar `tracesSampleRate: SENTRY_TRACES_SAMPLE_RATE`"
    )
    # tracesSampleRate NO debe ser literal 1.0 / 0.5 / etc.
    bad = re.search(r"tracesSampleRate\s*:\s*[01]\.\d+", block)
    assert bad is None, (
        f"Sentry.init contiene `tracesSampleRate` hardcoded: {bad.group(0)!r}. "
        "Usar `SENTRY_TRACES_SAMPLE_RATE` resuelto desde import.meta.env."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14: el slug del marker debe matchear
    al menos un archivo `tests/test_<slug>*.py`. Este test ancla el slug
    `p1_sentry_sample_cost` (su nombre lo provee implícito)."""
    src = _read(Path(__file__))
    assert "P1-SENTRY-SAMPLE-COST" in src
