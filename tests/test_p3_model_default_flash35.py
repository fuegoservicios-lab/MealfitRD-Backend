"""[P3-MODEL-DEFAULT-FLASH35 · 2026-05-19] Defaults de los 3 callsites
hardcoded a `gemini-3.1-pro-preview` migran a `gemini-3.5-flash`.

Razón del cambio:
    Decisión operacional 2026-05-19. Los 3 callsites del path PRO
    (`MEALFIT_CHAT_AGENT_MODEL`, `MEALFIT_CHAT_AGENT_SWAP_MODEL`,
    `MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL`) tenían default
    `gemini-3.1-pro-preview` — preview model con riesgo de deprecation
    silenciosa (audit 2026-05-11 documentó CB row stale 4.4 días). Se
    sustituye por `gemini-3.5-flash` (más estable, menor costo, latencia
    inferior). Convención P3-PREVIEW-MODEL-KNOB intacta: el SRE sigue
    pudiendo overridear via env var en EasyPanel sin redeploy.

Defensas que este test enforza (parser-based, NO importa los módulos):
    1. `agent.py:_chat_agent_model_name` default == `gemini-3.5-flash`.
    2. `agent.py:_chat_agent_swap_model_name` default == `gemini-3.5-flash`.
    3. `fact_extractor.py:_FACT_EXTRACTOR_PRIMARY_MODEL` default ==
       `gemini-3.5-flash`.
    4. Cero callsite hardcodeando el default viejo (`gemini-3.1-pro-preview`)
       como argumento de `_env_str(...)` en los 3 archivos productivos.
    5. Marker `_LAST_KNOWN_PFIX` en `app.py` apunta a este P-fix.

Patrón de los `*-lite-preview` (router/title/judge/reviewer/fact-checker/
proactive) NO se toca — esos siguen con default flash-lite-preview porque
son tareas schema-strict cheap-first; el cambio aplica solo al path PRO
que actualmente era pro-preview.

Cross-link convention (P2-HIST-AUDIT-14): el slug del marker
`P3-MODEL-DEFAULT-FLASH35` → `p3_model_default_flash35` matchea este
archivo `test_p3_model_default_flash35.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_FACT_EX = _BACKEND_ROOT / "fact_extractor.py"
_DB_PROFILES = _BACKEND_ROOT / "db_profiles.py"

_NEW_DEFAULT = "gemini-3.5-flash"
_OLD_DEFAULT = "gemini-3.1-pro-preview"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. agent.py — los 2 helpers PRO leen el nuevo default
# ---------------------------------------------------------------------------
_AGENT_KNOBS = [
    ("_chat_agent_model_name", "MEALFIT_CHAT_AGENT_MODEL"),
    ("_chat_agent_swap_model_name", "MEALFIT_CHAT_AGENT_SWAP_MODEL"),
]


@pytest.mark.parametrize("helper_name, knob_name", _AGENT_KNOBS)
def test_agent_pro_helpers_default_to_flash35(helper_name: str, knob_name: str):
    """`def <helper>() -> str:` debe leer `_env_str("<KNOB>", "gemini-3.5-flash")`."""
    src = _read(_AGENT_PY)
    # Localiza el body del helper.
    def_re = re.compile(rf"def\s+{re.escape(helper_name)}\s*\(\s*\)\s*->\s*str\s*:")
    m = def_re.search(src)
    assert m is not None, (
        f"P3-MODEL-DEFAULT-FLASH35 regresión: helper `def {helper_name}() -> str:` "
        f"no encontrado en agent.py. Si el helper fue renombrado, actualizar este test."
    )
    body_start = m.end()
    # Lee ~250 chars del body (suficiente para los 4 líneas del helper).
    body = src[body_start:body_start + 250]
    pat = re.compile(
        rf'_env_str\(\s*[\"\']{re.escape(knob_name)}[\"\']\s*,\s*[\"\']{re.escape(_NEW_DEFAULT)}[\"\']'
    )
    assert pat.search(body), (
        f"P3-MODEL-DEFAULT-FLASH35 regresión: helper `{helper_name}` no resuelve "
        f"`{knob_name}` con default `{_NEW_DEFAULT!r}`. Decisión 2026-05-19: el "
        f"path PRO sale del preview model y va a flash 3.5 (stable)."
    )


# ---------------------------------------------------------------------------
# 2. fact_extractor.py — el PRIMARY (PRO truth path) lee el nuevo default
# ---------------------------------------------------------------------------
def test_fact_extractor_primary_default_to_flash35():
    src = _read(_FACT_EX)
    pat = re.compile(
        rf'_env_str\(\s*[\"\']MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL[\"\']\s*,\s*[\"\']{re.escape(_NEW_DEFAULT)}[\"\']',
        re.DOTALL,
    )
    assert pat.search(src), (
        f"P3-MODEL-DEFAULT-FLASH35 regresión: knob "
        f"`MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL` no usa default `{_NEW_DEFAULT!r}` "
        f"en fact_extractor.py. Mismo patrón que los chat helpers."
    )


# ---------------------------------------------------------------------------
# 3. Cero referencia residual al default viejo como default activo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_AGENT_PY, _FACT_EX])
def test_no_old_default_as_env_str_fallback(path: Path):
    """El default viejo (`gemini-3.1-pro-preview`) NO debe aparecer como segundo
    arg de `_env_str(...)` en ninguno de los 2 archivos productivos. Comentarios
    histórico-narrativos que mencionan el modelo (incidentes, runbook) sí están
    permitidos — strip comentarios antes de buscar."""
    src = _read(path)
    no_comments = re.sub(r"#[^\n]*", "", src)
    pat = re.compile(
        rf'_env_str\([^)]*[\"\']{re.escape(_OLD_DEFAULT)}[\"\']',
        re.DOTALL,
    )
    bad = pat.findall(no_comments)
    assert not bad, (
        f"P3-MODEL-DEFAULT-FLASH35 regresión: {len(bad)} callsite(s) de "
        f"`_env_str(...)` en {path.name} aún usan `{_OLD_DEFAULT!r}` como default. "
        f"Migrar a `{_NEW_DEFAULT!r}`. Si necesitas conservar el viejo por A/B, "
        f"setea el env var explícitamente en EasyPanel, no en source."
    )


# ---------------------------------------------------------------------------
# 4. Marker `_LAST_KNOWN_PFIX` apunta a este P-fix
# ---------------------------------------------------------------------------
def test_last_known_pfix_matches_marker():
    """El marker `_LAST_KNOWN_PFIX` en `app.py` debe ser este P-fix mientras
    sea el último mergeado a HEAD. Si fue superado por otro, este test sigue
    pasando si el slug aparece en el archivo (anchor textual)."""
    app_src = _read(_APP_PY)
    assert "P3-MODEL-DEFAULT-FLASH35" in app_src, (
        "P3-MODEL-DEFAULT-FLASH35 regresión: marker no presente en app.py. "
        "Bumpear `_LAST_KNOWN_PFIX = 'P3-MODEL-DEFAULT-FLASH35 · 2026-05-19'` "
        "o asegurar que el slug aparece en un comentario subsecuente."
    )


# ---------------------------------------------------------------------------
# 5. Pricing del nuevo modelo presente en db_profiles (cost telemetry no-zero)
# ---------------------------------------------------------------------------
def test_pricing_entry_for_new_default():
    """`_DEFAULT_GEMINI_PRICING_MICROS_PER_M` debe tener entry para
    `gemini-3.5-flash` con los micros oficiales de la doc Google 2026-05-19.

    Sin esta entrada, `compute_gemini_cost_micros` retorna None y los eventos
    LLM persisten sin costo en `pipeline_metrics` → cost telemetry rota.

    Validación: localiza la línea que contiene la clave del modelo y verifica
    que tiene los 3 valores en micros (formato `N_NNN_NNN`). Tolerante a
    whitespace, estricta con cifras."""
    src = _read(_DB_PROFILES)
    # Línea con la entry — usamos line-grep porque el dict tiene una entry
    # por línea (comprobado en source 2026-05-19).
    matching_lines = [
        ln for ln in src.splitlines()
        if f'"{_NEW_DEFAULT}"' in ln and "input" in ln and "output" in ln
    ]
    assert matching_lines, (
        f"P3-MODEL-DEFAULT-FLASH35 regresión: entry `{_NEW_DEFAULT}` no "
        f"presente en `_DEFAULT_GEMINI_PRICING_MICROS_PER_M`. Añadir línea con "
        f"input=1_500_000, output=9_000_000, cached=150_000 (tier Estándar "
        f"Google AI doc 2026-05-19: $1.50/M, $9.00/M, $0.15/M)."
    )
    line = matching_lines[0]
    for expected in ("1_500_000", "9_000_000", "150_000"):
        assert expected in line, (
            f"P3-MODEL-DEFAULT-FLASH35 regresión: línea de pricing para "
            f"`{_NEW_DEFAULT}` no contiene el valor `{expected}`. Línea actual: "
            f"{line.strip()!r}. Esperado: input=1_500_000, output=9_000_000, "
            f"cached=150_000."
        )


# ---------------------------------------------------------------------------
# 6. Anchor presente en este test file (cross-link guard P2-HIST-AUDIT-14)
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    src = _read(Path(__file__))
    assert "P3-MODEL-DEFAULT-FLASH35" in src
