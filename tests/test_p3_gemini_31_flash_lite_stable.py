"""[P3-GEMINI-31-FLASH-LITE-STABLE · 2026-05-20] Test anti-regresión del
modelo `gemini-3.1-flash-lite` (sin sufijo `-preview`).

Bug observado:
    `gemini-3.1-flash-lite-preview` quedó en defaults de varios callsites
    productivos. La versión `-preview` de Google puede deprecarse sin aviso
    prolongado (incidente real 2026-05-11 con `gemini-3.1-pro-preview` que
    causó CB stale por 4.4 días). El modelo estable `gemini-3.1-flash-lite`
    (sin -preview) ya está disponible y es drop-in compatible.

Fix:
    Reemplazar todos los `gemini-3.1-flash-lite-preview` en código productivo
    backend por `gemini-3.1-flash-lite`. 42 reemplazos en 16 archivos
    (defaults de knobs + literal strings en tests anclados al default).

NOTA: este test bloquea la REGRESIÓN (reintroducir el `-preview` por
copy-paste accidental). El upgrade a `gemini-3.2-...` cuando Google publique
es libre — solo bloquea `gemini-3.1-flash-lite-preview` literal.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _scan_production_files():
    """Itera sobre archivos productivos del backend (excluye .git, __pycache__,
    .pyc, scripts/ y scratch/). Incluye tests porque están anchored al default."""
    skip_dirs = {".git", "__pycache__", "node_modules", ".pytest_cache"}
    skip_suffixes = {".pyc"}
    for p in _BACKEND_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix in skip_suffixes:
            continue
        # Skip este test mismo (auto-referencia el literal `preview` en docstring).
        if p.name == "test_p3_gemini_31_flash_lite_stable.py":
            continue
        yield p


def test_no_callsite_uses_preview_suffix():
    """[P3-GEMINI-31-FLASH-LITE-STABLE] Ningún archivo productivo debe
    contener `gemini-3.1-flash-lite-preview` (literal string).

    Comments narrativos describiendo el bug histórico (e.g., 'Pre-fix:
    usaba gemini-3.1-flash-lite-preview') son legítimos pero deben llevar
    el marker `P3-GEMINI-31-FLASH-LITE-STABLE-HISTORICAL` para distinguirse
    del default real.
    """
    OLD = "gemini-3.1-flash-lite-preview"
    violators = []
    for p in _scan_production_files():
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if OLD not in content:
            continue
        # Permitir solo si tiene el marker histórico CERCA.
        for line_no, line in enumerate(content.splitlines(), 1):
            if OLD not in line:
                continue
            # Mirar 3 líneas antes/después por el marker histórico.
            lines = content.splitlines()
            window_start = max(0, line_no - 4)
            window_end = min(len(lines), line_no + 3)
            window = "\n".join(lines[window_start:window_end])
            if "P3-GEMINI-31-FLASH-LITE-STABLE-HISTORICAL" not in window:
                violators.append(f"{p.relative_to(_BACKEND_ROOT)}:{line_no}: {line.strip()[:120]}")

    assert not violators, (
        f"{len(violators)} callsite(s) de `gemini-3.1-flash-lite-preview` "
        f"encontrados — reemplazar por `gemini-3.1-flash-lite` (sin sufijo). "
        f"Si es comment narrativo histórico, añadir marker "
        f"`P3-GEMINI-31-FLASH-LITE-STABLE-HISTORICAL` en las 3 líneas previas. "
        f"Ver P3-GEMINI-31-FLASH-LITE-STABLE · 2026-05-20.\n\n"
        + "\n".join(violators[:10])
    )


def test_new_model_id_present_in_defaults():
    """[P3-GEMINI-31-FLASH-LITE-STABLE] Verificación positiva: el nuevo
    string `gemini-3.1-flash-lite` debe aparecer en al menos los archivos
    productivos donde el viejo solía estar (agent.py, fact_extractor.py,
    sentiment_classifier.py, etc.)."""
    NEW = "gemini-3.1-flash-lite"
    expected_files = [
        "agent.py",
        "fact_extractor.py",
        "sentiment_classifier.py",
        "memory_manager.py",
        "ai_helpers.py",
        "proactive_agent.py",
    ]
    for fname in expected_files:
        p = _BACKEND_ROOT / fname
        assert p.exists(), f"Archivo esperado missing: {fname}"
        content = p.read_text(encoding="utf-8")
        # NOTE: `gemini-3.1-flash-lite` es prefix de `gemini-3.1-flash-lite-preview`,
        # así que un contains básico no distingue. Buscar la cadena seguida
        # por un char que NO sea letra/guión (boundary).
        assert re.search(r"gemini-3\.1-flash-lite(?![-a-zA-Z])", content), (
            f"{fname} NO contiene `gemini-3.1-flash-lite` sin `-preview` — "
            f"reemplazo incompleto. Ver P3-GEMINI-31-FLASH-LITE-STABLE."
        )
