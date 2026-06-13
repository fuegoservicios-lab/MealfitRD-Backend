"""[P2-NEW-FACTEX-PRIMARY-MODEL-KNOB · 2026-05-15] Anchor + regression guard.

`backend/fact_extractor.py` tiene 3 callsites a modelos LLM productivos:
  1. `should_extract_facts` (router gate, flash-lite-preview)
  2. `extract_facts` (PRO truth path)
  3. `_run_fact_pipeline` batch contradiction (PRO truth path)

Pre-fix los modelos estaban hardcoded inline:
  - `model="gemini-3.1-flash-lite"` (router)
  - `pro_model="gemini-3.1-pro-preview"` (extract + contradiction)

El SHADOW path ya tenía knob (P3-FACT-EXTRACTOR-SHADOW-AB · 2026-05-14)
pero el PRIMARY no — gap detectado en el audit 2026-05-15. Convención
canónica `P3-PREVIEW-MODEL-KNOB · 2026-05-12` exige knob para TODOS los
modelos preview de Google. Sin estos knobs, una deprecation de Google
tira la extracción de hechos hasta redeploy completo (45min Easypanel
cold start observado en incidente 2026-05-11 con `gemini-3.1-pro-preview`
4.4 días stale en circuit breaker).

Defensas que el test enforza:
  1. Anchor `P2-NEW-FACTEX-PRIMARY-MODEL-KNOB` presente en `fact_extractor.py`.
  2. Knobs `MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL` (default
     'gemini-3.5-flash' tras P3-MODEL-DEFAULT-FLASH35 · 2026-05-19, era
     'gemini-3.1-pro-preview') y `MEALFIT_FACT_EXTRACTOR_ROUTER_MODEL`
     (default 'gemini-3.1-flash-lite') resueltos via `_env_str`.
  3. Helpers `_fact_extractor_primary_model_name()` y
     `_fact_extractor_router_model_name()` definidos.
  4. Cero literales `pro_model="gemini-3.1-pro-preview"` en `_invoke_with_shadow`
     callsites — todos usan `_fact_extractor_primary_model_name()`.
  5. Cero literal `model="gemini-3.1-flash-lite"` en callsite del
     router — usa `_fact_extractor_router_model_name()`.
  6. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACT_EX = _REPO_ROOT / "backend" / "fact_extractor.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present_in_fact_extractor():
    src = _read(_FACT_EX)
    assert "P2-NEW-FACTEX-PRIMARY-MODEL-KNOB" in src, (
        "Falta anchor `P2-NEW-FACTEX-PRIMARY-MODEL-KNOB` en backend/fact_extractor.py."
    )


def test_primary_model_knob_registered():
    src = _read(_FACT_EX)
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] Default = constante DEEPSEEK_FLASH
    # de llm_provider (extracción es tarea aux barata, mismo modelo para
    # todos los tiers). Override sigue disponible via env var.
    pat = re.compile(
        r'_env_str\(\s*[\"\']MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL[\"\']\s*,\s*DEEPSEEK_FLASH',
        re.DOTALL,
    )
    assert pat.search(src), (
        "Knob `MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL` debe resolverse via "
        "`_env_str(\"MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL\", DEEPSEEK_FLASH)` "
        "(P0-DEEPSEEK-MIGRATION · 2026-06-12); override permite swap sin "
        "redeploy si se requiere otro modelo."
    )


def test_router_model_knob_registered():
    src = _read(_FACT_EX)
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] Default = DEEPSEEK_FLASH (gate
    # cheap-first del router).
    pat = re.compile(
        r'_env_str\(\s*[\"\']MEALFIT_FACT_EXTRACTOR_ROUTER_MODEL[\"\']\s*,\s*DEEPSEEK_FLASH',
        re.DOTALL,
    )
    assert pat.search(src), (
        "Knob `MEALFIT_FACT_EXTRACTOR_ROUTER_MODEL` debe resolverse via "
        "`_env_str(\"MEALFIT_FACT_EXTRACTOR_ROUTER_MODEL\", DEEPSEEK_FLASH)`."
    )


def test_helpers_defined():
    src = _read(_FACT_EX)
    assert "def _fact_extractor_primary_model_name(" in src, (
        "Helper `_fact_extractor_primary_model_name()` no encontrado. "
        "Centralizar en helper permite tests parser-based y consistencia."
    )
    assert "def _fact_extractor_router_model_name(" in src, (
        "Helper `_fact_extractor_router_model_name()` no encontrado."
    )


def test_zero_inline_primary_model_literal():
    """Cero `pro_model="gemini-..."` con literal hardcoded en callsites de
    `_invoke_with_shadow` — el helper debe ser quien lo provea.

    [P3-MODEL-DEFAULT-FLASH35 · 2026-05-19] Bloquea ambos el default viejo
    (`gemini-3.1-pro-preview`) y el nuevo (`gemini-3.5-flash`): cualquier
    literal inline rompe el patrón knob."""
    src = _read(_FACT_EX)
    # Permitido: el default literal dentro del `_env_str(...)` (línea del knob)
    # NO debe contar como callsite. Acotamos la búsqueda a contextos
    # `pro_model=` (kwarg de invocación), NO al `_env_str(...)` del knob.
    bad = re.findall(r'pro_model\s*=\s*[\"\']gemini-[a-z0-9.-]+[\"\']', src)
    assert not bad, (
        f"Encontrados {len(bad)} callsites con `pro_model=\"gemini-...\"` "
        f"hardcoded. Reemplazar por `pro_model=_fact_extractor_primary_model_name()`."
    )


def test_zero_inline_router_model_literal():
    """Cero `model="gemini-3.1-flash-lite"` en callsites de
    ChatGoogleGenerativeAI dentro de fact_extractor.py."""
    src = _read(_FACT_EX)
    bad = re.findall(r'\bmodel\s*=\s*[\"\']gemini-3\.1-flash-lite-preview[\"\']', src)
    assert not bad, (
        f"Encontrados {len(bad)} callsites con `model=\"gemini-3.1-flash-lite\"` "
        f"hardcoded. Reemplazar por `model=_fact_extractor_router_model_name()`."
    )


def test_callsites_use_helper_invocation():
    """Sanity: al menos 2 callsites deben invocar el helper PRIMARY (extract
    + contradiction) y al menos 1 el helper ROUTER (should_extract_facts)."""
    src = _read(_FACT_EX)
    primary_calls = re.findall(r"_fact_extractor_primary_model_name\(\)", src)
    router_calls = re.findall(r"_fact_extractor_router_model_name\(\)", src)
    assert len(primary_calls) >= 2, (
        f"Esperados ≥2 callsites usando _fact_extractor_primary_model_name() "
        f"(extract_facts + _run_fact_pipeline batch). Encontrados: {len(primary_calls)}."
    )
    assert len(router_calls) >= 1, (
        f"Esperado ≥1 callsite usando _fact_extractor_router_model_name() "
        f"(should_extract_facts). Encontrados: {len(router_calls)}."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14."""
    src = _read(Path(__file__))
    assert "P2-NEW-FACTEX-PRIMARY-MODEL-KNOB" in src
