"""[P2-NEW-15 · 2026-05-11] `beforeunload` listener para cancelar SSE
en tab close (Plan.jsx).

Bug original (re-audit 2026-05-11):
    `generateAIPlanStream` abre un SSE reader que lee chunks durante
    toda la pipeline (~2-5min). Si user cierra tab mid-stream:
      - Reader queda abierto en heap del frontend → memleak.
      - Backend sigue emitiendo events sin destino → cuota LLM quemada.

Fix:
    `useEffect` en `Plan` con `addEventListener('beforeunload')` que
    llama `cancelGeneration()` SI hay generación en vuelo
    (`globalGenerationPromise && globalAbortController`). NO se hace
    en el cleanup del useEffect principal porque StrictMode (dev) hace
    double-invoke y abortaría el promise compartido entre los dos
    mounts → UX rota en dev.

Estrategia del test (parser-based):
    1. Bloque marcado `[P2-NEW-15 · 2026-05-11]` existe.
    2. `useEffect` con dependency array `[]` (mount-only listener).
    3. `window.addEventListener('beforeunload', ...)` registrado.
    4. Handler llama `cancelGeneration()` con guard de generación en
       vuelo.
    5. Try/catch best-effort (nunca bloquear unload).
    6. Cleanup `removeEventListener` en return del useEffect.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAN_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Plan.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _PLAN_FP.read_text(encoding="utf-8")


def test_p2_new_15_block_present(src: str):
    assert "[P2-NEW-15 · 2026-05-11]" in src, (
        "P2-NEW-15 regresión: el bloque del beforeunload listener "
        "desapareció. Tab close mid-stream vuelve a generar memleak + "
        "quota burn."
    )


def test_useEffect_with_empty_deps(src: str):
    p2_idx = src.find("[P2-NEW-15 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3500]
    # Patrón: useEffect(() => { ... }, []);
    assert re.search(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{[\s\S]*?\},\s*\[\]\s*\)",
        block,
    ), (
        "P2-NEW-15 regresión: el listener ya no usa `useEffect(... , [])` "
        "(mount-only). Si se cambia a deps no-vacíos, se re-suscribe en "
        "cada render — leak de listeners."
    )


def test_beforeunload_listener_registered(src: str):
    p2_idx = src.find("[P2-NEW-15 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3500]
    assert re.search(
        r"window\.addEventListener\(\s*['\"]beforeunload['\"]\s*,",
        block,
    ), (
        "P2-NEW-15 regresión: `addEventListener('beforeunload', ...)` "
        "ya no está. Sin él, tab close mid-stream no cancela."
    )


def test_handler_calls_cancelGeneration_with_guard(src: str):
    p2_idx = src.find("[P2-NEW-15 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3500]
    # Guard: globalGenerationPromise && globalAbortController.
    assert re.search(
        r"globalGenerationPromise\s*&&\s*globalAbortController",
        block,
    ), (
        "P2-NEW-15 regresión: el handler ya no chequea el guard "
        "`globalGenerationPromise && globalAbortController`. Sin guard, "
        "llamamos cancelGeneration cuando no hay nada en vuelo "
        "(no-op pero ruido en logs)."
    )
    assert "cancelGeneration()" in block, (
        "P2-NEW-15 regresión: el handler ya no llama `cancelGeneration()`. "
        "Sin esa llamada, el beforeunload listener es no-op."
    )


def test_try_catch_in_handler(src: str):
    p2_idx = src.find("[P2-NEW-15 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3500]
    handler_idx = block.find("handleBeforeUnload")
    assert handler_idx > 0
    handler_block = block[handler_idx:handler_idx + 1500]
    assert "try {" in handler_block, (
        "P2-NEW-15 regresión: el handler ya no envuelve `cancelGeneration()` "
        "en try. Un error aquí podría bloquear el unload del tab."
    )
    assert "catch" in handler_block, (
        "P2-NEW-15 regresión: el handler ya no tiene catch."
    )


def test_cleanup_removes_listener(src: str):
    p2_idx = src.find("[P2-NEW-15 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3500]
    assert re.search(
        r"removeEventListener\(\s*['\"]beforeunload['\"]",
        block,
    ), (
        "P2-NEW-15 regresión: el cleanup ya no remueve el listener. "
        "Sin cleanup, el listener persiste tras unmount del componente."
    )
