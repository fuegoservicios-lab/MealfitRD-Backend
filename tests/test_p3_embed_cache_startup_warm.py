"""[P3-EMBED-CACHE-STARTUP-WARM · 2026-05-16] El cold init de `get_semantic_cache()`
(shopping_calculator.py) tarda ~35 batches × 3s = ~105s + 429s de Gemini cuando
Redis está frío. Antes este init corría LAZY en el primer call de
`get_shopping_list_delta`, bloqueando `/recalculate-shopping-list` por >100s
hasta que el browser timeout dispara 500/CORS al usuario.

Síntoma observado 2026-05-16 plan 4cc91584: user cambió duration (7/15/30 días)
→ recalc disparó embed cache init → bloqueó >100s → browser fetch timeout →
500 (sin matchear `_is_transient_db_error` porque la excepción venía del
timeout del fetch, no del backend).

Fix (2 partes):
  1. Startup warmer en `app.py` lifespan: spawn daemon thread que invoca
     `get_semantic_cache()` para que el init ocurra ANTES de cualquier
     request del usuario. Si Redis tiene cached vectors, el thread termina
     en ms. Si vacío, el init demora ~100s pero NO bloquea requests.
  2. Non-blocking lock acquire en `get_semantic_cache()` (timeout=0.05s):
     si el warmer está corriendo, requests concurrentes NO esperan al lock
     — caen al regex fast-path (P6-SEMANTIC-SKIP) en lugar de bloquear.
     La próxima call (post-init) leerá el cache instantáneo.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
_SHOPCALC = (_BACKEND_ROOT / "shopping_calculator.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix #1: startup warmer en app.py lifespan
# ---------------------------------------------------------------------------


def test_app_lifespan_spawns_warmer_thread():
    """El lifespan de FastAPI debe lanzar un daemon thread que invoque
    `get_semantic_cache()` ANTES de aceptar requests."""
    assert "P3-EMBED-CACHE-STARTUP-WARM" in _APP, (
        "Marker `P3-EMBED-CACHE-STARTUP-WARM` ausente en app.py — el warmer "
        "fue removido. Sin él, el primer recalc post-restart bloquea ~100s."
    )
    # Debe haber un import de threading + spawn de un Thread con target callable
    assert "import threading" in _APP, (
        "El warmer requiere `import threading` para spawn del daemon thread."
    )
    assert "_warm_semantic_cache_bg" in _APP, (
        "Función `_warm_semantic_cache_bg` ausente — el warmer fue refactoreado "
        "sin actualizar este test. Verificar que sigue invocando get_semantic_cache()."
    )


def test_warmer_invokes_get_semantic_cache():
    """El bg target debe invocar `get_semantic_cache()` (no otra función
    como `_init_embed_cache` directa, que se saltaría el lock + Redis check)."""
    # Localizar la función _warm_semantic_cache_bg y verificar que invoca get_semantic_cache
    idx = _APP.find("def _warm_semantic_cache_bg")
    assert idx > 0, "Function _warm_semantic_cache_bg no encontrada."
    end = _APP.find("\n        _t = threading.Thread", idx)
    body = _APP[idx:end if end > 0 else idx + 1500]
    assert "get_semantic_cache" in body, (
        "El warmer debe invocar `get_semantic_cache()` (la API pública que "
        "respeta el cooldown + Redis check). Atajos como `_init_embed_cache` "
        "directos saltarían la lógica defensive de cooldown."
    )


def test_warmer_is_daemon_thread():
    """El thread DEBE ser daemon=True para no bloquear el shutdown del proceso
    si Gemini cuelga la init por minutos. Sin daemon=True, un kill -SIGINT
    se queda esperando el join del thread."""
    idx = _APP.find("threading.Thread(target=_warm_semantic_cache_bg")
    assert idx > 0, "Spawn del Thread no encontrado."
    end = _APP.find(")", idx)
    spawn_call = _APP[idx:end + 1]
    assert "daemon=True" in spawn_call, (
        "El thread del warmer debe declararse `daemon=True` — sin esto, un "
        "shutdown del backend espera hasta que termine la init (~100s)."
    )


# ---------------------------------------------------------------------------
# Fix #2: non-blocking lock acquire en get_semantic_cache
# ---------------------------------------------------------------------------


def test_lock_acquire_uses_timeout():
    """`_semantic_cache_lock.acquire(timeout=...)` con timeout corto (≤1s)
    para que requests del usuario no esperen al warmer si está corriendo."""
    idx = _SHOPCALC.find("def get_semantic_cache")
    assert idx > 0
    body = _SHOPCALC[idx:idx + 3500]

    # Buscar el acquire con timeout
    m = re.search(r"_semantic_cache_lock\.acquire\(\s*timeout\s*=\s*([\d.]+)\s*\)", body)
    assert m, (
        "`get_semantic_cache` no usa `_semantic_cache_lock.acquire(timeout=...)`. "
        "Revierte P3-EMBED-CACHE-STARTUP-WARM fix #2: el lock bloqueante "
        "deja que las requests del usuario esperen ~100s al warmer."
    )
    timeout_s = float(m.group(1))
    assert 0.01 <= timeout_s <= 1.0, (
        f"Timeout del lock = {timeout_s}s, fuera de [0.01, 1.0]. <10ms es "
        "muy poco (carrera de threads pierde la ventana); >1s vuelve a "
        "bloquear requests."
    )


def test_lock_acquire_failure_returns_none():
    """Cuando el acquire timeout falla, debe retornar None (fast-path Regex)
    en lugar de continuar sin lock."""
    idx = _SHOPCALC.find("def get_semantic_cache")
    body = _SHOPCALC[idx:idx + 3500]
    # Buscar el branch `if not acquired:` con return None
    pat = re.compile(
        r"if not acquired\s*:\s*\n(?:[^\n]*\n){0,5}?\s*return None",
        re.DOTALL,
    )
    assert pat.search(body), (
        "El branch `if not acquired:` no retorna None — sin esto, el flujo "
        "continúa sin lock o sin garantía de cache, causando bugs."
    )


def test_lock_released_in_finally():
    """`_semantic_cache_lock.release()` DEBE estar en un `finally` para
    garantizar release aunque el init lance excepción."""
    idx = _SHOPCALC.find("def get_semantic_cache")
    assert idx > 0
    # Localizar la siguiente `def ` (función adyacente) para limitar el slice
    # al cuerpo COMPLETO de get_semantic_cache (sin tropezar con un finally
    # de otra función).
    next_def = _SHOPCALC.find("\ndef ", idx + 10)
    end = next_def if next_def > 0 else idx + 8000
    body = _SHOPCALC[idx:end]

    # Buscar pattern: `finally:` + `_semantic_cache_lock.release()` próximos
    pat = re.compile(
        r"finally\s*:\s*\n(?:[^\n]*\n){0,15}?\s*_semantic_cache_lock\.release\(\)",
        re.DOTALL,
    )
    assert pat.search(body), (
        "`_semantic_cache_lock.release()` no está dentro de un `finally`. "
        "Sin esto, una excepción en la init dejaría el lock retenido para "
        "siempre — todas las requests caerían al fast-path indefinidamente."
    )


def test_marker_present_in_shopcalc():
    """Marker `P3-EMBED-CACHE-STARTUP-WARM` presente en shopping_calculator.py
    para anclar el cambio del lock + finally."""
    assert "P3-EMBED-CACHE-STARTUP-WARM" in _SHOPCALC, (
        "Marker ausente en shopping_calculator.py — un refactor podría "
        "revertir el non-blocking acquire sin ningún signal."
    )
