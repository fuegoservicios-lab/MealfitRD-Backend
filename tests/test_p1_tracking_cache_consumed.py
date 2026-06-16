"""[P1-TRACKING-CACHE-CONSUMED · 2026-05-20] Test anti-regresión del cache
local de `consumed` en `TrackingProgress.jsx`.

Bug observado:
    Cuando el user navega Dashboard → Nevera/Plan → Dashboard, el componente
    `TrackingProgress` se desmonta (React Router default) y re-monta con
    `consumed = {calories: 0, ...}` → flash visible de macros en 0 durante
    los ~200-500ms del fetch a `/api/diary/consumed`. Reportado 2026-05-20:
    "cada vez que cambio de aparto y vuelvo aparecen las macros vacías
    unos milisegundos y después vuelven a su estado original".

Fix:
    Lazy initializer del useState lee `consumed` desde localStorage con
    clave compuesta `mealfit_tracking_consumed_<userId>_<YYYY-MM-DD>`.
    Si match → arranca con datos reales → fetch silencioso en background.

    Key con fecha de hoy = TTL implícito de 24h + invalidación automática
    en rollover de medianoche (la key de ayer ya no matchea hoy).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKING_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_consumed_initializer_reads_cache():
    """[P1-TRACKING-CACHE-CONSUMED] El initializer del useState `consumed`
    debe ser lazy `useState(() => ...)` que lee de localStorage, NO
    literal `useState({calories: 0, ...})`."""
    src = _read(_TRACKING_JSX)
    # Buscar useState(() => ...) para consumed.
    match = re.search(
        r"const\s*\[\s*consumed\s*,\s*setConsumed\s*\]\s*=\s*useState\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*\);",
        src,
        re.DOTALL,
    )
    assert match, (
        "useState lazy initializer para `consumed` no encontrado — bug "
        "del flash de macros en 0 al re-mount. Ver "
        "P1-TRACKING-CACHE-CONSUMED · 2026-05-20."
    )
    body = match.group(1)
    assert "safeLocalStorageGet" in body, (
        "Initializer NO lee de localStorage — no puede arrancar con cache."
    )
    assert re.search(
        r"_getConsumedCacheKey|_CONSUMED_CACHE_KEY_PREFIX|mealfit_tracking_consumed",
        body,
    ), "Initializer NO referencia la cache key esperada."


def test_consumed_cache_key_includes_user_and_date():
    """[P1-TRACKING-CACHE-CONSUMED] La cache key debe componerse de
    user_id + fecha-de-hoy. Sin esto, podría servir cache de OTRO user
    (logout/login) o de OTRO día (rollover medianoche)."""
    src = _read(_TRACKING_JSX)
    # Buscar la función o helper que construye la key.
    key_helper = re.search(
        r"const\s+_getConsumedCacheKey\s*=\s*\(?[^)]*\)?\s*=>\s*\{(.+?)\};",
        src,
        re.DOTALL,
    )
    assert key_helper, "Helper `_getConsumedCacheKey` no encontrado."
    body = key_helper.group(1)
    # Sanity: debe usar userId Y armar dateStr con año-mes-día.
    assert "userId" in body, "Helper no usa userId — riesgo cross-user."
    assert re.search(r"getFullYear\s*\(\)", body), (
        "Helper no compone la fecha (getFullYear). Sin fecha en la key, "
        "el cache no invalida en rollover de medianoche."
    )


def test_consumed_persisted_on_change_with_guard():
    """[P1-TRACKING-CACHE-CONSUMED] useEffect debe persistir `consumed`
    al change, PERO con guard contra el default vacío {0,0,0,0,0} para
    evitar bloquear la hidratación con datos frescos en próximos mounts."""
    src = _read(_TRACKING_JSX)
    # Buscar useEffect con deps [consumed, userId] o similar.
    persist_match = re.search(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*,\s*\[\s*consumed\s*,\s*userId\s*\]\s*\)",
        src,
        re.DOTALL,
    )
    assert persist_match, (
        "useEffect con deps `[consumed, userId]` ausente — `consumed` no se "
        "persiste al cambio y la próxima carga no encuentra cache fresco."
    )
    body = persist_match.group(1)
    assert "safeLocalStorageSet" in body, (
        "useEffect no llama `safeLocalStorageSet` — no persiste."
    )
    # Guard contra el default vacío para no sobreescribir el cache real con
    # un placeholder en 0. [P3-TRACKING-CACHE-EMPTY-FETCH · 2026-05-27]
    # reemplazó el guard heurístico `calories===0 && protein===0` (que también
    # bloqueaba un día legítimamente en 0) por un flag explícito `_fetched`
    # seteado SOLO tras el fetch real al server: `if (!consumed._fetched) return`.
    assert re.search(
        r"_fetched",
        body,
        re.DOTALL,
    ), (
        "Guard contra default vacío ausente. Debe usar el flag `_fetched` "
        "(set post-fetch del server) para no persistir el placeholder vacío "
        "y bloquear la hidratación con datos frescos. Ver "
        "P1-TRACKING-CACHE-CONSUMED · 2026-05-20 / "
        "P3-TRACKING-CACHE-EMPTY-FETCH · 2026-05-27."
    )


def test_loading_false_if_cache_hit():
    """[P1-TRACKING-CACHE-CONSUMED] `loading` debe inicializar en `false`
    si hay cache hit (no mostrar spinner cuando ya hay datos visibles)."""
    src = _read(_TRACKING_JSX)
    # Buscar useState(() => ...) para loading.
    loading_match = re.search(
        r"const\s*\[\s*loading\s*,\s*setLoading\s*\]\s*=\s*useState\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*\);",
        src,
        re.DOTALL,
    )
    assert loading_match, (
        "`loading` useState lazy initializer ausente. Para evitar spinner "
        "cuando hay cache, debe ser lazy condicional."
    )
    body = loading_match.group(1)
    assert "safeLocalStorageGet" in body, (
        "Initializer de `loading` no consulta cache — siempre arranca en true."
    )
    assert "return false" in body, (
        "Initializer no retorna false cuando hay cache hit."
    )
