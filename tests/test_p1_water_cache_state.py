"""[P1-WATER-CACHE-STATE · 2026-05-20] Anti-regresión del cache local
del state de hidratación en `WaterTracker.jsx`.

Bug observado:
    El card "Hidratación" muestra "0 de 8 vasos" momentáneamente cuando
    el user navega Nevera/Plan → Dashboard, antes del fetch a
    `/api/plans/water-intake`. Reportado 2026-05-20: "el texto de esto
    le pasa lo mismo" (refiriéndose al mismo bug del fix P1-TRACKING-CACHE-CONSUMED
    pero en otro card).

Fix:
    Lazy initializers de `glasses`, `goal`, `goalBasis` leen de localStorage
    con clave `mealfit_water_state_<YYYY-MM-DD>`. Si match → arranca con
    valores reales → fetch silencioso en background.

    Key con fecha = TTL implícito 24h + invalidación automática en
    rollover de medianoche.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WATER_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "WaterTracker.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_water_state_lazy_initializers_read_cache():
    """[P1-WATER-CACHE-STATE] `glasses`, `goal`, `goalBasis` deben usar
    lazy initializers que lean del cache. Sin esto, arrancan con 0/default
    y el user ve flash."""
    src = _read(_WATER_JSX)
    # readWaterStateFromCache debe estar definido.
    assert "readWaterStateFromCache" in src, (
        "Helper `readWaterStateFromCache` ausente. Sin él, no hay forma de "
        "hidratar el state desde cache. Ver P1-WATER-CACHE-STATE · 2026-05-20."
    )
    # Los 3 setters deben usar el cache.
    # Pattern: useState(() => _cachedState?.<field> ?? <default>).
    for field in ["glasses", "goal", "goalBasis"]:
        pattern = rf"const\s*\[\s*{field}\s*,\s*set\w+\s*\]\s*=\s*useState\(\s*\(\s*\)\s*=>\s*_cachedState"
        assert re.search(pattern, src), (
            f"useState de `{field}` NO usa lazy initializer con `_cachedState`. "
            f"Sin esto, arranca con default y el user ve flash. Ver "
            f"P1-WATER-CACHE-STATE."
        )


def test_water_cache_key_includes_date():
    """[P1-WATER-CACHE-STATE] La cache key debe incluir la fecha de hoy
    para invalidarse automáticamente en rollover de medianoche."""
    src = _read(_WATER_JSX)
    # Buscar referencia al prefix + getLocalDateString.
    assert "LS_WATER_CACHE_PREFIX" in src
    # El helper recibe `userId` para scoping per-usuario de la cache key
    # (evita servir cache del usuario A al B en logout/login); el arg es
    # opcional en el regex para tolerar la firma `()` o `(userId)`.
    helper = re.search(
        r"const\s+readWaterStateFromCache\s*=\s*\([^)]*\)\s*=>\s*\{(.+?)\};",
        src,
        re.DOTALL,
    )
    assert helper, "Helper readWaterStateFromCache no encontrado."
    body = helper.group(1)
    assert "getLocalDateString" in body, (
        "Cache key no usa fecha — no invalida en rollover medianoche."
    )


def test_water_state_persists_on_change():
    """[P1-WATER-CACHE-STATE] Debe existir useEffect que persista
    glasses/goal/goalBasis al cambio, para que próximo mount encuentre
    cache fresco."""
    src = _read(_WATER_JSX)
    # [2026-07-26] Antes el regex exigía la lista de deps EXACTA y en ORDEN
    # (`[glasses, goal, goalBasis, currentDate(, userId)?]`). Fallaba porque el código se volvió
    # MÁS correcto: hoy son `[glasses, goal, goalBasis, streak, currentDate, userId]` — `streak`
    # se persiste en el payload y `userId` está en la key, así que ambos DEBEN ser deps. Un test
    # que se rompe cuando añades la dependencia que faltaba está midiendo la forma, no la
    # invariante.
    #
    # Ahora: se localiza el effect por lo que HACE (escribe la cache) y se exige que sus deps
    # sean un SUPERCONJUNTO de las obligatorias. Añadir deps sigue siendo válido; quitar una de
    # las que se persisten, no.
    efectos = re.findall(r"useEffect\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*,\s*\[([^\]]*)\]\s*\)",
                         src, re.DOTALL)
    persist = None
    for cuerpo, deps in efectos:
        if ("setItem" in cuerpo or "safeLocalStorageSet" in cuerpo) and "_waterCacheKey" in cuerpo:
            persist = (cuerpo, deps)
            break
    assert persist, (
        "No hay useEffect que escriba la cache de agua (`_waterCacheKey` + setItem) — el state "
        "no se persiste y el próximo mount arranca en frío. Ver P1-WATER-CACHE-STATE."
    )
    body, deps_txt = persist
    _deps = {d.strip() for d in deps_txt.split(",") if d.strip()}
    _obligatorias = {"glasses", "goal", "goalBasis", "currentDate", "userId"}
    assert _obligatorias <= _deps, (
        f"al persist effect le faltan deps: {sorted(_obligatorias - _deps)}. Todo valor que se "
        f"escriba en la cache (o que forme la key) tiene que ser dep, o el cache queda stale. "
        f"Deps actuales: {sorted(_deps)}"
    )
    # Lo que se escribe debe estar cubierto: si el payload incluye `streak`, `streak` es dep.
    for _campo in ("glasses", "goal", "goalBasis", "streak"):
        if _campo in body:
            assert _campo in _deps, (
                f"`{_campo}` se escribe en la cache pero NO es dependencia del effect: se "
                f"persistiría un valor viejo."
            )
    assert "setItem" in body or "safeLocalStorageSet" in body, (
        "Persist effect no escribe a localStorage."
    )
    # La construcción de la key se extrajo al helper `_waterCacheKey(userId,
    # currentDate)` (que internamente usa LS_WATER_CACHE_PREFIX) para scoping
    # per-usuario; el persist effect lo invoca en vez de inlinear el prefix.
    assert (
        "LS_WATER_CACHE_PREFIX" in body
        or "mealfit_water_state_" in body
        or "_waterCacheKey" in body
    ), (
        "Persist effect no usa la cache key esperada (ni el prefix inline ni "
        "el helper `_waterCacheKey`)."
    )


def test_loading_false_if_cache_hit():
    """[P1-WATER-CACHE-STATE] `loading` debe inicializar false si hay cache
    hit, para evitar el spinner cuando ya hay datos visibles."""
    src = _read(_WATER_JSX)
    # Buscar useState(loading) con lazy initializer condicional al cache.
    loading_match = re.search(
        r"const\s*\[\s*loading\s*,\s*setLoading\s*\]\s*=\s*useState\(\s*\(\s*\)\s*=>\s*([^)]+)\)",
        src,
        re.DOTALL,
    )
    assert loading_match, "useState lazy initializer para `loading` ausente."
    body = loading_match.group(1)
    assert "_cachedState" in body, (
        "Loading initializer NO consulta cache — siempre arranca en true. "
        "Spinner visible cuando hay datos. Ver P1-WATER-CACHE-STATE."
    )
