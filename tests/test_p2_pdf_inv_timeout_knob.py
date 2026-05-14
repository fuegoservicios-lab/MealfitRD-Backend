"""[P2-PDF-INV-TIMEOUT-KNOB · 2026-05-14] Knob para el timeout de
`fetchFreshInventoryWithTimeout` en frontend.

Motivación (audit 2026-05-14):
    Los 4 callsites de `fetchFreshInventoryWithTimeout` en `Dashboard.jsx`
    (mount, focus, PDF download, restock) pasaban el literal `2000` ms.
    Si Supabase entraba en degradación tail-latency (incidente regional,
    pool exhausted, network blip), no había forma de subir el timeout
    sin redeploy del frontend (Vercel build). El cron P2-SHOPPING-3
    `_alert_pdf_stale_inventory_fallback_burst` detectaría el burst de
    fallbacks pero la mitigación requería rebuild + redeploy.

Fix:
    Nuevo helper `getInventoryFetchTimeoutMs()` en
    `shoppingHelpers.js` que lee `import.meta.env.VITE_INVENTORY_FETCH_TIMEOUT_MS`
    con clamp defensivo:
      - Default 2000 ms (comportamiento pre-knob preservado).
      - Mínimo 500 ms.
      - Máximo 10000 ms.
      - NaN/undefined/string vacío → default 2000.

    Los 4 callsites en `Dashboard.jsx` pasan ahora `getInventoryFetchTimeoutMs()`
    en lugar del literal `2000`.

Drift detection (parser-based):
    1. El helper existe en `shoppingHelpers.js` con los 3 bounds
       documentados (default 2000, min 500, max 10000).
    2. El env var `VITE_INVENTORY_FETCH_TIMEOUT_MS` se lee con
       `parseInt(import.meta.env...VITE_INVENTORY_FETCH_TIMEOUT_MS...)`.
    3. Los 4 callsites de `Dashboard.jsx` invocan el helper (no pasan
       `2000` literal como argumento posicional a
       `fetchFreshInventoryWithTimeout`).
    4. Cap CLAUDE.md: el bound máximo (10000) está documentado por
       inspección directa del helper.

Whitelist:
    El test legacy `frontend/src/__tests__/utils/shoppingHelpers.test.js`
    sigue pasando `2000` como segundo arg a `fetchFreshInventoryWithTimeout`
    para validar el comportamiento del helper interno (no es un callsite
    productivo). Está excluido del scan vía path-prefix.

Tooltip-anchor: P2-PDF-INV-TIMEOUT-KNOB-START | gap audit 2026-05-14
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHOPPING_HELPERS = _REPO_ROOT / "frontend" / "src" / "utils" / "shoppingHelpers.js"
_DASHBOARD = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


@pytest.fixture(scope="module")
def helpers_src() -> str:
    return _SHOPPING_HELPERS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dashboard_src() -> str:
    return _DASHBOARD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Helper exported con nombre canónico
# ---------------------------------------------------------------------------
def test_helper_exported(helpers_src: str):
    """`getInventoryFetchTimeoutMs` DEBE estar exportado como named export."""
    pattern = re.compile(
        r"export\s+const\s+getInventoryFetchTimeoutMs\s*=\s*\(\s*\)\s*=>",
    )
    assert pattern.search(helpers_src), (
        "P2-PDF-INV-TIMEOUT-KNOB regresión: "
        "`getInventoryFetchTimeoutMs` no exportado en shoppingHelpers.js. "
        "Sin él, los callsites no pueden leer el knob y caerían a literal "
        "`2000` o al default del param de `fetchFreshInventoryWithTimeout`. "
        "Fix: añadir `export const getInventoryFetchTimeoutMs = () => {...}`."
    )


# ---------------------------------------------------------------------------
# 2. Lee env var VITE_INVENTORY_FETCH_TIMEOUT_MS
# ---------------------------------------------------------------------------
def test_helper_reads_env_var(helpers_src: str):
    """El helper DEBE leer `import.meta.env.VITE_INVENTORY_FETCH_TIMEOUT_MS`
    via `parseInt`. Sin esto, el knob no se respeta y siempre vuelve al
    default.
    """
    pattern = re.compile(
        r"parseInt\s*\(\s*import\.meta\.env\s*\??\.\s*VITE_INVENTORY_FETCH_TIMEOUT_MS",
    )
    assert pattern.search(helpers_src), (
        "P2-PDF-INV-TIMEOUT-KNOB regresión: el helper "
        "`getInventoryFetchTimeoutMs` no llama "
        "`parseInt(import.meta.env.VITE_INVENTORY_FETCH_TIMEOUT_MS, 10)`. "
        "Sin esa lectura, el env var del knob queda no consumido."
    )


# ---------------------------------------------------------------------------
# 3. Clamp default 2000 ms
# ---------------------------------------------------------------------------
def test_helper_default_is_2000(helpers_src: str):
    """El default debe ser exactamente `2000` ms cuando el env var es
    NaN/undefined. Subir o bajar el default es una decisión que requiere
    coordinación operacional — este test ancla el contrato.
    """
    # Match: `: raw : 2000;` o `Number.isFinite(raw) ? raw : 2000`
    pattern = re.compile(
        r"Number\.isFinite\s*\(\s*raw\s*\)\s*\?\s*raw\s*:\s*2000",
    )
    assert pattern.search(helpers_src), (
        "P2-PDF-INV-TIMEOUT-KNOB regresión: el default del helper no es "
        "`2000` ms (pattern `Number.isFinite(raw) ? raw : 2000`). El "
        "default debe preservar el comportamiento pre-knob — cambiar "
        "este número rompe la simetría documentada con el test legacy "
        "shoppingHelpers.test.js que asume 2000."
    )


# ---------------------------------------------------------------------------
# 4. Clamp [500, 10000] explícito
# ---------------------------------------------------------------------------
def test_helper_clamps_bounds(helpers_src: str):
    """El helper DEBE aplicar clamps `[500, 10000]` defensive — por debajo
    de 500 ms casi todos los fetches caerían a stale fallback; arriba de
    10000 ms el usuario asume que la UI se colgó.
    """
    lower_pat = re.compile(r"if\s*\(\s*ms\s*<\s*500\s*\)\s*ms\s*=\s*500\s*;")
    upper_pat = re.compile(r"if\s*\(\s*ms\s*>\s*10000\s*\)\s*ms\s*=\s*10000\s*;")
    assert lower_pat.search(helpers_src), (
        "P2-PDF-INV-TIMEOUT-KNOB regresión: el helper perdió el clamp "
        "inferior `if (ms < 500) ms = 500;`. Sin él, un POST adversarial "
        "con `VITE_INVENTORY_FETCH_TIMEOUT_MS=1` haría que casi todos los "
        "fetches caigan a stale fallback (UX degradado)."
    )
    assert upper_pat.search(helpers_src), (
        "P2-PDF-INV-TIMEOUT-KNOB regresión: el helper perdió el clamp "
        "superior `if (ms > 10000) ms = 10000;`. Sin él, un knob mal "
        "calibrado (`=600000`) colgaría al usuario 10 min esperando un "
        "fetch que va a fallar igual."
    )


# ---------------------------------------------------------------------------
# 5. Dashboard.jsx importa el helper
# ---------------------------------------------------------------------------
def test_dashboard_imports_helper(dashboard_src: str):
    """El import named `getInventoryFetchTimeoutMs` desde
    `../utils/shoppingHelpers` DEBE aparecer en `Dashboard.jsx`."""
    # Tolerar otros named imports en la misma línea/destructure.
    pattern = re.compile(
        r"import\s*\{[^}]*\bgetInventoryFetchTimeoutMs\b[^}]*\}\s*from\s*['\"]\.\./utils/shoppingHelpers['\"]",
        re.DOTALL,
    )
    assert pattern.search(dashboard_src), (
        "P2-PDF-INV-TIMEOUT-KNOB regresión: Dashboard.jsx no importa "
        "`getInventoryFetchTimeoutMs` desde shoppingHelpers. Sin él, los "
        "callsites caerían al default del param `timeoutMs=2000` y el "
        "knob queda no consumido. Fix: añadir el named import."
    )


# ---------------------------------------------------------------------------
# 6. Cero callsites con `2000` literal posicional en Dashboard.jsx
# ---------------------------------------------------------------------------
def test_no_hardcoded_2000_in_dashboard_callsites(dashboard_src: str):
    """Ningún call a `fetchFreshInventoryWithTimeout(..., 2000)` en
    Dashboard.jsx — todos deben pasar `getInventoryFetchTimeoutMs()`.
    """
    # Match: cualquier call a `fetchFreshInventoryWithTimeout(...)` donde
    # el último arg sea `2000` literal (con `,` y posibles espacios/newlines
    # alrededor). Balancear paréntesis exactamente requiere parser real;
    # heurística: buscar `2000,?\s*\)` precedido por ≤500 chars sin
    # `fetchFreshInventoryWithTimeout(` adicional.
    pattern = re.compile(
        r"fetchFreshInventoryWithTimeout\s*\([^()]*?(?:\([^)]*\)[^()]*?)*,\s*2000\s*,?\s*\)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(dashboard_src))
    assert not matches, (
        f"P2-PDF-INV-TIMEOUT-KNOB regresión: encontrados "
        f"{len(matches)} callsites en Dashboard.jsx que aún pasan "
        f"`2000` literal a `fetchFreshInventoryWithTimeout(...)` "
        f"(offsets: {[m.start() for m in matches]}). Reemplazar el "
        f"literal por `getInventoryFetchTimeoutMs()` para que el knob "
        f"`VITE_INVENTORY_FETCH_TIMEOUT_MS` se respete en CADA callsite."
    )


# ---------------------------------------------------------------------------
# 7. Cobertura: los 4 callsites invocan el helper
# ---------------------------------------------------------------------------
def test_dashboard_invokes_helper_at_least_four_times(dashboard_src: str):
    """Los 4 callsites de `fetchFreshInventoryWithTimeout` en Dashboard.jsx
    (mount, focus, PDF, restock) deben pasar `getInventoryFetchTimeoutMs()`.
    """
    invocation_pat = re.compile(r"getInventoryFetchTimeoutMs\s*\(\s*\)")
    matches = list(invocation_pat.finditer(dashboard_src))
    assert len(matches) >= 4, (
        f"P2-PDF-INV-TIMEOUT-KNOB regresión: solo {len(matches)} "
        f"invocaciones de `getInventoryFetchTimeoutMs()` encontradas en "
        f"Dashboard.jsx (esperadas ≥4 para cubrir mount/focus/PDF/restock). "
        f"Algunos callsites todavía pasan literal o usan el default del "
        f"param. Audit los 4 callsites de `fetchFreshInventoryWithTimeout`."
    )


# ---------------------------------------------------------------------------
# 8. Cross-link slug (P2-HIST-AUDIT-14): este NO es el marker activo,
#    pero el slug del archivo se preserva para grep manual.
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Filename DEBE contener `p2_pdf_inv_timeout_knob` para grep manual."""
    expected_slug = "p2_pdf_inv_timeout_knob"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        f"El nombre de este archivo debe contener `{expected_slug}` para "
        f"grep manual del P-fix. Aunque el marker activo del bundle es "
        f"`P2-RECALC-GROCERY-DURATION-ENUM` (último alfabético), este "
        f"slug se mantiene para auditoría granular."
    )
