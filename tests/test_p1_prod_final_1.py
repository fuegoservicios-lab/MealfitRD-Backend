"""[P1-PROD-FINAL-1 · 2026-05-23] Regression guards for the production-readiness
P1 bundle that closed the audit findings of 2026-05-23 post-P1-HISTORY-ABORT.

Bundle scope:
  1. **P1-FRONTEND-LEGACY-LOCALSTORAGE-CRITICAL** — `Settings.jsx` lazy initializer
     y `Dashboard.jsx` push-onboarding useEffect migrados a `safeLocalStorageGet`.
     Pre-fix: raw `localStorage.getItem(...)` lanzaba `SecurityError` en iOS
     Private Mode dentro del mount/effect → página blanca (Settings) o
     onboarding silenciosamente roto (Dashboard).
  2. **P1-FRONTEND-RECIPES-LEGACY-AUTH** — `Recipes.jsx` `handleLogConsumption`
     migrado de `localStorage.getItem('supabase.auth.token')` + JSON.parse +
     fetch manual a `fetchWithAuth` (SSOT). Pre-fix usaba la key legacy de
     Supabase JS v1 (`'supabase.auth.token'`); v2 usa `sb-<ref>-auth-token`
     con shape distinto → `Bearer ` vacío → 401 silencioso.
  3. **P1-DASHBOARD-POLLING-ABORT** — el setInterval del Dashboard que
     polleaba `getPlanChunkStatus` cada 30s NO abortaba las fetches
     in-flight cuando el usuario navegaba fuera. AbortController scoped al
     useEffect cancela ambos (initial + setInterval) en cleanup. Mismo
     patrón que P1-HISTORY-ABORT (2026-05-23) aplicó a History.jsx.

Estrategia de los tests:
  - Parser-based (regex sobre source). NO ejecutamos JSX — los frontend
    tests vitest cubren el comportamiento; aquí ancoramos el contrato
    estructural para que un refactor inadvertido falle CI sin esperar
    al vitest.

Tooltip-anchors en source para detectar refactors:
  - `P1-FRONTEND-LEGACY-LOCALSTORAGE-CRITICAL · 2026-05-23` en
    Settings.jsx, Dashboard.jsx.
  - `P1-FRONTEND-RECIPES-LEGACY-AUTH · 2026-05-23` en Recipes.jsx.
  - `P1-DASHBOARD-POLLING-ABORT · 2026-05-23` en Dashboard.jsx,
    config/api.js.

P-fix umbrella anchor: `P1-PROD-FINAL-1` — slug `p1_prod_final_1` matchea
este archivo y satisface el cross-link enforcer `test_p2_hist_audit_14`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_SETTINGS_JSX = _FRONTEND_SRC / "pages" / "Settings.jsx"
_DASHBOARD_JSX = _FRONTEND_SRC / "pages" / "Dashboard.jsx"
_RECIPES_JSX = _FRONTEND_SRC / "pages" / "Recipes.jsx"
_API_JS = _FRONTEND_SRC / "config" / "api.js"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — P1-FRONTEND-LEGACY-LOCALSTORAGE-CRITICAL
# ---------------------------------------------------------------------------
def test_settings_notifications_lazy_init_uses_safe_localstorage():
    """`Settings.jsx` línea ~68: el lazy initializer de `notifications` debe
    usar `safeLocalStorageGet`, NO `localStorage.getItem` raw.

    Pre-fix lanzaba SecurityError en iOS Private Mode dentro del useState
    lazy init → todo el componente Settings crasheaba al mount.
    """
    text = _read(_SETTINGS_JSX)
    # Localizar el bloque del state notifications.
    # Permitimos espacios variables pero matcheamos contrato exacto.
    pattern = re.compile(
        r"useState\(\(\)\s*=>\s*\{\s*\n[^}]*"
        r"safeLocalStorageGet\(\s*['\"]mealfit_notifications['\"]\s*\)",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "Settings.jsx lazy initializer de `notifications` no usa "
        "safeLocalStorageGet. Si refactoreaste, asegúrate de mantener "
        "el wrapper defensivo (iOS Private Mode lanza SecurityError)."
    )
    # Y NO debe quedar la versión raw legacy.
    assert "localStorage.getItem('mealfit_notifications')" not in text, (
        "Settings.jsx tiene un `localStorage.getItem('mealfit_notifications')` "
        "raw — debe ser `safeLocalStorageGet('mealfit_notifications')`."
    )


def test_settings_imports_safe_localstorage_get():
    """`Settings.jsx` debe importar `safeLocalStorageGet` desde el helper SSOT."""
    text = _read(_SETTINGS_JSX)
    assert re.search(
        r"from\s+['\"]\.\./utils/safeLocalStorage['\"]",
        text,
    ), "Settings.jsx no importa desde utils/safeLocalStorage."
    assert "safeLocalStorageGet" in text, (
        "Settings.jsx no menciona safeLocalStorageGet (esperado en import)."
    )


def test_dashboard_push_onboarding_uses_safe_localstorage():
    """`Dashboard.jsx` useEffect del push-onboarding modal debe usar
    `safeLocalStorageGet('mealfit_push_onboarding_seen')`.

    Pre-fix: raw `localStorage.getItem(...)` lanzaba SecurityError en iOS
    Private Mode dentro del useEffect callback → onboarding modal nunca
    se disparaba para usuarios nuevos en ese entorno.
    """
    text = _read(_DASHBOARD_JSX)
    assert "safeLocalStorageGet('mealfit_push_onboarding_seen')" in text, (
        "Dashboard.jsx no usa safeLocalStorageGet para la key "
        "'mealfit_push_onboarding_seen'."
    )
    # NO debe haber raw getItem para esa key.
    assert (
        "localStorage.getItem('mealfit_push_onboarding_seen')" not in text
    ), (
        "Dashboard.jsx tiene un raw `localStorage.getItem('mealfit_push_"
        "onboarding_seen')` — debe ser `safeLocalStorageGet`."
    )


def test_dashboard_imports_safe_localstorage_get():
    """`Dashboard.jsx` debe importar `safeLocalStorageGet`."""
    text = _read(_DASHBOARD_JSX)
    assert re.search(
        r"safeLocalStorageGet.*from\s+['\"]\.\./utils/safeLocalStorage['\"]",
        text,
    ) or (
        "safeLocalStorageGet" in text
        and re.search(r"from\s+['\"]\.\./utils/safeLocalStorage['\"]", text)
    ), "Dashboard.jsx no importa safeLocalStorageGet desde utils/safeLocalStorage."


# ---------------------------------------------------------------------------
# Sección 2 — P1-FRONTEND-RECIPES-LEGACY-AUTH
# ---------------------------------------------------------------------------
def test_recipes_handle_log_consumption_uses_fetch_with_auth():
    """`Recipes.jsx::handleLogConsumption` debe usar `fetchWithAuth`, NO
    fetch manual + token extraction desde localStorage.

    Pre-fix usaba `localStorage.getItem('supabase.auth.token')` que es key
    legacy Supabase JS v1; v2 usa `sb-<project-ref>-auth-token` con shape
    distinto. Resultado: `Bearer ` vacío → backend 401 silencioso.
    """
    text = _read(_RECIPES_JSX)
    # Debe invocar fetchWithAuth contra /api/diary/consumed.
    assert "fetchWithAuth('/api/diary/consumed'" in text, (
        "Recipes.jsx::handleLogConsumption ya no llama "
        "fetchWithAuth('/api/diary/consumed', ...)."
    )
    # NO debe quedar la key legacy 'supabase.auth.token'.
    assert "'supabase.auth.token'" not in text, (
        "Recipes.jsx aún referencia la key legacy 'supabase.auth.token' "
        "(formato Supabase JS v1, no funciona con v2)."
    )
    # NO debe construir Authorization Bearer manualmente.
    assert "'Authorization': `Bearer ${jwt}`" not in text, (
        "Recipes.jsx aún construye el header Authorization manualmente — "
        "fetchWithAuth lo provee."
    )


# ---------------------------------------------------------------------------
# Sección 3 — P1-DASHBOARD-POLLING-ABORT
# ---------------------------------------------------------------------------
def test_get_plan_chunk_status_accepts_options():
    """`config/api.js::getPlanChunkStatus(planId, options)` debe aceptar
    `options` y forwardearlas a `fetchWithAuth`. Sin esto el AbortController
    del Dashboard no puede pasar el `signal`."""
    text = _read(_API_JS)
    assert re.search(
        r"export\s+const\s+getPlanChunkStatus\s*=\s*\("
        r"\s*planId\s*,\s*options\s*=\s*\{\s*\}\s*\)\s*=>\s*"
        r"fetchWithAuth\(\s*`[^`]+chunk-status`\s*,\s*options\s*\)",
        text,
    ), (
        "getPlanChunkStatus no acepta `options = {}` como segundo arg "
        "ni las forwardea a fetchWithAuth. Sin signal, el polling del "
        "Dashboard no puede ser abortado en unmount."
    )


def test_dashboard_polling_uses_abort_controller():
    """El useEffect del Dashboard que pollea chunk-status debe crear un
    `AbortController`, pasar `signal` a `getPlanChunkStatus`, y abortar en
    cleanup. Mismo patrón que P1-HISTORY-ABORT aplicó a History.jsx."""
    text = _read(_DASHBOARD_JSX)
    # Marker comment para identificar el useEffect correcto.
    assert "P1-DASHBOARD-POLLING-ABORT" in text, (
        "Dashboard.jsx no tiene el anchor `P1-DASHBOARD-POLLING-ABORT` "
        "— un refactor pudo haber removido el AbortController."
    )
    # Debe construir un AbortController.
    assert "new AbortController()" in text, (
        "Dashboard.jsx no instancia `new AbortController()` — el cleanup "
        "no puede abortar fetches in-flight."
    )
    # Debe pasar { signal } a getPlanChunkStatus.
    assert re.search(
        r"getPlanChunkStatus\([^,]+,\s*\{\s*signal\s*\}\s*\)",
        text,
    ), (
        "Dashboard.jsx no forwardea `{ signal }` a getPlanChunkStatus. "
        "El polling sigue siendo no-cancelable."
    )
    # Debe abortar en cleanup.
    assert "controller.abort()" in text, (
        "Dashboard.jsx no llama controller.abort() en cleanup — "
        "fetches in-flight sobreviven al unmount."
    )


def test_dashboard_polling_guards_aborted_before_setstate():
    """Tras el await `r.json()` debe haber un guard `signal.aborted` antes
    del setState. Sin esto, una response que llegó milisegundos antes del
    abort() seguiría llamando setChunkStatusInfo sobre unmounted."""
    text = _read(_DASHBOARD_JSX)
    # Buscar el patrón guard inmediatamente antes de setChunkStatusInfo
    # dentro del bloque del useEffect del polling.
    # Estrategia: contar guards `if (signal.aborted) return;` en el archivo
    # — debe haber ≥2 (uno antes del setState inicial, otro dentro del
    # setInterval) tras el cambio P1-DASHBOARD-POLLING-ABORT.
    guard_count = len(re.findall(r"if\s*\(\s*signal\.aborted\s*\)\s*return", text))
    assert guard_count >= 2, (
        f"Dashboard.jsx tiene {guard_count} guards `if (signal.aborted) "
        f"return` — esperaba >=2 (initial fetch + setInterval poll). "
        f"Si reduces el count, asegúrate de que ningún setState corra "
        f"sobre componente desmontado."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Marker bump
# ---------------------------------------------------------------------------
def test_last_known_pfix_marker_post_p1_prod_final_1():
    """`_LAST_KNOWN_PFIX` en app.py debe tener fecha >= 2026-05-23 (fecha
    de cierre del bundle P1-PROD-FINAL-1). El marker pudo ser superseded
    por un P-fix posterior (válido); lo que NO puede ocurrir es que se
    revierta a una fecha previa sin revertir también los 4 cambios del
    bundle (Settings.jsx safeLocalStorageGet, Dashboard.jsx 2x fixes,
    Recipes.jsx fetchWithAuth).

    [Relajado del exact-match `P1-PROD-FINAL-1` original 2026-05-23: el
    docstring del test ya anticipaba superseding por un bundle posterior.
    Las otras 9 assertions del archivo enforzan que los 4 fixes siguen
    vivos.]"""
    from datetime import date, datetime
    app_py = _REPO_ROOT / "backend" / "app.py"
    text = app_py.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    p1_prod_final_1_floor = date(2026, 5, 23)
    assert marker_date >= p1_prod_final_1_floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor "
        f"{p1_prod_final_1_floor} (P1-PROD-FINAL-1). Si revertiste el "
        f"marker debes también revertir los 4 fixes frontend del bundle "
        f"(los otros 9 tests de este archivo te avisarán)."
    )


# ---------------------------------------------------------------------------
# Sección 5 — Cap margin de CLAUDE.md tras la limpieza
# ---------------------------------------------------------------------------
def test_claude_md_has_breathing_room_after_cleanup():
    """Tras el cleanup de billing+webhook anti-patrones, CLAUDE.md debe
    estar al menos 800 chars bajo el cap (margen >= 1.5%). El cap está
    en test_p3_claudemd_cap.py — aquí solo enforzamos un floor de margen
    sano para que el próximo P-fix con tabla no tope inmediatamente."""
    claude_md = _REPO_ROOT / "CLAUDE.md"
    cap_test = _REPO_ROOT / "backend" / "tests" / "test_p3_claudemd_cap.py"
    cap_text = cap_test.read_text(encoding="utf-8")
    cap_match = re.search(r"_DEFAULT_CAP\s*=\s*(\d+)", cap_text)
    assert cap_match, "No pude parsear _DEFAULT_CAP del test cap."
    cap = int(cap_match.group(1))
    size = claude_md.stat().st_size
    margin = cap - size
    assert margin >= 800, (
        f"CLAUDE.md size={size}, cap={cap}, margin={margin} chars. "
        f"Esperaba >=800 chars de margen tras el cleanup P1-PROD-FINAL-1. "
        f"Si está más cerca del cap, repite la pasada de limpieza "
        f"estructural antes de añadir nuevo contenido."
    )
