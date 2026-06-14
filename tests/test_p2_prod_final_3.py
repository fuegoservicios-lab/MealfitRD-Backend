"""[P2-PROD-FINAL-3 · 2026-05-24] Test umbrella + cross-link del bundle P2.

Bundle 3-en-1 que cierra los 3 P2 residuales del audit prod-readiness
2026-05-24 (post P1-FRONTEND-FINAL-1, P1 limpio):

  GAP-1 P2-KNOBS-ENV-INT-NO-VALIDATOR:
    `backend/knobs.py::_env_int` carecía del parámetro `validator` opcional
    que `_env_float` sí soporta. Resultado: knobs int requerían clamp manual
    post-lectura (`_EMBEDDING_CACHE_MAXSIZE` lo hacía con max/min) o
    aceptaban valores absurdos silenciosamente (`MEALFIT_CB_FAILURE_THRESHOLD=0`
    abre el breaker eternamente). Fix: simetría con `_env_float` — validator
    opcional `(int) -> bool`, fail → WARNING + cae al default + parse_failed.

  GAP-2 P2-WATER-RETRY-NO-JITTER:
    `routers/plans.py::_execute_with_retry` usaba backoff constante
    350ms sin jitter ni exponencial. Thundering herd contra Supabase post-
    blip regional. Fix: backoff exponencial + jitter absoluto + 2 knobs
    `MEALFIT_WATER_RETRY_BACKOFF_BASE_S` y `MEALFIT_WATER_RETRY_JITTER_MAX_S`.

  GAP-3 P2-CUSTOM-MODALS-A11Y:
    3 modales custom sin `role="dialog"`, `aria-modal`, focus trap, ESC,
    restore focus: `PaymentModal.jsx` (CRÍTICO: surface PayPal),
    `LogoutConfirmModal.jsx`, restock modal inline en `Dashboard.jsx:4475+`.
    Fix: hook SSOT `frontend/src/hooks/useModalAccessibility.js` aplicado
    inline a los 3 (preserva layouts custom, especialmente el split-screen
    full-bleed de PaymentModal que NO encaja en Modal.jsx).

Verificaciones detalladas en este archivo + tests vitest específicos:
  - `frontend/src/__tests__/Modals.p2_a11y.test.js`

Cross-link guard P2-HIST-AUDIT-14: el slug `p2_prod_final_3` matchea
el marker `P2-PROD-FINAL-3`.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_APP_PY = _BACKEND / "app.py"
_KNOBS = _BACKEND / "knobs.py"
_CONSTANTS = _BACKEND / "constants.py"
_GRAPH_ORCH = _BACKEND / "graph_orchestrator.py"
_PLANS_ROUTER = _BACKEND / "routers" / "plans.py"
_FRONTEND = _REPO_ROOT / "frontend" / "src"
_USE_MODAL_A11Y = _FRONTEND / "hooks" / "useModalAccessibility.js"
_PAYMENT_MODAL = _FRONTEND / "components" / "dashboard" / "PaymentModal.jsx"
_LOGOUT_MODAL = _FRONTEND / "components" / "dashboard" / "LogoutConfirmModal.jsx"
_DASHBOARD = _FRONTEND / "pages" / "Dashboard.jsx"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — P2-KNOBS-ENV-INT-NO-VALIDATOR
# ---------------------------------------------------------------------------
def test_env_int_accepts_validator_param():
    """`_env_int` debe aceptar `validator: Optional[Callable[[int], bool]] = None`
    simétrico al `_env_float`. La signature exacta requiere el kw arg."""
    src = _read(_KNOBS)
    pat = re.compile(
        r"def\s+_env_int\s*\(\s*name:\s*str\s*,\s*default:\s*int\s*,\s*"
        r"validator:\s*Optional\[Callable\[\[int\]\s*,\s*bool\]\]\s*=\s*None",
        re.DOTALL,
    )
    assert pat.search(src), (
        "`_env_int` debe tener signature "
        "`def _env_int(name: str, default: int, validator: Optional[Callable[[int], bool]] = None)`. "
        "Sin el validator, knobs int críticos requieren clamp manual + permiten "
        "valores absurdos silenciosamente."
    )


def test_env_int_validator_logs_warning_and_falls_back():
    """El cuerpo del helper debe ejecutar el validator + caer al default con
    WARNING + parse_failed=True cuando retorna False."""
    src = _read(_KNOBS)
    # Buscamos el helper completo y verificamos las ramas críticas.
    fn_start = src.find("def _env_int(")
    assert fn_start >= 0
    fn_body = src[fn_start: fn_start + 2500]
    # Validator ejecutado.
    assert "validator(value)" in fn_body, (
        "`_env_int` body debe invocar `validator(value)` para chequear el rango."
    )
    # WARNING al fallar.
    assert re.search(r"fuera de rango permitido", fn_body), (
        "`_env_int` debe loguear WARNING `fuera de rango permitido` cuando validator False."
    )
    # parse_failed=True en fallback.
    assert re.search(r"parse_failed\s*=\s*True", fn_body), (
        "`_env_int` debe marcar `parse_failed=True` en _register_knob cuando validator False."
    )


def test_anchor_env_int_validator_in_knobs():
    src = _read(_KNOBS)
    assert "P2-KNOBS-ENV-INT-NO-VALIDATOR" in src, (
        "Falta anchor `P2-KNOBS-ENV-INT-NO-VALIDATOR` en backend/knobs.py."
    )


def test_embedding_cache_maxsize_uses_validator():
    """`_EMBEDDING_CACHE_MAXSIZE` migrado de clamp manual `max/min` a
    `_env_int(..., validator=lambda v: 100 <= v <= 100_000)`."""
    src = _read(_CONSTANTS)
    pat = re.compile(
        r"_EMBEDDING_CACHE_MAXSIZE\s*=\s*_knob_env_int_constants\s*\(\s*"
        r"[\"']MEALFIT_EMBEDDING_CACHE_MAXSIZE[\"']\s*,\s*5000\s*,\s*"
        r"validator\s*=\s*lambda\s+\w+\s*:\s*100\s*<=\s*\w+\s*<=\s*100_000",
        re.DOTALL,
    )
    assert pat.search(src), (
        "`_EMBEDDING_CACHE_MAXSIZE` debe usar `_env_int(..., validator=...)` "
        "en lugar de clamp manual con `max/min`. Migración P2-KNOBS-ENV-INT-NO-VALIDATOR."
    )


def test_cb_thresholds_use_validator():
    """`CB_FAILURE_THRESHOLD` y `CB_RESET_TIMEOUT_S` migrados al validator
    pattern. Pre-fix `=0` o negativo abría el breaker eternamente."""
    src = _read(_GRAPH_ORCH)
    pat_threshold = re.compile(
        r"CB_FAILURE_THRESHOLD\s*=\s*_env_int\s*\(\s*[\"']MEALFIT_CB_FAILURE_THRESHOLD[\"']\s*,\s*3\s*,\s*"
        r"validator\s*=\s*lambda\s+\w+\s*:\s*1\s*<=\s*\w+\s*<=\s*1000",
        re.DOTALL,
    )
    assert pat_threshold.search(src), (
        "`CB_FAILURE_THRESHOLD` debe usar `_env_int(..., validator=lambda v: 1 <= v <= 1000)`. "
        "Sin validator, `MEALFIT_CB_FAILURE_THRESHOLD=0` abre el breaker eternamente."
    )
    pat_timeout = re.compile(
        r"CB_RESET_TIMEOUT_S\s*=\s*_env_int\s*\(\s*[\"']MEALFIT_CB_RESET_TIMEOUT_S[\"']\s*,\s*30\s*,\s*"
        r"validator\s*=\s*lambda\s+\w+\s*:\s*1\s*<=\s*\w+\s*<=\s*86_400",
        re.DOTALL,
    )
    assert pat_timeout.search(src), (
        "`CB_RESET_TIMEOUT_S` debe usar `_env_int(..., validator=lambda v: 1 <= v <= 86_400)`. "
        "Sin validator, `MEALFIT_CB_RESET_TIMEOUT_S=0` deja el breaker open sin recovery."
    )


# ---------------------------------------------------------------------------
# Sección 2 — P2-WATER-RETRY-NO-JITTER
# ---------------------------------------------------------------------------
def test_anchor_water_retry_jitter_in_plans():
    src = _read(_PLANS_ROUTER)
    assert "P2-WATER-RETRY-NO-JITTER" in src, (
        "Falta anchor `P2-WATER-RETRY-NO-JITTER` en backend/routers/plans.py."
    )


def test_water_retry_uses_knob_with_clamp():
    """`_WATER_RETRY_BACKOFF_BASE_S` resuelto via `_env_float(name, 0.35,
    validator=lambda v: 0.05 <= v <= 5.0)`."""
    src = _read(_PLANS_ROUTER)
    base_pat = re.compile(
        r"_WATER_RETRY_BACKOFF_BASE_S\s*=\s*_env_float\s*\(\s*"
        r"[\"']MEALFIT_WATER_RETRY_BACKOFF_BASE_S[\"']\s*,\s*0\.35\s*,\s*"
        r"validator\s*=\s*lambda\s+\w+\s*:\s*0\.05\s*<=\s*\w+\s*<=\s*5\.0",
        re.DOTALL,
    )
    assert base_pat.search(src), (
        "`_WATER_RETRY_BACKOFF_BASE_S` debe usar `_env_float(..., validator=...)` "
        "con clamp [0.05, 5.0]."
    )
    jitter_pat = re.compile(
        r"_WATER_RETRY_JITTER_MAX_S\s*=\s*_env_float\s*\(\s*"
        r"[\"']MEALFIT_WATER_RETRY_JITTER_MAX_S[\"']\s*,\s*0\.1\s*,\s*"
        r"validator\s*=\s*lambda\s+\w+\s*:\s*0\.0\s*<=\s*\w+\s*<=\s*1\.0",
        re.DOTALL,
    )
    assert jitter_pat.search(src), (
        "`_WATER_RETRY_JITTER_MAX_S` debe usar `_env_float(..., validator=...)` "
        "con clamp [0.0, 1.0]."
    )


def test_water_retry_helper_uses_exponential_plus_jitter():
    """El helper `_execute_with_retry` calcula `sleep_s = base *
    (2 ** attempt) + random.uniform(0.0, jitter_max)`. Pre-fix era constante
    `_time.sleep(_WATER_RETRY_BACKOFF_S)` sin jitter."""
    src = _read(_PLANS_ROUTER)
    fn_start = src.find("def _execute_with_retry")
    assert fn_start >= 0
    fn_body = src[fn_start: fn_start + 3000]
    # Backoff exponencial: base * (2 ** attempt)
    assert re.search(
        r"_WATER_RETRY_BACKOFF_BASE_S\s*\*\s*\(\s*2\s*\*\*\s*attempt\s*\)",
        fn_body,
    ), "El helper debe calcular backoff exponencial `base * (2 ** attempt)`."
    # Jitter: random.uniform(0.0, jitter_max)
    assert re.search(
        r"_random\.uniform\s*\(\s*0\.0\s*,\s*_WATER_RETRY_JITTER_MAX_S\s*\)",
        fn_body,
    ), "El helper debe añadir jitter `random.uniform(0.0, jitter_max)`."


# ---------------------------------------------------------------------------
# Sección 3 — P2-CUSTOM-MODALS-A11Y
# ---------------------------------------------------------------------------
def test_use_modal_accessibility_hook_exists():
    """El hook SSOT `useModalAccessibility` debe existir en frontend/src/hooks/."""
    assert _USE_MODAL_A11Y.exists(), (
        "Falta `frontend/src/hooks/useModalAccessibility.js`. Hook SSOT "
        "para focus trap + ESC + restore focus + body overflow."
    )
    src = _read(_USE_MODAL_A11Y)
    assert "P2-CUSTOM-MODALS-A11Y" in src
    # Export named.
    assert re.search(r"export\s+function\s+useModalAccessibility", src), (
        "useModalAccessibility debe ser export named."
    )
    # ESC key handler.
    assert "Escape" in src, "Hook debe escuchar tecla Escape."
    # Focus trap (Tab key).
    assert "'Tab'" in src or '"Tab"' in src, "Hook debe manejar Tab para focus trap."
    # Restore focus.
    assert "triggerRef" in src, "Hook debe restaurar focus al trigger original."
    # Body overflow.
    assert "document.body.style.overflow" in src, "Hook debe lock body overflow."


def test_payment_modal_uses_a11y_hook():
    """PaymentModal debe importar y usar useModalAccessibility + tener
    role/aria-modal/aria-labelledby en el root."""
    src = _read(_PAYMENT_MODAL)
    assert "P2-CUSTOM-MODALS-A11Y" in src, (
        "Falta anchor `P2-CUSTOM-MODALS-A11Y` en PaymentModal.jsx."
    )
    assert re.search(
        r"import\s*\{[^}]*useModalAccessibility[^}]*\}\s*from\s*['\"]\.\./\.\./hooks/useModalAccessibility['\"]",
        src,
    ), "PaymentModal debe importar useModalAccessibility."
    assert "useModalAccessibility(" in src
    # role + aria
    assert re.search(r'role\s*=\s*["\']dialog["\']', src), (
        'PaymentModal root debe tener role="dialog".'
    )
    assert re.search(r'aria-modal\s*=\s*["\']true["\']', src), (
        'PaymentModal root debe tener aria-modal="true".'
    )
    assert re.search(r'aria-labelledby\s*=\s*["\']payment-modal-title["\']', src), (
        'PaymentModal root debe tener aria-labelledby apuntando a payment-modal-title.'
    )
    assert re.search(r'id\s*=\s*["\']payment-modal-title["\']', src), (
        "El heading principal debe tener id=`payment-modal-title`."
    )


def test_logout_confirm_modal_uses_a11y_hook():
    """LogoutConfirmModal debe importar y usar el hook + role/aria."""
    src = _read(_LOGOUT_MODAL)
    assert "P2-CUSTOM-MODALS-A11Y" in src
    assert re.search(
        r"import\s*\{[^}]*useModalAccessibility[^}]*\}\s*from\s*['\"]\.\./\.\./hooks/useModalAccessibility['\"]",
        src,
    ), "LogoutConfirmModal debe importar useModalAccessibility."
    assert "useModalAccessibility(" in src
    assert re.search(r'role\s*=\s*["\']dialog["\']', src)
    assert re.search(r'aria-modal\s*=\s*["\']true["\']', src)
    assert re.search(r'aria-labelledby\s*=\s*["\']logout-confirm-title["\']', src)
    assert re.search(r'id\s*=\s*["\']logout-confirm-title["\']', src)


def test_dashboard_restock_modal_uses_a11y_hook():
    """Dashboard.jsx restock modal inline debe usar el hook + role/aria."""
    src = _read(_DASHBOARD)
    assert "P2-CUSTOM-MODALS-A11Y" in src
    assert re.search(
        r"import\s*\{[^}]*useModalAccessibility[^}]*\}\s*from\s*['\"]\.\./hooks/useModalAccessibility['\"]",
        src,
    ), "Dashboard debe importar useModalAccessibility."
    # El hook se invoca con showRestockModal + isRestocking.
    assert re.search(
        r"useModalAccessibility\s*\(\s*\{[^}]*isOpen:\s*showRestockModal",
        src,
        re.DOTALL,
    ), "Dashboard debe invocar useModalAccessibility con isOpen=showRestockModal."
    # Restock modal root con role/aria.
    assert re.search(r'aria-labelledby\s*=\s*["\']restock-modal-title["\']', src), (
        'Restock modal debe tener aria-labelledby="restock-modal-title".'
    )
    assert re.search(r'id\s*=\s*["\']restock-modal-title["\']', src), (
        "El heading 'Confirmar compra' debe tener id=`restock-modal-title`."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Marker bumped + date-floor
# ---------------------------------------------------------------------------
def test_marker_bumped_to_p2_prod_final_3():
    """`_LAST_KNOWN_PFIX` debe estar bumpeado al cierre del bundle.
    Exact-match al cierre; RELAJADO a date-floor tras supersede (patrón
    emergente desde P1-PROD-FINAL-1). Superseded por P1-PROD-HARDEN-BUNDLE
    (2026-05-27) y bundles intermedios — ahora solo verifica no-regresión
    bajo el floor de este bundle. El exact-match vive en
    `test_p3_1_last_known_pfix_freshness` (floor global) y en el test del
    bundle vigente."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 24), (
        f"Marker `{marker}` (fecha {marker_date}) regresó por debajo del "
        f"floor de P2-PROD-FINAL-3 (2026-05-24)."
    )


def test_marker_date_meets_p2_prod_final_3_floor():
    """Date-floor sibling para futuros supersedes."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 24)
    assert marker_date >= floor, (
        f"Marker `{marker}` con fecha {marker_date} < floor {floor}."
    )


# ---------------------------------------------------------------------------
# Sección 5 — Cross-link guard P2-HIST-AUDIT-14
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    """El slug `p2_prod_final_3` matchea el marker `P2-PROD-FINAL-3`."""
    src = _read(Path(__file__))
    assert "P2-PROD-FINAL-3" in src
