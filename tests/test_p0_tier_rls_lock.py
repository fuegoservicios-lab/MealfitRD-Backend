"""[P0-TIER-RLS-LOCK · 2026-05-31] Cierre de la escalación de tier client-side.

Gap (audit frontend speed+security 2026-05-31):
    `AssessmentContext.jsx::upgradeUserPlan` tenía una rama `else` ("admin
    bypass") que hacía
        ``supabase.from('user_profiles').update({plan_tier: tier}).eq('id', userId)``
    — otorgando un tier arbitrario SIN verificación de pago. Combinado con la
    RLS de `user_profiles` (policy "Usuarios editan su propio perfil" permite
    UPDATE de cualquier columna de la propia fila) y el UPDATE a nivel de tabla
    de `authenticated`, CUALQUIER usuario logueado podía ejecutar desde la
    consola del browser:
        ``supabase.from('user_profiles').update({plan_tier:'ultra'}).eq('id', miId)``
    y auto-otorgarse tier ilimitado (o 'admin'), evadiendo TODO el billing
    server-side (`auth.verify_api_quota` deriva el límite de `plan_tier`;
    `agent.py` gatea features premium por `plan_tier`). Es el hueco simétrico
    client-side de I-Billing-1 (tier debe derivarse de PayPal en el backend).

Defensa en dos capas (este test ancla ambas):
    1. DB (capa real): trigger `BEFORE UPDATE` `trg_guard_user_profiles_entitlement`
       (función `guard_user_profiles_entitlement_columns`) que RAISE si un rol
       cliente (`authenticated`/`anon`) cambia plan_tier / paypal_* /
       subscription_*. Backend (postgres/service_role) exento. Migración SSOT
       `p0_user_profiles_entitlement_lock_2026_05_31.sql` en AMBOS dirs.
    2. Frontend: eliminada la rama `else`; `upgradeUserPlan` ahora fail-closed
       (throw) sin subscriptionId. Este test bloquea cualquier FUTURO callsite
       que reintroduzca un write client-side de columnas de billing.

Tooltip-anchor: P0-TIER-RLS-LOCK-START | gap audit 2026-05-31
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

# Columnas de entitlement/billing que el cliente JAMÁS debe escribir.
_BILLING_COLUMNS = (
    "plan_tier",
    "paypal_subscription_id",
    "paypal_plan_id",
    "subscription_status",
    "subscription_end_date",
)

# Detecta `supabase.from('user_profiles').update(` (cualquier whitespace).
_UPDATE_CALL = re.compile(
    r"supabase\s*\.\s*from\s*\(\s*['\"]user_profiles['\"]\s*\)\s*\.\s*update\s*\(",
    re.IGNORECASE | re.DOTALL,
)

# Migraciones SSOT (deben existir idénticas en ambos dirs).
_MIGRATION_DIRS = (
    _REPO_ROOT / "supabase" / "migrations",
    _REPO_ROOT / "backend" / "supabase" / "migrations",
)
_MIGRATION_NAME = "p0_user_profiles_entitlement_lock_2026_05_31.sql"


def _strip_js_comments(src: str) -> str:
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", src)
    no_line = re.sub(r"//[^\n]*", "", no_block)
    return no_line


def _iter_frontend_files():
    for f in _FRONTEND_SRC.rglob("*"):
        if not f.is_file() or f.suffix not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        parts = {p.lower() for p in f.parts}
        if "__tests__" in parts:
            continue
        nl = f.name.lower()
        if nl.endswith((".test.js", ".test.jsx", ".test.ts", ".test.tsx", ".d.ts")):
            continue
        yield f


# ---------------------------------------------------------------------------
# 1. Frontend NO escribe columnas de billing a user_profiles
# ---------------------------------------------------------------------------
def test_frontend_no_user_profiles_billing_column_write():
    """Falla si algún `supabase.from('user_profiles').update({...})` del
    frontend incluye una columna de entitlement/billing en el payload literal.

    El path legítimo (`updateUserProfile`) hace `.update(rest)` / `.update({
    health_profile })` / `.update({ full_name })` — ninguno contiene columnas
    de billing, así que no matchea. Solo un payload literal con plan_tier /
    paypal_* / subscription_* dispara la violación.
    """
    offenders: list[str] = []
    for f in _iter_frontend_files():
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        stripped = _strip_js_comments(src)
        for m in _UPDATE_CALL.finditer(stripped):
            # Ventana del payload: desde el `(` del update hasta ~400 chars
            # después (cubre objetos literales multilínea típicos).
            window = stripped[m.end(): m.end() + 400]
            for col in _BILLING_COLUMNS:
                # `col:` o `col :` como key del objeto literal.
                if re.search(r"\b" + re.escape(col) + r"\s*:", window):
                    line_no = stripped.count("\n", 0, m.start()) + 1
                    rel = f.relative_to(_REPO_ROOT)
                    offenders.append(f"  {rel}:{line_no} → update payload incluye '{col}'")
                    break

    assert not offenders, (
        "P0-TIER-RLS-LOCK violation: el frontend escribe una columna de "
        "entitlement/billing (plan_tier / paypal_* / subscription_*) a "
        "user_profiles vía supabase client. Eso evade el billing server-side "
        "(el tier DEBE derivarse de PayPal en /api/subscription/verify) y "
        "reabre la escalación de tier desde la consola del browser.\n\n"
        "Offenders:\n" + "\n".join(offenders) + "\n\n"
        "Cierre: enrutar el cambio por el backend (service_role). El cliente "
        "nunca debe escribir estas columnas — el trigger "
        "trg_guard_user_profiles_entitlement lo bloquea a nivel DB."
    )


# ---------------------------------------------------------------------------
# 2. Migración SSOT presente en AMBOS dirs e idéntica
# ---------------------------------------------------------------------------
def test_entitlement_lock_migration_present_and_synced():
    paths = [d / _MIGRATION_NAME for d in _MIGRATION_DIRS]
    for p in paths:
        assert p.exists(), (
            f"P0-TIER-RLS-LOCK: falta la migración SSOT en {p}. Debe existir "
            "idéntica en supabase/migrations Y backend/supabase/migrations "
            "(convención P3-MIGRATIONS-SSOT)."
        )
    contents = [p.read_text(encoding="utf-8") for p in paths]
    assert contents[0] == contents[1], (
        "P0-TIER-RLS-LOCK: la migración difiere entre los dos dirs SSOT. "
        "Mantenerlas byte-idénticas (P3-MIGRATIONS-SSOT)."
    )


# ---------------------------------------------------------------------------
# 3. La migración define el guard correcto (trigger + función + columnas)
# ---------------------------------------------------------------------------
def test_entitlement_lock_migration_contents():
    sql = (_MIGRATION_DIRS[0] / _MIGRATION_NAME).read_text(encoding="utf-8")
    low = sql.lower()

    assert "guard_user_profiles_entitlement_columns" in low, "falta la función guard"
    assert "trg_guard_user_profiles_entitlement" in low, "falta el trigger"
    assert "before update on public.user_profiles" in low, "el trigger no es BEFORE UPDATE de user_profiles"
    assert "security invoker" in low, (
        "el guard DEBE ser SECURITY INVOKER para leer current_user real; "
        "SECURITY DEFINER haría que current_user sea el owner y el guard nunca dispararía."
    )
    assert "set search_path = ''" in low, "falta SET search_path = '' (convención P3-NEW-2)"
    # Exime backend pero bloquea roles cliente.
    assert "current_user in ('authenticated', 'anon')" in low, (
        "el guard debe gatear por current_user IN ('authenticated','anon') "
        "para eximir backend (postgres/service_role)."
    )
    # Las 5 columnas protegidas.
    for col in _BILLING_COLUMNS:
        assert col in low, f"la migración no protege la columna '{col}'"


# ---------------------------------------------------------------------------
# 4. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P0-TIER-RLS-LOCK` para el
    cross-link test_p2_hist_audit_14_marker_test_link."""
    assert "p0_tier_rls_lock" in __file__.replace("\\", "/").lower()
