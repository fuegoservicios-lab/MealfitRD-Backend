"""[P1-VERCEL-SECURITY-HEADERS · 2026-05-12] Anchor + regression guard.

`frontend/vercel.json` debe declarar 6 security headers para `mealfitrd.com`.
Pre-fix solo tenía `rewrites` → exposición:
  - Sin HSTS: downgrade attack en redes hostiles roba sesión Supabase.
  - Sin X-Frame-Options: dashboard iframe-embedable → clickjacking.
  - Sin nosniff: navegadores antiguos hacían MIME-sniff sobre uploads.
  - Sin Referrer-Policy: URLs internas (con plan_id en path) leak via
    Referer header al navegar a links externos.
  - Sin Permissions-Policy: cualquier componente puede pedir camera/mic.
  - Sin CSP: reflected-XSS escala a exfiltración del JWT en localStorage.

Defensas que el test enforza:
  1. Headers presentes con los valores canónicos.
  2. CSP arranca en `Content-Security-Policy-Report-Only` (no enforce
     todavía — observación primero, promover tras 1 semana).
  3. CSP whitelist de hosts críticos para el funcionamiento de la app
     (Supabase, Sentry, PayPal, GA, Google Fonts). Si alguien rompe
     el whitelist eliminando un host crítico, los flows respectivos
     (pago, observabilidad, persistencia) quedan en silent-fail.

Test parser-based — escanea JSON directamente.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERCEL_JSON = _REPO_ROOT / "frontend" / "vercel.json"


def _load_vercel() -> dict:
    return json.loads(_VERCEL_JSON.read_text(encoding="utf-8"))


def _headers_for_path(cfg: dict, source_path: str) -> dict:
    """Devuelve dict {key: value} de los headers configurados para `source_path`.
    Falla si no hay entry matching."""
    for entry in cfg.get("headers", []):
        if entry.get("source") == source_path:
            return {h["key"]: h["value"] for h in entry.get("headers", [])}
    raise AssertionError(
        f"No hay entry `headers` con source={source_path!r} en vercel.json"
    )


def test_anchor_present_in_vercel_json():
    src = _VERCEL_JSON.read_text(encoding="utf-8")
    assert "P1-VERCEL-SECURITY-HEADERS" in src, (
        "Falta anchor `P1-VERCEL-SECURITY-HEADERS` en frontend/vercel.json. "
        "Sin anchor un futuro reader puede 'limpiar' la config asumiendo "
        "que los headers son boilerplate."
    )


def test_headers_block_exists():
    cfg = _load_vercel()
    assert "headers" in cfg and isinstance(cfg["headers"], list) and cfg["headers"], (
        "vercel.json debe tener key `headers` con al menos una entry."
    )


def test_strict_transport_security_present_with_long_max_age():
    """HSTS con max-age >= 1 año (63072000 = 2 años recomendado) +
    includeSubDomains + preload. Sin preload, navegadores nuevos no
    aprenden la política hasta la primera visita HTTPS exitosa."""
    h = _headers_for_path(_load_vercel(), "/(.*)")
    assert "Strict-Transport-Security" in h, "Falta header HSTS"
    val = h["Strict-Transport-Security"]
    m = re.search(r"max-age=(\d+)", val)
    assert m is not None, f"HSTS sin max-age: {val!r}"
    assert int(m.group(1)) >= 31536000, (
        f"HSTS max-age={m.group(1)} < 1 año (31536000s). Mínimo recomendado."
    )
    assert "includeSubDomains" in val, "HSTS sin includeSubDomains"
    assert "preload" in val, "HSTS sin preload"


def test_x_content_type_options_nosniff():
    h = _headers_for_path(_load_vercel(), "/(.*)")
    assert h.get("X-Content-Type-Options") == "nosniff", (
        "X-Content-Type-Options debe ser exactamente 'nosniff'"
    )


def test_x_frame_options_deny():
    """DENY (no SAMEORIGIN) — la app no necesita iframearse a sí misma.
    Si en el futuro algún flow lo requiere, cambiar a SAMEORIGIN
    explícitamente y documentar la razón."""
    h = _headers_for_path(_load_vercel(), "/(.*)")
    assert h.get("X-Frame-Options") == "DENY", (
        f"X-Frame-Options debe ser DENY, encontrado {h.get('X-Frame-Options')!r}"
    )


def test_referrer_policy_strict_origin():
    h = _headers_for_path(_load_vercel(), "/(.*)")
    assert h.get("Referrer-Policy") == "strict-origin-when-cross-origin", (
        "Referrer-Policy debe ser 'strict-origin-when-cross-origin' para "
        "evitar leak de URLs internas (que incluyen plan_id en path) "
        "via Referer header a links externos."
    )


def test_permissions_policy_disables_sensitive_apis():
    """camera/microphone/geolocation no usados por la app — deben quedar
    explícitamente vacíos `()` para evitar que un script third-party los
    pida. `payment` necesita PayPal habilitado."""
    h = _headers_for_path(_load_vercel(), "/(.*)")
    val = h.get("Permissions-Policy", "")
    for disabled in ("camera=()", "microphone=()", "geolocation=()"):
        assert disabled in val, (
            f"Permissions-Policy falta `{disabled}` (debe deshabilitar el API)"
        )


def test_csp_starts_as_report_only():
    """CSP debe estar en `Content-Security-Policy-Report-Only` inicialmente,
    NO en `Content-Security-Policy` enforced. Razón: en el primer rollout
    cualquier whitelist faltante rompería el flow afectado en producción
    sin warning. Report-Only emite violations al console del navegador para
    observación; promover tras 1 semana sin violations."""
    h = _headers_for_path(_load_vercel(), "/(.*)")
    assert "Content-Security-Policy-Report-Only" in h, (
        "CSP debe estar en Content-Security-Policy-Report-Only (no enforce). "
        "Promover a Content-Security-Policy enforced tras 1 semana de "
        "observación sin violations."
    )
    # No debería existir el header enforced todavía (until promoted)
    # Este assert se invierte cuando se promueva — comentar/eliminar después.
    assert "Content-Security-Policy" not in {
        k for k in h.keys() if k != "Content-Security-Policy-Report-Only"
    }, (
        "Existe `Content-Security-Policy` enforced. Si esto es intencional "
        "(promoción tras observación), actualizar el test eliminando este "
        "guard. Si no, dejar solo Report-Only hasta confirmar 0 violations."
    )


def test_csp_whitelists_critical_hosts():
    """Hosts críticos que SIN whitelist romperían flows core:
    - supabase.co  → DB + Auth + Realtime → app entera muerta.
    - sentry.io    → observabilidad → bugs en prod sin captura.
    - paypal.com   → billing → upgrades de tier rotos.
    - googletagmanager.com / google-analytics.com → marketing analytics.
    - posthog.com  → product analytics.
    - api.pwnedpasswords.com → [P3-HIBP-CSP · 2026-05-31] HIBP k-anonymity
      del chequeo de contraseña filtrada (Register + Reset, ver
      `checkLeakedPassword.js`). Sin el host, al promover la CSP a enforced el
      fetch a `/range/` se bloquea y el check DEGRADA-OPEN silencioso
      (`leaked:false`) → contraseñas comprometidas pasan sin señal. Es la
      contraparte del advisor aceptado `auth_leaked_password_protection`
      (implementado en frontend porque el toggle nativo requiere plan Pro).

    [P2-AUDIT-4 · 2026-05-15] `fonts.googleapis.com` y `fonts.gstatic.com`
    REMOVIDOS de esta lista. Convención P3-SELF-HOST-FONTS: el E2E test
    rechaza requests a esos hosts (fonts deben venir self-hosted desde el
    bundle). Tener los hosts en CSP era incoherencia documental: permitir
    un request que el E2E test luego rechaza. Si la app vuelve a necesitar
    tipografía remota, restaurar los hosts aquí + eliminar el guard E2E.
    """
    h = _headers_for_path(_load_vercel(), "/(.*)")
    csp = h["Content-Security-Policy-Report-Only"]
    critical_hosts = [
        "*.supabase.co",
        "*.sentry.io",
        "*.paypal.com",
        "googletagmanager.com",
        "google-analytics.com",
        # [P3-HIBP-CSP · 2026-05-31] connect-src para el chequeo HIBP de
        # contraseña filtrada — sin él el check degrada-open al pasar a enforced.
        "api.pwnedpasswords.com",
    ]
    missing = [host for host in critical_hosts if host not in csp]
    assert not missing, (
        f"CSP whitelist faltan hosts críticos: {missing}. "
        "Sin ellos los flows respectivos (DB/observabilidad/billing/marketing) "
        "harían silent-fail tras promover a enforced."
    )


def test_csp_disables_object_and_frame_ancestors():
    """object-src 'none' bloquea Flash/plugins (legacy XSS vector).
    frame-ancestors 'none' = equivalent moderno de X-Frame-Options DENY
    (browsers nuevos prefieren CSP sobre el header legacy)."""
    h = _headers_for_path(_load_vercel(), "/(.*)")
    csp = h["Content-Security-Policy-Report-Only"]
    assert "object-src 'none'" in csp, (
        "CSP debe tener `object-src 'none'` para bloquear plugins legacy."
    )
    assert "frame-ancestors 'none'" in csp, (
        "CSP debe tener `frame-ancestors 'none'` (equiv moderno de "
        "X-Frame-Options DENY)."
    )


def test_csp_upgrade_insecure_requests():
    """Forza upgrade automático de http:// → https:// para cualquier
    sub-resource (img, script, etc.) — defensa contra mixed-content."""
    h = _headers_for_path(_load_vercel(), "/(.*)")
    csp = h["Content-Security-Policy-Report-Only"]
    assert "upgrade-insecure-requests" in csp, (
        "CSP debe incluir `upgrade-insecure-requests` para forzar HTTPS "
        "en sub-resources."
    )


def test_rewrites_preserved():
    """Regression guard: añadir headers NO debe haber eliminado los
    rewrites SPA originales."""
    cfg = _load_vercel()
    rewrites = cfg.get("rewrites", [])
    assert any(
        r.get("destination") == "/index.html" for r in rewrites
    ), "vercel.json perdió el rewrite SPA → /index.html. Las rutas client-side se romperían."


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P1-VERCEL-SECURITY-HEADERS" in src
