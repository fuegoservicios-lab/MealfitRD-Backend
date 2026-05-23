"""[P0-PROD-AUDIT-1 · 2026-05-23] Audit blanket de cobertura de auth por endpoint.

Bug original (audit production-readiness 2026-05-23 — gap B-P0-2):
    El backend tiene ~92 endpoints distribuidos en `app.py` + 7 routers.
    La auditoría grep `Depends(get_verified_user_id|verify_api_quota)` mostró
    decoradores presentes en 23 sitios, pero NO había forma de garantizar
    que un nuevo endpoint añadido en un PR futuro tuviera cobertura de auth.

    Test blanket existente [`test_p1_audit_3_history_quota_exemption.py`]
    cubre solo 3 endpoints específicos del Historial (paywall exemption).
    NO existía un guard genérico que parsease TODOS los endpoints y los
    clasificara en: protegido / público intencional / IDOR.

    Modo de fallo concreto: alguien añade `@router.get("/api/inventory-list")`
    sin `Depends(get_verified_user_id)` y filtra inventario cross-user. RLS
    NO protege porque `SUPABASE_KEY = SERVICE_ROLE` bypassea RLS — `auth.py`
    es la ÚNICA capa (P0-AUDIT-1).

Fix:
    Este test escanea AST de `app.py` + `routers/*.py`, extrae cada
    `@(app|router).<method>(...)` decorator, y clasifica cada endpoint en
    uno de los buckets siguientes según el patrón canónico:

      1. **JWT_USER_SCOPED**: signature tiene `Depends(get_verified_user_id)`
         o `Depends(verify_api_quota)`. Cubre la mayoría de endpoints
         user-facing.
      2. **ADMIN_TOKEN**: body invoca `_verify_admin_token(...)` con el
         header de authorization. Cubre los `/admin/*` endpoints que se
         auth-gatean por `CRON_SECRET`.
      3. **WEBHOOK_HMAC**: body valida `WEBHOOK_SECRET` via
         `hmac.compare_digest`. Cubre webhooks externos firmados.
      4. **PUBLIC_INTENTIONAL**: en la allowlist explícita abajo con razón
         documentada. Cubre `/health`, `/ready`, `/health/version`, etc.
      5. **KNOWN_GAP**: en la allowlist con marker `KNOWN-GAP-<id>:` —
         endpoints que el audit identificó como gap real pendiente de fix.
         La presencia en este bucket NO falla el test (el operador conoce
         el gap), pero el conteo se publica para visibilidad.
      6. **UNCLASSIFIED**: si un endpoint no entra en ninguno de los
         buckets anteriores, el test FALLA loud. Esto es el corazón del
         enforcement: cualquier endpoint nuevo SIN auth visible debe
         (a) añadir el `Depends`, (b) añadir el inline-check, o
         (c) justificar la exención añadiendo entry a `_PUBLIC_ALLOWLIST`
         con razón clara.

Limitaciones (out of scope):
    - NO valida que los handlers internamente respeten `verified_user_id`
      (e.g. el handler podría retornar data cross-user aunque `Depends`
      esté presente). Eso requiere tests funcionales E2E por endpoint —
      cubierto parcialmente por `test_p3_next_1_i2_user_id_filter_contract.py`
      (SQL-level) y `test_p0_audit_1_auth_bypass.py` (auth function).
    - NO escanea rutas registradas via `app.include_router(...)` con
      prefix transformation — asumimos que prefix es opaco y el path
      dentro del router es el contrato.

Cómo añadir entries a la allowlist:
    Editar `_PUBLIC_ALLOWLIST` o `_KNOWN_GAPS` con (method, path) → reason.
    La razón debe explicar POR QUÉ el endpoint es legítimamente público
    O por qué el gap está aceptado/programado. PRs que añadan endpoints
    al allowlist deben justificar en review.

Tooltip-anchor: P0-PROD-AUDIT-1-AUTH-COVERAGE | audit 2026-05-23.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_ROUTERS_DIR = _BACKEND_ROOT / "routers"


# ---------------------------------------------------------------------------
# ALLOWLISTS — endpoints públicos intencionales o gaps conocidos.
# ---------------------------------------------------------------------------
# Estos endpoints NO tienen Depends/admin-gate/webhook-hmac. La razón debe
# ser clara para que un futuro mantenedor entienda por qué es seguro.
# (method, path) → reason
_PUBLIC_ALLOWLIST: dict[tuple[str, str], str] = {
    ("GET", "/"): "Root probe — devuelve string fijo, sin datos sensibles.",
    ("GET", "/health"): "Liveness probe público para load balancer/k8s. Solo devuelve {status: ok}.",
    ("GET", "/ready"): "Readiness probe público para load balancer/k8s. Solo devuelve {status: ready|not_ready}.",
    ("GET", "/health/version"): (
        "[P2-HEALTHZ-DEEP · 2026-05-12] Público sin auth para blackbox monitor "
        "externo (UptimeRobot). Expone solo markers operacionales — UUIDs "
        "hasheados via _hash_uuid_for_public(). Test [test_p2_prod_audit_3.py] "
        "ancla la sanitización."
    ),
    ("GET", "/admin/knobs"): (
        "[P3-5 · 2026-05-10] Snapshot del _KNOBS_REGISTRY. Documentado en "
        "app.py:1395 como público intencional — los valores son env vars "
        "MEALFIT_* que el operador conoce, no son secretos."
    ),
    ("GET", "/admin/cron-health"): (
        "[P0-2 · 2026-05-10] Diagnóstico operacional del scheduler. "
        "Documentado en app.py:1420 como público intencional — info de "
        "diagnóstico no sensible. Si reportara secrets, gatear con "
        "_verify_admin_token."
    ),
    # System health/observabilidad endpoints (router prefix /api/system/).
    # Todos son read-only para load balancers / k8s probes / Grafana, y los
    # UUIDs que exponen pasan por `_hash_uuid_for_public()` (sanitización
    # SHA-256[:12]). Convención P2-HEALTH-UID-STRIP · 2026-05-12. Sanity de la
    # sanitización: test_p2_prod_audit_3.py.
    ("GET", "/atomic-pool-health"): (
        "[P1-4 · 2026-05] Health del connection_pool — read-only. "
        "`last_user_id` sanitizado a `last_user_hash` via _hash_uuid_for_public(). "
        "Doc inline en routers/system.py:250."
    ),
    ("GET", "/chunk-queue-health"): (
        "[P1-5] Backlog/failures del worker de chunks — read-only. Sin UUIDs "
        "user-scoped en el response. Útil para dashboards Grafana."
    ),
    ("GET", "/pantry-tolerance-health"): (
        "[Pantry health] Snapshot de tolerancias acumuladas — read-only. "
        "Sin UUIDs en el response."
    ),
    ("GET", "/tz-fallback-health"): (
        "[TZ fallback] Métricas de fallback de timezone — read-only. Sin UUIDs."
    ),
    ("GET", "/health/plan-graph"): (
        "[Plan graph health] Estado del LangGraph compilado — read-only. Sin "
        "UUIDs. Probe extendido sobre /ready para diagnóstico granular."
    ),
    ("POST", "/cancel"): (
        "[P1-16 · P6-CANCEL-LOG] `/api/plans/cancel` — registra cancel de un "
        "session_id en in-memory set + limpia pending_pipeline KV. Diseño "
        "intencional sin auth documentado en routers/plans.py:4232 ('peor "
        "caso DoS de bajo impacto: registrar cancel para session_id que no "
        "existe es no-op'). Idempotente. Trade-off: latencia baja vs ataque "
        "teórico de DoS via cancel ajeno si session_id leak — el frontend "
        "valida ownership client-side de la generación activa antes de "
        "permitir clickear cancel."
    ),
}

# Gaps conocidos: endpoints que el audit identificó pero el fix queda como
# follow-up. La presencia aquí HACE pasar el test (el gap es conocido), pero
# el conteo se reporta para visibilidad. Cuando un gap se cierra, mover el
# entry a _PUBLIC_ALLOWLIST (si la decisión fue mantenerlo público) o
# eliminarlo (si se añadió auth gate).
_KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("GET", "/api/admin/test-proactive"): (
        "KNOWN-GAP-001: endpoint admin de test de notificaciones push que NO "
        "está gateado por _verify_admin_token. Riesgo: atacante con URL puede "
        "disparar push notifications de test al usuario fijo en el cron. "
        "Severidad baja (push notification spam, no IDOR), pero gap real. "
        "Follow-up: añadir `_verify_admin_token(request.headers.get(\"authorization\"))` "
        "al inicio del handler. Documentado en docs/runbooks/endpoint_auth_coverage.md."
    ),
}

# Patrones canónicos de auth — el test escanea el AST/source y matchea uno
# de estos para clasificar el endpoint.
_JWT_DEPS = {"get_verified_user_id", "verify_api_quota"}
_ADMIN_GATE_CALL = "_verify_admin_token"
_WEBHOOK_HMAC_MARKER = "compare_digest"  # hmac.compare_digest pattern (P0-WEBHOOK-1)
_WEBHOOK_SECRET_MARKER = "WEBHOOK_SECRET"


# ---------------------------------------------------------------------------
# AST extraction.
# ---------------------------------------------------------------------------

def _extract_endpoints_from_file(path: Path) -> list[dict]:
    """Parsea un .py y devuelve lista de endpoints con metadata.

    Cada endpoint: {
        'method': 'GET' | 'POST' | ...,
        'path': '/api/...',
        'fn_name': str,
        'signature_deps': set[str],  # nombres de funciones dentro de Depends(...)
        'body_calls': set[str],      # nombres de funciones llamadas en body
        'body_text': str,            # source text del body (para regex match)
        'file': str,
        'line': int,
    }
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    out: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            # Match: @app.METHOD("/path") o @router.METHOD("/path")
            if not isinstance(decorator, ast.Call):
                continue
            attr = decorator.func
            if not isinstance(attr, ast.Attribute):
                continue
            if not isinstance(attr.value, ast.Name):
                continue
            if attr.value.id not in {"app", "router"}:
                continue
            method = attr.attr.upper()
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            # Path es el primer argumento posicional (string literal).
            if not decorator.args:
                continue
            first_arg = decorator.args[0]
            path_val = _safe_string_constant(first_arg)
            if path_val is None:
                continue

            # Signature deps: extraer nombres dentro de `Depends(...)` en los
            # default values de los parámetros.
            sig_deps = _extract_signature_depends(node)

            # Body calls + texto: para detectar _verify_admin_token o
            # hmac.compare_digest invocados inline.
            body_calls = _extract_body_call_names(node)
            try:
                body_text = ast.unparse(node)
            except Exception:
                # Fallback: extraer source desde line range.
                lines = text.split("\n")
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 1)
                body_text = "\n".join(lines[start:end])

            out.append({
                "method": method,
                "path": path_val,
                "fn_name": node.name,
                "signature_deps": sig_deps,
                "body_calls": body_calls,
                "body_text": body_text,
                "file": path.name,
                "line": node.lineno,
            })
    return out


def _safe_string_constant(expr: ast.AST) -> Optional[str]:
    """Devuelve el string si `expr` es un Constant string literal; None si no."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _extract_signature_depends(fn_node: ast.AST) -> set[str]:
    """Extrae nombres de funciones que aparecen en `Depends(<name>)` dentro
    de los default values de la signature (args + kwonly args) y también en
    el `dependencies=[Depends(...), ...]` del decorator.
    """
    deps: set[str] = set()
    args = fn_node.args
    # defaults aplica a los últimos N args; kw_defaults aplica a kwonly args.
    for default in (args.defaults or []) + (args.kw_defaults or []):
        if default is None:
            continue
        deps.update(_extract_depends_names(default))
    # `dependencies=[Depends(...), ...]` en el decorator: ya lo cubre el caller
    # en una pasada separada (es kwarg del decorator).
    for decorator in getattr(fn_node, "decorator_list", []):
        if not isinstance(decorator, ast.Call):
            continue
        for kw in decorator.keywords:
            if kw.arg == "dependencies":
                deps.update(_extract_depends_names(kw.value))
    return deps


def _extract_depends_names(expr: ast.AST) -> set[str]:
    """Recorre expr buscando `Depends(<Name>)` y devuelve los `<Name>` strings."""
    out: set[str] = set()
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "Depends":
            if sub.args:
                first = sub.args[0]
                if isinstance(first, ast.Name):
                    out.add(first.id)
                elif isinstance(first, ast.Attribute):
                    out.add(first.attr)
    return out


def _extract_body_call_names(fn_node: ast.AST) -> set[str]:
    """Extrae nombres de funciones llamadas dentro del body (Name calls +
    Attribute calls). Usado para detectar `_verify_admin_token(...)` o
    `hmac.compare_digest(...)`.
    """
    out: set[str] = set()
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


# ---------------------------------------------------------------------------
# Clasificación.
# ---------------------------------------------------------------------------

def _classify_endpoint(ep: dict) -> str:
    """Devuelve uno de:
       'JWT_USER_SCOPED' / 'ADMIN_TOKEN' / 'WEBHOOK_HMAC' /
       'PUBLIC_INTENTIONAL' / 'KNOWN_GAP' / 'UNCLASSIFIED'.

    Pattern de `*_LIMITER` cuenta como JWT_USER_SCOPED porque la clase
    `RateLimiter.__call__` (`backend/rate_limiter.py:36`) tiene
    `Depends(get_verified_user_id)` como default value de su signature.
    FastAPI resuelve esa sub-dependency antes de invocar el limiter,
    así que un endpoint con `Depends(_*_LIMITER)` enforza la auth
    transitivamente. Convención canónica del repo.
    """
    key = (ep["method"], ep["path"])
    if key in _PUBLIC_ALLOWLIST:
        return "PUBLIC_INTENTIONAL"
    if key in _KNOWN_GAPS:
        return "KNOWN_GAP"
    if _JWT_DEPS & ep["signature_deps"]:
        return "JWT_USER_SCOPED"
    # Rate limiters transitively enforce get_verified_user_id (P0-AUDIT-1 pattern).
    if any(dep.endswith("_LIMITER") for dep in ep["signature_deps"]):
        return "JWT_USER_SCOPED"
    if _ADMIN_GATE_CALL in ep["body_calls"]:
        return "ADMIN_TOKEN"
    if _WEBHOOK_HMAC_MARKER in ep["body_calls"] and _WEBHOOK_SECRET_MARKER in ep["body_text"]:
        return "WEBHOOK_HMAC"
    return "UNCLASSIFIED"


def _collect_all_endpoints() -> list[dict]:
    """Recolecta endpoints de app.py + routers/*.py. Cachable a nivel módulo."""
    endpoints: list[dict] = []
    if _APP_PY.exists():
        endpoints.extend(_extract_endpoints_from_file(_APP_PY))
    if _ROUTERS_DIR.exists():
        for router_py in sorted(_ROUTERS_DIR.glob("*.py")):
            if router_py.name == "__init__.py":
                continue
            endpoints.extend(_extract_endpoints_from_file(router_py))
    return endpoints


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_endpoints() -> list[dict]:
    eps = _collect_all_endpoints()
    assert len(eps) > 0, (
        "Scan de endpoints retornó 0. Verifica que app.py + routers/ existen "
        f"en {_BACKEND_ROOT}. Si reorganizaste el repo, actualiza este test."
    )
    return eps


def test_baseline_endpoint_count_sanity(all_endpoints: list[dict]) -> None:
    """Sanity: contamos >= 50 endpoints. Si el scan retorna menos, probable
    refactor que rompió la heurística AST."""
    n = len(all_endpoints)
    assert n >= 50, (
        f"Solo {n} endpoints detectados — esperábamos >=50 (audit 2026-05-23 "
        f"contó ~92). Probable refactor que cambió el pattern @app/@router. "
        f"Actualizar heurística en este test o verificar que routers/ no esté vacío."
    )


def test_every_endpoint_classified(all_endpoints: list[dict]) -> None:
    """**Test principal**: cada endpoint cae en un bucket válido.

    Si este test falla, hay un endpoint sin clasificar (UNCLASSIFIED) →
    revisar el output y:
      (a) si el endpoint debe tener auth, añadir `Depends(get_verified_user_id)`
          o invocar `_verify_admin_token` en el body;
      (b) si es legítimamente público, añadir entry a `_PUBLIC_ALLOWLIST`
          con razón clara;
      (c) si es gap conocido aceptado, añadir entry a `_KNOWN_GAPS` con
          marker `KNOWN-GAP-NNN:` y razón.
    """
    unclassified = []
    for ep in all_endpoints:
        bucket = _classify_endpoint(ep)
        if bucket == "UNCLASSIFIED":
            unclassified.append(
                f"  - [{ep['method']:6s}] {ep['path']:50s}  "
                f"(fn={ep['fn_name']}, file={ep['file']}:{ep['line']})"
            )

    if unclassified:
        msg = (
            f"\n\n[P0-PROD-AUDIT-1] {len(unclassified)} endpoint(s) sin "
            f"cobertura de auth visible y NO en allowlist:\n\n"
            + "\n".join(unclassified)
            + "\n\nOpciones:\n"
            "  (a) Añadir `Depends(get_verified_user_id)` o `Depends(verify_api_quota)` a la signature.\n"
            "  (b) Invocar `_verify_admin_token(request.headers.get('authorization'))` al inicio del body.\n"
            "  (c) Añadir validación `hmac.compare_digest(..., WEBHOOK_SECRET)` en el body (webhook).\n"
            "  (d) Si es público intencional, añadir entry a `_PUBLIC_ALLOWLIST` en este test.\n"
            "  (e) Si es gap conocido pero aceptado por ahora, añadir entry a `_KNOWN_GAPS` con marker KNOWN-GAP-NNN.\n"
        )
        pytest.fail(msg)


def test_known_gaps_have_marker_id(all_endpoints: list[dict]) -> None:
    """Cada entry en `_KNOWN_GAPS` debe tener marker `KNOWN-GAP-NNN:` para
    cross-link con el tracker de follow-ups. Sin el marker, no hay forma de
    cerrar el gap auditable.
    """
    for key, reason in _KNOWN_GAPS.items():
        assert re.search(r"\bKNOWN-GAP-\d{3}\b", reason), (
            f"Entry _KNOWN_GAPS para {key} no tiene marker `KNOWN-GAP-NNN:`. "
            f"Razón actual: {reason!r}. Añadir marker para tracking."
        )


def test_allowlists_dont_overlap() -> None:
    """Un endpoint NO puede estar simultáneamente en `_PUBLIC_ALLOWLIST` y
    `_KNOWN_GAPS` — son buckets mutuamente excluyentes."""
    overlap = set(_PUBLIC_ALLOWLIST.keys()) & set(_KNOWN_GAPS.keys())
    assert not overlap, (
        f"Overlap entre _PUBLIC_ALLOWLIST y _KNOWN_GAPS: {overlap}. "
        f"Decidir un bucket por endpoint."
    )


def test_allowlist_entries_match_real_endpoints(all_endpoints: list[dict]) -> None:
    """Cada (method, path) en allowlist DEBE existir en el codebase. Si un
    endpoint allowlisteado se borra, el entry queda huérfano y oculta la
    intención.

    Falla loud — el operador debe limpiar el allowlist en el mismo PR que
    borra el endpoint.
    """
    real_keys = {(ep["method"], ep["path"]) for ep in all_endpoints}
    orphans_public = set(_PUBLIC_ALLOWLIST.keys()) - real_keys
    orphans_gaps = set(_KNOWN_GAPS.keys()) - real_keys
    orphans = orphans_public | orphans_gaps
    if orphans:
        pytest.fail(
            f"[P0-PROD-AUDIT-1] Entries huérfanos en allowlists "
            f"(endpoint borrado pero allowlist no limpiado): {orphans}. "
            f"Eliminar del `_PUBLIC_ALLOWLIST` / `_KNOWN_GAPS` en este test."
        )


def test_coverage_summary_printable(all_endpoints: list[dict], capsys) -> None:
    """Imprime breakdown por bucket — útil en CI logs para tracking de
    progreso (e.g. "KNOWN_GAP went from 1 to 0 = gap cerrado")."""
    buckets: dict[str, int] = {}
    for ep in all_endpoints:
        b = _classify_endpoint(ep)
        buckets[b] = buckets.get(b, 0) + 1

    total = sum(buckets.values())
    lines = [
        f"\n[P0-PROD-AUDIT-1] Endpoint auth coverage summary ({total} endpoints):",
    ]
    for bucket in sorted(buckets.keys()):
        pct = (buckets[bucket] / total) * 100 if total else 0
        lines.append(f"  {bucket:22s} {buckets[bucket]:4d}  ({pct:5.1f}%)")
    print("\n".join(lines))

    # Sanity: el bucket JWT_USER_SCOPED debe dominar (mayoría de endpoints
    # user-facing). Si baja drásticamente, probable refactor que rompió la
    # detección.
    assert buckets.get("JWT_USER_SCOPED", 0) >= 20, (
        f"Solo {buckets.get('JWT_USER_SCOPED', 0)} endpoints JWT_USER_SCOPED — "
        f"esperábamos >=20. Probable regresión en la detección AST."
    )


def test_known_gaps_count_is_documented() -> None:
    """El conteo de `_KNOWN_GAPS` debe estar publicado en el runbook
    de endpoint coverage. Si el conteo cambia (cierre de gap, descubrimiento
    de nuevo), actualizar el runbook en el mismo PR.

    Cierre del meta-gap: el operador no debería tener que ejecutar este test
    para saber cuántos gaps tiene el repo — debería verlo en el doc.
    """
    runbook = _BACKEND_ROOT / "docs" / "runbooks" / "endpoint_auth_coverage.md"
    assert runbook.exists(), (
        f"Runbook ausente en {runbook}. Crear con conteo actual de "
        f"_KNOWN_GAPS y SOP de cierre."
    )
    text = runbook.read_text(encoding="utf-8")
    # El doc debe mencionar el conteo exacto + cada KNOWN-GAP-NNN id
    # presente en _KNOWN_GAPS.
    for key, reason in _KNOWN_GAPS.items():
        m = re.search(r"\b(KNOWN-GAP-\d{3})\b", reason)
        assert m is not None, f"reason sin marker: {reason!r}"
        gap_id = m.group(1)
        assert gap_id in text, (
            f"Runbook endpoint_auth_coverage.md NO menciona `{gap_id}` "
            f"({key}). Añadir entry en el runbook para tracking."
        )
