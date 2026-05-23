# Endpoint auth coverage

> [P0-PROD-AUDIT-1 · 2026-05-23] SSOT operacional del audit de cobertura de
> auth por endpoint. Cierra el gap B-P0-2 del audit production-readiness:
> "Cobertura de auth por endpoint no auditable — 92 endpoints sin garantía
> de que un nuevo endpoint añadido en PR futuro tendrá `Depends(...)` o
> equivalente".

## Por qué importa

`backend/auth.py::get_verified_user_id` es la **única** capa de autenticación
del backend porque `SUPABASE_KEY = SERVICE_ROLE` bypassea RLS. Cualquier
endpoint sin uno de los patrones canónicos de auth abre IDOR universal sobre
`meal_plans` / `user_inventory` / `consumed_meals` / `user_facts` /
`health_profile`. RLS no es la red de seguridad — `auth.py` lo es.

Documentado a profundidad en CLAUDE.md → "Anti-patrones de autenticación
prohibidos" + `~/.claude/projects/.../memory/runbook_security_antipatterns.md`.

## Patrones canónicos de auth

El test [`tests/test_p0_prod_audit_1_endpoint_auth_coverage.py`](../../tests/test_p0_prod_audit_1_endpoint_auth_coverage.py)
clasifica cada endpoint en uno de los 5 buckets siguientes. Cualquier endpoint
que NO entre en ninguno (bucket `UNCLASSIFIED`) falla el test loud.

| Bucket | Cómo se detecta | Cuándo usar |
|---|---|---|
| `JWT_USER_SCOPED` | `Depends(get_verified_user_id)` o `Depends(verify_api_quota)` en signature | Endpoint user-facing — la mayoría de los `/api/...` deberían caer aquí |
| `ADMIN_TOKEN` | `_verify_admin_token(request.headers.get("authorization"))` en el body | Endpoints `/admin/*` operacionales (CRON_SECRET-gated) |
| `WEBHOOK_HMAC` | `hmac.compare_digest(..., WEBHOOK_SECRET)` en el body | Webhooks de terceros (Supabase trigger, PayPal) |
| `PUBLIC_INTENTIONAL` | Entry en `_PUBLIC_ALLOWLIST` con razón documentada | Probes de health/ready, snapshots no-sensibles (knobs registry, cron status) |
| `KNOWN_GAP` | Entry en `_KNOWN_GAPS` con marker `KNOWN-GAP-NNN:` | Gap auditado pero fix pospuesto — debe quedar tracked aquí |

## SOP: añadir un endpoint nuevo

1. **Decidir la categoría de auth**:
   - User-facing → `Depends(get_verified_user_id)` (read-only) o `Depends(verify_api_quota)` (consume LLM/credits).
   - Admin/operator → `_verify_admin_token(...)` al inicio del body.
   - Webhook firmado → `hmac.compare_digest(..., WEBHOOK_SECRET)` al inicio del body.
   - Público (probe) → añadir entry a `_PUBLIC_ALLOWLIST` en el test con razón.

2. **Verificar IDOR**: si el endpoint acepta `user_id` en path/body, el handler DEBE comparar `verified_user_id == user_id` o filtrar SQL por `user_id = verified_user_id`. Ver invariante I2 en CLAUDE.md.

3. **Ejecutar el guard local**:
   ```bash
   pytest tests/test_p0_prod_audit_1_endpoint_auth_coverage.py -v
   ```
   Si falla con `UNCLASSIFIED`, escoger una de las opciones del mensaje de error.

4. **Si añades a `_KNOWN_GAPS`**: incrementar el ID (`KNOWN-GAP-NNN`) + añadir entry a la tabla "Gaps conocidos" abajo + abrir issue para el fix.

## SOP: cerrar un KNOWN_GAP

1. Implementar el fix (añadir `Depends`/`_verify_admin_token`/HMAC verify).
2. Eliminar el entry de `_KNOWN_GAPS` en el test.
3. Eliminar la fila correspondiente en "Gaps conocidos" de este runbook.
4. Bumpear `_LAST_KNOWN_PFIX` con marker referenciando el cierre.

## Gaps conocidos (estado a 2026-05-23)

> Si el conteo cambia, actualizar AMBOS: el entry en `_KNOWN_GAPS` del test
> Y esta tabla en el mismo PR. El test
> `test_known_gaps_count_is_documented` enforza que cada `KNOWN-GAP-NNN`
> aparezca en este runbook.

| ID | Endpoint | Severidad | Descripción | Fix propuesto |
|---|---|---|---|---|
| `KNOWN-GAP-001` | `GET /api/admin/test-proactive` | Baja | Endpoint admin de test de push notifications sin `_verify_admin_token`. Atacante con URL puede disparar push de test (cron envía a usuario fijo). No es IDOR, solo spam de push. | Añadir `_verify_admin_token(request.headers.get("authorization"))` al inicio del handler en [`app.py`](../../app.py) (function `api_test_proactive`). |

## Endpoints públicos intencionales (estado a 2026-05-23)

> Cambios a esta lista deben justificarse en PR review. Promover algo de
> `_KNOWN_GAPS` a `_PUBLIC_ALLOWLIST` es una decisión de producto/seguridad,
> no técnica.

| Endpoint | Razón |
|---|---|
| `GET /` | Root probe — devuelve string fijo, sin datos sensibles. |
| `GET /health` | Liveness probe para load balancer/k8s. Solo `{status: ok}`. |
| `GET /ready` | Readiness probe para load balancer/k8s. Solo `{status: ready|not_ready, reason}`. |
| `GET /health/version` | [P2-HEALTHZ-DEEP] Blackbox monitor externo. UUIDs hasheados via `_hash_uuid_for_public()`. |
| `GET /admin/knobs` | [P3-5] Snapshot de `_KNOWN_REGISTRY`. Valores son env vars `MEALFIT_*` que el operador conoce. |
| `GET /admin/cron-health` | [P0-2] Diagnóstico operacional del scheduler. Info no sensible. |

## Cómo ejecutar el audit

```bash
# Test individual con output verbose (imprime breakdown por bucket).
pytest tests/test_p0_prod_audit_1_endpoint_auth_coverage.py -v -s

# Salida esperada (snapshot a 2026-05-23):
#   [P0-PROD-AUDIT-1] Endpoint auth coverage summary (~92 endpoints):
#     ADMIN_TOKEN            8  (  8.7%)
#     JWT_USER_SCOPED       ~70 ( 76.1%)
#     KNOWN_GAP              1  (  1.1%)
#     PUBLIC_INTENTIONAL     6  (  6.5%)
#     WEBHOOK_HMAC           2  (  2.2%)
#     # UNCLASSIFIED         0  (  0.0%)   ← FALLA si > 0
```

El gate del audit es: **`UNCLASSIFIED` debe ser 0**. `KNOWN_GAP` puede ser
>0 con SOP de cierre tracked aquí.

## Roadmap

- **P1**: añadir `_verify_admin_token` a `/api/admin/test-proactive` y eliminar `KNOWN-GAP-001`.
- **P1**: extender la clasificación para detectar pattern de validación inline manual (e.g. `if not verified_user_id: raise 401`), no solo `Depends`. Hoy se detecta solo via `Depends(...)` que es el patrón canónico.
- **P2**: integrar el output del audit (breakdown por bucket) en `/admin/cron-health` para visibilidad operacional sin correr pytest.
- **P2**: extender a frontend (escanear `fetchWithAuth` callsites para asegurar que todos los handlers backend invocados tienen counterpart auth en el test).
