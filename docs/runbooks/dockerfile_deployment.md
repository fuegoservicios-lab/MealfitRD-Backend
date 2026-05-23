# Dockerfile deployment runbook

> [P0-PROD-AUDIT-1 · 2026-05-23] SSOT operacional del [`Dockerfile`](../../Dockerfile) +
> [`.dockerignore`](../../.dockerignore). Gap cerrado: deploy productivo dependía
> 100% de auto-detect Nixpacks en EasyPanel; sin Dockerfile no había forma de
> reproducir la imagen localmente, debuggear builds rotos, o migrar a otra
> plataforma sin redescubrir convenciones.

## Cuándo aplica este runbook

- **Validar localmente** la misma imagen que corre en producción antes de pushear.
- **Migrar la plataforma de hosting** (EasyPanel → Fly.io / Railway / k8s / ECS).
- **Diagnosticar build roto en CI** (`docker build` falla y no sabes si es el
  Dockerfile, la dep, o el contexto).
- **Validar tamaño y superficie de seguridad** de la imagen final (CVE scan,
  audit non-root, healthcheck).

## Quick start

```bash
# Build local (desde la raíz del repo backend).
docker build -t mealfit-backend:dev .

# Run con variables de entorno desde .env local. Mapea puerto 3001.
docker run --rm -p 3001:3001 --env-file .env mealfit-backend:dev

# Verificar healthcheck (esperar ~30s para que pase `start-period`).
docker inspect --format='{{json .State.Health}}' \
    $(docker ps -q --filter ancestor=mealfit-backend:dev) | jq

# Probar liveness manualmente.
curl -fsS http://localhost:3001/health
# → {"status": "ok", "message": "MealfitRD AI Backend is running"}

# Probar readiness (puede tardar ~10-30s post-arranque por warm_plan_graph).
curl -fsS http://localhost:3001/ready
```

## Arquitectura del Dockerfile

```
python:3.12-slim (builder)
    │ apt-get install build-essential gcc libpq-dev
    │ python -m venv /opt/venv
    │ pip install -r requirements.txt
    ▼
python:3.12-slim (runtime)
    │ apt-get install dumb-init curl libpq5
    │ useradd appuser uid=10001
    │ COPY --from=builder /opt/venv /opt/venv
    │ COPY --chown=appuser . .
    │ USER appuser
    │ EXPOSE 3001
    │ HEALTHCHECK /health
    │ ENTRYPOINT dumb-init
    │ CMD uvicorn app:app --workers 1
    ▼
Final image (~150 MB)
```

### Decisiones de diseño

| Decisión | Razón |
|---|---|
| **Multistage build** | Imagen final ~150 MB vs ~450 MB single-stage. Build deps (gcc, build-essential) viven solo en el `builder` stage. |
| **Pin Python 3.12** | Matchea `pyrightconfig.json` + CI matrix. Cambios mayores requieren PR explícito — sin auto-detect. |
| **Non-root user (uid 10001)** | Defensa-en-profundidad contra RCE. Atacante con shell no puede escribir fuera de `/app`. |
| **`dumb-init` como PID 1** | Reapea zombies (procesos hijo de cron tasks) + propaga SIGTERM correctamente a uvicorn (graceful shutdown durante deploys rolling). |
| **`HEALTHCHECK` vs k8s probe** | `HEALTHCHECK` cubre plataformas sin probes nativos (Docker Compose, Fly.io machines). En k8s, configurar adicionalmente `livenessProbe` + `readinessProbe` apuntando a `/health` y `/ready`. |
| **`--workers 1` en uvicorn** | El scheduler APScheduler es in-memory (SSOT del wiring de crons en `cron_tasks.py:register_plan_chunk_scheduler`). Multi-worker rompería el SSOT. Para escalar, usar replicas horizontales (knob `MEALFIT_SCHEDULER_LEADER_*` soporta leader election cross-replica). |
| **`--proxy-headers` + `--forwarded-allow-ips=*`** | EasyPanel/Caddy/Traefik inyectan `X-Forwarded-For` / `X-Forwarded-Proto`. Sin esto, `request.client.host` reporta IP del proxy en lugar del cliente — rompe rate limiting per-IP. `*` confía en cualquier proxy upstream; cambiar a IP específica si se conoce. |

## SOP: deploy a EasyPanel (estado actual)

EasyPanel detecta el `Dockerfile` automáticamente y prefiere build vía Docker
sobre Nixpacks cuando ambos están disponibles. Migración esperada:

1. **Push** del PR que añade el `Dockerfile` a `main`.
2. **Trigger del redeploy** desde EasyPanel UI o via `force-redeploy` button.
3. **Observar el build log**: debe ver `Step 1/N : FROM python:3.12-slim AS builder`
   (no `Nixpacks build plan`). Si sigue usando Nixpacks, eliminar/renombrar el
   `nixpacks.toml` (si existe) o forzar Docker en Settings → Build.
4. **Validar el healthcheck** post-arranque: en EasyPanel, el servicio debe
   pasar a "Healthy" en <60s (start-period 30s + 1 check exitoso).
5. **Smoke test** del endpoint público:
   ```bash
   curl -fsS https://<easypanel-domain>/health/version | jq
   ```
   El campo `last_known_pfix` debe coincidir con `_LAST_KNOWN_PFIX` en HEAD.
   Si no coincide → cache hit de Nixpacks viejo (force rebuild without cache).

## SOP: migración a Fly.io

```bash
# Instalar flyctl + login.
brew install flyctl
flyctl auth login

# Crear app + Postgres connection vía Supabase (no usar fly postgres).
flyctl launch --dockerfile Dockerfile --no-deploy

# Inyectar todas las env vars de .env.example via:
flyctl secrets set $(cat .env | grep -v '^#' | xargs)

# Deploy.
flyctl deploy --strategy bluegreen

# Validar.
flyctl logs
curl -fsS https://<app>.fly.dev/health
```

Configurar en `fly.toml`:
```toml
[checks]
  [checks.liveness]
    type = "http"
    interval = "30s"
    timeout = "5s"
    grace_period = "30s"
    method = "get"
    path = "/health"
  [checks.readiness]
    type = "http"
    interval = "60s"
    timeout = "10s"
    grace_period = "60s"
    method = "get"
    path = "/ready"
```

## SOP: build roto en CI

### Síntoma: `Step N/M : RUN pip install ... → ERROR`

Probable: una dep en `requirements.txt` requiere C headers no instalados en el
builder. Verificar:

```bash
docker build --target builder -t mealfit-debug --progress=plain . 2>&1 | grep -i error
```

Si es `libpq-fe.h: No such file or directory` → falta `libpq-dev` (ya lo
instalamos; si reaparece, regression en el RUN apt-get del builder).

### Síntoma: imagen final >300 MB

Probable: olvido de excluir algo en `.dockerignore`. Diagnosticar:

```bash
docker build -t mealfit-bloat .
docker run --rm mealfit-bloat du -sh /app/* | sort -h | tail -20
```

Top hitters esperados: `cron_tasks.py` (~1.4 MB), `graph_orchestrator.py`
(~750 KB), `routers/plans.py` (~500 KB), `constants.py` (~150 KB), prompts/.
Cualquier cosa >2 MB es candidato a investigar (probable artefacto que debió
quedar en `.dockerignore`).

### Síntoma: HEALTHCHECK reporta "starting" indefinidamente

Probable: `app.py` arranca pero `/health` no responde en <5s. Diagnosticar:

```bash
docker logs <container-id> | tail -50
# Buscar tracebacks de cron_tasks.register_plan_chunk_scheduler o
# graph_orchestrator.warm_plan_graph().
```

Si `warm_plan_graph` tarda >30s consistentemente → subir `start-period` del
HEALTHCHECK a 60s (cambio en Dockerfile + redeploy).

## SOP: validar reproducibilidad

```bash
# Hash de la imagen actual.
docker build -t mealfit:run1 .
docker images --no-trunc --quiet mealfit:run1

# Re-build limpio (no cache).
docker build --no-cache -t mealfit:run2 .
docker images --no-trunc --quiet mealfit:run2

# Los hashes deberían coincidir si el Dockerfile es reproducible (mismo
# requirements.txt, mismo Python base image SHA).
diff <(docker history mealfit:run1) <(docker history mealfit:run2)
```

> Nota: Python 3.12-slim publica nuevas patch versions; para reproducibilidad
> estricta, pinear el SHA: `FROM python:3.12.7-slim@sha256:...`. Trade-off:
> ya no recibes patches de seguridad automáticos. Decisión actual: no pinear
> SHA → patches automáticos, reproducibilidad "bit-exact" sacrificada.

## Gaps conocidos / follow-ups

| Gap | Descripción | Severidad | Propuesta |
|---|---|---|---|
| `P1-DEPS-HASHES` | `requirements.txt` sin hashes (`--require-hashes`) → supply chain attack contra PyPI no detectada. | P1 | `pip-compile --generate-hashes` para producir lock file con SHA256 de cada wheel. |
| `P1-DOCKERFILE-SHA-PIN` | `python:3.12-slim` no pineado a SHA → image base puede cambiar entre builds. | P1 | Aceptar reproducibilidad "bit-exact" perdida en favor de patches automáticos; o pinear SHA + cron para bump mensual. |
| `P2-CI-DOCKER-BUILD` | CI no valida que el Dockerfile builde. Una regresión en `RUN pip install` solo se detecta en el deploy real a EasyPanel. | P2 | Cerrado vía nuevo job `backend-docker-build` añadido en `.github/workflows/ci.yml`. |
| `P2-DOCKER-SBOM` | No generamos SBOM (CycloneDX/SPDX) de la imagen. | P2 | `docker sbom mealfit-backend` o `syft mealfit-backend -o cyclonedx-json`. |

## Tests de regresión

- [`tests/test_p0_prod_audit_3_dockerfile_runtime.py`](../../tests/test_p0_prod_audit_3_dockerfile_runtime.py): valida invariantes del Dockerfile (non-root user, Python 3.12, HEALTHCHECK presente, `.dockerignore` excluye `.env`).
