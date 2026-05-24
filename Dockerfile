# [P0-PROD-AUDIT-1 · 2026-05-23] Dockerfile reproducible para el backend.
#
# Pre-fix: deploy dependía 100% de auto-detect Nixpacks en EasyPanel. Sin
# Dockerfile, (a) no había forma de validar localmente la misma imagen que
# corre en prod, (b) un cambio en heurísticas Nixpacks podía romper el deploy
# silenciosamente (e.g. switch de Python 3.11 → 3.13), (c) DR/migración a otra
# plataforma (Fly.io, Railway, k8s) requería re-discoverear todas las
# convenciones desde cero. Este Dockerfile cierra el gap haciendo el contrato
# explícito y portable.
#
# Diseño:
#   - **Multistage**: stage `builder` instala deps (gcc para psycopg binary,
#     wheels); stage `runtime` solo copia el venv resultante. Imagen final
#     ~150 MB en lugar de ~450 MB single-stage.
#   - **Pin de Python a 3.12**: matchea pyrightconfig.json + CI matrix.
#     Cambios mayores van vía PR explícito (no auto-detect).
#   - **Non-root user**: `appuser` (uid 10001) — defensa-en-profundidad contra
#     RCE en el worker (un atacante con shell no puede escribir fuera de /app).
#   - **HEALTHCHECK**: cura el liveness check de plataformas que no leen
#     `/health` HTTP (algunos schedulers de k8s/Fly).
#   - **dumb-init**: PID 1 que reapea zombies + propaga SIGTERM correctamente
#     a uvicorn (importante para graceful shutdown durante deploys rolling).
#   - **Layer cache**: `requirements.txt` copiado ANTES del resto → rebuilds
#     que solo tocan código no reinstalan pip.
#
# Build local:
#   docker build -t mealfit-backend:dev .
#
# Run local (env vars vía --env-file):
#   docker run --rm -p 3001:3001 --env-file .env mealfit-backend:dev
#
# Validar healthcheck:
#   docker inspect --format='{{json .State.Health}}' <container>
#
# Runbook completo: docs/runbooks/dockerfile_deployment.md

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps para psycopg[binary,pool] y otras wheels nativas. Solo en el
# builder — no llegan a la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# venv aislado para que el stage `runtime` lo copie tal cual.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# `--require-hashes` no se usa porque requirements.txt no tiene hashes (gap
# documentado para follow-up: P1-DEPS-HASHES). Mientras tanto, pip resolver
# valida versiones pinneadas exactas (==).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=3001 \
    ENVIRONMENT=production

# `dumb-init` reapea zombies + propaga SIGTERM/SIGINT correctamente a uvicorn
# durante graceful shutdown (workers de cron tienen ~30s para drenar antes
# del SIGKILL del orquestador). `curl` para el HEALTHCHECK.
# `libpq5` (runtime de libpq) requerido por psycopg en runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        dumb-init \
        curl \
        libpq5 \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Non-root user — defensa-en-profundidad. uid alto (>10000) evita colisiones
# con uids de host en bind-mounts accidentales.
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup \
        --create-home --home-dir /home/appuser \
        --shell /usr/sbin/nologin appuser

WORKDIR /app

# Copiar venv desde builder.
COPY --from=builder /opt/venv /opt/venv

# Copiar source code (excluido lo que .dockerignore filtra: tests/, venv/,
# .env, scratch/, htmlcov/, etc.). chown explícito para que appuser pueda
# leer/ejecutar todo (uvicorn workers necesitan read sobre cron_tasks.py).
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 3001

# HEALTHCHECK: /health es liveness puro (proceso vivo + atendiendo HTTP).
# `/ready` valida componentes downstream (LangGraph compilado) pero requiere
# más tiempo de start; lo dejamos al orquestador (k8s readinessProbe, Fly checks).
# start-period 30s da tiempo a uvicorn + warm_plan_graph en cold start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:${PORT}/health || exit 1

# dumb-init como PID 1; uvicorn como child. SIGTERM al contenedor → dumb-init
# → uvicorn graceful shutdown (workers terminan in-flight requests).
ENTRYPOINT ["dumb-init", "--"]

# Workers single — los cron jobs viven en el mismo proceso (APScheduler
# in-memory). Multi-worker romperia el SSOT del scheduler. Para escalar,
# usar replicas horizontales con un solo worker cada una (el knob
# `MEALFIT_SCHEDULER_LEADER_*` ya soporta leader election cross-replica).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3001", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
