"""[P0-PROD-AUDIT-1 · 2026-05-23] Invariantes del Dockerfile + .dockerignore.

Gap original (audit 2026-05-23 — B-P0-1):
    Deploy productivo dependía 100% de auto-detect Nixpacks en EasyPanel.
    Sin Dockerfile no había forma de:
      (a) validar localmente la misma imagen que corre en prod,
      (b) reproducir builds para debugging,
      (c) migrar a otra plataforma (Fly.io, k8s) sin redescubrir todo.

Fix:
    `Dockerfile` multistage + `.dockerignore` documentados con runbook
    operacional. Este test ancla las invariantes críticas de seguridad y
    reproducibilidad que NO deben regresionar.

Por qué un test (no solo el Dockerfile en sí):
    Es trivial añadir un `USER root` "para debugging" y olvidarlo. Es
    trivial borrar el `HEALTHCHECK` "porque k8s ya hace probe". Estas
    regresiones son silenciosas pero comprometen producción real. Este
    test las detecta loud en CI.

Tooltip-anchor: P0-PROD-AUDIT-1-DOCKERFILE | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DOCKERFILE = _BACKEND_ROOT / "Dockerfile"
_DOCKERIGNORE = _BACKEND_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert _DOCKERFILE.exists(), (
        f"Dockerfile ausente en {_DOCKERFILE}. Cierre del gap B-P0-1 perdido."
    )
    return _DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    assert _DOCKERIGNORE.exists(), (
        f".dockerignore ausente en {_DOCKERIGNORE}. Sin él, el build "
        "context inflama (>200 MB de venv/test_venv/htmlcov + posible leak "
        "de .env)."
    )
    return _DOCKERIGNORE.read_text(encoding="utf-8")


def test_dockerfile_uses_python_312(dockerfile_text: str) -> None:
    """Pin de Python 3.12 explícito. Matchea pyrightconfig.json + CI matrix.

    Si alguien sube a 3.13 o baja a 3.11 sin actualizar el resto del repo,
    inconsistencias en builtin behaviors (e.g. datetime.utcnow deprecation
    P3-DEPRECATED-UTCNOW) pueden causar fallas runtime no detectadas por
    tests sync.
    """
    assert re.search(r"^FROM\s+python:3\.12", dockerfile_text, re.MULTILINE), (
        "Dockerfile no usa `python:3.12-*` como base. Pin explícito "
        "requerido para matchear pyrightconfig.json y CI matrix."
    )


def test_dockerfile_is_multistage(dockerfile_text: str) -> None:
    """Multistage build (builder + runtime). Imagen final ~150 MB vs ~450 MB
    single-stage. Build deps (gcc, build-essential) NO deben llegar a la
    imagen productiva (superficie de ataque ampliada + size waste).
    """
    from_lines = re.findall(r"^FROM\s+\S+", dockerfile_text, re.MULTILINE)
    assert len(from_lines) >= 2, (
        f"Dockerfile NO es multistage (solo {len(from_lines)} FROM). "
        f"Regresión: build deps (gcc, libpq-dev) llegarían a la imagen final. "
        f"Restaurar el pattern `FROM ... AS builder` + `FROM ... AS runtime`."
    )


def test_dockerfile_runs_as_non_root(dockerfile_text: str) -> None:
    """`USER appuser` (o uid 10001) — defensa-en-profundidad contra RCE.

    Atacante con shell en un contenedor non-root NO puede escribir fuera
    de `/app` ni escalar al kernel. Si alguien añade `USER root` "para
    debugging" y olvida revertir, este test falla.
    """
    user_lines = re.findall(r"^USER\s+(\S+)", dockerfile_text, re.MULTILINE)
    assert user_lines, "Dockerfile NO tiene directiva `USER` — corre como root por default."
    last_user = user_lines[-1]
    assert last_user not in {"root", "0"}, (
        f"Dockerfile termina con `USER {last_user}` (root). Defensa contra "
        f"RCE perdida. Restaurar `USER appuser`."
    )


def test_dockerfile_has_healthcheck(dockerfile_text: str) -> None:
    """`HEALTHCHECK` presente. Cubre plataformas sin probes nativos
    (Docker Compose, Fly.io machines). Si alguien lo borra "porque k8s
    hace probe", los entornos sin probe nativo pierden liveness check.
    """
    assert re.search(r"^HEALTHCHECK\s+", dockerfile_text, re.MULTILINE), (
        "Dockerfile no tiene `HEALTHCHECK`. Restaurar — cubre entornos sin "
        "probe nativo del orquestador."
    )


def test_dockerfile_uses_dumb_init_or_tini(dockerfile_text: str) -> None:
    """PID 1 reaper (`dumb-init` o `tini`). Sin él, zombies de cron tasks
    se acumulan + SIGTERM al contenedor NO se propaga a uvicorn → graceful
    shutdown roto en deploys rolling.
    """
    has_dumb_init = "dumb-init" in dockerfile_text
    has_tini = re.search(r"\btini\b", dockerfile_text) is not None
    assert has_dumb_init or has_tini, (
        "Dockerfile no usa dumb-init ni tini como PID 1. Sin reaper, "
        "zombies acumulan + SIGTERM no propaga a uvicorn (graceful "
        "shutdown roto)."
    )


def test_dockerfile_pip_install_uses_requirements_txt(dockerfile_text: str) -> None:
    """`pip install -r requirements.txt` (no installs ad-hoc dispersos).
    Pin único de deps → reproducibilidad.
    """
    assert re.search(r"pip\s+install\s+.*-r\s+requirements\.txt", dockerfile_text), (
        "Dockerfile no instala desde `requirements.txt` con `pip install -r`. "
        "Sin SSOT de deps, builds divergen entre devs/CI/prod."
    )


def test_dockerfile_copies_requirements_before_source(dockerfile_text: str) -> None:
    """Layer cache optimization: `COPY requirements.txt` ANTES de
    `COPY . .` → rebuilds que solo tocan código NO reinstalan pip.

    Sin esta optimización, cada commit triggea reinstalación de ~200 MB
    de deps (psycopg, langchain, etc.) — CI 5x más lento.
    """
    lines = dockerfile_text.split("\n")
    requirements_idx = None
    source_copy_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s*COPY\s+.*requirements\.txt", line) and requirements_idx is None:
            requirements_idx = i
        # `COPY . .` o `COPY --chown=... . .` (copia source completo).
        if re.match(r"\s*COPY\s+(--chown=\S+\s+)?\.\s+\.", line) and source_copy_idx is None:
            source_copy_idx = i
    assert requirements_idx is not None, "Dockerfile no copia requirements.txt"
    assert source_copy_idx is not None, "Dockerfile no copia el source code completo"
    assert requirements_idx < source_copy_idx, (
        f"requirements.txt copiado DESPUÉS del source (idx {requirements_idx} >= {source_copy_idx}). "
        f"Layer cache busted: cada commit re-instala deps. Reordenar."
    )


def test_dockerignore_excludes_env_files(dockerignore_text: str) -> None:
    """`.env` y `.env.*` deben estar excluidos. Si alguien los borra del
    .dockerignore por error, el build copia secrets al filesystem de la
    imagen → leak en cualquier `docker save` o registry push.
    """
    assert re.search(r"^\.env\s*$", dockerignore_text, re.MULTILINE), (
        ".dockerignore NO excluye `.env`. Secret leak en imagen. CRÍTICO."
    )
    assert re.search(r"^\.env\.\*\s*$", dockerignore_text, re.MULTILINE), (
        ".dockerignore NO excluye `.env.*`. Risk: .env.production / .env.local "
        "leak. Restaurar."
    )


def test_dockerignore_excludes_tests_and_venvs(dockerignore_text: str) -> None:
    """Excluir `tests/`, `venv/`, `test_venv/`, `htmlcov/` — no son runtime,
    inflan el contexto de build innecesariamente.
    """
    must_exclude = ["tests/", "venv/", "test_venv/", "htmlcov/", "__pycache__/"]
    missing = []
    for pattern in must_exclude:
        # Aceptar exact match o como prefijo de línea.
        if not re.search(rf"^{re.escape(pattern)}\s*$", dockerignore_text, re.MULTILINE):
            missing.append(pattern)
    assert not missing, (
        f".dockerignore NO excluye: {missing}. Build context infla — "
        f"slowdown CI + waste de bandwidth en deploys."
    )


def test_dockerignore_excludes_git(dockerignore_text: str) -> None:
    """`.git/` excluido. Copiar metadata Git infla ~20 MB sin valor runtime
    (git_sha viene de env var inyectada por EasyPanel/CI, no del filesystem).
    """
    assert re.search(r"^\.git\s*$", dockerignore_text, re.MULTILINE) or \
           re.search(r"^\.git/\s*$", dockerignore_text, re.MULTILINE), (
        ".dockerignore NO excluye .git/. Bloat sin valor runtime."
    )


def test_dockerfile_documented_in_runbook() -> None:
    """El Dockerfile debe estar documentado en
    `docs/runbooks/dockerfile_deployment.md` con SOP de deploy y debugging.

    Sin el runbook, un SRE bajo presión NO sabe cómo migrar a otra plataforma
    o debuggear un build roto.
    """
    runbook = _BACKEND_ROOT / "docs" / "runbooks" / "dockerfile_deployment.md"
    assert runbook.exists(), (
        f"Runbook {runbook} ausente. Sin documentación del Dockerfile, "
        f"el cierre del gap B-P0-1 es incompleto (técnica sin contexto "
        f"operacional)."
    )
    text = runbook.read_text(encoding="utf-8")
    required_sections = ["Quick start", "decisiones de diseño",
                         "Dockerfile"]
    missing = [s for s in required_sections if s.lower() not in text.lower()]
    assert not missing, (
        f"Runbook dockerfile_deployment.md incompleto — secciones ausentes "
        f"(case-insensitive): {missing}."
    )
