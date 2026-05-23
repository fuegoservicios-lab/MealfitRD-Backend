"""[P0-PROD-AUDIT-1 · 2026-05-23] Guard que el job `pip-audit` está
configurado en CI.

Gap original (audit 2026-05-23 — B-P0-3):
    Ningún gate de CI ejecutaba auditoría de CVEs contra requirements.txt.
    Deps de 2024 (langchain==1.2.18, langgraph==1.1.10, psycopg==3.3.4)
    sin validación regular contra GitHub Advisory Database.

Fix:
    Job `backend-security-audit` añadido a `.github/workflows/ci.yml` con
    `continue-on-error: true` (mismo pattern que `frontend-lint`).

    Inicial NON-BLOCKING: baseline puede tener N warnings históricos;
    bloquear merge sobre baseline paralizaría. Roadmap: tras cleanup (bump
    o `--ignore-vuln` documentado), flippear a `continue-on-error: false`.

Por qué un test del CI yaml (no solo el yaml en sí):
    Es trivial borrar el job en un refactor cosmético del workflow o al
    extraer steps a un reusable workflow. Sin enforcement, el gap reabre
    silenciosamente. Este test detecta loud.

Tooltip-anchor: P0-PROD-AUDIT-1-PIP-AUDIT | audit 2026-05-23.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
# El workflow vive en `.github/workflows/ci.yml`. En estructura cross-repo
# meta-workspace, .github vive en el workspace root; en este repo, vive
# en el backend root. Intentar ambos.
_CI_YML_CANDIDATES = [
    _BACKEND_ROOT / ".github" / "workflows" / "ci.yml",
    _BACKEND_ROOT.parent / ".github" / "workflows" / "ci.yml",
]


def _read_ci_yml() -> str:
    for candidate in _CI_YML_CANDIDATES:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise AssertionError(
        f".github/workflows/ci.yml no encontrado. Buscado en: "
        f"{[str(c) for c in _CI_YML_CANDIDATES]}. Si moviste la ruta del "
        f"workflow, actualizar `_CI_YML_CANDIDATES` en este test."
    )


def test_pip_audit_job_present() -> None:
    """El job `backend-security-audit` (o equivalente con `pip-audit` en
    el step) debe existir en el workflow CI.

    Si alguien lo borra "porque siempre warns por base + base + base",
    el gap reabre y CVEs nuevos pasan silenciosos hasta deploy.
    """
    text = _read_ci_yml()
    has_job = "backend-security-audit:" in text
    has_step = "pip-audit" in text
    assert has_job and has_step, (
        f"CI workflow NO tiene job de pip-audit. "
        f"has_job={has_job}, has_step={has_step}. "
        f"Restaurar el job `backend-security-audit` que invoca `pip-audit -r requirements.txt`."
    )


def test_pip_audit_targets_requirements_txt() -> None:
    """El step debe escanear `requirements.txt` específicamente — no setup.py
    ni pyproject.toml (este repo no los usa). Si la convención cambia,
    actualizar AMBOS: el test y el workflow.
    """
    text = _read_ci_yml()
    assert "pip-audit -r requirements.txt" in text or \
           "pip-audit --requirement requirements.txt" in text or \
           "pip-audit -r backend/requirements.txt" in text, (
        "Step de pip-audit NO escanea requirements.txt. Probable que esté "
        "escaneando setup.py/pyproject.toml — este repo no los usa. "
        "Restaurar `pip-audit -r requirements.txt`."
    )


def test_pip_audit_job_is_initially_non_blocking() -> None:
    """El job debe tener `continue-on-error: true` inicialmente (baseline
    con N warnings históricos). Cuando flippeen a bloqueante (post-cleanup),
    actualizar este test loud.

    Documentar la decisión: si el test falla porque alguien lo flippeó,
    actualizar el test floor a `continue-on-error: false` Y agregar
    comentario justificando.
    """
    text = _read_ci_yml()
    # Buscar el bloque del job + 20 líneas alrededor para chequear el flag.
    idx = text.find("backend-security-audit:")
    assert idx != -1, "Job ausente — cubierto por test_pip_audit_job_present"
    block = text[idx:idx + 2000]  # ventana suficiente para el job completo
    assert "continue-on-error: true" in block, (
        "Job `backend-security-audit` NO tiene `continue-on-error: true`. "
        "Si decidiste flippear a bloqueante (gate hard), actualizar este "
        "test + documentar la decisión + asegurar que requirements.txt está "
        "limpio de CVEs conocidos."
    )


def test_docker_build_job_present() -> None:
    """El job `backend-docker-build` debe existir — valida que el Dockerfile
    builde en CI (no solo en local). Cierra el gap B-P0-1 secundario
    (Dockerfile podría regresionar y solo se detectaría en deploy real).
    """
    text = _read_ci_yml()
    assert "backend-docker-build:" in text, (
        "CI workflow NO tiene job `backend-docker-build`. Sin él, una "
        "regresión en el Dockerfile (e.g. RUN pip install fallando) solo "
        "se detecta en el deploy real a EasyPanel — diagnóstico tardío."
    )
    # El docker build debe validar el image como non-root + HEALTHCHECK.
    idx = text.find("backend-docker-build:")
    block = text[idx:idx + 3000]
    assert "non-root" in block.lower() or "appuser" in block.lower() or "10001" in block, (
        "Job `backend-docker-build` NO valida que la imagen corra como "
        "non-root. Sin esto, una regresión `USER root` accidental pasa "
        "silenciosamente. Añadir step de `docker inspect ... | grep -q appuser`."
    )
