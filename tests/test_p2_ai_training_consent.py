"""[P2-AI-TRAINING-CONSENT · 2026-07-04] Test ancla del consentimiento OPT-IN
para uso futuro de datos en entrenamiento de modelos propios de MealfitRD.

Por qué existe esta infraestructura ANTES que el pipeline de ML: el
consentimiento no es retroactivo — datos recolectados sin permiso explícito
no pueden usarse después. El flag se captura desde hoy.

Contratos:
  1. Migración SSOT en AMBOS dirs (migrations/ y backend/migrations/,
     convención P3-MIGRATIONS-SSOT), idéntica e idempotente, DEFAULT false.
  2. Endpoints GET/PATCH /api/user/preferences/ai-training con auth
     (get_verified_user_id) y default fail-secure FALSE en el GET.
  3. Gate SSOT del corpus futuro: db_profiles.get_ai_training_consented_user_ids
     filtra por `ai_training_consent IS TRUE` (fail-secure lista vacía).
  4. Frontend: toggle real en Configuración → Privacidad (PATCH al endpoint) y
     copy honesto ("hoy no entrena").
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND.parent
_MIGRATION_NAME = "p2_ai_training_consent_2026_07_04.sql"
_PREFS = _BACKEND / "routers" / "preferences.py"
_DB_PROFILES = _BACKEND / "db_profiles.py"
_SETTINGS = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    assert path.exists(), f"No existe {path} — ¿se renombró sin actualizar el test?"
    return path.read_text(encoding="utf-8")


def test_migration_in_both_dirs_identical_and_idempotent():
    backend_mig = _BACKEND / "migrations" / _MIGRATION_NAME
    root_mig = _REPO_ROOT / "migrations" / _MIGRATION_NAME
    src = _read(backend_mig)
    if root_mig.exists():  # workspace completo (en repo backend standalone no está)
        assert src == root_mig.read_text(encoding="utf-8"), (
            "P3-MIGRATIONS-SSOT: la migración difiere entre migrations/ y "
            "backend/migrations/ — sincronizar AMBOS dirs en el mismo commit."
        )
    assert "ADD COLUMN IF NOT EXISTS ai_training_consent" in src, "Falta IF NOT EXISTS (idempotencia)."
    assert re.search(r"DEFAULT\s+false", src, re.IGNORECASE), (
        "El consentimiento debe nacer FALSE (opt-in explícito, fail-secure)."
    )
    assert "RAISE EXCEPTION" in src, "Falta el sanity check DO $$ (patrón p2_next_4)."


def test_endpoints_wired_with_auth():
    src = _read(_PREFS)
    assert '@router.patch("/ai-training")' in src and '@router.get("/ai-training")' in src, (
        "Faltan los endpoints GET/PATCH /api/user/preferences/ai-training."
    )
    assert "update_ai_training_consent" in src
    # Default fail-secure en el GET: perfil ausente/None → False.
    assert re.search(r'ai_training_consent"\)\)\s*if\s+profile\s+else\s+False', src), (
        "El GET debe defaultear FALSE (opt-in fail-secure) con perfil ausente."
    )


def test_corpus_gate_ssot():
    src = _read(_DB_PROFILES)
    assert "def get_ai_training_consented_user_ids" in src, (
        "Falta el gate SSOT del corpus (get_ai_training_consented_user_ids)."
    )
    assert "ai_training_consent IS TRUE" in src, (
        "El gate debe filtrar por ai_training_consent IS TRUE."
    )
    assert "def update_ai_training_consent" in src
    # I2: la mutación filtra por id del usuario.
    m = re.search(r"def update_ai_training_consent.*?WHERE id = %s", src, re.DOTALL)
    assert m, "update_ai_training_consent debe filtrar WHERE id = %s (I2)."


def test_frontend_toggle_wired_and_honest():
    src = _read(_SETTINGS)
    assert "/api/user/preferences/ai-training" in src, (
        "El toggle de Privacidad debe leer/escribir el endpoint de consentimiento."
    )
    assert "handleToggleAiTraining" in src
    # Copy honesto: hoy NO se entrena (el consentimiento es para el futuro).
    assert re.search(r"no entrena", src), (
        "El copy debe dejar claro que HOY no se entrena con datos del usuario."
    )
