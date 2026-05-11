"""[P3-NEXT-2 · 2026-05-11] Anchor: `test_consecutive_zero_log_cascade.py`
fue borrado intencionalmente.

Contexto:
    El archivo `tests/test_consecutive_zero_log_cascade.py` llevaba marcado
    `@pytest.mark.skip` desde P0-5 con docstring reconociendo que el
    `_check_chunk_learning_ready` evolucionó (P0-2 v2 + P0-3 + P1-3 +
    P1-4) y el test original asumía early-return basado solo en
    `_consecutive_proxy_chunks`. El skip-reason explícitamente decía:

      "La cobertura real de `learning_proxy_exhausted` vive ahora en el
       flujo end-to-end de `test_chunked_learning_propagation` (P1-4
       logging_preference + P1-3 weak_window)."

    Mantenerlo skip-permanente era ruido en `pytest -v` y un test
    "rotten in place" — el costo de su skip-message superaba la
    señal de su nombre como recordatorio.

Decisión (audit conversacional 2026-05-11):
    Borrar el archivo. La cobertura sustituta ya existe en
    `test_chunked_learning_propagation` (que cubre learning_proxy_exhausted
    end-to-end con mocks completos de consumed_meals + inventory_activity
    + user_profiles.logging_preference + plan_chunk_queue de chunks
    previos).

Este test es un anchor:
    - Falla si el archivo borrado vuelve a aparecer (drift).
    - Documenta dónde vive la cobertura sustituta para que un futuro dev
      no lo recree.
"""
from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DELETED_FILE = (
    _REPO_ROOT / "backend" / "tests" / "test_consecutive_zero_log_cascade.py"
)
_REPLACEMENT_FILE = (
    _REPO_ROOT / "backend" / "tests" / "test_chunked_learning_propagation.py"
)


def test_obsolete_skipped_test_file_is_deleted():
    """El test obsoleto fue borrado en P3-NEXT-2. Si vuelve a aparecer,
    revisar si es intencional — la cobertura real ya está en
    `test_chunked_learning_propagation.py`."""
    assert not _DELETED_FILE.exists(), (
        f"P3-NEXT-2 violation: `{_DELETED_FILE.name}` reapareció. Fue "
        f"borrado intencionalmente el 2026-05-11 porque llevaba "
        f"@pytest.mark.skip permanente desde P0-5 — un test 'rotten in "
        f"place' que documentaba su propia obsolescencia. Si la "
        f"reescritura es intencional, considerar si la cobertura ya "
        f"existe en `test_chunked_learning_propagation.py` (P1-4 + P1-3) "
        f"antes de aceptar el archivo de vuelta."
    )


def test_replacement_coverage_file_still_exists():
    """La cobertura sustituta DEBE seguir existiendo. Si alguien borra
    `test_chunked_learning_propagation.py` sin reemplazo, perderíamos
    la cobertura E2E del gate de `learning_proxy_exhausted`."""
    assert _REPLACEMENT_FILE.exists(), (
        f"P3-NEXT-2 floor: `{_REPLACEMENT_FILE.name}` desapareció. Era "
        f"la cobertura sustituta del test obsoleto que borramos en "
        f"P3-NEXT-2. Si el rename es intencional, actualizar este anchor; "
        f"si fue borrado, restaurar la cobertura E2E de "
        f"`learning_proxy_exhausted` antes de continuar."
    )
