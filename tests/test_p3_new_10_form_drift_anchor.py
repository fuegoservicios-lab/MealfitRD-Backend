"""[P3-NEW-10 · 2026-05-11] Anchor explícito `[FORM-DRIFT-ANCHOR]` en
`frontend/src/config/formValidation.js`.

Por qué:
    El backend test `test_p0_form_6_required_fields_sync.py` ya detecta
    drift bidireccional entre `REQUIRED_FORM_FIELDS` (frontend) y
    `_REQUIRED_FORM_FIELDS` (backend). El gap que cierra P3-NEW-10 es
    DESCUBRIBILIDAD: un frontend dev modificando el array no
    necesariamente corre los tests backend, así que un anchor `[FORM-DRIFT-ANCHOR]`
    grep-able en el archivo del frontend les apunta directo al
    enforcement.

Tests:
    1. El anchor `[FORM-DRIFT-ANCHOR]` está presente en el archivo.
    2. El anchor menciona el test backend que enforza la simetría.
    3. El anchor menciona el SSOT backend (_REQUIRED_FORM_FIELDS).
    4. El comment de P3-NEW-10 incluye `dietType` como excepción.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORM_VAL_FP = _REPO_ROOT / "frontend" / "src" / "config" / "formValidation.js"


@pytest.fixture(scope="module")
def src() -> str:
    return _FORM_VAL_FP.read_text(encoding="utf-8")


def test_anchor_present(src: str):
    assert "[FORM-DRIFT-ANCHOR]" in src, (
        "P3-NEW-10 regresión: el anchor `[FORM-DRIFT-ANCHOR]` desapareció "
        "de formValidation.js. Sin él, un frontend dev grep-ando no "
        "encuentra rápido la documentación del contrato de drift."
    )


def test_anchor_mentions_backend_test(src: str):
    """El anchor debe nombrar explícitamente el test backend."""
    # Tomamos una ventana alrededor del anchor para validar.
    anchor_idx = src.find("[FORM-DRIFT-ANCHOR]")
    assert anchor_idx > 0
    block = src[anchor_idx:anchor_idx + 1500]
    assert "test_p0_form_6_required_fields_sync.py" in block, (
        "P3-NEW-10 regresión: el bloque del anchor ya no menciona el "
        "test backend que enforza la simetría. Sin esa referencia, un "
        "dev del frontend no sabe DÓNDE corre el guard."
    )


def test_anchor_mentions_backend_ssot(src: str):
    anchor_idx = src.find("[FORM-DRIFT-ANCHOR]")
    block = src[anchor_idx:anchor_idx + 1500]
    assert "_REQUIRED_FORM_FIELDS" in block, (
        "P3-NEW-10 regresión: el anchor ya no nombra el SSOT backend "
        "`_REQUIRED_FORM_FIELDS`. Sin ese nombre, un dev no sabe QUÉ "
        "actualizar en el backend si añade un field aquí."
    )


def test_anchor_documents_diettype_exception(src: str):
    anchor_idx = src.find("[FORM-DRIFT-ANCHOR]")
    block = src[anchor_idx:anchor_idx + 1500]
    assert "dietType" in block, (
        "P3-NEW-10 regresión: el anchor ya no documenta la excepción de "
        "`dietType` (frontend-only por legacy compat). Sin esta "
        "excepción documentada, un dev podría intentar 'arreglar' el "
        "test sync removiendo dietType del frontend o añadiéndolo al "
        "backend — ambos rompen casos legítimos."
    )


def test_p3_new_10_marker_present(src: str):
    assert "P3-NEW-10" in src, (
        "P3-NEW-10 regresión: el marker `P3-NEW-10` desapareció. Sin él, "
        "un revisor no puede correlacionar la addition con el audit "
        "2026-05-11 que la justificó."
    )
