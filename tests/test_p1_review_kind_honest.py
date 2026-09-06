# -*- coding: utf-8 -*-
"""[P1-REVIEW-KIND-HONEST · 2026-09-05] El informe cultural marcaba `human_signoff = true` en los seis perfiles,
y el fichero de firma dice, textualmente, revisor «Claude (revisión curatorial delegada por el dueño)». La
revisión automatizada vale como control de calidad del catálogo; registrarla como humana, no.

Tres campos separados, ninguno deducido de otro, y la clínica en False EXPLÍCITO: un campo ausente se lee como
«no aplica», y aquí sí aplica — simplemente no existe todavía."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import cultural_benchmark as cb  # noqa: E402

_REVIEW = _BACKEND / "data" / "registry" / "cultural_curation_review_v1.json"
_REPORT = _BACKEND / "data" / "registry" / "cultural_benchmark_v1.json"


def test_el_registro_de_firma_declara_que_es_automatica():
    rec = json.loads(_REVIEW.read_text(encoding="utf-8"))
    assert rec.get("kind") == "automated", "la firma de Claude no puede registrarse como humana"
    assert "Claude" in str(rec.get("reviewer")), "si cambia el revisor, hay que revisar este contrato"
    assert rec.get("kind_note"), "el registro explica qué NO es"


def test_sin_kind_se_asume_automatica_nunca_humana():
    """El default seguro: un registro antiguo sin `kind` no asciende a humano por omisión."""
    src = (_BACKEND / "cultural_benchmark.py").read_text(encoding="utf-8")
    assert 'rec.get("kind") or "automated"' in src


@pytest.mark.skipif(not (_BACKEND / "data" / "registry" / "dish_registry_es_v1.json").exists(),
                    reason="snapshots no compilados")
def test_el_informe_vivo_separa_los_tres_tipos():
    rep = cb.run_benchmark()
    for pid, e in rep["profiles"].items():
        rv = e["review"]
        assert rv["automated_review"] is True, pid
        assert rv["human_cultural_review"] is False, pid
        assert rv["clinical_review"] is False, pid
        assert "human_signoff" not in rv, f"{pid}: el campo que mentía no vuelve"


def test_el_informe_committed_ya_no_afirma_revision_humana():
    if not _REPORT.exists():
        pytest.skip("informe no committed")
    saved = json.loads(_REPORT.read_text(encoding="utf-8"))
    crudo = json.dumps(saved, ensure_ascii=False)
    assert "human_signoff" not in crudo, "el informe publicado conserva la afirmación falsa"
    for pid, e in (saved.get("profiles") or {}).items():
        assert e["review"]["automated_review"] is True, pid
        assert e["review"]["human_cultural_review"] is False, pid


@pytest.mark.skipif(not (_BACKEND / "data" / "registry" / "dish_registry_es_v1.json").exists(),
                    reason="snapshots no compilados")
def test_el_gate_pasa_sin_firma_humana_porque_es_un_gate_automatico():
    """Lo que el roadmap 2.6 pedía: un gate automático puede pasar sin persona detrás SI se llama automático."""
    rep = cb.run_benchmark()
    assert rep["gate_ok"] is True
    assert all(e["review"]["human_cultural_review"] is False for e in rep["profiles"].values())
    md = cb.render_markdown(rep)
    assert "automated" in md, "la tabla dice de qué tipo es cada firma"
