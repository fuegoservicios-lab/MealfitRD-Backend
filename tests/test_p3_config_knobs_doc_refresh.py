"""[P3-CONFIG-KNOBS-DOC-REFRESH · 2026-07-09] Fase 4 (limpieza de config, alcance SEGURO = doc +
comentario, cero cambio de comportamiento). El doc `knobs_reference.md` afirmaba "~161 knobs" (stale:
el registry registra ~420 at import y hay ~900 nombres MEALFIT_ distintos en source). El doc por diseño
NO mirror-ea los knobs (drift garantizado) — así que el fix es (a) corregir el conteo stale, (b) añadir
un script repetible `scripts/dump_knobs.py` (la "regeneración desde el registry" real), y (c) arreglar el
comentario stale del budget del micro-closer (decía 80 kcal, el default real es 120).

Parser-based + smoke del script. NO promueve flags ni borra ramas (eso arriesgaría rollback + WIP).
"""
import os
import subprocess
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def test_doc_count_no_longer_stale():
    doc = _read(_BACKEND, "docs", "knobs_reference.md")
    # el conteo stale "~161" no debe seguir presentándose como el número actual de knobs
    assert "~161 env vars" not in doc, "el conteo stale '~161 env vars' debe corregirse"
    # debe referir al mecanismo de conteo vivo (registry) con un número realista (≥400)
    assert "get_knobs_registry_snapshot" in doc, "debe apuntar al snapshot del registry (fuente viva)"
    assert "dump_knobs" in doc, "debe referenciar el script repetible scripts/dump_knobs.py"


def test_doc_keeps_required_sections():
    """No romper el contrato de test_p2_prod_final_2 (secciones requeridas)."""
    doc = _read(_BACKEND, "docs", "knobs_reference.md")
    for req in ("P2-KNOBS-OPERATIONAL-DOC", "_KNOBS_REGISTRY", "/health/version",
                "get_knobs_registry_snapshot", "MEALFIT_SHOPPING_COHERENCE_GUARD",
                "MEALFIT_CB_FAILURE_THRESHOLD", "Cómo añadir un knob nuevo"):
        assert req in doc, f"el doc debe conservar la sección requerida: {req}"


def test_dump_knobs_script_exists_and_runs():
    path = os.path.join(_BACKEND, "scripts", "dump_knobs.py")
    assert os.path.exists(path), "falta scripts/dump_knobs.py (regeneración desde el registry)"
    # smoke: corre y emite al menos un knob conocido
    r = subprocess.run([sys.executable, path], cwd=_BACKEND, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, f"dump_knobs.py falló: {r.stderr[-500:]}"
    # un knob module-scope (registrado at import); los in-function no aparecen hasta ejecutarse.
    assert "MEALFIT_ADVERSARIAL_PAID_ONLY" in r.stdout, "el dump debe listar knobs registrados at import"
    assert "registrados at import" in r.stdout, "el dump debe reportar el conteo del registry"


def test_micro_closer_budget_comment_not_stale():
    src = _read(_BACKEND, "graph_orchestrator.py")
    # el comentario del tope del closer no debe decir "(80) kcal" (default real = 120)
    assert "MAX_KCAL_PER_DAY (80) kcal" not in src, "comentario stale: el default del closer es 120, no 80"
    # y el default real debe seguir siendo 120
    assert '_env_int("MEALFIT_MICRONUTRIENT_CLOSER_MAX_KCAL_PER_DAY", 120)' in src


def test_marker_present():
    assert "P3-CONFIG-KNOBS-DOC-REFRESH" in _read(_BACKEND, "docs", "knobs_reference.md")
