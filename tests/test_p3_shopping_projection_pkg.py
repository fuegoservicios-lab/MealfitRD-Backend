"""[P3-SHOPPING-PROJECTION-PKG · 2026-09-05] Primera extracción real hacia `shopping/`: la proyección de compras
(read model, reproyección con huella, estado UI) vive en `shopping/projection/` y `plan_jobs.py` queda como
outbox puro que re-exporta. Y los god files quedan CONGELADOS en tamaño (roadmap §11 / Fase 9): crecer más
de un 3 % sobre la foto de hoy exige extraer, no apilar.
"""
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def test_a_el_paquete_existe_y_plan_jobs_reexporta_los_mismos_objetos():
    import plan_jobs as pj
    from shopping.projection import (build_shopping_projection, classify_projection_jobs, enqueue_shopping_reprojection,
                                     shopping_list_fingerprint)
    from shopping.projection import read_model, reprojection, status
    assert pj.build_shopping_projection is build_shopping_projection is read_model.build_shopping_projection
    assert pj.classify_projection_jobs is classify_projection_jobs is status.classify_projection_jobs
    assert pj.enqueue_shopping_reprojection is enqueue_shopping_reprojection is reprojection.enqueue_shopping_reprojection
    assert pj.shopping_list_fingerprint is shopping_list_fingerprint is reprojection.shopping_list_fingerprint


def test_b_plan_jobs_ya_no_define_el_read_model_y_el_paquete_no_importa_plan_jobs_al_cargar():
    src = (_BACKEND / "plan_jobs.py").read_text(encoding="utf-8")
    for fn in ("def build_shopping_projection(", "def classify_projection_jobs(", "def shopping_list_fingerprint(",
               "def enqueue_shopping_reprojection(", "def _compact_row(", "def _row_cost("):
        assert fn not in src, fn
    assert "from shopping.projection.read_model import" in src
    for mod in ("read_model", "reprojection", "status"):
        txt = (_BACKEND / "shopping" / "projection" / f"{mod}.py").read_text(encoding="utf-8")
        head = txt.split("def ", 1)[0]
        assert "import plan_jobs" not in head, f"{mod}: importar plan_jobs al cargar sería un ciclo (plan_jobs importa el paquete)"
    # el consumidor sigue en el outbox y resuelve el read model por el nombre re-exportado (los tests lo parchean ahí)
    assert "def _consume_shopping_projection(" in src and "projection = build_shopping_projection(" in src


def test_c_la_reproyeccion_usa_las_primitivas_del_outbox_en_tiempo_de_llamada(monkeypatch):
    import plan_jobs as pj
    calls = []
    monkeypatch.setattr(pj, "plan_jobs_enabled", lambda: calls.append("enabled") or False)
    assert pj.enqueue_shopping_reprojection("e45e649c-231d-493a-adbf-af8aa8b73ce8", "f47126cb-e137-4003-9db3-cbec22b02d59", reason="t") is None
    assert calls == ["enabled"], "el paquete consulta plan_jobs.plan_jobs_enabled (parcheable) y no una copia importada"


# ----------------------------------------------------------------------------- god files congelados
# Foto 2026-09-05 (líneas) + 3 %. Superarlo NO se arregla subiendo el número: se arregla extrayendo.
_CEILINGS = {
    "graph_orchestrator.py": 53_100,   # 51 572
    "cron_tasks.py": 36_550,           # 35 479
    "routers/plans.py": 18_100,        # 17 549
    "shopping_calculator.py": 14_400,  # 13 968
    "plan_jobs.py": 800,               # outbox puro: si crece, es que algo volvió a colarse
}
_FRONTEND_CEILINGS = {
    "pages/Dashboard.jsx": 10_800,          # 10 485
    "context/AssessmentContext.jsx": 4_700,  # 4 528
}


@pytest.mark.parametrize("rel,ceiling", sorted(_CEILINGS.items()))
def test_d_god_files_backend_congelados(rel, ceiling):
    n = (_BACKEND / rel).read_text(encoding="utf-8", errors="replace").count("\n")
    assert n <= ceiling, f"{rel}: {n} líneas > tope {ceiling}. Extrae un módulo (roadmap 2.5 §11), no subas el tope."


@pytest.mark.parametrize("rel,ceiling", sorted(_FRONTEND_CEILINGS.items()))
def test_e_god_files_frontend_congelados(rel, ceiling):
    for base in (_BACKEND.parents[0], _BACKEND.parent):
        p = base / "frontend" / "src" / rel
        if p.exists():
            n = p.read_text(encoding="utf-8", errors="replace").count("\n")
            assert n <= ceiling, f"{rel}: {n} líneas > tope {ceiling}. Extrae un componente/hook, no subas el tope."
            return
    pytest.skip("frontend hermano no disponible")
