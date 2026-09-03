"""[P1-PLANDATA-ID-RECALC · 2026-09-02] Ancla backend del fix frontend.

Medido en prod: 24 × `POST /api/plans/adopt-guest-plan → 409` en un día, uno por carga de
página, siempre desde el navegador del dueño. Causa: el recálculo de la lista (que corre
al cargar) devuelve `plan_data` SIN `id`, y cuatro sitios lo adoptaban crudo en estado y
en `mealfit_plan`; en la siguiente carga el backstop `P1-GUEST-ADOPT-SELFHEAL` veía «días
sin id» y lo tomaba por plan de invitado. Misma clase que P1-PLANDATA-ID-HYDRATE-2.

Tooltip-anchor: P1-PLANDATA-ID-RECALC | conservarPlanId en los escritores de plan_data
"""
import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
RAW = [
    re.compile(r"setPlanData\((result|rd)\.plan_data\)"),
    re.compile(r"safeLocalStorageSet\('mealfit_plan', JSON\.stringify\((result|rd)\.plan_data\)\)"),
    re.compile(r"safeLocalStorageSet\('mealfit_plan', (result|rd)\.plan_data\)"),
]


def _src(rel: str) -> str:
    p = FRONTEND / rel
    if not p.exists():
        pytest.skip(f"frontend no visible desde este checkout: {rel}")
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", ["src/pages/Dashboard.jsx", "src/context/AssessmentContext.jsx"])
def test_no_raw_plan_data_adoption(rel):
    src = _src(rel)
    for rx in RAW:
        m = rx.search(src)
        assert m is None, f"{rel}: forma cruda «{m and m.group(0)}» — envuelve con conservarPlanId(...)"


def test_dashboard_marks_the_fix_and_imports_helper():
    src = _src("src/pages/Dashboard.jsx")
    assert "P1-PLANDATA-ID-RECALC" in src
    assert re.search(r"import \{[^}]*\bconservarPlanId\b[^}]*\} from '\.\./context/AssessmentContext'", src)


def test_pantry_keeps_its_own_id_heal():
    src = _src("src/pages/Pantry.jsx")
    assert re.search(r"if \(result\.plan_data\.id == null\) \{\s*const _healId = planData\?\.id \?\? _serverKnownPlanId;", src)


def test_vitest_blanket_exists():
    assert (FRONTEND / "src/__tests__/PlanDataIdRecalc.source.test.js").exists()
