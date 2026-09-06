# -*- coding: utf-8 -*-
"""[I20 · dato ausente ≠ cero · 2026-09-06] Entrega parcial de los contratos I20–I27.

Tres incidentes en un día, la misma forma: `int(x or -1)` sobre un `attempts` que valía 0,
`generation_status = 'complete'` sobre un estado que ninguna fila tiene, y
`canonicalize_unit(unit) or "unidad"` sobre una unidad que el mapa no conoce. Los tres
mecanismos funcionaban, corrían y no se quejaban; lo que estaba mal era **el dato que esperaban
encontrar**, y por eso ningún test los cazó — todos les daban el dato que sí existía.

Este test ancla el documento a los tres casos: si alguno se revierte, falla aquí **además** de en
su propio test. Un contrato que solo vive en un `.md` es una intención.

Doc: `backend/docs/invariante_dato_ausente.md`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_DOC = _BACKEND / "docs" / "invariante_dato_ausente.md"


def test_el_documento_existe_y_nombra_los_tres_casos():
    assert _DOC.exists(), "se borró el doc del contrato I20"
    txt = _DOC.read_text(encoding="utf-8")
    for marker in ("P0-FILL-FENCED", "P1-QUALITY-SWEEP-STATUS", "P1-UNKNOWN-UNIT-NOT-WHOLE"):
        assert marker in txt, f"el doc dejó de citar {marker}, que es su evidencia"


# ── caso 1: attempts=0 es un valor, no una ausencia ──────────────────────────────────────────
def test_el_fencing_distingue_cero_de_ausente():
    """`int(x or -1)` con `attempts=0` daba -1 y el fence rechazaba TODA primera escritura.
    Vivo 40 minutos en producción; los 8 tests que existían pasaban porque usaban `attempts=3`."""
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.find("_chunk_fence")
    assert i > 0, "desapareció el fence del chunk"
    trozo = src[i:i + 4000]
    assert "is not None" in trozo, (
        "el fence volvió a colapsar ausente y cero: `attempts=0` es un valor legítimo")
    assert not re.search(r"int\(\s*_at_raw\s+or\s", trozo), "volvió el `int(x or …)`"


# ── caso 2: un predicado que no puede casar ──────────────────────────────────────────────────
def test_el_barrido_no_pide_un_estado_inexistente():
    """Cero planes en `complete` en toda la base: el barrido corría, resolvía 0 y «0 resueltos»
    es exactamente lo que se espera de un barrido sano. La ausencia se leía como salud."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.find("def _resolve_stale_plan_quality_alerts")
    assert i > 0
    j = re.search(r"\n(?:async )?def ", src[i + 10:])
    cuerpo = src[i:i + 10 + (j.start() if j else len(src))]
    assert "complete_partial" in cuerpo, "el barrido volvió a mirar solo `complete`"


# ── caso 3: no lo sé ≠ el default ────────────────────────────────────────────────────────────
def test_una_unidad_desconocida_no_colapsa_al_default():
    from nutrition_db import IngredientNutritionDB, NutritionInfo
    db = IngredientNutritionDB.__new__(IngredientNutritionDB)
    info = NutritionInfo(name="Cilantro", kcal=20, protein=1, carbs=3, fats=0,
                         density_g_per_unit=50.0)
    assert db.to_grams(3, "zzz", info) is None, "una unidad desconocida volvió a pesar «una pieza»"
    assert db.to_grams(3, "unidad", info) == 150.0, "«sin unidad» dejó de significar la pieza entera"


# ── la regla, no solo los casos ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("regla", [
    "x if x is not None else default",
    "cuántas filas lo tienen",
    "no lo sé",
    "antes de desplegarlo",
])
def test_el_doc_enuncia_las_cuatro_aplicaciones(regla):
    """Los cuatro puntos de «Cómo se aplica». Sin ellos el doc es una anécdota de tres bugs en vez
    de un contrato que alguien pueda seguir en código nuevo."""
    txt = _DOC.read_text(encoding="utf-8")
    assert regla in txt, f"el doc perdió la regla {regla!r}"
