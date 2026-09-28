"""[P1-PLAN-LOTE-721 · 2026-09-28] La ficha de un plato registrado — el contrato entre el cliente y el servidor.

La mitad del servidor la ancla `test_p1_plan_lote_720.py`; la del cliente, `frontend/src/__tests__/lote721.test.jsx`.
Aquí va lo que ninguno de los dos ve solo, porque vive en la COSTURA:

  1. El vocabulario del origen. [P1-PLAN-LOTE-762] La ficha ya no PINTA el origen (el dueño: «que no aparezcan
     detalles innecesarios»); solo lo usa para decidir si la receta del plan es la de ese plato (`plan_meal`), y ese
     valor tiene que seguir existiendo en `ficha_comida.ORIGENES_DE_COMIDA`.
  2. Lo que la ficha pide existe: `GET /api/diary/meal/{id}` y «Registrar otra vez» con los campos de
     `RepeatMealRequest`; el componedor manda `origin` y `ManualMealRequest` lo acepta.
  3. La foto NO viaja: la promesa de la Política de Privacidad («No retenemos la imagen una vez procesada») sigue
     siendo verdad. El almacén del dispositivo no habla con la red y el servidor no tiene ni ruta ni columna de foto
     para la comida.

Tooltip-anchor: P1-PLAN-LOTE-721
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import ficha_comida as fc

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _f(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip(f"frontend ausente: {rel}")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def _b(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_el_origen_que_la_ficha_usa_es_del_vocabulario_del_servidor():
    ficha = _f("src/components/dashboard/FichaDeComida.jsx")
    usados = set(re.findall(r"fuente === '(\w+)'", ficha))
    assert usados, "la ficha ya no mira el origen: revisar este contrato"
    assert usados <= set(fc.ORIGENES_DE_COMIDA)
    assert "getOrigenes" not in ficha   # [762] las etiquetas de origen ya no se pintan


def test_lo_que_la_ficha_pide_existe():
    ficha = _f("src/components/dashboard/FichaDeComida.jsx")
    assert "fetchWithAuth(`/api/diary/meal/${meal.id}`)" in ficha
    diary = _b("routers/diary.py")
    assert '@router.get("/meal/{meal_id}")' in diary
    assert 'APIRouter(prefix="/api/diary"' in diary or "prefix=\"/api/diary\"" in diary

    # «Registrar otra vez»: los campos que manda son los de RepeatMealRequest
    assert "JSON.stringify({ source_meal_id: meal.id, meal_type: meal.meal_type || undefined, days_ago: 0 })" in ficha
    from routers.diary import RepeatMealRequest
    assert {"source_meal_id", "meal_type", "days_ago"} <= set(RepeatMealRequest.model_fields)


def test_el_origen_estimado_cruza_de_un_lado_al_otro():
    assert "origin: lines.some((l) => l.estimated) ? 'estimate' : undefined," in _f("src/components/dashboard/LogMealModal.jsx")
    from routers.diary import ManualMealRequest
    assert "origin" in ManualMealRequest.model_fields


def test_la_foto_no_viaja():
    almacen = _f("src/utils/fotosDeComidas.js")
    assert not re.search(r"fetchWithAuth|fetch\(|XMLHttpRequest|sendBeacon|/api/", almacen)
    # [P1-PLAN-LOTE-762] el pie «Solo en este dispositivo» salió de la ficha; la promesa sigue: la foto sale del
    # almacén del dispositivo y la ficha no la pide a ninguna ruta
    ficha = _f("src/components/dashboard/FichaDeComida.jsx")
    assert "useFotoDeComida(userId, meal?.id, 'foto')" in ficha
    assert not re.search(r"fetchWithAuth\([^)]*(foto|photo)", ficha, re.I) and "photo_url" not in ficha
    # el servidor no tiene dónde guardarla: ni columna en la migración del lote ni en la ficha que devuelve
    mig = _b("migrations/p1_plan_lote_720_consumed_meals_origen_2026_09_28.sql")
    assert not re.search(r"image|photo_url|foto", mig.split("ADD COLUMN", 1)[1], re.I)
    assert "image" not in _b("ficha_comida.py").split("def ficha_de_comida", 1)[1]
