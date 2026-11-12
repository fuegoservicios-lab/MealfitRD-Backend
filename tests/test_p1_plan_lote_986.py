"""[P1-PLAN-LOTE-986 · 2026-11-11] Una comida de OTRA franja principal no se fusiona en esta.

Caso vivo (cuenta de prueba): «El arroz está cocinado con aceite de coco y este es mi almuerzo, en el desayuno me comí
dos bagel con jamón…» — los bagels (meal_type=desayuno) acabaron dentro del almuerzo: una fila de 1.050 kcal, y el
coach dijo que estaban como desayuno.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import db_facts  # noqa: E402
from agent import _detectar_intencion_parte_de_comida, _intentar_fusionar_alimento_en_comida_existente  # noqa: E402

_TEXTO = ("El arroz está cocinado con aceite de coco y este es mi almuerzo, en el desayuno me comí dos bagel con "
          "jamón y quedo cada uno\n1 taza")


def _hoy(meal_type, minutos=1):
    t = (datetime.now(timezone.utc) - timedelta(minutes=minutos)).isoformat()
    return [{"id": f"{meal_type}-1", "meal_name": "Arroz blanco con aceite de coco y pollo guisado", "meal_type": meal_type,
             "calories": 440, "protein": 26, "carbs": 50, "healthy_fats": 14, "ingredients": [],
             "created_at": t, "consumed_at": t}]


_BAGELS = {"meal_name": "Dos bagels con jamón", "meal_type": "desayuno", "calories": 610, "protein": 30,
           "carbs": 99, "healthy_fats": 12, "ingredients": ["Bagels (2 unidades, ≈100 g cada uno)", "Jamón (≈56 g)"]}


def test_el_detector_toma_la_franja_de_la_frase():
    assert _detectar_intencion_parte_de_comida([HumanMessage(content=_TEXTO)]) == "desayuno"
    assert _detectar_intencion_parte_de_comida([HumanMessage(content="fue parte del almuerzo")]) == "almuerzo"


def test_los_bagels_no_entran_en_el_almuerzo(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _hoy("almuerzo"))
    assert _intentar_fusionar_alimento_en_comida_existente("u", dict(_BAGELS), [HumanMessage(content=_TEXTO)]) is None


def test_otra_franja_principal_no_se_fusiona_aunque_diga_parte_de(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _hoy("almuerzo"))
    msgs = [HumanMessage(content="esto lo comí en la cena, no es parte del almuerzo")]
    assert _intentar_fusionar_alimento_en_comida_existente("u", {**_BAGELS, "meal_type": "cena"}, msgs) is None
    # sin que el usuario nombre la otra franja, la adición sí entra (meal_type puesto por la hora)
    assert _intentar_fusionar_alimento_en_comida_existente(
        "u", {**_BAGELS, "meal_type": "cena"}, [HumanMessage(content="fue parte del almuerzo")]) is not None


def test_misma_franja_o_snack_sigue_fusionando(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _hoy("almuerzo"))
    msgs = [HumanMessage(content="esto fue parte del almuerzo")]
    for mt in ("almuerzo", "merienda", None):
        args = {**_BAGELS, "meal_type": mt} if mt else {k: v for k, v in _BAGELS.items() if k != "meal_type"}
        fused = _intentar_fusionar_alimento_en_comida_existente("u", args, msgs)
        assert fused is not None and fused["meal_id"] == "almuerzo-1", mt


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 986
