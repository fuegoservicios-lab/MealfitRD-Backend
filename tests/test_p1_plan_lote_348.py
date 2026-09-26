# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-348 · 2026-09-26] «Descríbelo y lo calculo»: el plato armado a mano, separado en ingredientes.

El dueño: «¿Qué comiste? es poco flexible: nadie pudiera crear un plato desde cero… si no le tiró foto y no quiere
escribirle al coach». El componedor ya juntaba varios alimentos, pero la estimación por texto devolvía UN número para
todo. Ahora `POST /api/diary/consumed/estimate-plate` separa el texto en ingredientes (gramos + macros de cada uno) y
cada uno llega como línea EDITABLE.

Casa SOLO con los PLATOS del catálogo (cocinados, `per_100g` del plato terminado). Los alimentos del catálogo NO: su
`per_100g` es del alimento CRUDO/seco y la IA estima lo que se comió (cocido) — «200 g de arroz» casado con el arroz
crudo daría ~720 kcal en vez de ~260 (la lección de los lotes 282-286: la base del número).
Tooltip-anchor: P1-PLAN-LOTE-348
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

_DISHES = {
    "moro": {"label": "Moro de habichuelas", "finished_g": 230, "per_100g": {"kcal": 172.6, "protein": 5, "carbs": 30, "fats": 3.5},
             "constituents": [{"name": "Arroz blanco", "g": 80}]},
}


def _item(name, grams, kcal, p=0, c=0, f=0):
    return {"name": name, "grams": grams, "calories": kcal, "protein": p, "carbs": c, "healthy_fats": f}


def test_un_plato_del_catalogo_casa_y_el_resto_queda_estimado(monkeypatch):
    import plato_descrito as pd
    import food_search
    monkeypatch.setattr(food_search, "load_dishes", lambda: _DISHES)
    lineas = pd.lineas_del_plato([_item("moro de habichuelas", 200, 999), _item("Pollo guisado", 150, 260, 30, 4, 12)])
    moro, pollo = lineas
    assert moro["ref"] == "dish:moro" and moro["unit"] == "g" and moro["qty"] == 200
    assert round(moro["macros"]["kcal"]) == 345            # del CATÁLOGO (172.6 × 2), no los 999 de la IA
    assert pollo["ref"] == "custom" and pollo["estimated"] is True and pollo["grams"] == 150
    assert pollo["macros"] == {"kcal": 260.0, "protein": 30.0, "carbs": 4.0, "fats": 12.0}


def test_un_alimento_crudo_del_catalogo_no_se_casa(monkeypatch):
    import plato_descrito as pd
    import food_search
    monkeypatch.setattr(food_search, "load_dishes", lambda: _DISHES)
    (arroz,) = pd.lineas_del_plato([_item("Arroz blanco", 200, 260, 5, 56, 1)])
    assert arroz["ref"] == "custom" and round(arroz["macros"]["kcal"]) == 260


def test_basura_topes_y_maximo_de_lineas(monkeypatch):
    import plato_descrito as pd
    import food_search
    monkeypatch.setattr(food_search, "load_dishes", lambda: {})
    items = [_item("", 100, 100), {"name": "x"}] + [_item(f"cosa {i}", 99999, 99999) for i in range(20)]
    lineas = pd.lineas_del_plato(items)
    assert len(lineas) == 12
    assert all((l["grams"] or 0) <= 2000 and l["macros"]["kcal"] <= 5000 for l in lineas)   # sin gramos: None


def test_el_prompt_ofrece_los_platos_del_catalogo_por_su_nombre(monkeypatch):
    import plato_descrito as pd
    import food_search
    monkeypatch.setattr(food_search, "load_dishes", lambda: _DISHES)
    p = pd.prompt_del_sistema()
    assert "Moro de habichuelas" in p and "grams" in p and "items" in p


def test_el_endpoint_devuelve_las_lineas_y_falla_suave(monkeypatch):
    from routers import diary
    import plato_descrito as pd
    import food_search
    monkeypatch.setattr(food_search, "load_dishes", lambda: _DISHES)

    async def bien(texto, idioma, uid):
        return {"name": "Moro con pollo", "items": [_item("Moro de habichuelas", 230, 0), _item("Pollo guisado", 150, 260)]}

    monkeypatch.setattr(pd, "estimar_con_ia", bien)
    r = asyncio.run(diary.api_estimate_plate(diary.EstimatePlateRequest(text="moro con pollo guisado"), verified_user_id="u" * 8))
    assert r["name"] == "Moro con pollo" and [l["ref"] for l in r["lineas"]] == ["dish:moro", "custom"]

    async def mal(texto, idioma, uid):
        raise TimeoutError("sin IA")

    monkeypatch.setattr(pd, "estimar_con_ia", mal)
    r = asyncio.run(diary.api_estimate_plate(diary.EstimatePlateRequest(text="moro con pollo guisado"), verified_user_id="u" * 8))
    assert r["operation_failed"] is True and r["error_code"] == "estimate_unavailable"


def test_exento_de_cuota_y_con_limitador():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    i = src.index('@router.post("/consumed/estimate-plate")')
    firma = src[i:i + 400]
    assert "_ESTIMATE_PLATE_LIMITER" in firma and "verify_api_quota" not in firma


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 348
