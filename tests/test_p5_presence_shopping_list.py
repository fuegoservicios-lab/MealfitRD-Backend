"""[P5-PRESENCE-SHOPPING-LIST · 2026-06-23] La lista de compras es un ESPEJO VIVO de la Nevera
(spec del owner): un ítem aparece SOLO si está AUSENTE de la Nevera; presente en cualquier
cantidad → oculto; todo presente → lista vacía; se agota un ítem → reaparece.

Cambios anclados (frontend-only; el backend sigue persistiendo is_restocked pero ya NO suprime
contenido — limpieza de _build_hybrid spawneada como follow-up):
  1. shoppingHelpers.js: getCanonicalIngredientSet + getDeltaSourceList (membresía = set
     canónico COMPLETO weekly, así un ítem recortado del ciclo por restock se sigue chequeando).
  2. Dashboard.jsx buildDeltaShoppingList: binario presencia (present→hide / absent→show);
     ELIMINADA la supresión por ventana is_restocked (isPostRestockRotation + _staleDedup).
  3. Los 3 sitios de fuente del delta usan getDeltaSourceList.
"""
import os
import re

_FE = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src")
_DASH = open(os.path.join(_FE, "pages", "Dashboard.jsx"), encoding="utf-8").read()
_HELPERS = open(os.path.join(_FE, "utils", "shoppingHelpers.js"), encoding="utf-8").read()


def test_helpers_exist():
    assert "export const getCanonicalIngredientSet" in _HELPERS
    assert "export const getDeltaSourceList" in _HELPERS
    assert "P5-PRESENCE-SHOPPING-LIST" in _HELPERS


def test_presence_binary_in_delta():
    # El loop debe ocultar lo presente y mostrar lo ausente.
    assert "MODELO DE PRESENCIA" in _DASH
    assert re.search(r"if\s*\(\s*_invQty\s*>\s*0\s*\)", _DASH), "present→hide branch ausente"
    # El comentario debe documentar que un ítem agotado reaparece.
    assert "se agota la leche" in _DASH or "reaparece" in _DASH


def test_time_window_suppression_removed():
    # isPostRestockRotation / _staleDedup ya NO deben existir como CÓDIGO (solo en comentarios).
    # Heurística: no debe haber una asignación `const isPostRestockRotation =` ni `const _staleDedup =`.
    assert not re.search(r"const\s+isPostRestockRotation\s*=", _DASH), "isPostRestockRotation debe estar eliminado"
    assert not re.search(r"const\s+_staleDedup\s*=", _DASH), "_staleDedup debe estar eliminado"


def test_forward_looking_filter_present():
    # [P5-PRESENCE-FORWARD-LOOKING] Un agotado reaparece SOLO si el plan restante lo usa.
    assert "P5-PRESENCE-FORWARD-LOOKING" in _DASH
    assert "remainingNeedsSet" in _DASH
    # Construye el set desde las comidas de hoy en adelante (computeRollingWindow → todayPlanDayIndex).
    assert "computeRollingWindow(" in _DASH
    # Ciclo terminado → set vacío → nada reaparece.
    assert re.search(r"daysLeft\s*<=\s*0", _DASH), "debe vaciar la lista cuando el ciclo terminó"
    # Fail-open: el comentario debe documentar que ante datos raros NO filtra (no esconde lo necesario).
    assert "FAIL-OPEN" in _DASH or "fail-open" in _DASH


def test_three_sources_use_delta_source_list():
    # Los 3 sitios (botón verde, PDF, restock) deben sourcing por getDeltaSourceList.
    assert _DASH.count("getDeltaSourceList(") >= 3, "los 3 sitios de fuente deben usar getDeltaSourceList"
    # Y el import debe incluirlo.
    assert "getDeltaSourceList" in _DASH.split("from '../utils/shoppingHelpers'")[0].rsplit("import", 1)[-1] \
        or "getDeltaSourceList," in _DASH
