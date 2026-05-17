"""[P3-NAN-INF-SANITIZE · 2026-05-16] Defensa contra
`ValueError: Out of range float values are not JSON compliant` en
`/recalculate-shopping-list`.

Síntoma observado log backend 2026-05-16 21:04:24 (plan 4cc91584):

```
✅ [RECALC] Listas recalculadas exitosamente ×1 personas
INFO: 127.0.0.1:53208 - "POST /api/plans/recalculate-shopping-list HTTP/1.1" 500
ERROR: Exception in ASGI application
ValueError: Out of range float values are not JSON compliant
```

El handler completaba toda la lógica de negocio exitosamente, pero el
serializador JSON de Starlette/FastAPI rechazaba un NaN/Inf presente en
`merged_plan_data`. El cliente recibía 500 + CORS error secundario.

Causa probable (sin confirmar el callsite exacto): el aggregator/scaling
de shopping_calculator hace alguna división por zero cuando un ítem tiene
`package_size=0` en master_ingredients, o un `multiplier=0` se cuela en
una fórmula. Esto es el bug RAÍZ; este helper es defense-in-depth para
que un NaN downstream no crashe la API.

Patrón de fix: `_sanitize_floats_for_json` recorre el dict de response y
reemplaza NaN/Inf con `None` (JSON-compliant). El frontend ya trata `null`
como "valor ausente" en estos campos.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Anchor parser-based: helper declarado + aplicado al return del recalc
# ---------------------------------------------------------------------------


def test_helper_declared_with_marker():
    """`_sanitize_floats_for_json` debe existir + marker P3-NAN-INF-SANITIZE."""
    assert "def _sanitize_floats_for_json(" in _PLANS, (
        "Helper `_sanitize_floats_for_json` no declarado en routers/plans.py."
    )
    assert "P3-NAN-INF-SANITIZE" in _PLANS, (
        "Marker `P3-NAN-INF-SANITIZE` ausente — refactor cosmético podría "
        "borrar el por qué del sanitize."
    )


def test_helper_uses_math_isnan_and_isinf():
    """El helper debe chequear AMBOS NaN y Inf — Inf solo cubriría parte
    del problema (división por cero produce inf; 0/0 produce nan)."""
    idx = _PLANS.find("def _sanitize_floats_for_json(")
    end = _PLANS.find("\n\n", idx)
    body = _PLANS[idx:end if end > 0 else idx + 1500]
    assert "isnan" in body, "Helper no chequea `isnan` — NaN pasaría al JSON."
    assert "isinf" in body, "Helper no chequea `isinf` — Inf pasaría al JSON."


def test_helper_recursive_on_dict_list_tuple():
    """El helper debe recursar sobre containers — un NaN dentro de un
    `plan_data['days'][0]['meals'][1]['calories']` no se sanitiza si solo
    se chequea el top-level."""
    idx = _PLANS.find("def _sanitize_floats_for_json(")
    end = _PLANS.find("\n\n", idx)
    body = _PLANS[idx:end if end > 0 else idx + 1500]
    assert "isinstance(obj, dict)" in body, (
        "Helper no recursa sobre dict — NaN dentro de plan_data no se sanitiza."
    )
    assert "isinstance(obj, list)" in body, (
        "Helper no recursa sobre list — NaN dentro de aggregated_shopping_list no se sanitiza."
    )


def test_recalc_endpoint_applies_sanitize_to_response():
    """El endpoint `/recalculate-shopping-list` DEBE invocar
    `_sanitize_floats_for_json` sobre `merged_plan_data` antes de retornar."""
    # Localizar el return final del handler
    idx = _PLANS.find('@router.post("/recalculate-shopping-list")')
    next_router = _PLANS.find("@router.", idx + 50)
    body = _PLANS[idx:next_router if next_router > 0 else idx + 8000]

    # Buscar `_sanitize_floats_for_json(merged_plan_data)` en el return
    assert "_sanitize_floats_for_json(merged_plan_data)" in body, (
        "Endpoint NO sanitiza `merged_plan_data` antes de retornar — "
        "revierte P3-NAN-INF-SANITIZE. Sin esto, un NaN downstream "
        "(división por zero del aggregator) crashea con 500."
    )


# ---------------------------------------------------------------------------
# Tests funcionales del helper (import directo)
# ---------------------------------------------------------------------------


def _load_helper():
    """Importa `_sanitize_floats_for_json` de routers/plans.py SIN cargar
    el resto del módulo (que requiere DB, FastAPI app, etc.).

    Estrategia: extraer el bloque del helper como texto, ejecutarlo en
    un namespace aislado con `math` disponible. Más liviano que importar
    `routers.plans` (que tiene side-effects).
    """
    import textwrap
    idx = _PLANS.find("def _sanitize_floats_for_json(")
    end = _PLANS.find("\n\n", idx)
    code = textwrap.dedent(_PLANS[idx:end if end > 0 else idx + 1500])
    ns = {"_math": math}  # nombre que usa el helper
    exec(code, ns)
    return ns["_sanitize_floats_for_json"]


def test_helper_replaces_nan_with_none():
    fn = _load_helper()
    assert fn(float("nan")) is None
    assert fn(float("inf")) is None
    assert fn(float("-inf")) is None


def test_helper_preserves_normal_floats_and_ints():
    fn = _load_helper()
    assert fn(1.5) == 1.5
    assert fn(0.0) == 0.0
    assert fn(-3.7) == -3.7
    assert fn(42) == 42
    assert fn("text") == "text"
    assert fn(None) is None
    assert fn(True) is True


def test_helper_recurses_through_nested_structures():
    fn = _load_helper()
    payload = {
        "ok": 1.5,
        "bad": float("nan"),
        "list": [1, 2, float("inf"), {"nested": float("-inf")}],
        "deep": {"a": {"b": [float("nan"), "ok", 0]}},
    }
    out = fn(payload)
    assert out["ok"] == 1.5
    assert out["bad"] is None
    assert out["list"][2] is None
    assert out["list"][3]["nested"] is None
    assert out["deep"]["a"]["b"][0] is None
    assert out["deep"]["a"]["b"][1] == "ok"


def test_helper_is_idempotent():
    """Aplicar el sanitize dos veces debe dar el mismo resultado que
    aplicarlo una sola vez. None se preserva como None."""
    fn = _load_helper()
    payload = {"a": float("nan"), "b": [float("inf"), 1.0]}
    once = fn(payload)
    twice = fn(once)
    assert once == twice


def test_helper_result_is_json_serializable():
    """Sanity end-to-end: tras el sanitize, json.dumps NO debe levantar
    `ValueError: Out of range float values are not JSON compliant`."""
    import json
    fn = _load_helper()
    payload = {
        "plan_data": {
            "days": [{"meals": [{"calories": float("nan"), "protein": 1.5}]}],
            "aggregated_shopping_list": [{"qty": float("inf"), "name": "Foo"}],
        }
    }
    sanitized = fn(payload)
    # Con allow_nan=False (default de Starlette JSONResponse en versiones
    # nuevas), json.dumps debe completar sin ValueError.
    json.dumps(sanitized, allow_nan=False)
