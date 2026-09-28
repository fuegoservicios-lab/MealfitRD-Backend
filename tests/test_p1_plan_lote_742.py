# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-742 · 2026-09-28] El borrado de la Nevera al cambiar de usuario sin arrastrar pantryCache al arranque.

Tras 629/681/702 el arranque medía 148,5 kB gz sobre un techo de 148 (mealfitrd-ia-9b). `pantryCache.js` entraba sólo
porque AssessmentContext llama a `borrarCacheDeInventario`, y ese borrado tiene que ser síncrono (P1-XTAB-CACHE-LEAK).
El estado del inventario vive ahora en `inventarioEnMemoria.js` (sólo depende de safeLocalStorage); pantryCache lo
comparte como `_INV.entrada` y re-exporta el borrado. El comportamiento lo prueba `frontend/src/__tests__/lote742.test.js`.

tooltip-anchor: P1-PLAN-LOTE-742
"""
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_el_arranque_importa_el_modulo_minimo():
    if not _SRC.exists():
        pytest.skip("frontend ausente (repo hermano)")
    ctx = (_SRC / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert "import { borrarCacheDeInventario } from '../utils/inventarioEnMemoria';" in ctx
    assert "utils/pantryCache'" not in ctx
    cache = (_SRC / "utils" / "pantryCache.js").read_text(encoding="utf-8")
    assert "from './inventarioEnMemoria';" in cache and "let _inventoryEntry" not in cache
