# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-124 · 2026-09-19] «Lo que más registras» sale al instante y el escáner también desde el diario.

El dueño: «lo que más registras dura 1 segundo más o menos para aparecer cada vez que salgo y entro, y lo de registrar
comida mediante ver días anteriores no aparece la manera de agregar mediante la cámara o foto».

  · «Registrar comida» se monta al abrirse y nacía con `frequent = []`: cada apertura enseñaba la hoja sin la lista y un
    segundo después la empujaba dentro. Misma clase que «Calculando tus metas…» (lote 112). Ahora nace con la última
    lista buena (`frontend/src/utils/frequentFoodsCache.js`: por usuario, borrada en `_clearUserScopedCaches`) y vuelve
    a pedir por detrás; una respuesta mala no la pisa y una idéntica ni toca el estado. El usuario sale de `userId` o
    de `mealfit_user_id` (lo que la app ya recuerda al iniciar sesión): los dos montajes con contrato de texto
    (TrackingProgress, Dashboard) no se tocaron.
  · El diario montaba el componedor SIN `onScan`, y sin él no pinta «Buscar o escribir / Escanear con foto». Ahora cede
    el paso al escáner, que nace YA en el día que se mira (`initialDaysAgo`, hasta 7: el tope de
    `POST /api/diary/consumed`). El aviso «Quedó en el diario de…» ya no llama «antier» a hace cinco días.
  · Los chips de «¿Cuándo?» vivían copiados en el componedor y el escáner y habían divergido: `dayOptions.js`.

Visto en el arnés: diario → Registrar comida → «Escanear con foto» → «Escanear comida» sobre el diario.
Contrato fino: `frontend/src/__tests__/lote124.test.jsx`. Cero cambios de backend: el endpoint ya aceptaba 0..7."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_lo_que_mas_registras_nace_con_la_lista_recordada():
    lm = _front("src/components/dashboard/LogMealModal.jsx")
    assert "useState(() => readFrequentFoodsCache(uid) || []);" in lm
    assert "writeFrequentFoodsCache(uid, d.items);" in lm
    assert "mismaListaFrecuente(prev, d.items) ? prev : d.items" in lm, "una respuesta idéntica no mueve la lista"
    assert "safeLocalStorageGet('mealfit_user_id', null)" in lm and "u !== 'guest'" in lm
    cache = _front("src/utils/frequentFoodsCache.js")
    assert "if (guardado?.userId === userId && Array.isArray(guardado.items))" in cache, "la caché es POR usuario"
    assert "clearFrequentFoodsCache();" in _front("src/context/AssessmentContext.jsx"), "y se borra al cerrar sesión"


def test_el_escaner_tambien_desde_el_diario_y_en_el_dia_que_se_mira():
    dh = _front("src/components/dashboard/DiaryHistory.jsx")
    assert "onScan={pasarAlEscaner}" in dh
    assert "<ScanMealModal isOpen onClose={cerrarEscaner} userId={userId || 'guest'} initialDaysAgo={atras} />" in dh
    assert "if (registrando || escaneando) return;" in dh, "con cualquiera de las dos hojas encima, las teclas son suyas"
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    assert "useState(() => normalizarDiasAtras(initialDaysAgo));" in sm
    assert "const dia = nombreDelDiaAtras(t, daysAgo);" in sm
    diary = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "days_ago: int = Field(default=0, ge=0, le=7)" in diary, "el escáner no promete más días que el backend"


def test_los_chips_del_dia_en_un_solo_sitio():
    assert "export const getDayOptionsCon = (t, daysAgo) => {" in _front("src/components/dashboard/dayOptions.js")
    for rel in ("src/components/dashboard/LogMealModal.jsx", "src/components/dashboard/ScanMealModal.jsx"):
        src = _front(rel)
        assert "from './dayOptions';" in src, rel
        assert "const _getDayOptions = (t) => [" not in src, f"{rel}: volvió la copia local"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 124
