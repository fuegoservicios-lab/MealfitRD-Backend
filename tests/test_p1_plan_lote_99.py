# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-99 · 2026-09-18] El componedor «Registrar comida», rehecho para el teléfono.

El dueño, con captura: «incómodo de interactuar y entender… un cambio radical». Lo que cambió para la mano: hoja
inferior con cabecera y pie fijos y cuerpo desplazable; preguntas con chips («¿Qué comida es?», «¿Cuándo?») en vez de
dos desplegables sin etiqueta; «Lo que más registras» en lista; campos a 1rem (iOS hace zoom por debajo de 16px);
el pie dice por qué «Registrar» está apagado. Ancla cross-repo; el contrato fino vive en `LogMealModal.lote99.test.jsx`."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_DASH = "src/components/dashboard/"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _regla(css: str, sel: str) -> str:
    i = css.index("\n" + sel + " {")
    return re.sub(r"\s+", " ", css[i:css.index("}", i)])


def test_hoja_inferior_con_pie_fijo():
    css = _front(_DASH + "LogMealModal.module.css")
    assert "display: flex; flex-direction: column;" in _regla(css, ".panel")
    assert "max-height: 94dvh;" in _regla(css, ".panel")
    assert "align-items: flex-end;" in _regla(css, ".overlay")
    assert "flex: 1 1 auto; min-height: 0; overflow-y: auto;" in _regla(css, ".body")
    assert "env(safe-area-inset-bottom, 0px)" in _regla(css, ".footer")
    assert "@media (min-width: 641px) {" in css, "en escritorio sigue centrado"


def test_preguntas_con_chips_y_sin_desplegables():
    jsx = _front(_DASH + "LogMealModal.jsx")
    for k in ("¿Qué comiste?", "¿Qué comida es?", "¿Cuándo?", "Añade al menos un alimento para registrar."):
        assert "t('" + k + "')" in jsx, k
    assert "<Chips label={t('Tipo de comida')}" in jsx and "<Chips label={t('Día')}" in jsx
    assert "className={styles.select}" not in jsx and "styles.selectors" not in jsx
    # la única <select> que queda es la unidad de cada línea, con el tratamiento oscuro de P1-LOGMEAL-SELECT-DARK
    css = _front(_DASH + "LogMealModal.module.css")
    assert "color-scheme: light;" in _regla(css, ".lineUnit")
    assert ':global(html[data-theme="dark"]) .lineUnit {' in css


def test_campos_a_1rem_y_frecuentes_en_lista():
    css = _front(_DASH + "LogMealModal.module.css")
    for sel in (".search", ".lineQty", ".lineUnit", ".nameInput", ".macroInput"):
        assert "font-size: 1rem;" in _regla(css, sel), sel
    assert "overflow-x: auto" not in css
    assert "flex-direction: column;" in _regla(css, ".frequentList")


def test_extra_a_secas_y_los_catalogos():
    shared = _front(_DASH + "mealLogShared.js")
    assert "label: t('Extra')" in shared and "Extra (fuera del plan)" not in shared
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        d = json.loads(_front("src/i18n/locales/" + loc + ".json"))
        assert "Extra (fuera del plan)" not in d
        for k in ("¿Qué comiste?", "¿Qué comida es?", "¿Cuándo?", "Extra", "Ponle nombre (opcional)"):
            assert d.get(k), loc + ": " + k


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 99
    assert "P1-PLAN-LOTE-99" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
