"""[P1-MICROS-EMPTY-QUIET · 2026-08-21] El panel de micros era un muro de 0%
cuando el plan no tiene platos.

Reportado con captura: plan pausado sin platos → 15 tarjetas «por mejorar» al
0% con dosis de suplementos y precauciones clínicas sobre comidas que NO
existen. El reporte de micros es el promedio de lo que aportan los platos: sin
platos, todo-ceros — vacuo.

El gate es DATA-driven (`panel.every(valor == 0)`), no el estado de la cola:
un reporte todo-ceros es vacuo sea cual sea la causa, y cualquier plato real
aporta algo a alguno de los 17 micros. El estado compacto conserva cabecera +
una línea («cuándo se llenará») en vez de desaparecer el panel.

La conducta la ancla el companion vitest (render con todo-ceros vs valores
reales); este test protege la existencia del gate y su copy en los catálogos
desde el gate de CI backend.

Tooltip-anchor: P1-MICROS-EMPTY-QUIET
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_METER = (
    _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "MicronutrientMeter.jsx"
)
_VITEST_COMPANION = (
    _REPO_ROOT
    / "frontend"
    / "src"
    / "__tests__"
    / "MicronutrientMeter.p1_micros_empty_quiet.test.jsx"
)
_LOCALES_DIR = _REPO_ROOT / "frontend" / "src" / "i18n" / "locales"

_NOTE_KEY = (
    "Aún no hay platos que medir. Cuando tu plan tenga comidas, aquí verás "
    "cuánto aportan a cada micronutriente."
)


def test_all_zero_gate_exists():
    src = _METER.read_text(encoding="utf-8")
    assert "_allZero" in src and "panel.every" in src, (
        "P1-MICROS-EMPTY-QUIET: desapareció el gate todo-ceros — el panel vuelve "
        "a renderizar 15 tarjetas al 0% con dosis clínicas sobre platos que no existen."
    )
    assert "emptyNote" in src, (
        "P1-MICROS-EMPTY-QUIET: el estado compacto perdió su nota — un panel que "
        "se esfuma deja al usuario preguntándose dónde quedó."
    )


def test_vitest_companion_exists():
    assert _VITEST_COMPANION.exists(), (
        "P1-MICROS-EMPTY-QUIET: falta el companion vitest (todo-ceros → compacto; "
        "cualquier valor real → panel completo intacto)."
    )
    assert "P1-MICROS-EMPTY-QUIET" in _VITEST_COMPANION.read_text(encoding="utf-8")


def test_note_key_translated_in_all_catalogs():
    """La clave ES el texto español (P1-I18N-DASHBOARD): cambiar el copy del
    componente sin tocar los catálogos huerfana la traducción EN SILENCIO."""
    src = _METER.read_text(encoding="utf-8")
    assert _NOTE_KEY in src, (
        "P1-MICROS-EMPTY-QUIET: el copy de la nota cambió en el componente. "
        "Actualiza también esta constante Y los 4 catálogos (la clave es el texto)."
    )
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        data = json.loads((_LOCALES_DIR / f"{loc}.json").read_text(encoding="utf-8"))
        assert _NOTE_KEY in data and data[_NOTE_KEY].strip(), (
            f"P1-MICROS-EMPTY-QUIET: {loc} no traduce la nota del estado compacto."
        )
