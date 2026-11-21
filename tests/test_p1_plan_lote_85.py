# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-85 · 2026-09-17] Un solo botón «Registrar comida» en Progreso en Tiempo Real; las dos vías (buscar o
escribir / escanear con foto) viven DENTRO del componedor como un selector de dos opciones. El dueño: «desde afuera las dos
opciones se ve un poco raro estéticamente». Los dos catálogos afectados pierden las claves huérfanas y ganan las nuevas."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_la_tarjeta_tiene_un_solo_boton():
    tp = _front("src/components/dashboard/TrackingProgress.jsx")
    assert "scanBtnSecondary" not in tp and "t('Escanear comida con la cámara')" not in tp
    assert tp.count("{t('Registrar comida')}") == 1
    # [P1-PLAN-LOTE-382] …y le pasa la comida y el día del componedor
    assert ("const handleLogToScan = useCallback((d) => { setLogOpen(false); setScanOpen(d || true); }, []);" in tp
            or "const handleLogToScan = useCallback(() => { setLogOpen(false); setScanOpen(true); }, []);" in tp)
    assert "scanBtnSecondary" not in _front("src/components/dashboard/TrackingProgress.module.css")


def test_las_dos_vias_viven_dentro_del_componedor():
    # [P1-PLAN-LOTE-1147] la foto sigue dentro del componedor, ahora como botón de la caja (sin pestañas)
    lm = _front("src/components/dashboard/LogMealModal.jsx")
    assert "onClick={() => onScan({ mealType, daysAgo })}" in lm and "aria-label={t('Escanear con foto')}" in lm
    assert "t('Foto')" not in lm and 'className={styles.modes}' not in lm


@pytest.mark.parametrize("loc", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_los_catalogos_sin_huerfanas_y_con_las_nuevas(loc):
    d = json.loads(_front(f"src/i18n/locales/{loc}.json"))
    # [P1-PLAN-LOTE-1146] «Foto» volvió con uso propio (la pestaña «Mover» de Compartir, lote 1144): ya no es huérfana
    assert "Escanear comida con la cámara" not in d
    assert d.get("Escanear con foto")
    assert "Buscar o escribir" not in d and "Cómo registrar" not in d   # [1147] las pestañas se fueron


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 85
    assert "P1-PLAN-LOTE-85" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
