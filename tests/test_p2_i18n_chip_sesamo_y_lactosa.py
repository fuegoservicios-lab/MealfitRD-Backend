"""[P2-I18N-CHIP-SESAMO-Y-LACTOSA-AUSENTES · 2026-08-23] Dos clases que el motor SÍ
bloqueaba y que el formulario no tenía cómo pedirle: sésamo (uno de los 14 alérgenos
obligatorios de la UE; el catálogo trae Ajonjolí, Tahini, Hummus, Aceite de sésamo) y
lactosa (clase propia, más ESTRECHA que lácteos a propósito). Es
``P2-ALLERGEN-CHIPS-REACH-ENGINE`` aplicado a lo que faltaba, con el mismo criterio: no
que el chip EXISTA, sino que su ``val`` llegue al motor y proteja — medido contra
``clinical_backstop_for_meal`` antes de añadirlos.

Las filas «cada chip bloquea su alérgeno» viven en la lista canónica del test hermano;
aquí van las invariantes que justifican que sean chips PROPIOS y no alias de un vecino.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_QALLERGIES = _REPO / "frontend" / "src" / "components" / "assessment" / "questions" / "QAllergies.jsx"
_LOCALES = _REPO / "frontend" / "src" / "i18n" / "locales"
_MARKER = "P2-I18N-CHIP-SESAMO-Y-LACTOSA-AUSENTES"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def chips() -> list:
    if not _QALLERGIES.is_file():
        pytest.skip("QAllergies.jsx no está en este árbol")
    src = _QALLERGIES.read_text(encoding="utf-8", errors="replace")
    return re.findall(r'\{\s*val:\s*"([^"]+)"', src)


def _bloquea(go, chip: str, ingrediente: str) -> bool:
    plato = {"name": "Prueba", "ingredients": [ingrediente], "preparation_steps": []}
    return bool(go.clinical_backstop_for_meal(plato, allergies=[chip]))


def test_los_dos_chips_existen_con_el_val_sin_acento(chips):
    """`val` es el identificador que consume el motor; sigue la convención de sus vecinos
    (`Lacteos`, `Mani`): sin acento."""
    assert "Sesamo" in chips, f"sigue sin chip de sésamo: {chips} [{_MARKER}]"
    assert "Lactosa" in chips, f"sigue sin chip de lactosa: {chips} [{_MARKER}]"
    assert "Sésamo" not in chips, "el val lleva acento: rompe la convención y el motor lo normaliza igual"


@pytest.mark.parametrize("alimento", ["30 g de Tahini", "20 g de Ajonjolí", "100 g de Hummus", "10 ml de Aceite de sésamo"])
def test_sesamo_es_chip_propio_porque_frutos_secos_no_lo_cubre(go, alimento):
    """El sésamo es una semilla y el motor lo separa de los frutos de cáscara a propósito.
    Si `Frutos Secos` empezara a bloquearlo, el chip propio sobraría — y este test lo diría."""
    assert _bloquea(go, "Sesamo", alimento), f"`Sesamo` NO bloquea {alimento!r} [{_MARKER}]"
    assert not _bloquea(go, "Frutos Secos", alimento), (
        f"«Frutos Secos» bloquea {alimento!r}: si es deliberado, el chip de sésamo pasa a ser redundante")


def test_lactosa_es_mas_estrecha_que_lacteos_y_por_eso_es_chip_propio(go):
    """El intolerante a la lactosa tolera lo que el alérgico a la proteína láctea no. Las dos
    clases bloquean la leche; sólo `Lacteos` debe ser el superconjunto."""
    assert _bloquea(go, "Lactosa", "200 ml de Leche entera")
    assert _bloquea(go, "Lacteos", "200 ml de Leche entera")
    lactosa = go._expand_allergy_declarations(["Lactosa"])
    lacteos = go._expand_allergy_declarations(["Lacteos"])
    assert lactosa and lacteos
    assert lactosa < lacteos, (
        f"`Lactosa` ya no es subconjunto ESTRICTO de `Lacteos` — si se igualaron, el chip propio "
        f"miente: {sorted(lactosa - lacteos)} [{_MARKER}]")


def test_los_chips_no_se_confunden_con_un_pollo(go):
    """Control: ninguno de los dos bloquea comida sin su alérgeno."""
    assert not _bloquea(go, "Sesamo", "150 g de Pechuga de pollo")
    assert not _bloquea(go, "Lactosa", "150 g de Pechuga de pollo")


@pytest.mark.parametrize("locale", ["en-US", "fr-FR", "it-IT", "pt-BR"])
def test_las_etiquetas_tienen_traduccion(locale):
    """La etiqueta que se pinta (`t('Sésamo')`, `t('Lactosa')`) tiene que existir en los cuatro
    catálogos; si no, el chip nuevo sale en español en una pantalla que ya no lo está."""
    f = _LOCALES / f"{locale}.json"
    if not f.is_file():
        pytest.skip("catálogos no están en este árbol")
    src = f.read_text(encoding="utf-8")
    for clave in ('"Sésamo":', '"Lactosa":'):
        assert clave in src, f"{locale} sin {clave} [{_MARKER}]"
