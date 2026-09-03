"""[P2-ALLERGEN-CHIPS-REACH-ENGINE · 2026-08-21] El motor sabía bloquear pescado y maní; el
formulario no tenía cómo pedírselo.

El paso de alergias ofrecía SEIS chips — Lácteos, Gluten, Huevo, Mariscos, Nueces, Soya — y el
backstop clínico modela nueve clases, entre ellas `pescado`, `mani` y `sesamo`. La capacidad existía
y era inalcanzable desde la interfaz: el patrón inerte de toda esta ola, esta vez en el sitio donde
más caro sale.

DOS HUECOS MEDIDOS, y el segundo es el grave:

  · **Pescado sin chip.** El marisco tenía el suyo y el pescado no, en un beta cuyo primer país es
    España — donde el catálogo trae boquerones, anchoas, merluza, bacalao y trucha. El alérgico
    tenía que ESCRIBIRLO en el campo libre. Que funcione si lo escribes no es lo mismo que
    ofrecerlo: la mitad de la seguridad de un chip es que el usuario no tiene que acordarse.

  · **«Nueces» NO cubre el maní.** Verificado contra el motor:
        chip `Frutos Secos` + «50 g de Maní»  →  **pasa**
    El cacahuete es una legumbre, no un fruto de cáscara, y el motor los separa a propósito (hay un
    test de esta misma ola que lo exige). Así que un alérgico al maní marcando el único chip que le
    suena se queda **sin protección y creyendo que la tiene** — peor que no tener chip, porque el
    chip le dio una respuesta.

LA AUDITORÍA DECÍA OTRA COSA, y la medición la refuta. Su P2-13 afirmaba que «el chip Mariscos deja
pasar pescado beta (Boquerones, Anchoas, Trucha)». Es cierto que los deja pasar, y es CORRECTO: los
boquerones son pescado, no marisco. Medido chip a chip, la separación está completa — `Mariscos`
bloquea los ocho mariscos del catálogo y `Pescado` bloquea los cinco pescados beta. El defecto no
era la clasificación: era que uno de los dos chips no existía.

LO QUE NO SE HACE. No se añaden los catorce alérgenos del Reglamento UE 1169/2011. Apio, mostaza,
sulfitos y altramuces no los modela el motor, así que un chip sería un botón que no protege — la
misma mentira, con más botones. Primero la clase en el backstop, después el chip; ese orden es el
que este test fija.

La forma del test es deliberada: **no comprueba que el literal esté en el JSX**, comprueba que cada
chip del formulario BLOQUEA de verdad su alérgeno al pasar por el backstop. Un chip que existe y no
protege es exactamente el defecto que se está cerrando.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_QALLERGIES = _REPO / "frontend" / "src" / "components" / "assessment" / "questions" / "QAllergies.jsx"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def chips() -> list:
    """Los `val` de los chips, leídos del componente. Es el valor que consume el motor clínico —
    la etiqueta traducida no viaja al backend."""
    if not _QALLERGIES.is_file():
        pytest.skip("QAllergies.jsx no está en este árbol")
    src = _QALLERGIES.read_text(encoding="utf-8", errors="replace")
    return re.findall(r'\{\s*val:\s*"([^"]+)"', src)


def _bloquea(go, chip: str, ingrediente: str) -> bool:
    plato = {"name": "Prueba", "ingredients": [ingrediente], "preparation_steps": []}
    return bool(go.clinical_backstop_for_meal(plato, allergies=[chip]))


# ── Cada chip protege de verdad ─────────────────────────────────────────────────────────────────

_CHIP_Y_SU_ALIMENTO = [
    ("Lacteos", "200 ml de Leche entera"),
    ("Gluten", "100 g de Pan blanco familiar"),
    ("Huevo", "2 Huevos"),
    ("Mariscos", "150 g de Camarones"),
    ("Frutos Secos", "30 g de Almendras fileteadas"),
    ("Soya", "100 g de Soya texturizada"),
    ("Pescado", "150 g de Boquerones"),
    ("Mani", "30 g de Mantequilla de maní"),
    # [P2-I18N-CHIP-SESAMO-Y-LACTOSA-AUSENTES · 2026-08-23] los dos que quedaban con clase y sin chip.
    ("Sesamo", "30 g de Tahini"),
    ("Lactosa", "200 ml de Leche entera"),
]


@pytest.mark.parametrize("chip,ingrediente", _CHIP_Y_SU_ALIMENTO)
def test_cada_chip_bloquea_su_alergeno(go, chips, chip, ingrediente):
    """La prueba que importa: no que el chip EXISTA, sino que llegue al motor. Un botón que dice
    «soy alérgico» y no protege es peor que no tenerlo — le da al usuario una respuesta."""
    assert chip in chips, f"el chip {chip!r} no está en el formulario: {chips}"
    assert _bloquea(go, chip, ingrediente), (
        f"el chip {chip!r} NO bloquea {ingrediente!r}: existe en la interfaz y no protege"
    )


def test_el_pescado_tiene_chip(chips):
    """El marisco lo tenía y el pescado no, en un beta cuyo primer país trae boquerones, anchoas,
    merluza, bacalao y trucha en el catálogo."""
    assert "Pescado" in chips, f"sigue sin chip de pescado: {chips}"


def test_el_mani_tiene_chip_propio(chips, go):
    """El hueco grave. `Frutos Secos` NO cubre el maní —el cacahuete es una legumbre— así que el
    alérgico marcaba el único chip que le sonaba y se quedaba sin protección CREYENDO que la
    tenía."""
    assert "Mani" in chips, f"sigue sin chip de maní: {chips}"
    assert not _bloquea(go, "Frutos Secos", "50 g de Maní"), (
        "«Frutos Secos» empezó a bloquear el maní: si es deliberado, revisa el test de esta ola "
        "que exige la separación legumbre/fruto de cáscara antes de cambiar esto"
    )


# ── La separación marisco/pescado ya era correcta ───────────────────────────────────────────────

@pytest.mark.parametrize("pez", ["Boquerones", "Anchoas", "Trucha", "Merluza", "Bacalao"])
def test_el_chip_de_mariscos_deja_pasar_el_pescado_a_proposito(go, pez):
    """Refuta el diagnóstico de la auditoría, que llamaba defecto a esto. Los boquerones son
    pescado, no marisco: que `Mariscos` los deje pasar es la clasificación correcta. Lo que faltaba
    era el otro chip."""
    assert not _bloquea(go, "Mariscos", f"150 g de {pez}")
    assert _bloquea(go, "Pescado", f"150 g de {pez}"), f"{pez} escapa al chip de pescado"


@pytest.mark.parametrize("bicho", ["Camarones", "Gambas", "Almejas", "Pulpo", "Mejillones"])
def test_el_chip_de_pescado_deja_pasar_el_marisco_a_proposito(go, bicho):
    """La simétrica, para que nadie 'arregle' la asimetría fundiendo las dos clases."""
    assert _bloquea(go, "Mariscos", f"150 g de {bicho}")


# ── Lo que NO se añade ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("eu", ["apio", "mostaza", "sulfitos", "altramuz"])
def test_no_hay_chip_para_un_alergeno_que_el_motor_no_modela(go, chips, eu):
    """Cuatro de los catorce del Reglamento UE 1169/2011 no tienen clase en el backstop. Un chip
    sin clase es un botón que no protege: la misma mentira con más botones. El orden correcto es
    clase primero, chip después — y este test lo fija en esa dirección: si alguien añade el chip
    sin la clase, falla."""
    tiene_clase = any(eu in k for k in go._ALLERGEN_SYNONYMS)
    tiene_chip = any(eu in c.lower() for c in chips)
    assert not (tiene_chip and not tiene_clase), (
        f"hay chip de {eu!r} pero el motor no lo modela: el usuario lo marcaría y no le protegería"
    )
