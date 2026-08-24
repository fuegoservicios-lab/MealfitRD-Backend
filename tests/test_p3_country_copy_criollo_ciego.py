"""[P3-COUNTRY-COPY-CRIOLLO-CIEGO · 2026-08-23] Dos textos criollos del Dashboard salían a
pantalla sin ninguna condición de país.

  1. «Afinando sabores criollos…» — una de las OCHO frases de relleno de la cola de paciencia
     del overlay de cocinado (`_cookingStagesTail`). No sale al pulsar «Cambiar plato» sin
     más: vive en la cola, así que aparece en las esperas largas. Un español, un mexicano o un
     estadounidense esperaba SU plan leyendo que le estaban afinando sabores criollos.

  2. «Algunos platos compuestos (ej. sancocho, mangú)…» — el aviso de `composite_dish_unresolved`
     ilustraba el problema del usuario con dos platos que fuera de RD no ha comido nunca. El
     aviso funciona sin ellos: lo accionable es «usa Cambiar Plato», no el nombre del guiso.

Las otras dos apariciones medidas de léxico dominicano en pantalla SÍ aciertan y quedan fuera
de este guard a propósito: `QCountry.jsx` describe el catálogo dominicano DENTRO de la tarjeta
de RD, y `ScanMealModal.jsx` avisa —gateado— de que el escáner está calibrado con cocina
dominicana.

EL ALCANCE ES `Dashboard.jsx` Y ESO ES LA PROPIEDAD, no una comodidad: el Dashboard es la
pantalla que ven los SEIS países sin ninguna bifurcación de país en su copy. Un léxico que
sólo es cierto en uno de los seis no puede vivir ahí sin condición. (El catálogo de 60 platos
del diario —«Plato criollo · ración»— es otro gap con otro diagnóstico: los platos son
dominicanos DE VERDAD, así que el problema es el catálogo, no la etiqueta.)

Los comentarios se ELIMINAN antes de medir. Los dos arreglos llevan encima una nota que CITA
el texto viejo entre comillas angulares; sin barrido, el guard se aprobaría a sí mismo con la
explicación del defecto que persigue.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_DASHBOARD = _REPO / "frontend" / "src" / "pages" / "Dashboard.jsx"
_LOCALES = _REPO / "frontend" / "src" / "i18n" / "locales"
_IDIOMAS = ("en-US", "pt-BR", "fr-FR", "it-IT")

# Léxico dominicano que no puede aparecer en copy sin condición de país.
_LEXICO_DO = ("criollo", "criolla", "sancocho", "mangú", "mangu")

_NUEVAS = (
    "Afinando los sabores…",
    "Algunos platos compuestos no se pudieron desglosar en ingredientes con precisión, "
    "así que sus macros y su lista de compras son aproximados. Usa Cambiar Plato si "
    "necesitas más exactitud.",
)
_VIEJAS = (
    "Afinando sabores criollos…",
    "Algunos platos compuestos (ej. sancocho, mangú) no se pudieron desglosar en ingredientes "
    "con precisión, así que sus macros y su lista de compras son aproximados. Usa Cambiar "
    "Plato si necesitas más exactitud.",
)


def _sin_comentarios(src: str) -> str:
    out = []
    i, n = 0, len(src)
    comilla = None
    while i < n:
        c = src[i]
        if comilla:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if c == comilla:
                comilla = None
            i += 1
            continue
        if c in "\"'`":
            comilla = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


_RX_T = re.compile(r"\bt\(\s*(['\"])((?:\.|(?!\1).)*)\1", re.S)


def _literales_de_t(src: str):
    return [m.group(2) for m in _RX_T.finditer(_sin_comentarios(src))]


def test_el_barrido_de_comentarios_no_deja_pasar_la_cita_del_defecto():
    """Control del propio guard: las notas de los dos arreglos citan el texto viejo, así que
    si el barrido fallara los dos tests de abajo pasarían por la explicación."""
    crudo = _DASHBOARD.read_text(encoding="utf-8", errors="replace")
    limpio = _sin_comentarios(crudo)
    assert "sabores criollos" in crudo, (
        "la nota que documenta el arreglo desapareció: sin ella este control no mide nada"
    )
    assert "sabores criollos" not in limpio, "el barrido de comentarios no funcionó"


@pytest.mark.parametrize("palabra", _LEXICO_DO)
def test_ningun_copy_del_dashboard_lleva_lexico_dominicano(palabra):
    """La propiedad: el Dashboard lo ven los SEIS países con el MISMO copy."""
    culpables = [s for s in _literales_de_t(_DASHBOARD.read_text(encoding="utf-8", errors="replace"))
                 if palabra in s.lower()]
    assert not culpables, (
        f"copy del Dashboard con léxico dominicano ({palabra!r}) y sin condición de país: "
        f"{culpables}"
    )


@pytest.mark.parametrize("nueva", _NUEVAS)
def test_el_copy_neutro_esta_en_el_codigo(nueva):
    """Que no esté el viejo no basta: alguien podría haberlo borrado y dejar el hueco."""
    assert nueva in _sin_comentarios(
        _DASHBOARD.read_text(encoding="utf-8", errors="replace")), nueva


@pytest.mark.parametrize("idioma", _IDIOMAS)
def test_el_copy_neutro_esta_traducido_en_los_cuatro_catalogos(idioma):
    """La clave ES el texto español (P1-I18N-DASHBOARD): cambiar el copy sin mover el catálogo
    huerfana la traducción EN SILENCIO y el italiano lee español."""
    cat = json.loads((_LOCALES / f"{idioma}.json").read_text(encoding="utf-8"))
    for nueva in _NUEVAS:
        assert cat.get(nueva), f"{idioma}: falta la traducción de {nueva[:48]!r}"
    for vieja in _VIEJAS:
        assert vieja not in cat, (
            f"{idioma}: quedó huérfana la traducción del copy viejo {vieja[:48]!r}"
        )
