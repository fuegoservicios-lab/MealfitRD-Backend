# -*- coding: utf-8 -*-
"""[P2-GUINEITO-SINGULAR · 2026-09-07] El diminutivo estaba curado sólo en plural.

`Guineo verde` llevaba tres alias y los tres eran PLURAL — `guineítos verdes`, `guineitos
verdes`, `guineos verdes`. Quien curó el set escribió la forma que tenía delante y no la otra, y
«medio guineíto verde» (10 comidas vivas) no resolvía a ningún alimento. **Un nombre que no
resuelve es invisible para toda la capa culinaria**: ni V3 lo reclama, ni V7e puede comparar
cantidades sobre él.

Medido antes de tocar el catálogo, simulando el alias en memoria sobre las 1.194 comidas con las
once capas: **V7e 159 → 162 (+3), el resto sin cambio, cero disparos perdidos**. Los tres son la
misma forma —la lista compra medio guineíto y el paso manda aplastar dos— y uno es el mangú que
el dueño marcó a mano en el juicio ciego. El alias no inventa capturas: destapa las que ya
existían detrás de un nombre que el motor no sabía leer.

Ese `+3` es lo que separa este cambio del que se revirtió el mismo día: la lectura de «medio»
escrito con palabra cubría el 6,9 % de la flota y dio **+0 disparos**. La prevalencia de un
patrón no es la prevalencia de un defecto, y aquí sí hay defecto detrás.
"""
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_BASELINE = _BACKEND / "scripts" / "data" / "do_corpus_retarget_baseline_2026_08_18.json"
_SEED = _BACKEND / "scripts" / "seed_guineito_singular_2026_09_07.py"


@pytest.fixture(scope="module")
def mapping():
    """El retarget DO commiteado: dice a qué alimento resuelve cada forma del corpus vivo."""
    return json.loads(_BASELINE.read_text(encoding="utf-8"))["mapping"]


@pytest.mark.parametrize("forma", ["guineíto verde", "guineítos verdes", "guineos verdes"])
def test_el_diminutivo_resuelve_en_singular_y_en_plural(forma, mapping):
    """El singular es el que faltaba; los plurales ya estaban y no pueden perderse."""
    assert forma in mapping, f"{forma!r} no resuelve a ningún alimento del catálogo"


def test_resuelve_al_guineo_VERDE_y_no_al_maduro(mapping):
    """`Guineo` (maduro) y `Guineo verde` son filas distintas con macros distintas.

    Colapsar el diminutivo al guineo a secas sería peor que no tenerlo: el motor razonaría sobre
    el alimento equivocado en silencio, que es exactamente lo que hacía el yogurt griego cuando
    resolvía al normal."""
    assert mapping["guineíto verde"] == "Guineo verde"


def test_el_alias_es_frase_completa_nunca_el_diminutivo_suelto():
    """«guineito» a secas colisionaría con `Guineo` en cualquier texto que no especifique.

    Es la clase de error —el nombre corto que se traga al largo— que este repo lleva 19 veces
    documentada, así que el alias entra en dos palabras a propósito."""
    src = _SEED.read_text(encoding="utf-8")
    inicio = src.index("NUEVOS = [")
    nuevos = src[inicio:src.index("]", inicio)]
    for suelto in ('"guineito"', "'guineito'", '"guineíto"', "'guineíto'"):
        assert suelto not in nuevos, "el diminutivo suelto colisiona con `Guineo`"
    assert "guineito verde" in nuevos and "guineíto verde" in nuevos


def test_el_seed_rechaza_un_alias_que_ya_reclama_otro_alimento():
    """Ancla textual: el script comprueba colisiones ANTES de escribir.

    Repartir una identidad ambigua es peor que no tenerla — misma regla que aplica
    `build_culinary_index` al descartar `mariscos` y `nueces`."""
    src = _SEED.read_text(encoding="utf-8")
    assert "COLISIÓN" in src, "el seed dejó de comprobar colisiones antes de escribir"
    assert "--aplicar" in src, "el seed debe simular por defecto y escribir sólo bajo bandera"
