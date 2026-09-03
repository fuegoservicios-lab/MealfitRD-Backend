"""[P2-LANDING-AREASERVED-APEX · 2026-08-23] G70: el landing público declaraba UN país en sus
datos estructurados y la app declaraba SEIS, para la MISMA entidad.

MEDIDO con curl el 2026-08-23:

    https://bioboros.com/       → "areaServed":{"@type":"Country","name":"República Dominicana"}
    https://app.bioboros.com/   → array de 6 países
    Los dos bloques declaran     "url":"https://bioboros.com/"   ← la misma Organization

Gana por canonicidad y por volumen la que dice un país. Un usuario en España que busca «plan de
comidas IA España» no encontraba a Bioboros como servicio de su país — y eso no es un detalle de
SEO: es la contradicción entre lo que el negocio vende y lo que declara.

DOS SITIOS, no uno: `jsonld.py` genera las páginas interiores, pero la portada
(`bioboros/index.html`) está escrita A MANO y no pasa por el generador. Corregir sólo el script
habría dejado la página más visitada diciendo un país — la misma trampa que ya mordió con los
enlaces de redes.

DÓNDE VIVE EL SSOT: los CÓDIGOS de país son de `constants.COUNTRY_PROFILES`, en este repo. El
apex es otro repositorio y no puede importarlo, así que allí viven los NOMBRES públicos y la
correspondencia se verifica desde aquí: si entra un séptimo país, este test se pone rojo.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_APEX = _BACKEND.parent.parent / "bioboros-cinematic"
_APP_INDEX = _BACKEND.parent / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(
    not _APEX.is_dir(),
    reason="el repo del apex no está clonado junto a este (es un repo hermano)",
)


def _leer(p: Path) -> str:
    return io.open(p, encoding="utf-8").read()


def _nombres_esperados() -> set[str]:
    import constants
    return {p["name_es"] for p in constants.COUNTRY_PROFILES.values()}


def test_el_generador_del_apex_declara_todos_los_paises():
    src = _leer(_APEX / "jsonld.py")
    m = re.search(r"PAISES_SERVIDOS = \(([^)]*)\)", src, re.S)
    assert m, "desapareció la constante PAISES_SERVIDOS del apex"
    declarados = set(re.findall(r'"([^"]+)"', m.group(1)))
    esperados = _nombres_esperados()
    assert declarados == esperados, (
        f"el apex declara {sorted(declarados)} y COUNTRY_PROFILES dice {sorted(esperados)}"
    )


def test_el_generador_emite_una_LISTA_y_no_un_pais_suelto():
    """Un dict `{"@type":"Country"}` es exactamente el defecto: schema.org lo lee como uno."""
    src = _leer(_APEX / "jsonld.py")
    codigo = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    assert '"areaServed": [{"@type": "Country", "name": n} for n in PAISES_SERVIDOS]' in codigo, (
        "el apex volvió a emitir un solo país en areaServed"
    )


def test_la_portada_escrita_a_mano_tambien():
    """La portada NO pasa por el generador: es HTML a mano. Corregir sólo el script deja la
    página más visitada diciendo un país."""
    html = _leer(_APEX / "bioboros" / "index.html")
    i = html.find('"areaServed"')
    assert i > 0, "la portada perdió el bloque areaServed"
    bloque = html[i:i + 500]
    assert bloque.lstrip('"areaServed":').lstrip().startswith("["), (
        "la portada declara areaServed como objeto único, no como lista"
    )
    for nombre in _nombres_esperados():
        assert nombre in bloque, f"la portada no declara «{nombre}»"


def test_la_app_y_el_apex_dicen_lo_MISMO():
    """Son la misma Organization con la misma `url`: si difieren, una de las dos miente y el
    buscador elige por su cuenta cuál."""
    html_app = _leer(_APP_INDEX)
    i = html_app.find('"areaServed"')
    assert i > 0
    en_app = set(re.findall(r'"name":\s*"([^"]+)"', html_app[i:i + 700]))
    en_apex = set(re.findall(r'"([^"]+)"', re.search(
        r"PAISES_SERVIDOS = \(([^)]*)\)", _leer(_APEX / "jsonld.py"), re.S).group(1)))
    faltan = en_apex - en_app
    sobran = en_app - en_apex
    assert not faltan and not sobran, (
        f"app y apex discrepan — sólo en el apex: {sorted(faltan)}; sólo en la app: {sorted(sobran)}"
    )
