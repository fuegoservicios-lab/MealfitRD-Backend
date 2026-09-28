"""[P1-LANDING-CIFRAS · 2026-09-28] `/llms.txt` del apex lo sirve el LANDING, no React.

POR QUÉ. El landing (`bioboros-cinematic`) genera ya su propio `llms.txt` desde las
mismas páginas que su sitemap. Pero nginx sólo entrega un fichero del landing en la
raíz si tiene su `location` exacto hacia `/var/www/bioboros-v2/current` —así viven
`/sitemap.xml`, `/robots.txt` y `/release.txt`—. Sin él, `/llms.txt` cae en el
`location /` del apex y lo sirve el build de React: un fichero viejo, generado por
`frontend/scripts/build-sitemap.mjs`, que lista `/funciones` y `/precision`, dos
rutas que responden 301 desde el 18 de agosto.

Medido el 28-sep con `curl https://bioboros.com/llms.txt`: 200 con el de React
(«Este fichero se GENERA desde `scripts/build-sitemap.mjs`»).

ORDEN DE DESPLIEGUE, y es lo que este test no puede comprobar: esta `location` tiene
que estar APLICADA en el VPS antes de desplegar la rama del landing que trae el
`llms.txt` nuevo. `scripts/verify-live.sh` del landing compara `/llms.txt` contra su
manifiesto y, si sigue respondiendo React, el despliegue se revierte solo.

Parser-based sobre la fuente real (`infra/nginx/mealfit.conf`, el SSOT que
`infra/verificar-nginx.sh` compara contra `nginx -T`). Tooltip-anchor en el conf:
`LLMS-TXT-DEL-LANDING`.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONF = Path(__file__).resolve().parents[1] / "infra" / "nginx" / "mealfit.conf"
_RAIZ_LANDING = "/var/www/bioboros-v2/current"


def _bloque_apex() -> str:
    """El primer `server` del apex (el de 443 con `root /var/www/bioboros/current`)."""
    src = _CONF.read_text(encoding="utf-8", errors="replace")
    i = src.index("server_name bioboros.com www.bioboros.com _;")
    j = src.find("server_name", i + 10)
    return src[i: j if j > 0 else len(src)]


def _location_exacta(bloque: str, ruta: str) -> str | None:
    m = re.search(r"location\s+=\s+" + re.escape(ruta) + r"\s*\{(.*?)\}", bloque, re.S)
    return m.group(1) if m else None


def test_llms_txt_tiene_location_exacta_hacia_el_landing() -> None:
    cuerpo = _location_exacta(_bloque_apex(), "/llms.txt")
    assert cuerpo is not None, (
        "El apex no tiene `location = /llms.txt`: cae en `location /` y lo sirve React "
        "(el fichero viejo con /funciones y /precision), aunque el landing ya genere el suyo."
    )
    assert re.search(r"root\s+" + re.escape(_RAIZ_LANDING) + r"\s*;", cuerpo), (
        f"`/llms.txt` tiene que salir de la raíz del landing ({_RAIZ_LANDING})."
    )
    assert re.search(r"try_files\s+/llms\.txt\s+=404\s*;", cuerpo), (
        "Sin `try_files /llms.txt =404` un fichero ausente vuelve a caer al SPA con 200: "
        "el mismo soft-404 que tuvo /release.txt."
    )


def test_llms_txt_se_sirve_como_texto_y_sin_cache_larga() -> None:
    cuerpo = _location_exacta(_bloque_apex(), "/llms.txt") or ""
    assert re.search(r"default_type\s+text/plain\s*;", cuerpo), (
        "llms.txt es texto: sin `default_type text/plain` nginx no sabe qué tipo darle."
    )
    assert 'add_header Cache-Control "no-cache"' in cuerpo, (
        "Como /sitemap.xml y /release.txt: describe el despliegue vigente y no lleva huella."
    )


def test_los_ficheros_raiz_del_landing_tienen_su_location() -> None:
    """Los cuatro ficheros de la raíz que el landing publica, en el mismo sitio."""
    bloque = _bloque_apex()
    for ruta in ("/sitemap.xml", "/robots.txt", "/release.txt", "/llms.txt"):
        cuerpo = _location_exacta(bloque, ruta)
        assert cuerpo is not None and _RAIZ_LANDING in cuerpo, (
            f"{ruta} no sale del landing: caería en React."
        )


def test_tooltip_anchor_presente() -> None:
    assert "LLMS-TXT-DEL-LANDING" in _CONF.read_text(encoding="utf-8", errors="replace")
