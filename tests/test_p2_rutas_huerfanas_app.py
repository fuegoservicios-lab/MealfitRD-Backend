"""[P2-RUTAS-HUERFANAS-APP · 2026-08-23] Las rutas de marketing del apex NO pueden
quedarse sirviendo el SPA en `app.bioboros.com`.

P1-LEGAL-UNA-SOLA-COPIA cerró 15 de las 19 con un `location` regex que redirige al apex
preservando `$request_uri`. Pero esa lista SALE DEL SITEMAP DEL APEX, así que solo cubre
las rutas que existen allí con el MISMO nombre. Las tres que cambian de nombre —o que no
existen en el apex— se quedaron fuera y devolvían **200 con el React congelado**:

    /funciones   → en el apex es `como-funciona`
    /precision   → en el apex es `research`
    /cookies     → el apex lo resuelve con 301 a `/privacy#cookies` (P3-COOKIES-MERGE)

Medido con curl contra producción el 2026-08-23, no inferido: 15 daban 301 y estas tres
200. La peor era `/cookies`, que servía el `<title>` del LOGIN en una dirección legal:
para una persona el rodeo acaba bien (React la manda a /privacy), pero un buscador o la
vista previa de un enlace leen el HTML servido, no el resultado de ejecutar JavaScript.

Este guard es parser-based sobre `infra/nginx/mealfit.conf` porque el fichero es la
FUENTE de lo que se despliega (`deploy-mealfit.ps1 infra` lo sube tal cual y corre
`nginx -t`). No sustituye a medir producción; ancla la decisión para que un futuro
`location` reordenado o una lista "simplificada" no reabra el agujero en silencio.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CONF = Path(__file__).resolve().parents[1] / "infra" / "nginx" / "mealfit.conf"

#: ruta huérfana → destino canónico en el apex
_DESTINOS = {
    "funciones": "https://bioboros.com/como-funciona",
    "precision": "https://bioboros.com/research",
    "cookies": "https://bioboros.com/privacy#cookies",
}


def _conf() -> str:
    return _CONF.read_text(encoding="utf-8", errors="replace")


def _bloque_app() -> str:
    """El `server` de app.bioboros.com. Las reglas tienen que vivir AHÍ: en el server
    del apex no harían nada, y en el de otro host redirigirían lo que no toca."""
    src = _conf()
    i = src.index("server_name app.bioboros.com;")
    # hasta el siguiente server_name (o el final): suficiente para acotar el bloque
    j = src.find("server_name", i + 10)
    return src[i: j if j > 0 else len(src)]


@pytest.mark.parametrize("ruta,destino", sorted(_DESTINOS.items()))
def test_la_ruta_huerfana_redirige_al_apex(ruta: str, destino: str):
    bloque = _bloque_app()
    m = re.search(
        r"location\s+~\s+\^/" + ruta + r"/\?\$\s*\{\s*return\s+301\s+\"?([^\";]+)\"?\s*;",
        bloque,
    )
    assert m, (
        f"/{ruta} no tiene regla de redirección en el server de app.bioboros.com. "
        f"Sin ella el catch-all del SPA devuelve 200 con el React congelado."
    )
    assert m.group(1) == destino, f"/{ruta} redirige a {m.group(1)!r}, se esperaba {destino!r}"


def test_el_destino_con_fragmento_va_entre_comillas():
    """En nginx `#` abre un comentario: sin comillas, `/privacy#cookies` se convierte en
    `/privacy` y el fragmento se pierde EN SILENCIO — la regla sigue pareciendo correcta.
    Es la misma familia que un comentario que satisface a su propio guard."""
    bloque = _bloque_app()
    m = re.search(r"location\s+~\s+\^/cookies/\?\$\s*\{\s*return\s+301\s+(\S+)\s*;", bloque)
    assert m, "falta la regla de /cookies"
    assert m.group(1).startswith('"') and m.group(1).endswith('"'), (
        f"el destino de /cookies debe ir ENTRE COMILLAS (lleva `#`), y está como {m.group(1)!r}"
    )


def test_no_se_pisan_las_rutas_de_la_aplicacion():
    """El agujero se cierra por ARRIBA, no por abajo: estas reglas no pueden capturar
    ninguna ruta que la app necesite servir."""
    bloque = _bloque_app()
    patrones = re.findall(r"location\s+~\s+(\^/\S+?/\?\$)", bloque)
    vivas = ["/login", "/dashboard", "/assessment", "/plan", "/configuracion", "/history", "/register"]
    for pat in patrones:
        rx = re.compile(pat)
        for ruta in vivas:
            assert not rx.match(ruta), f"la regla {pat} captura {ruta}, que la app tiene que servir"
