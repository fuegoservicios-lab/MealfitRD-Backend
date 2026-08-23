"""[P2-DEV-ESPEJO-APEX · 2026-08-23] El servidor de desarrollo y nginx tienen que
redirigir LAS MISMAS rutas de marketing.

El dueño abrió `localhost:5173/about` tres veces creyendo que el landing local estaba
desactualizado. No lo estaba: el landing público lo genera OTRO proyecto
(`bioboros-cinematic`, HTML estático) y lo que servía el dev era la ruta React
equivalente, congelada en el diseño anterior. En producción nginx ya la redirigía; en
desarrollo no. *Un entorno que se comporta distinto del de verdad en una ruta concreta
no es más cómodo: cuesta una vuelta cada vez que alguien la pisa.*

Este guard ata las dos listas. Si mañana se añade una página al apex y sólo se mete en
nginx, el dev vuelve a mentir — y al revés. Parser-based sobre las dos FUENTES reales:
`backend/infra/nginx/mealfit.conf` y `frontend/vite.config.js`.
"""

from __future__ import annotations

import re
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
_NGINX = _RAIZ / "backend" / "infra" / "nginx" / "mealfit.conf"
_VITE = _RAIZ / "frontend" / "vite.config.js"


def _bloque_app_nginx() -> str:
    src = _NGINX.read_text(encoding="utf-8", errors="replace")
    i = src.index("server_name app.bioboros.com;")
    j = src.find("server_name", i + 10)
    return src[i: j if j > 0 else len(src)]


def _rutas_nginx() -> set[str]:
    """Las de passthrough (la lista larga) + las tres que cambian de nombre."""
    bloque = _bloque_app_nginx()
    m = re.search(r"location\s+~\s+\^/\(([^)]+)\)\(/\.\*\)\?\$", bloque)
    assert m, "no se encontró la lista de passthrough en el server de app.bioboros.com"
    rutas = set(m.group(1).split("|"))
    rutas |= set(re.findall(r"location\s+~\s+\^/([a-z-]+)/\?\$", bloque))
    return rutas


def _rutas_vite() -> set[str]:
    src = _VITE.read_text(encoding="utf-8", errors="replace")
    i = src.index("espejo-apex-en-desarrollo")
    bloque = src[i: i + 2500]
    paso = re.search(r"const PASO = new Set\(\[(.*?)\]\)", bloque, re.S)
    assert paso, "no se encontró PASO en el plugin del espejo"
    rutas = set(re.findall(r"'([a-z-]+)'", paso.group(1)))
    renombra = re.search(r"const RENOMBRA = \{(.*?)\}", bloque, re.S)
    assert renombra, "no se encontró RENOMBRA en el plugin del espejo"
    rutas |= set(re.findall(r"^\s*([a-z]+):", renombra.group(1), re.M))
    return rutas


def test_las_dos_listas_cubren_las_mismas_rutas():
    n, v = _rutas_nginx(), _rutas_vite()
    assert n == v, (
        "nginx y el servidor de desarrollo redirigen conjuntos DISTINTOS de rutas.\n"
        f"  solo en nginx: {sorted(n - v) or '—'}\n"
        f"  solo en vite:  {sorted(v - n) or '—'}\n"
        "Las dos listas describen la MISMA decisión (el landing vive en el apex); "
        "separarlas devuelve el entorno de desarrollo que miente."
    )


def test_las_tres_renombradas_apuntan_al_mismo_destino():
    """Las que cambian de nombre son las fáciles de desincronizar: el destino no se
    deduce de la ruta."""
    bloque = _bloque_app_nginx()
    src = _VITE.read_text(encoding="utf-8", errors="replace")
    for ruta, destino in (("funciones", "/como-funciona"), ("precision", "/research"), ("cookies", "/privacy#cookies")):
        m = re.search(
            r"location\s+~\s+\^/" + ruta + r"/\?\$\s*\{\s*return\s+301\s+\"?https://bioboros\.com([^\";]+)\"?\s*;",
            bloque,
        )
        assert m, f"nginx no redirige /{ruta}"
        assert m.group(1) == destino, f"nginx manda /{ruta} a {m.group(1)!r}, se esperaba {destino!r}"
        assert re.search(ruta + r":\s*'" + re.escape(destino) + r"'", src), (
            f"el plugin del espejo no manda /{ruta} a {destino!r}"
        )


def test_el_plugin_no_toca_el_build():
    """`apply: 'serve'`: sin eso, el build de producción heredaría un middleware que sólo
    tiene sentido en el servidor de desarrollo."""
    src = _VITE.read_text(encoding="utf-8", errors="replace")
    i = src.index("espejo-apex-en-desarrollo")
    assert re.search(r"apply:\s*'serve'", src[i: i + 400]), "el espejo debe llevar apply: 'serve'"
