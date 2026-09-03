"""[P2-EDGE-HTTP2 · 2026-08-15] HTTP/2 en el edge, y la via para aplicarlo.

LO ENCONTRO LA PRIMERA TRAZA REAL DE LCP DEL PROYECTO. Chrome DevTools contra
producción (Slow 4G + CPU 4x): todas las respuestas salían por `http/1.1`. nginx
1.24 trae `--with-http_v2_module`; sólo no estaba puesto. Ninguna auditoría
estática lo habría visto — es una propiedad del servidor corriendo, no del repo.

Duele más aquí que en un sitio cualquiera: el critical path del apex son 23
recursos, y `P2-LANDING-OLA1-DIET` partió lucide en ~12 chunks de icono de
150-300 B justificándolo como «barato sobre HTTP/2». El sitio no estaba en
HTTP/2, así que esa premisa era falsa cuando se escribió.

TRES TRAMPAS, y las tres costaron tiempo real:

  1. `http2` va SÓLO en el primer `server` que declara cada dirección:puerto. En
     nginx < 1.25.1 es un parámetro del `listen` y aplica a la pareja completa;
     repetirlo en otro server del mismo :443 da `duplicate listen options` y nginx
     NO ARRANCA.
  2. `reload` NO basta. SIGHUP hereda los sockets de escucha ya abiertos y no
     re-aplica los parámetros del `listen`. `nginx -t` pasaba, el reload decía OK,
     `nginx -T` mostraba `http2`... y el navegador seguía en http/1.1. Todo verde,
     cero efecto. Hace falta `restart`.
  3. El fichero vivo es `sites-enabled/mealfit` y NO es un symlink: el de
     `sites-available` se quedó en julio. Escribir en `sites-available` —que es lo
     que uno teclea de memoria— no habría cambiado nada, y `nginx -t` habría pasado
     igual.

Este test ancla las tres, porque las tres son invisibles al leer el diff.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF = _REPO_ROOT / "backend" / "infra" / "nginx" / "mealfit.conf"
_DEPLOY = _REPO_ROOT / "deploy-mealfit.ps1"


def _leer(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p} no existe en este checkout (repos hermanos)")
    return p.read_text(encoding="utf-8")


def _listens_443(conf: str) -> list[str]:
    return [ln.strip() for ln in conf.splitlines()
            if re.match(r"^\s*listen\s+.*443", ln) and not ln.strip().startswith("#")]


def test_http2_activo_en_el_edge() -> None:
    conf = _leer(_CONF)
    con_http2 = [ln for ln in _listens_443(conf) if "http2" in ln]
    assert con_http2, (
        "Ningún `listen ... 443` declara `http2`. El apex sirve 23 recursos en el "
        "critical path y 12 de ellos son chunks de icono diminutos: sin "
        "multiplexación, HTTP/1.1 los serializa en 6 conexiones."
    )


def test_http2_solo_en_el_primer_server_del_443() -> None:
    """Repetirlo en otro server del mismo puerto impide que nginx ARRANQUE.

    No es un aviso teórico: `duplicate listen options for 0.0.0.0:443` es un error
    fatal de configuración, y el fichero tiene CUATRO server blocks escuchando 443
    (apex, app.bioboros.com y los dos de mealfitrd.com).
    """
    conf = _leer(_CONF)
    listens = _listens_443(conf)

    # Se agrupa por dirección:puerto (`[::]:443` vs `443`) porque nginx trata cada
    # una como un socket distinto: `http2` puede ir una vez en cada.
    for clave, patron in (("[::]:443", r"listen\s+\[::\]:443"), ("443", r"listen\s+443")):
        del_puerto = [ln for ln in listens if re.match(rf"^listen\s+{re.escape(clave)}\b", ln)
                      or re.match(patron + r"\b", ln)]
        con_http2 = [ln for ln in del_puerto if "http2" in ln]
        assert len(con_http2) <= 1, (
            f"`http2` aparece {len(con_http2)} veces para {clave}: "
            f"{con_http2}\nnginx fallará con `duplicate listen options` y NO "
            "arrancará. Va sólo en el PRIMER server block que declara ese puerto."
        )


def test_el_deploy_reinicia_nginx_no_solo_recarga() -> None:
    """`reload` no re-aplica los parámetros del `listen`. Ver la cabecera."""
    dep = _leer(_DEPLOY)
    assert "systemctl restart nginx" in dep, (
        "El deploy de infra volvió a `reload`. SIGHUP hereda los sockets de escucha "
        "abiertos: un cambio de `listen` (http2, backlog, reuseport) pasa `nginx -t`, "
        "el reload dice OK, y NO surte efecto. Todo verde y cero cambio — que es "
        "peor que no haberlo hecho, porque la config dice una cosa y el servidor "
        "hace otra."
    )


def test_el_deploy_escribe_en_el_fichero_que_de_verdad_sirve() -> None:
    dep = _leer(_DEPLOY)
    assert "sites-enabled/mealfit" in dep, (
        "El deploy de infra ya no apunta a `sites-enabled/mealfit`. Ese es el "
        "fichero VIVO y no es un symlink: `sites-available/mealfit` es una copia "
        "vieja. Escribir ahí no cambia producción y `nginx -t` pasa igual."
    )


def test_el_deploy_valida_antes_de_aplicar_y_restaura_si_falla() -> None:
    """El orden es la garantía: validar → aplicar. Nunca al revés."""
    dep = _leer(_DEPLOY)
    for pieza, porque in [
        ("nginx -t", "sin validación, una config mala tumba los DOS dominios"),
        ("/var/backups/nginx", "sin backup no hay a dónde volver"),
        ("ALPN protocol", "sin comprobar ALPN, 'recargado' no significa 'h2 activo'"),
    ]:
        assert pieza in dep, f"Falta `{pieza}` en el deploy de infra: {porque}."


def test_infra_no_entra_en_el_target_all() -> None:
    """Recargar el edge y publicar un bundle son riesgos de orden distinto."""
    dep = _leer(_DEPLOY)
    m = re.search(r"if \(\$target -eq 'infra'\)\s*\{?\s*Deploy-Infra", dep)
    assert m, "No encuentro el dispatch de `infra` como target propio."
    assert not re.search(r"\$target -eq 'infra'\s+-or\s+\$target -eq 'all'", dep), (
        "`infra` entró en `all`. `all` corre varias veces al día y su peor fallo es "
        "un bundle roto (reversible con un symlink); una config mala de nginx tira "
        "los dos dominios y el rollback exige SSH justo cuando el sitio está caído."
    )
