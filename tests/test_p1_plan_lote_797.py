"""[P1-PLAN-LOTE-797 · 2026-09-28] certbot renovaba los certificados y nginx seguía sirviendo el viejo.

El 28-sep, tres dominios (bioboros.com, app.bioboros.com, www) servían un certificado que caducaba en días aunque certbot
ya tenía el renovado en disco: nada recargaba nginx tras la renovación. El arreglo en el VPS es un deploy-hook
(`/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh`); este test ancla su fuente versionada en `infra/` para que
una reconstrucción del VPS no lo pierda (P2-INFRA-EDGE-SSOT: lo que vive solo en el disco del VPS no se reproduce).
"""
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_HOOK = _BACKEND / "infra" / "letsencrypt" / "renewal-hooks" / "deploy" / "reload-nginx.sh"


def test_el_hook_de_renovacion_recarga_nginx():
    assert _HOOK.is_file(), _HOOK
    raw = _HOOK.read_bytes()
    assert raw.startswith(b"#!/bin/sh\n"), "sin shebang, certbot no lo ejecuta"
    assert b"\r" not in raw, "un CRLF rompe /bin/sh en el VPS"
    lineas = [l.strip() for l in raw.decode("utf-8").splitlines() if l.strip() and not l.startswith("#")]
    assert lineas == ["systemctl reload nginx"], lineas


def test_el_readme_de_infra_dice_donde_va():
    readme = (_BACKEND / "infra" / "README.md").read_text(encoding="utf-8")
    assert "`letsencrypt/renewal-hooks/deploy/reload-nginx.sh`" in readme
    assert "`/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh`" in readme
