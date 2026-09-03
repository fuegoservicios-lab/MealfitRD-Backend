"""[P2-SUPERMARKET-TOKEN-SPLIT · 2026-08-14] El secreto del catálogo deja de ser
el secreto maestro.

EL DEFECTO. `/supermercado` es una página PÚBLICA del apex con un editor dentro:
el admin teclea un token en un formulario y edita precios. Ese token era
`CRON_SECRET` — el mismo que abre TODOS los `/admin/*`, incluido
`POST /api/system/admin/account/purge-data`, que ejecuta `delete_account_data`
sobre un `user_id` arbitrario del body y cuyo propio docstring dice «purga TODA
la data user-scoped (33 tablas)».

POR QUÉ ES P2 Y NO P1, que importa para no exagerar el arreglo: no hay vector
explotable hoy. Cero sinks XSS en la superficie pública (los dos
`dangerouslySetInnerHTML` reciben path SVG hardcodeada y `LazyMarkdown` siempre
lleva `rehypeSanitize`), la comparación es `hmac.compare_digest` (sin canal de
timing) y a 60 intentos/min un secreto de alta entropía es inalcanzable.

Lo que se cierra es RADIO DE DAÑO y ROTACIÓN. Hoy el token del catálogo y el de
los crons son la misma cadena: rotar uno rota el otro, y ese secreto se teclea y
se guarda en el navegador de un origen que carga posthog-js y @sentry/react en
runtime. Probabilidad baja × impacto catastrófico.

⚠️ EL PRECIO, que hay que aceptar explícitamente: pasan a ser DOS secretos que
rotar. Esto no es gratis y por eso queda escrito aquí y en la fila de CLAUDE.md.

⚠️ Y LA COMPATIBILIDAD NO ES OPCIONAL: durante el despliegue, el backend nuevo
convive con navegadores que ya tienen el `CRON_SECRET` guardado en
sessionStorage. Si el gate nuevo sólo aceptara el token nuevo, el editor se
rompería para el operador en mitad del rollout. Se aceptan AMBOS mientras
`SUPERMARKET_ADMIN_TOKEN` no esté configurado, y el día que lo esté, el viejo
deja de valer.

Tooltip-anchor: P2-SUPERMARKET-TOKEN-SPLIT
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

_ROUTER = Path(__file__).resolve().parent.parent / "routers" / "supermarket.py"


@pytest.fixture()
def sm():
    import routers.supermarket as modulo
    return modulo


@pytest.fixture()
def entorno(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "el-secreto-maestro-de-los-crons")
    monkeypatch.delenv("SUPERMARKET_ADMIN_TOKEN", raising=False)
    return monkeypatch


def _bearer(t):
    return f"Bearer {t}"


# ---------------------------------------------------------------------------
# 1. El gate propio existe y NO es el maestro
# ---------------------------------------------------------------------------

def test_el_catalogo_tiene_su_propio_gate(sm):
    assert hasattr(sm, "_verify_supermarket_token"), (
        "[P2-SUPERMARKET-TOKEN-SPLIT] Falta `_verify_supermarket_token`. Sin un "
        "gate propio, el editor de una página pública sigue pidiendo el secreto "
        "que abre `purge-data` sobre 33 tablas."
    )


def test_las_mutaciones_ya_no_llaman_al_gate_maestro():
    """El barrido estructural: 3 mutaciones admin + el `include_inactive` del listado."""
    src = _ROUTER.read_text(encoding="utf-8")
    # Sin comentarios: la explicación del cambio nombra el gate viejo.
    import re
    codigo = re.sub(r"#.*$", "", src, flags=re.MULTILINE)
    codigo = re.sub(r'""".*?"""', "", codigo, flags=re.DOTALL)
    assert "_verify_admin_token(" not in codigo, (
        "[P2-SUPERMARKET-TOKEN-SPLIT] `supermarket.py` vuelve a invocar el gate "
        "maestro. Cada llamada es una superficie pública pidiendo el `CRON_SECRET`."
    )


# ---------------------------------------------------------------------------
# 2. Comportamiento del gate
# ---------------------------------------------------------------------------

def test_acepta_su_propio_token_cuando_esta_configurado(sm, entorno):
    entorno.setenv("SUPERMARKET_ADMIN_TOKEN", "token-solo-del-catalogo")
    sm._verify_supermarket_token(_bearer("token-solo-del-catalogo"))  # no levanta


def test_con_token_propio_configurado_el_MAESTRO_deja_de_valer(sm, entorno):
    """La razón de ser del P-fix: el CRON_SECRET no debe abrir el catálogo."""
    entorno.setenv("SUPERMARKET_ADMIN_TOKEN", "token-solo-del-catalogo")
    with pytest.raises(HTTPException) as e:
        sm._verify_supermarket_token(_bearer("el-secreto-maestro-de-los-crons"))
    assert e.value.status_code == 403


def test_sin_token_propio_acepta_el_maestro_para_no_romper_el_rollout(sm, entorno):
    """Compatibilidad durante el despliegue, y sólo mientras la env var falte."""
    sm._verify_supermarket_token(_bearer("el-secreto-maestro-de-los-crons"))  # no levanta


def test_rechaza_un_token_cualquiera(sm, entorno):
    with pytest.raises(HTTPException) as e:
        sm._verify_supermarket_token(_bearer("no-soy-nadie"))
    assert e.value.status_code == 403


def test_rechaza_sin_cabecera(sm, entorno):
    for cabecera in (None, "", "Basic abc", "token-pelado"):
        with pytest.raises(HTTPException) as e:
            sm._verify_supermarket_token(cabecera)
        assert e.value.status_code == 401


def test_fail_secure_si_no_hay_ningun_secreto(sm, monkeypatch):
    """Sin secretos configurados NO se abre: se apaga."""
    monkeypatch.delenv("CRON_SECRET", raising=False)
    monkeypatch.delenv("SUPERMARKET_ADMIN_TOKEN", raising=False)
    with pytest.raises(HTTPException) as e:
        sm._verify_supermarket_token(_bearer("lo-que-sea"))
    assert e.value.status_code == 503, (
        "Sin secreto el gate debe responder 503 (deshabilitado), nunca dejar pasar."
    )


def test_la_comparacion_es_constant_time():
    """Un `!=` plano corta en el primer byte distinto: recupera el secreto byte a byte."""
    import re
    src = _ROUTER.read_text(encoding="utf-8")
    cuerpo = re.search(r"def _verify_supermarket_token.*?(?=\ndef |\n@)", src, re.DOTALL)
    assert cuerpo, "[P2-SUPERMARKET-TOKEN-SPLIT] No se encontró el gate."
    assert "compare_digest" in cuerpo.group(0), (
        "[P2-SUPERMARKET-TOKEN-SPLIT] El gate no compara en tiempo constante. "
        "Es el mismo patrón que P1-ADMIN-TOKEN-CONSTTIME fijó para el maestro."
    )


def test_el_limitador_admin_sigue_puesto():
    """El gate cambia; el freno anti-fuerza-bruta no se toca."""
    src = _ROUTER.read_text(encoding="utf-8")
    assert src.count("_check_admin_rate_limit(request)") >= 3, (
        "[P2-SUPERMARKET-TOKEN-SPLIT] Desapareció `_check_admin_rate_limit` de "
        "alguna mutación. Sin él, 60 intentos/min dejan de ser el techo."
    )
