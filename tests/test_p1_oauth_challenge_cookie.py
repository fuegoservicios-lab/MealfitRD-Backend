"""[P1-OAUTH-CHALLENGE-COOKIE · 2026-08-10] El verifier de Neon NO se canjea en el servidor.

EL HALLAZGO, EN NÚMEROS DE PRODUCCIÓN
─────────────────────────────────────
El dueño reportó que «Continuar con Google» exigía DOS clics. Los logs del VPS,
30 días: **24 canjes intentados, 0 con éxito**, todos HTTP 400, desde el
2026-07-11. El arreglo escrito el 2026-07-03 para evitar ese doble clic nunca
funcionó ni una vez, y nada lo avisó porque nadie miraba ese contador.

Preguntándole a Neon directamente, con la misma petición y el mismo verifier de
prueba:

    sin cookie de challenge  → {"code":"SESSION_CHALLENGE_COOKIE_NOT_FOUND"}
    con cookie de challenge  → {"code":"VERIFICATION_NOT_FOUND"}

El segundo error ya es «ese verifier no existe»: superó el control de la cookie.
Es decir, el verifier NO es autosuficiente — va emparejado a una cookie
`__Secure-neon-auth.session_challange` que Neon deja en el NAVEGADOR y en SU
dominio. **Ningún proceso de servidor la tendrá jamás**, así que no es un
problema de red, de plazo ni de reintentos: era imposible por construcción.

Antes de esto hubo TRES intentos de arreglo del mismo síntoma —reintentos,
más reintentos (de 4 a 8), y el canje server-side—, los tres tratándolo como un
problema de TIEMPO. Esperar más no trae una cookie que no existe de este lado.

    LECCIÓN: cuando un arreglo no arregla, comprueba si su premisa es
    siquiera posible antes de subirle los plazos.

Este fichero impide que el canje server-side vuelva a introducirse.
Tooltip-anchor: P1-OAUTH-CHALLENGE-COOKIE
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_FRONT = _BACKEND.parent / "frontend" / "src"


def _sin_comentarios_py(src: str) -> str:
    """Quita docstrings y comentarios: las notas EXPLICAN el patrón prohibido, y
    un escáner que las lea acusaría al arreglo de ser el defecto."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    return re.sub(r"#[^\n]*", "", src)


def test_ningun_endpoint_canjea_el_verifier_contra_neon():
    """El patrón prohibido: pedirle a Neon `/get-session` con el verifier desde el
    backend. Devuelve 400 SIEMPRE — la cookie de challenge es del navegador."""
    for py in (_BACKEND / "routers").glob("*.py"):
        limpio = _sin_comentarios_py(py.read_text(encoding="utf-8"))
        assert "neon_auth_session_verifier" not in limpio, (
            f"P1-OAUTH-CHALLENGE-COOKIE: {py.name} vuelve a canjear el verifier server-side. "
            "Neon responde SESSION_CHALLENGE_COOKIE_NOT_FOUND porque esa cookie vive en el "
            "navegador: fueron 24 fallos de 24 durante un mes. El canje va en el cliente."
        )


def test_el_canje_del_cliente_manda_las_credenciales():
    """Sin `credentials: 'include'` la cookie de challenge no viaja y el canje del
    navegador cae en el MISMO 400 que el del servidor."""
    fps = (_FRONT / "utils" / "firstPartySession.js").read_text(encoding="utf-8")
    bloque = fps[fps.index("export async function adoptOAuthVerifierFirstParty"):]
    assert "get-session?neon_auth_session_verifier=" in bloque
    assert "credentials: 'include'" in bloque, (
        "P1-OAUTH-CHALLENGE-COOKIE: el canje perdió `credentials: 'include'`. Sin eso la "
        "cookie de challenge no viaja y Neon vuelve a rechazar el canje."
    )


def test_el_servidor_sigue_siendo_quien_decide():
    """Mover el canje al cliente NO significa confiar en el cliente: el backend
    valida el Bearer contra el JWKS antes de emitir la sesión propia."""
    src = (_BACKEND / "routers" / "auth_session.py").read_text(encoding="utf-8")
    bloque = src[src.index("async def oauth_adopt"):]
    bloque = bloque[:bloque.index("@router") if "@router" in bloque[10:] else len(bloque)]
    assert "get_neon_bearer_user_id" in bloque, (
        "P1-OAUTH-CHALLENGE-COOKIE: el endpoint dejó de validar el Bearer. El cliente "
        "canjea, pero quien decide si hay sesión es el servidor."
    )


def test_el_contador_por_flujo_sigue_vivo():
    """El defecto vivió un mes porque su contador no se miraba. El endpoint dedicado
    existe para que el retorno de Google tenga log propio y contable — si esto se
    fusiona con /session, se pierde la única señal que lo delata."""
    src = (_BACKEND / "routers" / "auth_session.py").read_text(encoding="utf-8")
    assert '@router.post("/oauth/adopt")' in src, "el endpoint dedicado del retorno OAuth desapareció"
    assert "P1-OAUTH-FIRST-PARTY] retorno de Google adoptado" in src, (
        "P1-OAUTH-CHALLENGE-COOKIE: se perdió el log de ÉXITO del retorno OAuth. Es el "
        "contador que estuvo en 0/24 sin que nadie se enterara."
    )
