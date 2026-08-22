"""[P1-I18N-CHAT-TITULOS-SERVIDOR · 2026-08-22] El backend decidia el TEXTO del rotulo de
la conversacion y el cliente lo pintaba crudo.

`get_user_chat_sessions` componia «Nuevo Chat» y «Interaccion con imagen o sistema» en
espanol duro. Con la app en ingles, la barra lateral del Agente listaba «Nuevo Chat»,
«Nuevo Chat», «Generar plan para mi objetivo: lose_fat»… bajo un encabezado de grupo que SI
decia «Today» — el unico rotulo traducido de esa columna, lo que hacia el contraste mas
evidente todavia.

MEDIDO en produccion: 171 de 186 sesiones vivas llevaban uno de esos dos titulos.

LA DISTINCION QUE ARREGLA ESTO, y que es la misma frontera de todo el sistema: un rotulo
GENERICO no es contenido, es INTERFAZ. El backend devuelve ahora un DISCRIMINADOR
(`title_key`: 'empty' | 'image_or_system') con `title: None`, y el cliente resuelve con
`t()`. Lo que SI es contenido --el `[SYSTEM_TITLE]`, que ya nace en el idioma del usuario
via `build_title_language_directive`, y el texto que el usuario escribio-- sigue viajando
tal cual en `title` y no se traduce.

Y hay una razon para que decida el CLIENTE y no el servidor: el idioma activo puede haber
cambiado despues del ultimo mensaje de esa sesion. El cliente sabe cual es AHORA.

LO QUE NO SE TOCA: `'Generando titulo...'` en el frontend. Parece copy y NO lo es -- es un
CENTINELA de estado que `AgentPage` y `SidebarRecientes` comparan por igualdad para decidir
si pintan el esqueleto de carga. Traducirlo romperia esa comparacion en silencio, que es
exactamente la clase de dano que la frontera «lo que el motor usa como identificador no se
traduce» existe para evitar. Aparece en el trinquete de espanol sin envolver, y ahi debe
seguir.

tooltip-anchor: P1-I18N-CHAT-TITULOS-SERVIDOR
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_DB_CHAT = _BACKEND / "db_chat.py"
_SIDEBAR = _ROOT / "frontend" / "src" / "components" / "agent" / "SidebarRecientes.jsx"

_MARKER = "P1-I18N-CHAT-TITULOS-SERVIDOR"

# Los rotulos genericos que el backend NO puede volver a componer.
_ROTULOS_PROHIBIDOS = ["Nuevo Chat", "Nuevo chat", "Interacción con imagen o sistema"]


def _fuente_backend() -> str:
    return io.open(_DB_CHAT, encoding="utf-8").read()


def _sin_comentarios_py(src: str) -> str:
    """Un rotulo citado en un comentario no es codigo. La sustitucion preserva la longitud
    para que los numeros de linea sigan siendo los del fichero."""
    fuera = []
    for linea in src.split("\n"):
        i = linea.find("#")
        fuera.append(linea if i == -1 else linea[:i] + " " * (len(linea) - i))
    return "\n".join(fuera)


@pytest.mark.parametrize("rotulo", _ROTULOS_PROHIBIDOS)
def test_el_backend_no_compone_el_rotulo_generico(rotulo: str) -> None:
    src = _sin_comentarios_py(_fuente_backend())
    # `s["title"] = "Nuevo Chat"` y variantes.
    patron = re.compile(r'\[\s*["\']title["\']\s*\]\s*=\s*["\']' + re.escape(rotulo))
    m = patron.search(src)
    assert not m, (
        f"`db_chat.py` vuelve a componer el rótulo «{rotulo}» en español. El cliente lo "
        f"pinta crudo, así que en inglés la barra lateral del Agente lo lista tal cual. "
        f"Devuelve `title: None` + `title_key`. [{_MARKER}]"
    )


def test_el_backend_devuelve_el_discriminador() -> None:
    """La otra mitad: sin `title_key`, el cliente no tiene con qué resolver y todo cae al
    mismo rótulo — que es peor que el español, porque además pierde la distinción."""
    src = _sin_comentarios_py(_fuente_backend())
    for clave in ("empty", "image_or_system"):
        assert re.search(r'["\']title_key["\']\s*\]\s*=\s*["\']' + clave, src), (
            f"`db_chat.py` no emite `title_key = '{clave}'`. [{_MARKER}]"
        )


def test_el_contenido_real_sigue_viajando_sin_traducir() -> None:
    """La frontera. El `[SYSTEM_TITLE]` y lo que el usuario escribió NO son interfaz."""
    src = _sin_comentarios_py(_fuente_backend())
    assert "[SYSTEM_TITLE] " in src, (
        f"desapareció la rama que conserva el título real generado por el LLM [{_MARKER}]"
    )
    # Esa rama pone `title_key = None`: el cliente NO debe sustituirlo por un genérico.
    assert re.search(r'\[SYSTEM_TITLE\][\s\S]{0,400}?title_key["\']\s*\]\s*=\s*None', src), (
        f"la rama de `[SYSTEM_TITLE]` no marca `title_key = None`, así que el cliente podría "
        f"pisar un título REAL con un rótulo genérico. [{_MARKER}]"
    )


def test_el_cliente_resuelve_por_la_clave() -> None:
    if not _SIDEBAR.exists():
        pytest.skip(f"no existe {_SIDEBAR} (¿repo hermano sin clonar?)")
    src = io.open(_SIDEBAR, encoding="utf-8").read()
    assert "title_key" in src, (
        f"`SidebarRecientes.jsx` no lee `title_key`: el backend dejó de mandar el texto y "
        f"nadie lo resuelve, así que la columna se queda sin rótulo. [{_MARKER}]"
    )
    for clave in ("empty", "image_or_system"):
        assert clave in src, f"falta el caso `{clave}` en el cliente [{_MARKER}]"


def test_el_centinela_de_generando_titulo_NO_se_traduce() -> None:
    """`'Generando título...'` parece copy y es un CENTINELA de estado.

    `AgentPage` y `SidebarRecientes` lo comparan por igualdad para decidir si pintan el
    esqueleto de carga. Envolverlo en `t()` rompería esa comparación EN SILENCIO — misma
    clase de daño que traducir un nombre de alimento.
    """
    if not _SIDEBAR.exists():
        pytest.skip("sin frontend")
    src = io.open(_SIDEBAR, encoding="utf-8").read()
    assert "'Generando título...'" in src, (
        f"el centinela cambió de forma; revisa las comparaciones de AgentPage antes de "
        f"tocarlo [{_MARKER}]"
    )
    assert not re.search(r"t\(\s*['\"]Generando título\.\.\.['\"]", src), (
        f"el centinela `'Generando título...'` se envolvió en `t()`. Se compara por "
        f"IGUALDAD en dos ficheros: traducirlo rompe el esqueleto de carga sin que nada "
        f"avise. [{_MARKER}]"
    )
