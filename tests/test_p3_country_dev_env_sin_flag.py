"""[P3-COUNTRY-DEV-ENV-SIN-FLAG · 2026-08-23] `npm run dev` levantaba la app con el sistema de
países APAGADO, y sin ninguna señal de que eso estaba pasando.

`VITE_COUNTRY_SYSTEM` es build-time (Vite inlinea las `VITE_*` al construir). `.env.production`
y `.env.native` la traen a `true` desde el flip del 2026-08-18; `.env.example` —el fichero del
que sale el `.env` de cualquiera que clone el repo— tenía CERO ocurrencias. Quien reproduce en
local el bug de un usuario de España veía un wizard SIN paso de país, una Configuración SIN
sección de país y un Dashboard que nunca entra en modo beta: otro producto, sin aviso.

`frontend/.env` NO es el fichero de este gap: está gitignorado (`.gitignore:32`), no es un
artefacto versionable y no hay forma de vigilarlo. El gap versionable es `.env.example`.

Este guard mide TRES cosas y la tercera es la que importa a las 3 de la mañana: que el fichero
diga que apagarla ES el rollback. Una variable declarada sin decir qué apaga es una constante
que alguien puede cambiar sin saber lo que hace — la misma lección que
`P3-I18N-AUTOLOCALE-INDESCUBRIBLE` dejó escrita dos líneas más abajo en este mismo fichero.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_FRONT = _REPO / "frontend"
_EJEMPLO = _FRONT / ".env.example"
_VAR = "VITE_COUNTRY_SYSTEM"


def _asignaciones(ruta: Path):
    """[(var, valor)] de las líneas de asignación (ignora comentarios)."""
    out = []
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        s = linea.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out.append((k.strip(), v.strip()))
    return out


def test_el_ejemplo_declara_la_bandera_del_sistema_de_paises():
    """RED pre-fix: 0 ocurrencias. Un `.env` copiado del ejemplo levantaba el mundo pre-flip."""
    declaradas = dict(_asignaciones(_EJEMPLO))
    assert _VAR in declaradas, (
        f"{_EJEMPLO.name} no declara {_VAR}: `npm run dev` arranca con el sistema de países "
        f"apagado y sin señal de que está mirando otro producto"
    )


def test_el_ejemplo_arranca_con_el_mismo_valor_que_produccion():
    """Un ejemplo que no reproduce producción es peor que no tenerlo: da falsa confianza."""
    valor_ejemplo = dict(_asignaciones(_EJEMPLO)).get(_VAR)
    valor_prod = dict(_asignaciones(_FRONT / ".env.production")).get(_VAR)
    assert valor_prod, ".env.production dejó de declarar la bandera — ¿se revirtió el flip?"
    assert valor_ejemplo == valor_prod, (
        f"{_VAR}: el ejemplo dice {valor_ejemplo!r} y producción {valor_prod!r}. Un dev que "
        f"copie el ejemplo NO reproduce lo que ve el usuario."
    )


def test_el_bloque_nombra_su_par_del_backend_y_dice_que_apagarla_es_el_rollback():
    """Las dos banderas se apagan JUNTAS: el backend seguiría sellando país con el frontend
    apagado, y el frontend ofrecería países que el backend ya no honra. Sin esa frase, quien
    tenga que revertir en un incidente apagará una sola."""
    texto = _EJEMPLO.read_text(encoding="utf-8", errors="replace")
    i = texto.index(f"\n{_VAR}=")
    # El bloque de comentario que precede a la asignación: desde la línea en blanco anterior.
    inicio = texto.rfind("\n\n", 0, i)
    bloque = texto[inicio:i]
    assert "MEALFIT_COUNTRY_SYSTEM" in bloque, (
        "el bloque no nombra su par del backend: se apagarían por separado"
    )
    assert re.search(r"rollback", bloque, re.I), (
        "el bloque no dice que apagarla ES el rollback del sistema de países"
    )


@pytest.mark.parametrize("fichero", (".env.production", ".env.native"))
def test_los_env_del_flip_siguen_encendidos(fichero):
    """Control: si alguien apaga el flip de verdad, este test lo dice en vez de dejar que el
    test de paridad de arriba se ponga verde por el lado equivocado."""
    valor = dict(_asignaciones(_FRONT / fichero)).get(_VAR)
    assert str(valor).lower() in ("1", "true"), f"{fichero}: {_VAR}={valor!r}"
