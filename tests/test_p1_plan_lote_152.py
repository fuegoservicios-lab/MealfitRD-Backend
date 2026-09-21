# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-152 · 2026-09-21] El origen de la app de Android en la lista CORS.

Bioboros nace como app de Android para la tanda de pruebas con testers. Su WebView sirve la app desde
`https://localhost` (`androidScheme: 'https'`, fijado explícito en `capacitor.config.ts`), y ese origen NO
estaba permitido: la app habría arrancado y muerto en la primera llamada —login, diario, avisos, todo— con un
«Disallowed CORS origin» que en un teléfono no se ve por ningún lado.

Lo llamativo es que el agujero estaba ANUNCIADO: el comentario de la entrada de iOS, de agosto, terminaba
diciendo «Android sería https://localhost». *Una nota que describe el siguiente caso no lo cubre; solo
demuestra que alguien lo vio venir.*
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _codigo() -> str:
    """El cuerpo de la llamada a `add_middleware(CORSMiddleware, ...)`, sin los comentarios de arriba.

    Buscar `allow_origins=[` a pelo en el fichero entero no sirve: el bloque de comentarios que precede a la
    llamada CITA la configuración vieja (`allow_headers=["*"]`) para explicar por qué se cerró, y un `index()`
    ingenuo se queda con la cita en vez de con el código."""
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    i = app.index("    CORSMiddleware,")
    return app[i:app.index("\n)", i)]


def _lista_cors() -> str:
    c = _codigo()
    i = c.index("allow_origins=[")
    return c[i:c.index("allow_credentials=", i)]


def test_el_origen_de_android_esta_permitido():
    assert '"https://localhost"' in _lista_cors()


def test_el_de_ios_sigue_estando():
    """El de Android se AÑADE; si alguien lo cambia por el otro, el iPhone deja de hablar con la API."""
    assert '"capacitor://localhost"' in _lista_cors()


def test_no_se_abre_a_http_localhost():
    """`http://localhost` a secas permitiría a cualquier servidor local del puerto 80 hablar con la API.

    Los de desarrollo llevan puerto (`:5173`, `:5174`) y son otra cosa."""
    lista = _lista_cors()
    assert '"http://localhost"' not in lista
    assert '"http://localhost:5173"' in lista, "los de desarrollo con puerto sí siguen"


def test_la_cabecera_de_sesion_sigue_permitida():
    """En nativo la cookie no viaja: `X-MF-Session` es el ÚNICO mecanismo de sesión, en iOS y en Android."""
    c = _codigo()
    i = c.index("allow_headers=[")
    assert '"X-MF-Session"' in c[i:]


def test_el_marcador_va_con_su_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P1-PLAN-LOTE-(\d+) · \d{4}-\d{2}-\d{2}"', app)
    assert m, "el marcador cambió de forma"
    assert int(m.group(1)) >= 152, "el marcador nunca baja"
