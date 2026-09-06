# -*- coding: utf-8 -*-
"""[P1-DEPLOY-ARGS-STRICT · 2026-09-06] Un argumento desconocido no puede degradar a «lo de siempre».

El 06-sep se tecleó `.\\deploy-mealfit.ps1 -Backend`. Ese switch no existe —el objetivo es
posicional— y al invocar por `-File` PowerShell **ignora el sobrante en silencio**, así que
`$target` se quedó en su default `all`: un error de tecleo desplegó el frontend además del
backend y publicó una release que nadie pidió. Como el deploy empaqueta el **árbol de trabajo**,
lo de más puede ser código sin commitear y sin gate — que fue exactamente lo que pasó.

El modo de fallo no es «se despliega de menos». Es **«se despliega de más, sin decirlo»**, que es
la clase de fallo que nadie descubre hasta que ya ocurrió.

⚠️ Este test mira un fichero que vive FUERA del repo del backend (la raíz del workspace). Esa es
la lección de «un guard que lee fuera del repo es verde en CI y rojo en local» aplicada al revés:
en CI el fichero no existe y el test **se salta**; en local, donde sí está, comprueba. Por eso
tampoco bumpea `_LAST_KNOWN_PFIX`: el binario del backend no cambió con este P-fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_DEPLOY = _BACKEND.parent / "deploy-mealfit.ps1"
pytestmark = pytest.mark.skipif(
    not _DEPLOY.exists(),
    reason="deploy-mealfit.ps1 vive en la raíz del workspace, fuera del repo del backend")


@pytest.fixture(scope="module")
def script() -> str:
    return _DEPLOY.read_text(encoding="utf-8", errors="replace")


def test_recoge_los_argumentos_sobrantes(script):
    """Sin `ValueFromRemainingArguments` no hay nada que comprobar: el sobrante ni llega."""
    assert "ValueFromRemainingArguments" in script, (
        "el script dejó de recoger los argumentos sobrantes; un switch inventado volvería a "
        "ignorarse y el objetivo caería a su default 'all'")


def test_falla_ante_un_argumento_desconocido(script):
    """`throw`, no `Write-Warning`: el aviso que no detiene el deploy no habría evitado nada."""
    i = script.find("ValueFromRemainingArguments")
    trozo = script[i:i + 1600]
    assert "throw" in trozo, "el sobrante se recoge pero no detiene el deploy"
    assert "NADA se desplego" in trozo or "NADA se desplegó" in trozo, (
        "el mensaje debe decir explícitamente que no se desplegó nada")


def test_el_mensaje_ensena_la_forma_correcta(script):
    """Un error que solo dice «argumento inválido» obliga a abrir el script. Este dice cómo."""
    i = script.find("ValueFromRemainingArguments")
    trozo = script[i:i + 1600]
    for forma in ("deploy-mealfit.ps1 backend", "deploy-mealfit.ps1 frontend",
                  "deploy-mealfit.ps1 all", "-SkipTests"):
        assert forma in trozo, f"el mensaje de error no menciona {forma!r}"


def test_el_mensaje_no_lleva_comillas_invertidas(script):
    """La comilla invertida es el ESCAPE de PowerShell. En la primera versión de este mensaje
    escribí el default entre comillas invertidas y salió «su default ll»: la secuencia se comió
    la 'a' y emitió un BEL. Un mensaje de error corrompido en el momento del incidente es
    justo cuando menos se puede permitir."""
    i = script.find("[P1-DEPLOY-ARGS-STRICT] Argumento")
    assert i > 0, "cambió el texto del error; actualiza este test junto al cambio"
    fin = script.find('"@', i)
    assert fin > i, "no se encontró el cierre del here-string"
    assert "`" not in script[i:fin], (
        "hay una comilla invertida dentro del here-string de comillas dobles: PowerShell la "
        "interpretará como escape y corromperá el mensaje")


def test_el_objetivo_sigue_siendo_posicional_y_acotado(script):
    """El guard no sustituye al `ValidateSet`: son capas distintas. `ValidateSet` acota un valor
    posicional válido; el guard caza lo que ni siquiera llega a ser un valor."""
    assert "[ValidateSet('backend','frontend','all','infra')]" in script
    assert "[string]$target = 'all'" in script
