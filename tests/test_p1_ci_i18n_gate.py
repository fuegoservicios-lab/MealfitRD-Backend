"""[P1-CI-I18N-GATE · 2026-08-20] El chequeo de catalogos no estaba en el gate.

El 2026-08-20 el dueno reporto OCHO superficies distintas en espanol con la app en
ingles: dias de la semana, fecha del modal, titulo del plan, pestanas, fecha de la
tarjeta, slots de Recetas, submenu «Mas informacion», nombres de plan y el splash.

Todas llegaron por CAPTURA. Ninguna por CI. `npm run i18n:check` existia y funcionaba
--de hecho cazo varias cosas ese mismo dia-- pero solo corria cuando alguien se
acordaba de escribirlo a mano. Una defensa que depende de que alguien la invoque no es
una defensa del sistema: es una costumbre.

POR QUE ESTRICTO, que es la decision que de verdad importa aqui. MEDIDO quitando una
traduccion de `en-US.json`:

    npm run i18n:check          -> exit 0     <- no habria cazado NADA de hoy
    npm run i18n:check:strict   -> exit 1

Sin `--strict`, una clave sin traduccion no tumba nada: el texto cae al espanol y la
pantalla queda a medias EN SILENCIO -- literalmente la forma de los ocho reportes.
Anadir el paso en modo permisivo habria sido teatro: un guard en el gate que da verde
justo en el caso que motivo ponerlo.

Encenderlo no crea deuda: el repo esta al 100% en los 4 idiomas (0 huerfanas, 0
faltantes). Solo obliga a que una cadena nueva nazca traducida.

tooltip-anchor: P1-CI-I18N-GATE
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
_GATE = _ROOT / "scripts" / "run_ci.ps1"
_GATE_ESPEJO = _BACKEND / "scripts" / "run_ci.ps1"
_PKG = _ROOT / "frontend" / "package.json"


def _gate() -> str:
    return io.open(_GATE, encoding="utf-8").read()


def test_las_dos_copias_ssot_siguen_identicas():
    assert _GATE.read_bytes() == _GATE_ESPEJO.read_bytes()


def test_el_gate_tiene_un_paso_de_i18n():
    assert 'Run-Step "Frontend i18n"' in _gate(), (
        "el chequeo de catalogos volvio a quedarse fuera del gate: los fallos de "
        "traduccion vuelven a llegar por captura")


def test_el_paso_corre_en_modo_ESTRICTO_por_defecto():
    """La mitad del P-fix. Medido: sin `--strict`, una clave sin traducir sale 0 y el
    texto cae al espanol en silencio -- que es la forma de los 8 reportes del dia."""
    s = _gate()
    assert "npm run i18n:check:strict" in s, "el paso perdio el --strict: da verde con claves sin traducir"
    # Y el permisivo SOLO detras del knob, nunca como default.
    m = re.search(r'\$strict = \$env:MEALFIT_CI_I18N_STRICT(.*?)\n\s*\}', s, re.S)
    assert m, "falta la escotilla por knob"
    rama = m.group(1)
    assert 'if ($strict -eq "0"' in rama, "el permisivo no esta gateado por el knob"


def test_el_paso_va_ANTES_de_vitest():
    """~2 s frente a ~2 min, y su fallo es de una linea. Fallar rapido y con la causa
    a la vista es la diferencia entre arreglarlo y volver a lanzarlo a ciegas."""
    s = _gate()
    assert s.index('Run-Step "Frontend i18n"') < s.index('Run-Step "Frontend vitest"')


def test_el_paso_respeta_SkipFrontend():
    """Es un chequeo de frontend: un `-SkipBackend` no debe arrastrarlo, pero un
    `-SkipFrontend` si."""
    s = _gate()
    i_skip = s.index("if (-not $SkipFrontend)")
    i_i18n = s.index('Run-Step "Frontend i18n"')
    i_build = s.index("if (-not $SkipBuild)")
    assert i_skip < i_i18n < i_build, "el paso i18n quedo fuera del bloque de frontend"


def test_los_dos_scripts_npm_existen():
    """El gate llama por nombre: si alguien renombra el script de package.json, el paso
    fallaria por 'missing script' y pareceria un fallo de traduccion."""
    pkg = json.loads(io.open(_PKG, encoding="utf-8").read())
    scripts = pkg.get("scripts", {})
    assert "i18n:check" in scripts
    assert "i18n:check:strict" in scripts
    assert "--strict" in scripts["i18n:check:strict"]


def test_documenta_por_que_estricto():
    """Sin la medicion escrita, el proximo que vea el gate rojo por una clave sin
    traducir lo baja a permisivo «para desbloquear» y desarma justo lo que servia."""
    s = _gate()
    assert "ocho superficies" in s.lower()
    # Lo que se ancla es la MEDICION, no una palabra: los dos codigos de salida son el
    # hecho del que depende toda la decision. Sin ellos, «¿por que estricto?» hay que
    # re-descubrirlo quitando una clave a mano.
    assert "npm run i18n:check          -> exit 0" in s, (
        "falta la medicion que justifica el --strict")
    assert "npm run i18n:check:strict   -> exit 1" in s
