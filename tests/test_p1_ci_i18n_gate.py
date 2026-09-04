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

import pytest

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


# ============================================================
# [P2-I18N-DEPLOY-ESCOTILLA · 2026-08-21] La escotilla es PEGAJOSA
# ============================================================
#
# `MEALFIT_CI_I18N_STRICT=0` baja el chequeo a permisivo, y el propio comentario de
# `run_ci.ps1` dice «sirve para una tanda larga a medio traducir; no para desplegar».
# Pero en PowerShell una variable de entorno vive TODA la sesion: quien la puso por la
# manana la sigue teniendo puesta cuando despliega por la tarde. Y el unico efecto
# observable era un `Write-Host` amarillo a 2.000 lineas del final de un log de 20 min.
#
# Hasta hoy este fichero ni siquiera ABRIA `deploy-mealfit.ps1`, que es donde el
# descuido se paga: el deploy invocaba el gate sin mencionar la variable.

_DEPLOY = _ROOT / "deploy-mealfit.ps1"

# La INVOCACION del gate, no cualquier mencion: `scripts/run_ci.ps1` aparece antes
# en prosa (el bloque que explica que hace el gate), asi que anclar en la primera
# aparicion medía la documentacion y no el codigo.
_INVOCA_GATE = re.compile(r"&\s*pwsh\b.*run_ci\.ps1")


def _deploy() -> str:
    if not _DEPLOY.exists():
        pytest.skip(f"{_DEPLOY} no existe en este checkout")
    return _DEPLOY.read_text(encoding="utf-8")


def test_el_deploy_aborta_con_la_escotilla_puesta():
    s = _deploy()
    assert "MEALFIT_CI_I18N_STRICT" in s, (
        "deploy-mealfit.ps1 no menciona la escotilla del gate de i18n. Con la variable "
        "puesta en la sesion, el chequeo corre PERMISIVO y una pantalla a medio traducir "
        "se despliega en silencio. [P2-I18N-DEPLOY-ESCOTILLA]"
    )
    i_var = s.index("MEALFIT_CI_I18N_STRICT")
    m = re.search(_INVOCA_GATE, s)
    assert m, "no encontre la invocacion de run_ci.ps1 en deploy-mealfit.ps1"
    i_gate = m.start()
    assert i_var < i_gate, (
        "la comprobacion de la escotilla esta DESPUES de invocar el gate: para cuando "
        "salta, el gate permisivo ya dio verde. [P2-I18N-DEPLOY-ESCOTILLA]"
    )


def test_el_deploy_ofrece_la_valvula_por_invocacion():
    """No se prohibe la escotilla: el caso legitimo existe. Se exige DECIRLO.

    `-SkipTests` deja rastro POR INVOCACION en vez de por sesion, y esa es exactamente
    la diferencia: una la escribes cada vez, la otra se te olvida puesta.
    """
    s = _deploy()
    i_var = s.index("MEALFIT_CI_I18N_STRICT")
    bloque = s[i_var: re.search(_INVOCA_GATE, s).start()]
    assert "-SkipTests" in bloque, (
        "el aborto por la escotilla no le dice al operador cual es la salida legitima. "
        "Un guard que bloquea sin ofrecer la alternativa se acaba desactivando. "
        "[P2-I18N-DEPLOY-ESCOTILLA]"
    )
    assert "throw" in bloque, (
        "la escotilla solo AVISA en el deploy; tiene que abortar. [P2-I18N-DEPLOY-ESCOTILLA]"
    )
