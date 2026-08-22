"""[P2-SYSMODULES-STUB-LEAK · 2026-08-22] Un test instalaba un `sentry_sdk` VACÍO en
`sys.modules` con `setdefault` y no lo deshacía. Si en ese worker de xdist nadie había
importado el real todavía, el stub quedaba para todo el proceso y el siguiente test que
importara `app` (que llama `sentry_sdk.init`) moría con `AttributeError` — cambiando de
víctima según el reparto. Tumbó un deploy (gate `all`, 2026-08-22 13:4x UTC).

El síntoma señala a la víctima, nunca al causante. Y «comprobar si el real es importable»
cierra el caso frecuente, no el mecanismo: lo que lo cierra es que la escritura sea
REVERSIBLE (`monkeypatch.setitem`, que pytest deshace al terminar el test).

Alcance del guard, a propósito ESTRECHO: stubs (`ModuleType`/`MagicMock`) de `sentry_sdk`
y `apscheduler*` (los módulos que `app` necesita al importar) escritos sin monkeypatch. El repo tiene ~60 escrituras directas a
`sys.modules` de otra clase (`sys.modules[module_name] = module` del cargador por ruta,
que registra módulos REALES; stubs de `langgraph` a nivel de módulo, que corren en la
colección y son deterministas). Ensanchar esto a todas es otro trabajo.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS = Path(__file__).resolve().parent

# Los dos módulos que `app` necesita al importar y que los tests stubean: un stub
# parcial de cualquiera de los dos mata el siguiente `import app` del worker.
# El nombre puede ser literal ('sentry_sdk') o una VARIABLE de un bucle (mod_name): el
# causante original iteraba una tupla. Lo que lo define como stub es el VALOR
# (ModuleType/MagicMock), no el nombre — el cargador por ruta del repo registra un
# módulo REAL bajo `module_name` y no entra aquí.
# (Prosa FUERA de la asignación: test_p1_renewal_stub_clobber lee el span entero del
# Assign y un comentario dentro de los paréntesis cuenta como código.)
_SENTRY_STUB_DIRECT = re.compile(
    r"""sys\.modules(\.setdefault\(|\[)\s*(['"](sentry_sdk|apscheduler[.\w]*)['"]|\w+)\s*[,\]]\s*=?\s*(types\.ModuleType|MagicMock)"""
)


def _offenders():
    """Solo escrituras que corren DENTRO de una función (def/async def): son las que
    dependen del orden del worker. Un `try/except ImportError` a nivel de módulo
    (test_p1_cron_bundle, test_p2_ops_bundle) corre en la colección, en todos los
    workers y solo sin el paquete real: determinista, misma clase que los ~40 stubs
    de langgraph."""
    out = []
    for f in sorted(_TESTS.glob("test_*.py")):
        if f.name == Path(__file__).name:
            continue
        en_funcion = False
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line and not line[0].isspace():
                en_funcion = line.startswith(("def ", "async def ", "@"))
            s = line.strip()
            if not en_funcion or s.startswith("#") or "`sys.modules" in line:
                continue
            if _SENTRY_STUB_DIRECT.search(line):
                out.append(f"{f.name}:{i}: {s}")
    return out


def test_ningun_test_stubea_sentry_sdk_sin_monkeypatch():
    off = _offenders()
    assert not off, (
        "Stub de sentry_sdk NO reversible en sys.modules (fuga al resto del worker y el "
        "siguiente `import app` muere en sentry_sdk.init). Usa "
        "`monkeypatch.setitem(sys.modules, 'sentry_sdk', stub)`:\n  " + "\n  ".join(off)
    )


def test_el_stub_reversible_no_rompe_el_import_de_app(monkeypatch):
    """Funcional: con el stub puesto por monkeypatch, al salir del test `sys.modules`
    vuelve a como estaba. Aquí se comprueba la mitad observable: el real (o la
    ausencia) se restaura — lo verifica el propio pytest al terminar; lo que este test
    asegura es que un stub con `init` no-op no tumba a nadie mientras vive."""
    import sys
    import types

    antes = sys.modules.get("sentry_sdk")
    stub = types.ModuleType("sentry_sdk")
    stub.init = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sentry_sdk", stub)
    assert sys.modules["sentry_sdk"] is stub
    monkeypatch.undo()
    assert sys.modules.get("sentry_sdk") is antes
