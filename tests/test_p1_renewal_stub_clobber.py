"""[P1-RENEWAL-STUB-CLOBBER · 2026-07-30] Un stub incondicional a nivel de módulo envenena la
sesión ENTERA de pytest — desde la COLECCIÓN, no desde la ejecución.

El caso: `test_renewal_pantry_empty.py` hacía `sys.modules['langchain_core'] = MagicMock()` en el
cuerpo del módulo. pytest importa TODOS los ficheros al coleccionar antes de correr el primer test,
así que el langchain_core REAL quedaba reemplazado para toda la sesión y cualquier import tardío
(dentro de una función de test) recibía el MagicMock.

La firma que lo hizo indetectable — y que costó CUATRO hipótesis:
  · 3 tests de diary_dinner_slot verdes SOLOS, verdes en test_p1_* (7.492), verdes en [a-o]+p0
    (1.626), verdes con el prefijo alfabético completo (478 ficheros)…
  · …y rojos ÚNICAMENTE en la corrida total: el envenenador es alfabéticamente POSTERIOR a sus
    víctimas, así que ningún prefijo lo incluía. **El veneno no era de orden de ejecución sino de
    orden de COLECCIÓN.**
  · En pequeño ni siquiera arranca: con el envenenador coleccionado primero, la colección revienta
    con `AttributeError: __path__` (un MagicMock no es un paquete).

La regla que este blanket fija: en el CUERPO de un módulo de tests, `sys.modules[...] = ...` solo
condicional (el patrón del conftest: probar el import real primero; stub únicamente si falta la
dependencia). Dentro de funciones/fixtures se permite — ahí hay teardown posible.
"""
from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def test_ningun_modulo_de_tests_clobbea_sys_modules_incondicionalmente():
    malos = []
    for f in sorted(_TESTS.glob("test_*.py")):
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:                     # SOLO el cuerpo del módulo (import-time)
            if isinstance(node, ast.Assign):
                seg = ast.get_source_segment(src, node) or ""
                if "sys.modules[" in seg:
                    malos.append(f"{f.name}: {seg.splitlines()[0][:80]}")
    assert not malos, (
        "asignación INCONDICIONAL a sys.modules en el cuerpo de un módulo de tests — se ejecuta en "
        "la COLECCIÓN y reemplaza el paquete real para TODA la sesión (así se envenenaron los 3 "
        "tests de diary_dinner_slot desde un fichero alfabéticamente posterior). Usa el patrón "
        "condicional (__import__ real primero, stub solo en ImportError):\n  " + "\n  ".join(malos))


def test_los_dos_renewal_usan_el_patron_condicional():
    for name in ("test_renewal_pantry_empty.py", "test_renewal_15d.py"):
        src = (_TESTS / name).read_text(encoding="utf-8")
        assert "if _stub_mod not in sys.modules:" in src, f"{name} perdió el guard condicional"
        assert "except ImportError:" in src, f"{name} debe stubbear SOLO si el paquete real falta"
