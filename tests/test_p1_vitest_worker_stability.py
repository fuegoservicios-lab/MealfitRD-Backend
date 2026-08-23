"""[P1-VITEST-WORKER-STABILITY · 2026-08-20] La suite del frontend reportaba menos
archivos de los que tiene, sin decirlo.

EL SINTOMA. Tres corridas seguidas de `npx vitest run` sobre el mismo arbol dieron
**247, 258 y 265** archivos. En medio, "Worker exited unexpectedly".

LO GRAVE NO ES QUE FALLEN, ES COMO FALLAN. Un archivo cuyo worker muere antes de
ejecutarlo no aparece como fallo: no aparece. El resumen dice "255 passed" sin
mencionar que faltan diez, y un total menor se lee exactamente igual de verde que el
total completo. El gate del deploy (`deploy-mealfit.ps1` -> `scripts/run_ci.ps1`) se
apoya en ese exit code.

Es la MISMA forma del falso verde que `P1-CI-GATE-INCONCLUSIVE` cerro el dia anterior
en pytest, al otro lado del mismo gate: alli el cache de `--lf` vacio deseleccionaba
todo y salia 0; aqui son archivos que nunca corren. Dos mecanismos distintos, una sola
leccion: **cuando una suite puede ejecutar menos de lo que cree, "verde" deja de
significar "paso todo"**.

LA CAUSA. `vite.config.js` no fijaba pool ni workers, asi que vitest arranca
`nucleos - 1` forks (~11 aqui) y cada uno monta jsdom mas la app entera. Con 16,9 GB y
~3,6 GB libres, eso es presion de memoria suficiente para matar workers. Mismo error
que `-n 8` en pytest, que ya se habia medido en este repo.

MEDIDO, NO SUPUESTO. Con `maxWorkers=4`, dos corridas consecutivas: 265/265 archivos,
2.697/2.697 tests, exit 0, cero errores. Coste ~130 s frente a ~98 s. Treinta segundos
por una cifra fiable es barato; lo caro es desplegar con la suite a medias.

tooltip-anchor: P1-VITEST-WORKER-STABILITY
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG = _ROOT / "frontend" / "vite.config.js"
_TESTS_DIR = _ROOT / "frontend" / "src"


def _config() -> str:
    return io.open(_CONFIG, encoding="utf-8").read()


def _solo_codigo(s: str) -> str:
    """Quita las lineas que son COMENTARIO ENTERO (`//`).

    Hace falta porque la prosa de `vite.config.js` CITA `maxWorkers` y los numeros
    medidos: un `in` sobre el fuente crudo pasaria aunque alguien borrara la linea de
    codigo. Es el patron que mordio seis veces el 2026-08-19.

    NO se filtran los bloques `/* */`, y la primera version que lo intentaba fallaba
    por eso: `vite.config.js` contiene globs como `'**/node_modules/**'`, cuyo `/*`
    abre un comentario falso que se traga hasta el siguiente `*/` -- codigo real
    incluido. Aqui produjo un falso NEGATIVO (el test fallaba con el codigo correcto),
    pero en una asercion de tipo "esto no debe aparecer" el mismo filtro daria un
    falso VERDE. Un filtro de comentarios tiene que ser conservador o no estar.
    """
    return "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("//"))


def test_el_tope_de_workers_esta_puesto_en_codigo():
    codigo = _solo_codigo(_config())
    assert re.search(r"\bmaxWorkers\s*:", codigo), (
        "sin tope, vitest arranca ~11 forks y los workers mueren: la suite ejecuta "
        "MENOS archivos de los que tiene y el resumen no lo dice")


def test_el_tope_es_conservador():
    """4 es lo medido en esta maquina. Un numero alto reintroduce el problema; el
    guard no fija el valor exacto, solo que no vuelva a ser 'todos los nucleos'."""
    codigo = _solo_codigo(_config())
    m = re.search(r"maxWorkers\s*:\s*Number\(process\.env\.VITEST_MAX_WORKERS\)\s*\|\|\s*(\d+)", codigo)
    assert m, "el default debe ser un literal legible detras del knob"
    assert 1 <= int(m.group(1)) <= 6, (
        f"maxWorkers por defecto = {m.group(1)}: demasiado alto para 16 GB con jsdom")


def test_hay_knob_para_otra_maquina():
    """Convencion del repo: lo que puede necesitar cambiar sin tocar codigo va como
    knob. Otra maquina con mas RAM puede subirlo sin editar el config."""
    assert "VITEST_MAX_WORKERS" in _solo_codigo(_config())


def test_documenta_la_medicion_que_lo_destapo():
    """Sin la nota, alguien ve '4 workers en una maquina de 12 nucleos' y lo sube
    'para que corra mas rapido'. La cifra que importa no es la duracion: es que las
    tres corridas daban 247, 258 y 265 archivos."""
    texto = _config()
    assert "247" in texto and "265" in texto, (
        "debe conservar los conteos que prueban que la suite ejecutaba de menos")
    assert "Worker exited unexpectedly" in texto


def test_el_conteo_esperado_de_archivos_sigue_siendo_verificable():
    """Ancla del numero: si el conteo real se aleja mucho del que documenta el
    config, la nota deja de servir para detectar una corrida incompleta."""
    reales = sum(1 for _ in _TESTS_DIR.rglob("*.test.js")) + \
        sum(1 for _ in _TESTS_DIR.rglob("*.test.jsx"))
    assert reales >= 200, f"solo {reales} archivos de test encontrados: ¿ruta mala?"
    documentado = 319  # 2026-08-23: la suite creció con las olas de i18n/países (la nota del config lo registra)
    assert abs(reales - documentado) <= 40, (
        f"el config documenta {documentado} archivos y hay {reales}: actualiza la nota "
        "o la referencia deja de detectar una corrida incompleta")
