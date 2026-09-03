"""[P1-DEMO-NO-ES-TEST · 2026-08-19] En `scripts/` no vive nada llamado `test_*`.

POR QUÉ. Dos ficheros de `backend/scripts/` se llamaban `test_medical_reviewer.py`
y `test_semantic_cache.py`. Ninguno era una prueba: el primero invoca
`review_plan_node` y el segundo `run_plan_pipeline` —la generación completa de un
plan, la operación más cara del sistema—. Sus propias fichas del README los
describían como manuales.

Pero el nombre `test_*` no es una etiqueta descriptiva: es una INSTRUCCIÓN para
pytest. Lo que pasó de verdad:

  - El del reviewer tumbó el gate del despliegue con «async def functions are not
    natively supported», un mensaje que habla del plugin que falta y no de la
    causa. Nada se desplegó por un fichero que nunca debió recogerse.
  - El del semantic cache sobrevivió sólo porque esa corrida lo DESELECCIONABA.
    No estaba protegido, estaba escondido — y un `pytest scripts/` sin filtro lo
    habría ejecutado y facturado, justo lo que la directiva de gasto prohíbe.

Este guard es barato y cierra la clase entera: mira NOMBRES, no contenido, así
que no puede equivocarse sobre lo que un fichero hace.
"""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_scripts_no_contiene_ficheros_test():
    intrusos = sorted(p.name for p in SCRIPTS.glob("test_*.py"))
    assert not intrusos, (
        "[P1-DEMO-NO-ES-TEST] Ficheros en backend/scripts/ que pytest recogerá "
        f"como pruebas: {intrusos}.\n\n"
        "Las pruebas viven en backend/tests/. Si esto es una demostración manual, "
        "renómbrala a `demo_*.py`; si es una prueba de verdad, muévela a tests/.\n\n"
        "No es cosmético: los dos casos que motivaron esto invocaban el proveedor "
        "REAL, uno de ellos generando un plan entero. Un fichero que se llama "
        "`test_*` y gasta dinero es una factura esperando a que alguien corra la "
        "suite sin mirar."
    )


def test_las_demos_declaran_que_no_son_pruebas():
    """Y que las que se renombraron sigan explicando por qué.

    Sin esto, alguien puede borrar la cabecera, dejar el fichero con aire de
    prueba y volver a la ambigüedad que costó un despliegue — sólo que esta vez
    sin el nombre delator.
    """
    faltan = []
    for p in sorted(SCRIPTS.glob("demo_*.py")):
        if "NO una prueba" not in p.read_text(encoding="utf-8"):
            faltan.append(p.name)
    assert not faltan, (
        f"[P1-DEMO-NO-ES-TEST] Estas demos no declaran que no son pruebas: {faltan}. "
        "La cabecera es lo que impide que alguien las mueva a tests/ de vuelta."
    )
