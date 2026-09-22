"""[P1-PLAN-LOTE-154 · 2026-09-22] El atajo del formulario dejaba de prometer lo que no podía cumplir.

REPORTE DEL DUEÑO (con captura): paso 2 de 26, «¿Cómo quieres que la IA arme tu plan?» marcada con
asterisco y sin elegir, y debajo el botón «Saltar a la última pregunta». Encima, el aviso de un
intento anterior: «Antes de saltar, completa: Cómo arma tu plan la IA». Sus palabras: «ni siquiera
debe aparecerme este botón… y aunque la llene no debería dejarme saltar otras preguntas que son
obligatorias de llenar».

CAUSA. La validación existía y funcionaba —ese aviso ES ella— pero corría un click DESPUÉS del
botón. Quién ve el atajo lo decidía `canSkip`, que mira la HISTORIA del usuario (¿llegó más lejos
en esta visita?, ¿ya tiene un plan?, ¿viene de la rama corta del contador?); quién puede saltar de
verdad lo decidía `handleSkipToLastStep`, que mira el CONTRATO de campos obligatorios y rebota al
primero que falte. Dos respuestas a la misma pregunta, y la que el usuario ve es la que no manda.

  *Una puerta que se anuncia abierta y está cerrada no es una validación que funciona: es un
  botón que miente, aunque nadie llegue a cruzarla.*

Es la hermana de P1-SKIP-RESPECTS-BUDGET por el otro extremo: allí una regla del paso no protegía
a quien no pasaba por el paso; aquí la regla del salto no llegaba a tiempo de decidir si el salto
se ofrece siquiera.

ARREGLO. El botón se pinta con `canSkipToEnd` = `canSkip` Y nada obligatorio pendiente en SU rama
Y el formulario cifrado ya leído Y sin condición fuera de alcance. Las dos quejas se cierran a la
vez porque el contrato cubre TODAS las obligatorias, no la del paso en curso.

La validación del handler se CONSERVA: esconder el botón no es cerrar la puerta. Este fichero
vigila las dos capas — que la de arriba exista y que la de abajo no se retire creyéndola redundante.

Tooltip-anchor: P1-PLAN-LOTE-154
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FLOW = (_REPO_ROOT / "frontend" / "src" / "components" / "assessment"
         / "InteractiveAssessmentFlow.jsx")
_TEST_FRONT = _REPO_ROOT / "frontend" / "src" / "__tests__" / "lote154.test.jsx"


def _src() -> str:
    if not _FLOW.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({_FLOW}).")
    return _FLOW.read_text(encoding="utf-8")


def test_el_boton_cuelga_de_can_skip_to_end_y_no_de_can_skip():
    """El JSX del atajo no puede volver a colgar de `canSkip` a secas.

    `canSkip` sigue existiendo y sigue gobernando el bloque de navegación entero (con él aparece
    «Siguiente Paso» aunque el paso esté vacío, para que quien vuelve pueda moverse): lo que no
    puede es decidir SOLO si se ofrece el salto.
    """
    src = _src()
    assert "{canSkipToEnd && currentStep < steps.length - 1 && (" in src, (
        "El botón «Saltar a la última pregunta» debe pintarse con `canSkipToEnd`. Si vuelve a "
        "`canSkip`, reaparece en un paso con la obligatoria en blanco — el reporte del 22-sep."
    )


def test_la_condicion_mira_el_contrato_de_su_rama():
    """`canSkipToEnd` exige las cuatro cosas, y el contrato es el DE LA RAMA.

    Lo de la rama no es un detalle: con el contrato del plan (22 campos) en modo contador, el
    salto exigía «Tu horario cotidiano», un campo cuyo paso no existe en esa rama — la lección de
    P1-TRACKING-SKIP-CONTRACT, que aquí se habría repetido al copiar la comprobación.
    """
    src = _src()
    inicio = src.find("const _faltaAlgoObligatorio")
    assert inicio != -1, "Falta el cálculo `_faltaAlgoObligatorio`."
    fin = src.find("const canSkipToEnd")
    assert fin != -1 and fin > inicio, "Falta `canSkipToEnd` o quedó antes de lo que consume."
    bloque = src[inicio:fin + 400]

    for fragmento, porque in (
        ("findFirstIncompleteFieldFor(formData, TRACKING_REQUIRED_FIELDS)",
         "la rama corta se valida contra SU contrato"),
        ("findFirstIncompleteField(formData)",
         "la rama del plan se valida contra el contrato completo"),
        ("!loadingSensitive",
         "mientras el formData cifrado se descifra, el contrato ve alergias vacías"),
        ("!_faltaAlgoObligatorio",
         "es la condición del reporte: nada obligatorio pendiente"),
        ("!hasOutOfScopeMedical(formData)",
         "el gate clínico también sobrevive al atajo (P1-OUTSCOPE-SKIP-GATE)"),
    ):
        assert fragmento in bloque, f"`canSkipToEnd` perdió `{fragmento}`: {porque}."


def test_el_handler_conserva_su_propia_validacion():
    """La segunda capa no se retira por redundante.

    Esconder el botón es UX; el handler es la puerta. Si mañana alguien afloja la condición de
    arriba —o se llega aquí por otra vía— el salto tiene que seguir rebotando al primer campo
    incompleto. La duplicación es deliberada y la forma del guard de `loadingSensitive` está
    fijada aparte por los tests de P1-14.
    """
    src = _src()
    inicio = src.find("const handleSkipToLastStep")
    assert inicio != -1, "Desapareció `handleSkipToLastStep`."
    # El salto final del handler. Se busca DESDE el handler: la misma llamada aparece antes en el
    # fichero (el clamp de P1-FORM-AUDIT-BATCH) y cortar desde el principio da un trozo vacío.
    fin = src.find("setCurrentStep(steps.length - 1);", inicio)
    assert fin > inicio, "El handler ya no termina saltando a la última pregunta."
    cuerpo = src[inicio:fin]
    assert len(cuerpo) > 200, "El cuerpo extraído es sospechosamente corto: revisa el corte."

    for fragmento in ("loadingSensitive", "findFirstIncompleteField", "hasOutOfScopeMedical(formData)"):
        assert fragmento in cuerpo, (
            f"`handleSkipToLastStep` perdió `{fragmento}`. Esconder el botón NO sustituye a la "
            f"validación del salto."
        )


def test_el_caso_del_dueno_esta_medido_en_el_frontend():
    """El contrato de verdad se mide montando el componente, no leyendo el fichero.

    Estos asserts de texto son el ancla que sobrevive a un renombre; lo que demuestra la conducta
    es `frontend/src/__tests__/lote154.test.jsx`, que monta el wizard con una obligatoria en
    blanco y comprueba que el botón no existe. Si alguien lo borra, este test lo acusa.
    """
    if not _TEST_FRONT.exists():
        pytest.skip(f"El árbol del frontend no está junto a este backend ({_TEST_FRONT}).")
    contenido = _TEST_FRONT.read_text(encoding="utf-8")
    assert "EL CASO DE LA CAPTURA" in contenido
    assert "LA SEGUNDA QUEJA" in contenido
    assert "botonSaltar()).toBeNull()" in contenido
    assert "botonSaltar()).toBeInTheDocument()" in contenido, (
        "El atajo tiene que seguir apareciendo con el formulario completo: el arreglo no es "
        "quitar el botón."
    )
