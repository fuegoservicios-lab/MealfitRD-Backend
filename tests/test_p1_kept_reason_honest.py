"""[P1-KEPT-REASON-HONEST · 2026-08-05] La rama PARCIAL de regenerate-day dice por
qué se conservó cada slot, en vez de dejar que el cliente lo adivine.

POR QUÉ EXISTE. `P2-REGEN-DAY-HONEST-CODE` (2026-07-10) ya distinguía la causa real
—Nevera sin inventario vs guardrail del LLM agotando retries— pero SOLO en la rama
donde no se regeneró nada. En la rama parcial (algunos platos cambiaron y otros no)
la respuesta llevaba únicamente los nombres de los slots, y el frontend rellenaba
el motivo con un literal fijo: «tu Nevera no daba para cambiarlos».

Medido en producción el 2026-08-05, ventana de 2 horas: 26 de los 28 reintentos
fueron `guardrail_rejection` (macros fuera de banda) y CERO por despensa, con la
Nevera del usuario llena. El aviso mandaba a comprar comida para arreglar un
problema de porciones — el mismo daño que aquel P-fix documentó ("erosiona
confianza, visto en vivo con Nevera llena"), en la rama que se quedó fuera.

tooltip-anchor: P1-KEPT-REASON-HONEST
"""
import io
import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PLANS = _BACKEND_ROOT / "routers" / "plans.py"


def _sin_comentarios(bloque: str) -> str:
    """Descarta líneas de comentario.

    La primera versión del test hermano (`test_p1_upcoming_fetchall`) pasaba con el
    arreglo borrado porque el comentario citaba literalmente el símbolo buscado. Un
    test que certifica TEXTO en vez de la DECISIÓN no prueba nada.
    """
    return "\n".join(
        linea for linea in bloque.splitlines()
        if not linea.lstrip().startswith("#")
    )


def _success_return_block() -> str:
    """El dict de retorno de la rama de ÉXITO (la que lleva `slots_kept`)."""
    src = io.open(_PLANS, encoding="utf-8").read()
    start = src.index('"slots_kept": slots_kept,')
    end = src.index('"band_score": _band_score,', start)
    return _sin_comentarios(src[start:end])


def test_la_respuesta_parcial_declara_el_motivo():
    """La rama parcial emite `slots_kept_reason`."""
    bloque = _success_return_block()
    assert '"slots_kept_reason"' in bloque, (
        "La respuesta de regenerate-day no declara `slots_kept_reason`. Sin él, el "
        "cliente no puede saber si el slot se conservó por Nevera o por guardrail y "
        "vuelve a culpar siempre a la Nevera."
    )


def test_el_motivo_usa_la_causa_real_no_una_constante():
    """Se deriva de `_kept_reasons`, la misma fuente que la rama de fallo total.

    Un literal fijo (`"pantry"` siempre) satisfaría al test anterior y reproduciría
    exactamente el bug que este P-fix cierra.
    """
    bloque = _success_return_block()
    assert "_kept_reasons" in bloque, (
        "`slots_kept_reason` no se deriva de `_kept_reasons`. Si es una constante, "
        "el motivo vuelve a ser una suposición."
    )
    assert "SWAP_STRICT_PANTRY_NO_INVENTORY" in bloque and "ERRORES DE DESPENSA" in bloque, (
        "El criterio de 'pantry' debe ser el MISMO que usa la rama de fallo total "
        "(P2-REGEN-DAY-HONEST-CODE). Dos criterios distintos para la misma pregunta "
        "es como se desincronizan."
    )


def test_ambos_valores_son_alcanzables():
    """`pantry` y `ai` existen los dos, y `None` cuando no se conservó nada."""
    bloque = _success_return_block()
    assert re.search(r'"pantry"', bloque), "Falta el valor 'pantry'."
    assert re.search(r'"ai"', bloque), (
        "Falta el valor 'ai'. Si solo existe 'pantry', el arreglo es cosmético: el "
        "caso medido en producción (guardrail de macros) seguiría sin nombre."
    )
    assert "if slots_kept else None" in bloque, (
        "Sin slots conservados el motivo debe ser None, no un valor por defecto que "
        "el cliente pueda interpretar como una causa real."
    )


def test_la_rama_de_fallo_total_conserva_su_clasificacion():
    """No romper lo que P2-REGEN-DAY-HONEST-CODE ya hacía bien."""
    src = io.open(_PLANS, encoding="utf-8").read()
    assert '"error_code": "pantry_insufficient_for_goal"' in src
    assert '"error_code": "ai_exhausted_retries"' in src
