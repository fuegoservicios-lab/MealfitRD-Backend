# -*- coding: utf-8 -*-
"""[P1-I18N-DEAD-VEREDICTO · 2026-09-08] Una fila muerta por un veredicto viejo no bloquea para siempre.

Secuela inmediata de `P1-I18N-RECONCILE`, y la encontré **ejercitando mi propio arreglo contra
producción**: el barrido decía «1 candidato, 0 encolados».

La cadena: `already_enriched` («no quedaba nada por traducir») se clasificaba como fallo
reintentable y mató un job con `attempts=5` **por haber hecho el trabajo** — la alerta de
dead-letter lo dice con esas palabras: `attempts=5 error=already_enriched`. Corregido el
vocabulario, la fila muerta seguía reteniendo su `dedup_key`: `enqueue_plan_job` hace
`ON CONFLICT DO NOTHING` y `maybe_enqueue_display_i18n` sólo cuenta como «ya en cola» los estados
vivos. Resultado: el cron encontraría el plan cada 20 minutos y no podría encolar nada, **para
siempre**. Un no-op perpetuo que desde fuera parece vivo — exactamente el defecto que el P-fix
padre vino a cerrar, reproducido dentro de su propio arreglo.

Va en su propio fichero a propósito: «reintentar el veredicto correcto» y «no quedarse bloqueado
por un veredicto viejo» son dos contratos, y borrar uno debe fallar su test, no el del otro.
"""
import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- el veredicto que quedó mal
def test_repara_los_veredictos_que_el_vocabulario_VIEJO_dejo_mal():
    """Una fila `dead` con un `error_code` que HOY es terminal bloquearía su `dedup_key` para siempre.

    Lo destapé ejercitando mi propio arreglo contra producción: el barrido encontraba el plan y
    encolaba **cero**. La cadena: `already_enriched` mató un job con `attempts=5` *por haber hecho
    el trabajo*; corregido el vocabulario, la fila muerta seguía ahí, y `enqueue_plan_job` hace
    `ON CONFLICT DO NOTHING` sobre `dedup_key` mientras `maybe_enqueue_display_i18n` sólo cuenta
    como «ya en cola» los estados vivos. Resultado: «1 candidato, 0 encolados» cada 20 minutos,
    para siempre — *un no-op perpetuo que parece vivo desde fuera*, que es exactamente el defecto
    que este P-fix vino a cerrar.
    """
    src = (_BACKEND / "plan_jobs.py").read_text(encoding="utf-8")
    fn = next((n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "reconcile_missing_display_i18n"), None)
    assert fn is not None
    doc = ast.get_docstring(fn, clean=False)
    sql = " ".join(s for s in
                   (n.value for n in ast.walk(fn)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != doc))
    assert "status IN ('failed', 'dead')" in sql, (
        "no repara las filas que el vocabulario viejo dejó mal: su `dedup_key` bloquea el "
        "re-encolado y el barrido queda en no-op perpetuo")
    assert "error_code = ANY(" in sql, (
        "debe acotar la reparación a los `error_code` que HOY son terminales — revivir un "
        "dead-letter con error transitorio quemaría LLM en bucle")
    # Por AST, no por un recorte de N caracteres: la primera versión miraba los primeros 2.600 y
    # dejó de ver el bloque en cuanto lo moví unas líneas. Un instrumento atado a una distancia
    # arbitraria mide la distancia, no el contrato.
    usados = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "_DONE_SKIPS" in usados, (
        "la lista de códigos reparables debe salir de `_DONE_SKIPS`, no de un literal aparte: "
        "así, añadir una palabra terminal repara sus filas históricas sin tocar dos sitios")


def test_la_reparacion_va_ANTES_de_buscar_candidatos():
    """Si repara después de consultar, la primera pasada informa «1 candidato, 0 encolados» — una
    línea de log que describe un problema ya resuelto. El orden es parte del contrato."""
    src = (_BACKEND / "plan_jobs.py").read_text(encoding="utf-8")
    cuerpo = src.split("def reconcile_missing_display_i18n")[1]
    i_rep = cuerpo.find("status IN ('failed', 'dead')")
    i_sel = cuerpo.find("_I18N_PENDIENTES_SQL")
    assert 0 < i_rep < i_sel, (
        "la reparación de veredictos corre DESPUÉS de seleccionar candidatos: el barrido reporta "
        "un candidato que él mismo acaba de resolver")
