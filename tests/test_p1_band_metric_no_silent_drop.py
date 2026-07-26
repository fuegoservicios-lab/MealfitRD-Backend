"""[P1-BAND-METRIC-NO-SILENT-DROP · 2026-07-26] Las corridas problemáticas desaparecían del A/B.

El emit de `pipeline_metrics.node='clinical_band'` estaba envuelto en `if _band_val is not None:`.
Cuando el score de banda no se podía calcular, **se descartaba la fila entera** — y con ella el
`retries` de esa corrida.

Medido: el plan `b2464ffe` (2026-07-26 14:55) **reintentó una vez** por «avena al vapor como cena» y
«proteínas asignadas omitidas», y no dejó fila de `clinical_band`; su log tampoco imprimió
«CLINICAL BAND SCORE», o sea `_band_val` era None.

⚠️ El sesgo **no es aleatorio**: la corrida que se perdió es justo del tipo que el A/B necesita
contar. Un experimento que pierde selectivamente sus casos difíciles reporta el sistema mejor de lo
que es.

El caso completo vive en `test_p1_band_cohort_on_final.py` (sección 4), que es el archivo del fix
hermano sobre el mismo emit. Este archivo existe para que el marker `P1-BAND-METRIC-NO-SILENT-DROP`
tenga su cross-link propio (`test_p2_hist_audit_14_marker_test_link`) y para anclar la CLASE.
"""
import graph_orchestrator as go


def test_ningun_emit_de_metrica_va_envuelto_en_un_guard_de_valor():
    """Ancla la CLASE, no la instancia: un `if <valor> is not None:` alrededor de un emit convierte
    "no pude medir esto" en "esto no ocurrió". Si hace falta el guard, que sea sobre el CAMPO y no
    sobre la fila."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('"node": "clinical_band",')
    assert "if _band_val is not None:" not in src[max(0, i - 1600):i]


def test_la_corrida_sin_banda_conserva_sus_reintentos():
    """Lo que hace útil la fila aunque el score falte: `retries` y la cohorte siguen ahí."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('"node": "clinical_band",')
    # Se corta en el CIERRE del emit, no a N bytes: una ventana fija se queda corta en cuanto
    # alguien añade un comentario (me pasó al escribir este mismo test) y el assert falla por el
    # tamaño, no por el código.
    bloque = src[i:src.index("})", i)]
    assert '"retries": final_state.get("attempt", 1) - 1' in bloque
    assert '"daygen_model_cohort"' in bloque
    assert '"band_unavailable": _band_val is None' in bloque
