"""[P1-SLOT-DRIFT-OBSERVABLE · 2026-08-05] `slot_drift` aterriza en algún sitio.

POR QUÉ EXISTE. `P2-SLOT-DRIFT-TELEMETRY` (2026-07-29) construyó el cómputo de la
desviación por slot con un objetivo explícito en su docstring — "el reparto
fisiológico es hoy inobservable en producción... medir primero, mover el reparto
después" — y los dos levers que moverían el reparto nacieron OFF bajo
`MEALFIT_SLOT_AWARE_DAY_REPAIR` esperando esa medición.

Pero la clave se calculaba y se tiraba: entra en el dict de
`compute_clinical_band_score` y su único consumidor, el gate de retry, lee
`score`/`per_macro` y descarta el resto. Medido contra producción el 2026-08-05:
CERO menciones de `slot_drift` en 24 h de journal y CERO filas en
`pipeline_metrics`. La decisión aplazada no podía tomarse nunca porque la
evidencia no aterrizaba en ninguna parte.

Es la clase "el sistema mide X y nadie consume X", ya registrada en este repo.

tooltip-anchor: P1-SLOT-DRIFT-OBSERVABLE
"""
import io
import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_GO = _BACKEND_ROOT / "graph_orchestrator.py"


def _src() -> str:
    return io.open(_GO, encoding="utf-8").read()


def _sin_comentarios(bloque: str) -> str:
    """Un comentario que cite el símbolo buscado haría pasar el test con el
    arreglo borrado — ya ocurrió con `test_p1_upcoming_fetchall`."""
    return "\n".join(
        linea for linea in bloque.splitlines()
        if not linea.lstrip().startswith("#")
    )


def test_el_computo_sigue_existiendo():
    """Ancla del sitio: si `_compute_slot_drift` desaparece, avisa aquí y no en
    el test de abajo verificando el vacío."""
    assert "def _compute_slot_drift(" in _src()


def test_existe_un_emisor_de_slot_drift():
    assert "def _emit_slot_drift_metric_best_effort(" in _src(), (
        "No hay emisor de `slot_drift`. El cómputo sin emisor es una medición que "
        "nadie puede leer."
    )


def test_el_emisor_escribe_a_pipeline_metrics():
    src = _src()
    start = src.index("def _emit_slot_drift_metric_best_effort(")
    end = src.index("def _compute_slot_drift(", start)
    bloque = _sin_comentarios(src[start:end])
    assert "INSERT INTO pipeline_metrics" in bloque, (
        "El emisor no persiste a `pipeline_metrics`: la medición seguiría sin poder "
        "consultarse para decidir sobre el reparto."
    )
    assert re.search(r'["\']slot_drift["\']', bloque), (
        "El row debe identificarse como `slot_drift` para poder agruparlo sin "
        "parsear texto libre."
    )


def test_el_emisor_se_invoca_de_verdad():
    """Un emisor que nadie llama es el mismo bug con un paso más."""
    src = _src()
    llamadas = [
        m for m in re.finditer(r"_emit_slot_drift_metric_best_effort\(", src)
    ]
    # 1 definición + al menos 1 invocación.
    assert len(llamadas) >= 2, (
        "`_emit_slot_drift_metric_best_effort` está definido pero no se invoca. "
        "El cómputo seguiría tirándose."
    )
    # La invocación vive junto al cómputo del band score, que es donde nace el dato.
    idx_gate = src.index("_bsr = compute_clinical_band_score(plan, {})")
    ventana = _sin_comentarios(src[idx_gate:idx_gate + 2500])
    assert "_emit_slot_drift_metric_best_effort(" in ventana, (
        "La invocación no está donde se calcula el band score; si el dato se emite "
        "desde otro sitio, actualiza este ancla conscientemente."
    )


def test_es_best_effort_y_no_puede_tumbar_la_generacion():
    src = _src()
    start = src.index("def _emit_slot_drift_metric_best_effort(")
    end = src.index("def _compute_slot_drift(", start)
    bloque = src[start:end]
    assert "except Exception" in bloque, (
        "El emisor debe tragarse sus fallos: es telemetría, no puede romper una "
        "generación de plan."
    )
