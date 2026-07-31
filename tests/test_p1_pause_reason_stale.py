"""[P1-PAUSE-REASON-STALE · 2026-07-30] Un motivo de pausa resuelto no describe la pausa actual.

Encontrado inspeccionando el `pipeline_snapshot` del chunk del owner mientras se
investigaba P1-SYNTH-LESSON-NOT-STUB: estaba pausado por `synthesis_ratio_exceeded`
pero arrastraba `_pantry_pause_reason: "empty_pantry_proactive"` de una pausa de
nevera **ya resuelta el día anterior** (`_pantry_pause_resolved_at` puesto).

Eso no es basura inerte. El cron de recovery resuelve el motivo así:

    pause_reason = snap.get("_pantry_pause_reason") or snap.get("_pause_reason") or ...

o sea que **la clave caduca gana sobre la vigente** y el chunk toma la rama
equivocada del recovery.

Causa: de los 7 sitios que sellaban `_pantry_pause_resolution`, solo UNO
(`prev_chunk_concluded`) limpiaba las claves de la pausa viva — y con el comentario
correcto al lado. Los otros 6 sellaban `resolved_at` y dejaban el motivo puesto.

Medido en producción: 4 de 89 chunks con motivo caduco (los 4 del mismo plan, todos
aún `pending` — o sea que la mala clasificación estaba por ocurrir, no ya ocurrida).
"""
import re
from pathlib import Path

CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"


# ------------------------------------------------------- el helper hace las dos cosas

def test_resolver_sella_y_limpia():
    """Sellar la resolución y borrar el motivo van juntos, o no sirve de nada."""
    from cron_tasks import _resolve_pantry_pause_markers

    snap = {
        "_pantry_pause_reason": "empty_pantry_proactive",
        "_pantry_pause_started_at": "2026-07-28T20:30:06+00:00",
        "_pantry_pause_ttl_hours": 12,
        "_pantry_pause_reminder_hours": 4,
        "_pantry_pause_reminders": 2,
        "_pantry_pause_last_reminder_at": "2026-07-29T08:31:05+00:00",
        "otra_cosa": "intacta",
    }
    fuera = _resolve_pantry_pause_markers(snap, "degraded_flexible_meal")

    assert fuera is snap, "debe mutar y devolver el mismo dict (encadenable)"
    assert snap["_pantry_pause_resolution"] == "degraded_flexible_meal"
    assert snap["_pantry_pause_resolved_at"], "falta el sello temporal"
    for k in ("_pantry_pause_reason", "_pantry_pause_started_at", "_pantry_pause_ttl_hours",
              "_pantry_pause_reminder_hours", "_pantry_pause_reminders",
              "_pantry_pause_last_reminder_at"):
        assert k not in snap, f"{k} describe una pausa viva y sobrevivió a su resolución"
    assert snap["otra_cosa"] == "intacta", "no debe tocar nada más del snapshot"


def test_ningun_sitio_sella_la_resolucion_a_mano():
    """Blanket: el sello va SIEMPRE por el helper.

    Sin esto, el octavo sitio que alguien añada repite el bug — que es exactamente
    cómo llegamos aquí (7 sitios, 1 correcto).
    """
    src = CRON.read_text(encoding="utf-8")
    directas = [
        (n, l.strip()) for n, l in enumerate(src.split("\n"), start=1)
        if '["_pantry_pause_resolution"] = ' in l
    ]
    # La única permitida es la de dentro del propio helper.
    assert len(directas) == 1, (
        "P1-PAUSE-REASON-STALE regresión: el sello de `_pantry_pause_resolution` debe "
        "pasar por `_resolve_pantry_pause_markers`, que además limpia el motivo. "
        "Asignaciones directas encontradas:\n  "
        + "\n  ".join(f"linea {n}: {t}" for n, t in directas)
    )
    n_helper = directas[0][0]
    inicio_helper = src[:src.index("def _resolve_pantry_pause_markers")].count("\n") + 1
    fin_helper = inicio_helper + src[src.index("def _resolve_pantry_pause_markers"):].split(
        "\ndef ")[0].count("\n")
    assert inicio_helper <= n_helper <= fin_helper, (
        f"la única asignación directa (línea {n_helper}) debería estar DENTRO del helper "
        f"(líneas {inicio_helper}-{fin_helper})"
    )


# --------------------------------------------- el lector, para filas ya escritas

def _resolver_motivo(snap: dict) -> str:
    """Réplica del guard del cron, extraída del fuente para no divergir del original.

    Anclada al tooltip-anchor y al orden relativo, no a una ventana de bytes.
    """
    src = CRON.read_text(encoding="utf-8")
    i = src.find("# tooltip-anchor: [P1-PAUSE-REASON-STALE] motivo caduco no gana")
    assert i != -1, "desapareció el tooltip-anchor del guard en cron_tasks.py"
    j = src.find("pause_reason = str(", i)
    k = src.find("\n", j)
    fragmento = src[src.find("\n", i) + 1: k]
    # Des-indentar y ejecutar el fragmento real con `snap` en scope.
    lineas = [l[12:] if l.startswith(" " * 12) else l.lstrip() for l in fragmento.split("\n")]
    ns = {"snap": snap, "str": str}
    exec("\n".join(lineas), ns)          # noqa: S102 — ejecutamos el guard REAL de prod
    return ns["pause_reason"]


def test_motivo_de_nevera_ya_resuelto_no_gana():
    """El caso exacto del chunk del owner."""
    motivo = _resolver_motivo({
        "_pantry_pause_reason": "empty_pantry_proactive",
        "_pantry_pause_started_at": "2026-07-28T20:30:06+00:00",
        "_pantry_pause_resolved_at": "2026-07-29T20:32:40+00:00",   # resuelta DESPUÉS
        "_pause_reason": "synthesis_ratio_exceeded",
    })
    assert motivo == "synthesis_ratio_exceeded", (
        f"el motivo caduco de nevera ganó sobre el vigente: {motivo!r}"
    )


def test_pausa_de_nevera_VIVA_sigue_ganando():
    """El falso positivo simétrico, que mi primera versión del guard introducía.

    Tras el fix, una pausa nueva escribe `started_at` fresco pero el `resolved_at` de
    la ANTERIOR sigue en el snapshot. Anular por la mera presencia de `resolved_at`
    mataría el motivo vivo — cambiar un error por otro.
    """
    motivo = _resolver_motivo({
        "_pantry_pause_reason": "empty_pantry",
        "_pantry_pause_started_at": "2026-07-30T10:00:00+00:00",    # empezó DESPUÉS
        "_pantry_pause_resolved_at": "2026-07-29T20:32:40+00:00",   # de resolverse la vieja
        "_pause_reason": "algo_anterior",
    })
    assert motivo == "empty_pantry", (
        f"una pausa de nevera VIVA debe ganar; el guard la anuló: {motivo!r}"
    )


def test_sin_ninguna_marca_cae_al_default():
    assert _resolver_motivo({}) == "empty_pantry"
    assert _resolver_motivo({"_pause_reason": "tz_unresolved"}) == "tz_unresolved"
