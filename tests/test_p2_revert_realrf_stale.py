"""[P2-REVERT-REALRF-STALE · 2026-07-31] (audit solver+seeder v6 · P2 / F5) Tras el revert de
Nevera, el relevel de grasas recortaba sobre dicts huérfanos.

En `update_plan_day` (regenerate-day) el bloque `[P1-UPDATE-MACRO-PARITY]` hace:

    _real_rf = [m for m in new_meals if isinstance(m, dict)]   # ALIAS de los dicts vivos
    ...
    _rdi_rd(_real_rf, ...)          # el refine 5g los muta IN-PLACE
    if _exc_rf:                     # el refine rompió el ledger de la Nevera
        new_meals[:] = _pre_rf      # REVERT por asignación de SLICE
    ...
    _trim_fats_rd(_real_rf, ...)    # <-- recorta sobre los de ANTES del revert

`new_meals[:] = _pre_rf` sustituye el CONTENIDO de la lista por los deepcopies. `_real_rf` sigue
apuntando a los dicts viejos —los que el refine mutó— que ya NO están en el día que se persiste.
`_trim_day_fats_to_target` muta in-place, así que su recorte cae en objetos que nadie guarda: el
relevel de grasas queda INERTE exactamente en la rama donde el día quedó fuera de banda.

Este fichero prueba las dos mitades: (1) la mecánica de Python que hace posible el bug, para que
la regla estructural no se lea como superstición; (2) que el código de producción re-deriva la
lista tras el revert.

Anchor de producción: P2-REVERT-REALRF-STALE.
"""
import re
from pathlib import Path

PLANS = Path(__file__).resolve().parent.parent / "routers" / "plans.py"


def test_la_asignacion_de_slice_deja_huerfano_al_alias():
    """La mecánica exacta del bug, sin misterio: slice-assign no reapunta los alias."""
    vivos = [{"n": "a", "fats": 30}, {"n": "b", "fats": 20}]
    alias = [m for m in vivos if isinstance(m, dict)]
    respaldo = [dict(m) for m in vivos]

    vivos[:] = respaldo  # el revert
    for m in alias:      # el trim, sobre el alias
        m["fats"] = 0

    assert [m["fats"] for m in vivos] == [30, 20], (
        "el trim sobre el alias NO llega al día vivo — ésta es la razón del fix"
    )
    alias_fresco = [m for m in vivos if isinstance(m, dict)]
    for m in alias_fresco:
        m["fats"] = 0
    assert [m["fats"] for m in vivos] == [0, 0], "re-derivar sí alcanza al día vivo"


def test_el_revert_re_deriva_la_lista_viva():
    """tooltip-anchor de producción: P2-REVERT-REALRF-STALE

    El endpoint no es unit-testeable sin DB (vive dentro de `update_plan_day`, tras auth, fetch de
    plan y ledger de Nevera), así que la regla se ancla estructuralmente: la re-derivación debe ir
    PEGADA al revert, no en el callsite del trim — así cubre también lo que se añada después.
    """
    src = PLANS.read_text(encoding="utf-8", errors="ignore")
    i = src.index("new_meals[:] = _pre_rf")
    ventana = src[i: i + 1200]
    codigo = "\n".join(l for l in ventana.splitlines() if not l.lstrip().startswith("#"))

    assert re.search(r"_real_rf\s*=\s*\[m for m in new_meals", codigo), (
        "tras `new_meals[:] = _pre_rf` hay que re-derivar `_real_rf`: si no, el relevel de grasas "
        "recorta dicts que ya no están en el día que se persiste"
    )
    assert "P2-REVERT-REALRF-STALE" in ventana, "falta el tooltip-anchor junto al revert"


def test_el_trim_de_grasas_sigue_recibiendo_real_rf():
    """Guard de la otra mitad: si alguien cambia el argumento del trim, este anclaje lo avisa."""
    src = PLANS.read_text(encoding="utf-8", errors="ignore")
    assert "_trim_fats_rd(_real_rf" in src, (
        "el relevel de grasas del regenerate-day ya no recibe `_real_rf`; revisar que su nueva "
        "fuente sea el día VIVO post-revert"
    )
