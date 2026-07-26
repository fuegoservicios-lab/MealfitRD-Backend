"""[P1-SKELETON-FIDELITY-FILLER · 2026-07-26] "agua" no prueba que haya atún.

Validando `P1-SKELETON-FIDELITY-PLURAL` apareció el error CONTRARIO en el mismo matcher. El
chequeo es `any(token)`, y los tokens de `'atún en agua'` son `atún` y **`agua`**:

    'atún en agua'  vs  "1 huevo, harina, media taza de agua tibia"  →  True

Cualquier receta que mencione agua hacía creer al gate que el atún estaba presente, **tapando una
omisión real**. Verificado sobre el plan vivo `a3b9510e`: matcher=True con "atún" ausente del día.

⚠️ Este repo ya aprendió esto con el MISMO alimento, en la función hermana:

    [P2-STEM-FILLER-TOKENS · 2026-07-06] "agua/aceite/lata jamás son evidencia de uso (el
    'atún en agua' quedaba sin paso porque la masa de las arepitas usaba agua real)"

Estaba arreglado en `_ensure_ingredients_used_in_recipe` y no en `_skeleton_protein_present`.
Mismo bug, otra función — buscar al hermano antes de dar por nuevo un defecto.

Importa la dirección: el fix de plural hizo el matcher MÁS permisivo; éste lo hace menos, y en el
eje correcto (un token de relleno nunca identifica un alimento).
"""
import pytest

import graph_orchestrator as go


# ───────────── 3. tokens de relleno [P1-SKELETON-FIDELITY-FILLER] ─────────────
#
# El chequeo es `any(token)`, así que `'atún en agua'` matcheaba por **`agua`**: cualquier receta
# que mencione agua hacía creer al gate que el atún estaba presente — tapando una omisión REAL.
# Verificado sobre el plan vivo a3b9510e: matcher=True con "atún" ausente del día.
#
# ⚠️ Este repo ya aprendió esto con el MISMO alimento en la función hermana
# (P2-STEM-FILLER-TOKENS, 2026-07-06: *"el 'atún en agua' quedaba sin paso porque la masa de las
# arepitas usaba agua real"*). Estaba arreglado allí y no aquí.

@pytest.mark.parametrize("asignado,texto_del_dia", [
    ("atún en agua", "1 huevo, 3 cdas de harina, media taza de agua tibia"),
    ("sardinas en lata", "1 lata de habichuelas rojas"),
    ("atún en aceite", "1 cda de aceite de oliva y lechuga"),
])
def test_el_relleno_no_prueba_que_la_proteina_este(asignado, texto_del_dia):
    assert go._skeleton_protein_present(asignado, texto_del_dia) is False


@pytest.mark.parametrize("asignado,texto_del_dia", [
    ("atún en agua", "110 g de atún en agua escurrido"),
    ("sardinas en lata", "40 g de sardinas en lata"),
])
def test_con_la_proteina_REAL_sigue_matcheando(asignado, texto_del_dia):
    assert go._skeleton_protein_present(asignado, texto_del_dia) is True


def test_alimento_cuyo_nombre_EMPIEZA_por_relleno_no_se_queda_sin_tokens():
    """'aceite de oliva': filtrar 'aceite' deja 'oliva', que sí identifica. El fallback protege
    el caso extremo en que el filtro vaciara la lista."""
    assert go._skeleton_protein_present("aceite de oliva", "1 cda de aceite de oliva") is True


def test_knob_de_rollback_filler(monkeypatch):
    monkeypatch.setattr(go, "SKELETON_FIDELITY_FILLER", False)
    assert go._skeleton_protein_present("atún en agua", "media taza de agua tibia") is True


def test_la_lista_de_relleno_no_contiene_alimentos():
    """Meter un alimento aquí lo volvería invisible para el gate en todos los planes."""
    for sospechoso in ("atun", "atún", "sardinas", "pollo", "huevo", "queso", "leche", "res"):
        assert sospechoso not in go._SKELETON_PROTEIN_FILLER_TOKENS
