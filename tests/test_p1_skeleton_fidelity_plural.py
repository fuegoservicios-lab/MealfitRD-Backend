"""[P1-SKELETON-FIDELITY-PLURAL · 2026-07-26] "huevos" ≠ "1 huevo" para el matcher.

`omitió múltiples proteínas clave asignadas` es la razón de rechazo **#1 de las últimas dos
semanas** (19 de 114 razones), y cada rechazo cuesta una regeneración completa del plan.

Ranking medido sobre el journal completo (1.293 razones de rechazo desde 2026-06-12):

    45 días:  proteína repetida el mismo día  261   ← dominaba; ya mitigada
    14 días:  proteínas OMITIDAS               19   > proteína repetida (17)
     7 días:  proteínas OMITIDAS                6   > proteína repetida (5)

Al ejecutar `_skeleton_protein_present` con las cadenas EXACTAS del log:

    'huevos'                             vs "1 huevo entero"   → False  ← falso omitido
    'huevos enteros (desayuno/merienda)' vs "1 huevo batido"   → False  ← falso omitido
    'edamame'                            vs "habichuelas"      → False  ← correcto, otro alimento
    'mozzarella (despensa)'              vs "queso mozzarella" → True
    'chuleta de cerdo'                   vs "costilla de cerdo"→ True

El skeleton asigna en PLURAL y el día lista en SINGULAR. `\bhuevos\b` no casa con "huevo", y
`'huevos'` aparece en casi todos los casos que reporta el journal. El resto del matcher
(paréntesis, alternativas con `/`, frontera de palabra) ya funcionaba.

Mismo patrón de stems que `P1-REVERSE-COH-PLURAL` aplica en la coherencia inversa — el repo ya
resolvió esto en otro sitio.

⚠️ La dirección del riesgo: este cambio hace el matcher MÁS permisivo, o sea que podría tapar una
omisión real. Por eso los casos negativos (alimento genuinamente distinto) están anclados abajo:
son la mitad que importa.
"""
import pytest

import graph_orchestrator as go


# ───────────── 1. los falsos omitidos medidos ─────────────

@pytest.mark.parametrize("asignado,texto_del_dia", [
    ("huevos", "1 huevo entero, 50 g de avena"),
    ("huevos enteros (desayuno/merienda)", "1 huevo batido con sal"),
    ("huevos", "revoltillo con 1 huevo y papas"),
    ("claras", "3 claras de huevo"),
    ("lentejas", "1 taza de lenteja cocida"),
])
def test_plural_asignado_vs_singular_en_el_dia(asignado, texto_del_dia):
    assert go._skeleton_protein_present(asignado, texto_del_dia) is True


@pytest.mark.parametrize("asignado,texto_del_dia", [
    ("huevos", "2 huevos, pan integral"),
    ("queso ricotta", "52 g de ricotta"),
    ("mozzarella (despensa)", "15 g de queso mozzarella"),
    ("yogurt natural (despensa)", "1 taza de yogurt griego"),
    ("maní/mantequilla de maní", "2 cdas de mantequilla de maní"),
    ("pechuga de pollo (almuerzo/cena)", "150 g de pechuga de pollo"),
    ("chuleta de cerdo", "120 g de costilla de cerdo"),
    ("carne de res", "100 g de res guisada"),
    ("queso de hoja", "15 g de queso de hoja cocido"),
])
def test_lo_que_ya_funcionaba_sigue_funcionando(asignado, texto_del_dia):
    assert go._skeleton_protein_present(asignado, texto_del_dia) is True


# ───────────── 2. la mitad que importa: seguir detectando la omisión REAL ─────────────

@pytest.mark.parametrize("asignado,texto_del_dia", [
    ("edamame", "80 g de habichuelas rojas"),
    ("salmón", "150 g de pechuga de pollo"),
    ("queso ricotta", "2 huevos y avena"),
    ("cerdo", "150 g de pescado blanco al vapor"),
    ("atún", "1 taza de yogurt griego con fresas"),
])
def test_omision_REAL_sigue_flageada(asignado, texto_del_dia):
    assert go._skeleton_protein_present(asignado, texto_del_dia) is False


def test_no_matchea_dentro_de_otra_palabra():
    """La frontera de palabra es load-bearing: 'pollo' NO puede matchear 'repollo'."""
    assert go._skeleton_protein_present("pollo", "1 taza de repollo morado rallado") is False


def test_el_plural_opcional_no_rompe_la_frontera():
    """El sufijo `(?:s|es)?` no puede convertirse en un comodín que abra substrings."""
    assert go._skeleton_protein_present("res", "150 g de repollo y arroz") is False
    assert go._skeleton_protein_present("pavo", "2 tazas de pavos reales decorativos") is True


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


# ───────────── 4. knobs ─────────────

def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(go, "SKELETON_FIDELITY_PLURAL", False)
    assert go._skeleton_protein_present("huevos", "1 huevo entero") is False
    assert go._skeleton_protein_present("huevos", "2 huevos") is True


def test_knob_de_rollback_filler(monkeypatch):
    monkeypatch.setattr(go, "SKELETON_FIDELITY_FILLER", False)
    assert go._skeleton_protein_present("atún en agua", "media taza de agua tibia") is True


def test_la_lista_de_relleno_no_contiene_alimentos():
    """Meter un alimento aquí lo volvería invisible para el gate en todos los planes."""
    for sospechoso in ("atun", "atún", "sardinas", "pollo", "huevo", "queso", "leche", "res"):
        assert sospechoso not in go._SKELETON_PROTEIN_FILLER_TOKENS
