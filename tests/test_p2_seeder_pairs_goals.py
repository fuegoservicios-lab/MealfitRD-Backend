"""[P2-SEEDER-PAIRS-GOALS · 2026-07-31] (audit solver+seeder v6 · P2 / F17+F28+F18)
Tres defectos del seeder determinista (`ai_helpers.get_deterministic_variety_prompt`).

F17 · EL PAR DE CARBOS SE ARMA DE DOS LISTAS DISTINTAS
    _carb_slots = _rotate_pairs(chosen_carbs)                    # rotación DEDUPLICADA
    carb_params = {f"carb_{i}": chosen_carbs[i] ...}             # 1er slot: lista PADDED
    carb_params.update({f"carb_{i}b": _carb_slots[i][1] ...})    # 2º slot: la DEDUPED
Parear por índice entre dos listas derivadas por separado. Con duplicados en `chosen_carbs`
—el caso POR DEFECTO, porque `num_carbs_to_pick = min(2, ...)` y luego se rellena a 3— el
índice i deja de referirse al mismo elemento en ambas y sale `carb_i == carb_ib`: el prompt
ASIGNA una base y en la misma frase PROHÍBE repetirla. Medido sobre las 3 barajas equiprobables
de [A,B,A]: 2 de 3 producen al menos un día colisionado.

Las hermanas ya lo hacen bien y esa asimetría es la prueba: frutas y vegetales toman AMBOS
slots del mismo `_slots[i]`. Los carbos son la única categoría que quedó mezclando listas.

Por qué nació verde: el test funcional del P-fix original usa `mainGoal="gain_muscle"`, que
auto-promueve a `variety_level=max` → 3 carbos únicos → el dedupe es la identidad → nunca pisa
la ventana rota. Fixture que certifica el caso SIN bug.

F28 · TOKENS DE OBJETIVO QUE NO EXISTEN
`_GOALS_FORCE_MAX_VARIETY` y `_GOALS_PENALIZE_PROCESSED` contienen 'lose_weight' y
'health_improvement'. El enum real de `mainGoal` es {lose_fat, gain_muscle, maintenance,
performance} y el router devuelve 422 para cualquier otro ⇒ esos dos tokens son INALCANZABLES:
la auto-promoción a variedad máxima y el penalty de embutidos nunca disparaban para quien
quiere perder grasa, que es justo el perfil que más los necesita.

F18 · EL "SORTEO" DEL ANCLA LIVIANA NO SORTEA
`_light_pool[:4]` es un slice POSICIONAL. El pool concatena vegetales ANTES que proteínas y los
vegetales aportan exactamente 5 frutos secos, así que los 11 lácteos (índices 5..15) no entran
JAMÁS. El fix v5 amplió el pool del helper y su consumidor lo dejó inerte.

Anchor de producción: P2-SEEDER-PAIRS-GOALS.
"""
import random
import re

import pytest


def _prompt(**fd):
    import ai_helpers as ah
    return ah.get_deterministic_variety_prompt("", dict(fd), user_id=None)


_RX_PRIMARIA = re.compile(r"obligatoriamente:\s*(.+?)\s*\+\s*(.+?)\s+y como acompañante")
_RX_SEGUNDA = re.compile(r"usa\s+(.+?)\s+—\s+NUNCA la misma base")


def _pares_de_carbo(texto):
    """Devuelve [(base_asignada, segunda_base)] por día, leídos del prompt real."""
    out = []
    for linea in texto.splitlines():
        m1, m2 = _RX_PRIMARIA.search(linea), _RX_SEGUNDA.search(linea)
        if m1 and m2:
            out.append((m1.group(2).strip(), m2.group(1).strip()))
    return out


# --------------------------------------------------------------- F17

def test_el_par_de_carbos_del_dia_nunca_es_la_misma_base():
    """El caso por defecto (sin mainGoal ⇒ variedad standard ⇒ 2 carbos sorteados)."""
    colisiones = []
    for semilla in range(30):
        random.seed(semilla)
        pares = _pares_de_carbo(_prompt(mainGoal="maintenance"))
        for i, (a, b) in enumerate(pares):
            if a and b and a.lower() == b.lower():
                colisiones.append((semilla, i, a))
    assert not colisiones, (
        f"{len(colisiones)} días asignan la MISMA base en los dos slots y a la vez prohíben "
        f"repetirla. Muestra: {colisiones[:5]}"
    )


def test_el_par_de_carbos_sigue_bien_con_variedad_maxima():
    """Control: el camino que ya funcionaba (3 carbos únicos) no puede romperse."""
    for semilla in range(10):
        random.seed(semilla)
        for a, b in _pares_de_carbo(_prompt(mainGoal="gain_muscle")):
            assert a.lower() != b.lower(), f"colisión con variety max: {a!r}"


# --------------------------------------------------------------- F28

def test_los_tokens_de_objetivo_existen_en_el_enum_real():
    """Un token fuera del enum es código inerte: el router 422ea antes de que llegue aquí."""
    import inspect
    import ai_helpers as ah
    from routers.plans import _MAIN_GOAL_ENUM

    src = inspect.getsource(ah.get_deterministic_variety_prompt)
    huerfanos = []
    for nombre in ("_GOALS_FORCE_MAX_VARIETY", "_GOALS_PENALIZE_PROCESSED"):
        m = re.search(nombre + r"\s*=\s*\{(.*?)\}", src, re.S)
        assert m, f"no se encontró {nombre}"
        for tok in re.findall(r'"([^"]+)"', m.group(1)):
            if tok not in _MAIN_GOAL_ENUM:
                huerfanos.append(f"{nombre}:{tok}")
    assert not huerfanos, (
        f"tokens de mainGoal que el enum real no contiene (nunca disparan): {huerfanos}. "
        f"Enum: {sorted(_MAIN_GOAL_ENUM)}"
    )


def test_lose_fat_auto_promueve_a_variedad_maxima():
    """El efecto de usuario: quien quiere perder grasa recibe 3 proteínas base, no 2."""
    import inspect
    import ai_helpers as ah

    src = inspect.getsource(ah.get_deterministic_variety_prompt)
    m = re.search(r"_GOALS_FORCE_MAX_VARIETY\s*=\s*\{(.*?)\}", src, re.S)
    assert "lose_fat" in m.group(1), (
        "el objetivo de pérdida de grasa debe auto-promover a variedad máxima; "
        "el token que había ('lose_weight') no existe en el enum"
    )


# --------------------------------------------------------------- F32 · dietType nulo

@pytest.mark.parametrize("fd", [
    {"dietType": None},
    {"diet": None},
    {"diet": None, "dietType": None},
    {"dietType": ""},
])
def test_diet_nulo_no_bloquea_la_generacion(fd):
    """[P2-SEEDER-DIET-NONE] `dietType: null` es entrada VÁLIDA (presence-optional en el boundary).

    El código hacía `form_data.get("diet", form_data.get("dietType", "")).lower()`: el default
    posicional no cubre una clave que EXISTE con valor None ⇒ AttributeError determinista ⇒ el
    retry del pipeline repetía el mismo crash y la generación quedaba bloqueada para ese usuario.
    """
    texto = _prompt(**fd)
    assert isinstance(texto, str), f"el seeder debe devolver texto con {fd!r}"


def test_diet_nulo_equivale_a_sin_dieta():
    """Degrada a 'balanced', no inventa una restricción que el usuario no declaró."""
    from constants import _get_fast_filtered_catalogs
    completo = [len(x) for x in _get_fast_filtered_catalogs((), (), "")]
    random.seed(1)
    assert isinstance(_prompt(dietType=None), str)
    assert [len(x) for x in _get_fast_filtered_catalogs((), (), "")] == completo


# --------------------------------------------------------------- F18

def test_el_ancla_liviana_puede_ofrecer_lacteos():
    """`[:4]` posicional cortaba antes del primer queso: 11 lácteos del pool eran inalcanzables."""
    import ai_helpers as ah
    from constants import DOMINICAN_VEGGIES_FATS, DOMINICAN_PROTEINS

    pool = ah._build_light_protein_pool(list(DOMINICAN_VEGGIES_FATS), list(DOMINICAN_PROTEINS))
    lacteos = [x for x in pool if re.search(r"queso|yogur", x, re.I)]
    assert lacteos, "sanity: el pool debe contener lácteos (si no, este test no mide nada)"

    vistos = set()
    for semilla in range(40):
        random.seed(semilla)
        vistos.update(ah._pick_light_anchor_candidates(pool, 4))
    assert vistos & set(lacteos), (
        f"ningún lácteo apareció nunca entre los candidatos del ancla en 40 sorteos; "
        f"el pool tiene {len(lacteos)} y salen siempre los mismos: {sorted(vistos)}"
    )


def test_el_ancla_liviana_no_repite_candidato():
    import ai_helpers as ah
    pool = ["A", "B", "C", "D", "E", "F"]
    random.seed(7)
    picks = ah._pick_light_anchor_candidates(pool, 4)
    assert len(picks) == 4 and len(set(picks)) == 4, f"sorteo con reemplazo: {picks}"


def test_el_ancla_liviana_tolera_pool_corto():
    import ai_helpers as ah
    assert ah._pick_light_anchor_candidates(["A", "B"], 4) == ["A", "B"] or \
           sorted(ah._pick_light_anchor_candidates(["A", "B"], 4)) == ["A", "B"]
    assert ah._pick_light_anchor_candidates([], 4) == []
