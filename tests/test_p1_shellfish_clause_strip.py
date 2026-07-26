"""[P1-SHELLFISH-CLAUSE-STRIP · 2026-07-26] "Desecha los filetes que no se abran".

Plan vivo `2b3be84e`, D3 «Filete de pescado blanco al Ajillo sobre Puré Cremoso de Ñame»:

    "Lava bien filete de pescado blanco (si son frescos, retira las barbas)"
    "cocina 5-6 minutos, hasta que filete de pescado blanco se abran (desecha los que no se abran)"

Es una receta de **mejillones** cuya proteína se sustituyó sin reescribir las cláusulas del
molusco. El plural del verbo lo delata: `se abran` concuerda con los mejillones originales, no con
el filete. `_rewrite_recipe_steps_after_subs` cambia el SUSTANTIVO —hace lo que fue diseñado a
hacer— pero nadie retira las instrucciones de técnica del alimento viejo.

Un usuario que lea "desecha los que no se abran" concluye que el sistema no sabe cocinar. Es más
dañino para la confianza que cualquier desvío de macros.

## Por qué este SÍ y la concordancia de género NO

Medido sobre 100 comidas de 9 planes: **2 ocurrencias (1%)**, ambas en esa receta. Es una
incidencia parecida a la de los defectos de gramática que descarté hoy — pero la diferencia es la
ambigüedad, no la frecuencia:

  · `se abran` / `retira las barbas` en un plato sin concha **no tienen lectura legítima**. Cero
    falsos positivos posibles.
  · La concordancia de género necesita el núcleo del sintagma («Costilla de Cerdo Guisada» es
    correcto y mi detector lo marcaba). Ahí un fixer determinista rompe más de lo que arregla.

Por eso este pase sólo actúa cuando el plato **no** lleva molusco, y ante la duda no toca nada.
"""
import pytest

import graph_orchestrator as go


_PASO_1 = ("Mise en place: Lava bien filete de pescado blanco (si son frescos, retira las barbas). "
           "Pela el ñame y córtalo en trozos medianos.")
_PASO_2 = ("El Toque de Fuego: agrega filete de pescado blanco y el jugo de limón y tapa la sartén; "
           "cocina 5-6 minutos, hasta que filete de pescado blanco se abran (desecha los que no se abran).")


def _plato(pasos, nombre="Filete de pescado blanco al Ajillo", ings=None):
    return [{"day": 1, "meals": [{"name": nombre, "recipe": list(pasos),
                                  "ingredients": ings or ["1 filete de pescado", "1 pedazo de ñame"]}]}]


# ───────────── 1. el caso reportado ─────────────

def test_quita_el_parentesis_de_las_barbas():
    d = _plato([_PASO_1])
    assert go._strip_shellfish_only_clauses(d) == 1
    s = d[0]["meals"][0]["recipe"][0]
    assert "barbas" not in s
    assert s.startswith("Mise en place: Lava bien filete de pescado blanco.")


def test_reescribe_hasta_que_se_abran():
    d = _plato([_PASO_2])
    assert go._strip_shellfish_only_clauses(d) == 1
    s = d[0]["meals"][0]["recipe"][0]
    assert "se abran" not in s
    assert "desecha" not in s.lower()
    assert "hasta que esté bien cocido" in s


def test_no_deja_espacios_ni_puntuacion_rota():
    d = _plato([_PASO_1, _PASO_2])
    go._strip_shellfish_only_clauses(d)
    for s in d[0]["meals"][0]["recipe"]:
        assert " ." not in s and " ," not in s and "  " not in s, repr(s)
        assert s == s.strip()


def test_idempotente():
    d = _plato([_PASO_1, _PASO_2])
    assert go._strip_shellfish_only_clauses(d) == 2
    assert go._strip_shellfish_only_clauses(d) == 0


# ───────────── 2. la mitad que importa: NO tocar moluscos de verdad ─────────────

@pytest.mark.parametrize("nombre,ings", [
    ("Mejillones al Ajillo", ["500 g de mejillones"]),
    ("Guiso de Almejas", ["300 g de almejas frescas"]),
    ("Chipichipi al vapor", ["400 g de chipichipi"]),
    ("Filete al Ajillo", ["1 filete", "200 g de mejillones"]),   # concha en INGREDIENTES
])
def test_un_plato_CON_concha_no_se_toca(nombre, ings):
    d = _plato([_PASO_1, _PASO_2], nombre=nombre, ings=ings)
    antes = list(d[0]["meals"][0]["recipe"])
    assert go._strip_shellfish_only_clauses(d) == 0
    assert d[0]["meals"][0]["recipe"] == antes


def test_fail_safe_ante_la_duda():
    """Si no se puede decidir, `_meal_has_shellfish` devuelve True y el pase no toca nada."""
    assert go._meal_has_shellfish({"name": None, "ingredients": None}) in (True, False)
    assert go._strip_shellfish_only_clauses(None) == 0
    assert go._strip_shellfish_only_clauses([{"meals": [{"recipe": "no es lista"}]}]) == 0


def test_no_toca_pasos_normales():
    d = _plato(["Mise en place: pica la cebolla y el ajo.",
                "El Toque de Fuego: saltea 5 minutos hasta que esté dorado.",
                "Montaje: sirve caliente."])
    antes = list(d[0]["meals"][0]["recipe"])
    assert go._strip_shellfish_only_clauses(d) == 0
    assert d[0]["meals"][0]["recipe"] == antes


def test_solo_toca_recipe():
    d = _plato([_PASO_1])
    ings_antes = list(d[0]["meals"][0]["ingredients"])
    go._strip_shellfish_only_clauses(d)
    assert d[0]["meals"][0]["ingredients"] == ings_antes


# ───────────── 3. cableado ─────────────

def test_corre_en_el_finalize():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def finalize_plan_data_coherence(")
    j = src.index('return (total, ", ".join(parts))', i)
    assert "_strip_shellfish_only_clauses(days)" in src[i:j]


def test_knob_de_rollback():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'SHELLFISH_CLAUSE_STRIP = _env_bool("MEALFIT_SHELLFISH_CLAUSE_STRIP", True)' in src
