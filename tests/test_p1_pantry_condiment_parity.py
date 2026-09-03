"""[P1-PANTRY-CONDIMENT-PARITY · 2026-08-22] El prompt promete condimentos que el validador no exime.

## La contradicción, literal

`build_pantry_correction_context` ([prompts/plan_generator.py](../prompts/plan_generator.py))
le dice al modelo, en el bloque de CORRECCIÓN OBLIGATORIA:

> «Condimentos básicos (sal, pimienta, aceite, ajo, **cebolla**, cilantro) están siempre
> permitidos.»

Y el prompt del catálogo ([graph_orchestrator.py](../graph_orchestrator.py)) añade
«comino, cúrcuma, laurel, tomillo, curry, cebolla en polvo» y una EXCEPCIÓN explícita de
repostería: «SÍ puedes usar polvo de hornear, levadura, bicarbonato y vainilla … aunque
no estén en la lista».

`constants._ALLOWED_CONDIMENTS` —la tupla que decide qué NO tiene que estar en la
nevera— tenía **once** palabras y no incluía **ninguna** de esas.

Consecuencia medida en el plan real `2245eb45`: el bloque de 4 días murió con
«Ingredientes COMPLETAMENTE INEXISTENTES en inventario: ¼ cdta de polvo de hornear,
½ cdta de comino, 1 cebolla, ½ hoja de laurel». El sistema autorizó por escrito
exactamente lo que después castigó. El modelo obedeció y perdió.

## Por qué se arregla del lado del validador

Porque el prompt es el **contrato ofrecido**: si el sistema promete algo y luego lo
penaliza, el fallo está en quien juzga, no en quien obedeció. Y porque un cuarto de
cucharadita de polvo de hornear tumbaba cuatro días de menú ya pagados al LLM.

## Lo que este archivo NO hace

No fusiona `_ALLOWED_CONDIMENTS` con `culinary_coherence.CONDIMENT_EXEMPT`. Parecen la
misma lista y no lo son: la primera responde «¿tiene que existir este ingrediente en la
nevera?» y la segunda «¿necesita este ingrediente un método de cocción?». Colapsarlas
sería escribir la cuarta tabla que `P1-DIET-CANON-SSOT` prohíbe, y por la peor de las
razones: que se parecen.

Tampoco toca el gate de CANTIDAD. Exentar a la cebolla del gate no la borra de la lista
de compras — el agregador la sigue costeando (aparece en la lista de 48 ítems del plan
del incidente). La exención sólo significa «no te niegues a cocinar porque falte», que
es justo lo que el prompt prometía.
"""
import re

import pytest

from constants import _ALLOWED_CONDIMENTS, validate_ingredients_against_pantry


# Lo que los prompts AUTORIZAN por su nombre. Cada entrada lleva de dónde sale, para que
# quien la borre del prompt sepa que debe borrarla aquí (y viceversa).
AUTORIZADOS_POR_EL_PROMPT = {
    "sal": "plan_generator.build_pantry_correction_context",
    "pimienta": "plan_generator.build_pantry_correction_context",
    "aceite": "plan_generator.build_pantry_correction_context",
    "ajo": "plan_generator.build_pantry_correction_context",
    "cebolla": "plan_generator.build_pantry_correction_context + day_generator:44",
    "cilantro": "plan_generator.build_pantry_correction_context",
    "perejil": "day_generator:44",
    "oregano": "day_generator:44",
    "comino": "graph_orchestrator:5430 (P1-SPICES-CATALOG-SYNC)",
    "curcuma": "graph_orchestrator:5430",
    "laurel": "graph_orchestrator:5430",
    "tomillo": "graph_orchestrator:5430",
    "curry": "graph_orchestrator:5430",
    "polvo de hornear": "graph_orchestrator:5440 (P1-BAKING-STAPLES)",
    "levadura": "graph_orchestrator:5440",
    "bicarbonato": "graph_orchestrator:5440",
    "vainilla": "graph_orchestrator:5440",
}


class TestParidadPromptValidador:
    @pytest.mark.parametrize("condimento", sorted(AUTORIZADOS_POR_EL_PROMPT))
    def test_lo_que_el_prompt_promete_el_validador_lo_exime(self, condimento):
        origen = AUTORIZADOS_POR_EL_PROMPT[condimento]
        assert condimento in _ALLOWED_CONDIMENTS, (
            f"«{condimento}» está autorizado en {origen} pero el validador lo trata como "
            f"alimento inexistente. Autorizar algo y castigarlo luego fue lo que mató al "
            f"chunk 2 del plan 2245eb45."
        )

    def test_el_caso_literal_del_incidente(self):
        """Las 4 líneas que aparecían en la violación real ya no bloquean."""
        nevera = ["Huevo", "Papa", "Espinacas", "Avena", "Yogurt"]
        for linea in ("¼ cdta de polvo de hornear", "½ cdta de comino",
                      "1 cebolla", "½ hoja de laurel"):
            veredicto = validate_ingredients_against_pantry(
                [linea], nevera, strict_quantities=False
            )
            assert veredicto is True, f"«{linea}» sigue bloqueando: {veredicto}"

    def test_un_alimento_de_verdad_sigue_bloqueando(self):
        """La exención es de CONDIMENTOS. Si esto pasa, la abrimos demasiado."""
        nevera = ["Huevo", "Papa"]
        veredicto = validate_ingredients_against_pantry(
            ["200 g de pechuga de pollo"], nevera, strict_quantities=False
        )
        assert veredicto is not True, "el pollo NO es un condimento"


class TestNoAbrimosLaPuertaDeMas:
    """El repo ha cerrado 15 bugs de subcadena («sal»⊂«salsa», «pollo»⊂«repollo»).

    La tupla se compila a regex con `\\b` y plural opcional; estos casos anclan que
    añadir palabras no reintrodujo la clase.
    """

    @pytest.mark.parametrize("impostor", [
        "200 g de salchicha",      # contiene "sal"
        "150 g de repollo",        # contiene "pollo"
        "100 g de cebollín",       # contiene "cebolla"? NO: difiere en la 7ª letra
        "2 cdas de currywurst",    # contiene "curry"
    ])
    def test_no_se_cuela_por_subcadena(self, impostor):
        veredicto = validate_ingredients_against_pantry(
            [impostor], ["Huevo"], strict_quantities=False
        )
        assert veredicto is not True, (
            f"«{impostor}» pasó como condimento — vuelve la clase de bug de subcadena"
        )

    def test_las_regex_exigen_palabra_completa(self):
        from constants import _ALLOWED_CONDIMENTS_RES
        assert len(_ALLOWED_CONDIMENTS_RES) == len(_ALLOWED_CONDIMENTS)
        for rx in _ALLOWED_CONDIMENTS_RES:
            assert rx.pattern.startswith(r"\b"), f"{rx.pattern} sin borde izquierdo"
            assert rx.pattern.endswith(r"\b"), f"{rx.pattern} sin borde derecho"


class TestNoNaceUnaCuartaTabla:
    def test_no_fusionamos_con_la_lista_culinaria(self):
        """Responden preguntas distintas; parecerse no es razón para colapsarlas."""
        from culinary_coherence import CONDIMENT_EXEMPT
        assert set(_ALLOWED_CONDIMENTS) != set(CONDIMENT_EXEMPT), (
            "si alguien las igualó, comprobó el parecido y no el propósito: una decide "
            "existencia en nevera, la otra si hace falta método de cocción"
        )
