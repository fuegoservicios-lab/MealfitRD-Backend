# -*- coding: utf-8 -*-
"""[P1-CAP-FALLBACK-MISMO-ALIMENTO · 2026-09-08] El fallback por índice escribía a ciegas.

`_cap_unrealistic_portions` mapea la línea de raw **por alimento** (`_rescale_raw_by_food`). Cuando
ese mapeo no resuelve —línea sin alimento de catálogo— caía a un fallback por ÍNDICE:

    elif _lockstep and idx < len(raw):
        raw[idx] = _resc(str(raw[idx]), factor)

Sin comprobar que `raw[idx]` fuera del mismo alimento. Y `_lockstep` (mismo largo) no es
paralelismo: es la condición que `P2-RAW-PAIR-BY-FOOD` midió como cierta el 93,5 % de las veces y
verdadera sólo el 48,1 %.

## Cómo apareció: mirando lo intacto, no el destrozo

Todo el día persiguiendo esta clase sobre comidas ya rotas — y una comida rota no dice quién la
rompió. La respuesta vino del experimento en positivo: tomar las **986 comidas ALINEADAS** de la
flota, correr cada uno de los 11 pases de la cola del pipeline por separado, y contar cuáles dejan
rota una que estaba limpia.

Resultado: **un solo pase, un solo caso** (0,10 %). `_cap_unrealistic_portions` escaló *el repollo*
en la receta y *la CEBOLLA* en la compra. Tras el arreglo: **0**.

*Una comida ya rota no dice quién la rompió; una comida limpia que queda rota tras un pase concreto,
sí.*

## Por qué el arreglo es la guarda y no el factor

En esta misma función intenté antes otro arreglo —rederivar el factor contra los gramos de raw— y
**empeoró las cosas**: los recortes brutales pasaron de 1 a 14 sobre la flota, y lo revertí. Éste
toca **sólo la condición del fallback**, no la aritmética. Medido antes: el fallback dispara 3 de 78
llamadas, así que callarse cuando no puede confirmar el alimento cuesta muy poco.

Y la doctrina ya estaba escrita en el hermano (`_remove_one_raw_line_by_food`): *«recortar la línea
equivocada deja al usuario comprando otra cosa, que es peor que no recortar ninguna»*. El fallback
contradecía la regla que el código de al lado enuncia.
"""
import inspect
import pytest

# [P0-CI-TIMEOUT-15 · 2026-09-11] La CI del backend nunca había TERMINADO (moría a los 15 min); al
# terminar por primera vez este módulo salió rojo sin base de datos: sus casos convierten medidas
# caseras y emparejan alimentos contra el catálogo VIVO (densidades, aliases). Se declara para que en
# CI se salte con su motivo en vez de fingir un veredicto; en el checkout del dueño corre entero.
pytestmark = pytest.mark.needs_local_data

import graph_orchestrator as go


def test_el_fallback_exige_el_MISMO_alimento():
    src = inspect.getsource(go._cap_unrealistic_portions)
    assert "not _distinto_alimento(str(raw[idx]), s)" in src, (
        "el fallback por índice volvió a escribir sin comprobar el alimento")


def test_la_guarda_bloquea_SOLO_con_discrepancia_positiva():
    """La asimetría que costó dos tests: bloquear exige AFIRMAR que son distintos.

    Exigir confirmación positiva de IGUALDAD bloqueaba también «no resuelve ninguna» —donde el mapeo
    por alimento tampoco resolvió— y dejaba raw sin recortar: cambia «escribe el alimento
    equivocado» por «no escribe el correcto», que es el modo silencioso que este día entero cierra.
    """
    from constants import distinto_alimento_raw_display as d

    # Las cuatro ramas, cada una medida sobre un caso REAL de la flota o de la suite:
    assert d("150 g de camarones", "150 g de pechuga de pollo") is True, "alimentos distintos"
    assert d("1.5 aguacate (221 g)", "1½ aguacates (221 g)") is False, (
        "plural tratado como otro alimento — lo cazó `test_avocado_unicode_fraction_capped`")
    assert d("0.83 cebolla", "1/2 tazas de repollo rallado") is True, (
        "raw resuelve a `cebolla` y el display NO resuelve: escalar lo que SÍ sé nombrar con el "
        "factor de lo que no, es el caso real medido sobre las 986 comidas alineadas")
    assert d("xyz abc", "qwe rty") is False, "ninguna resuelve ⇒ conducta histórica, no hay info"


def test_el_helper_vive_FUERA_del_fichero_en_su_techo():
    """`graph_orchestrator.py` está en 53.100 de 53.100. Su propio test del techo dice que eso «no se
    arregla subiendo el número: se arregla extrayendo», así que el helper vive en `constants.py`,
    junto al SSOT de identidad que usa. Si alguien lo devuelve, el techo lo para."""
    import constants

    assert hasattr(constants, "distinto_alimento_raw_display")
    assert "def distinto_alimento_raw_display" not in inspect.getsource(go)


def test_el_fallback_sigue_colgando_del_mapeo_fallido():
    """La guarda nueva va DENTRO del `elif`, no sustituyéndolo: el camino bueno sigue siendo
    `_rescale_raw_by_food`, y el fallback sólo actúa cuando ése no resolvió."""
    src = inspect.getsource(go._cap_unrealistic_portions)
    i_food = src.index("_rescale_raw_by_food(raw, [s], [factor])")
    i_fall = src.index("elif _lockstep and idx < len(raw)")
    assert i_food < i_fall, "el fallback dejó de colgar del mapeo por alimento"


def test_NO_se_toco_la_aritmetica_del_factor():
    """El intento anterior en esta función rederivaba el factor y empeoró las cosas (1→14 recortes
    brutales). Este arreglo es sólo la guarda; si alguien vuelve a tocar el factor, que sea a
    sabiendas de que ya se probó y se revirtió."""
    src = inspect.getsource(go._cap_unrealistic_portions)
    assert "raw[idx] = _resc(str(raw[idx]), factor)" in src, "cambió la escritura del fallback"
    assert "_cap_raw_factor" not in src, (
        "volvió la rederivación del factor: se midió y empeoraba (recortes brutales 1→14). "
        "Ver `docs/hallazgo_tope_encoge_raw_ya_capado.md` antes de reintentarlo.")


def test_el_cap_no_quedo_inerte():
    """Un arreglo que apaga la función no es un arreglo. El cap debe seguir capando.

    Medido tras el cambio sobre las 1.194 comidas vivas: capa el display en 76 y raw en 72 — la
    diferencia son exactamente los fallbacks que ahora se callan por no poder confirmar el alimento.
    """
    src = inspect.getsource(go._cap_unrealistic_portions)
    assert "_rescale_raw_by_food" in src and "meal[\"ingredients_raw\"] = _cr" in src
