"""[P2-TRACKING-COUNTRY-INERT · 2026-08-21] En modo contador el país se preguntaba y no hacía nada.

La auditoría lo midió así: `plan_mode.py`, `routers/diary.py`, `routers/user_data.py` y
`food_search.py` tienen **cero** menciones de `country`. La rama contador del wizard sí lo pregunta
—el step está en `_trackingSteps`— así que era una pregunta write-only: se contesta, se persiste, y
no gobierna nada. El patrón inerte que este repo ya pagó dos veces.

QUÉ CAMBIÓ, Y NO FUE AQUÍ. Tres P-fixes de esta misma ola le dieron trabajo al país en superficies
que el modo contador SÍ usa:

  · `suggest_foods_for_nutrient` (P2-SUGGEST-FOODS-COUNTRY) — el coach está disponible en modo
    contador, y era la tool que le ofrecía chiles mexicanos a un español.
  · el aviso de calibración del escáner (P2-VISION-COUNTRY-COPY) — el diario se llena escaneando.
  · `_local_date_str_for_user` (P2-LOCAL-DATE-STR-UTC4) — el huso con el que se corta el día del
    diario y del contador de agua.

O sea que el país dejó de ser inerte por el camino, no por un arreglo dirigido. Este fichero lo
ancla: si alguien deshace esos cableados, vuelve a serlo y nadie se entera.

LO QUE SÍ SE ARREGLÓ AQUÍ ES LA PROMESA. El step del país dice «Adapta tus platos, medidas y —donde
ya está listo— los precios del súper». En la rama contador **no hay platos generados, ni lista de la
compra, ni precios**: se le hacía una pregunta al usuario con una promesa que esa rama no puede
cumplir. La misma rama ya sobreescribe el copy de otro step por esta razón exacta («en esta rama no
hay plan por diseño»); el del país se le empareja.

Y LO QUE DELIBERADAMENTE SIGUE SIN PAÍS: el catálogo del diario. `P2-DIARY-CATALOG-COUNTRY` lo
decidió con su razón — el diario es RETROSPECTIVO, así que filtrar le quitaría a un dominicano en
España el plato que se acaba de comer. Que `food_search.py` y `routers/diary.py` sigan en cero
menciones no es deuda: es esa decisión.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FLOW = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
         / "InteractiveAssessmentFlow.jsx")


@pytest.fixture(scope="module")
def flow() -> str:
    if not _FLOW.is_file():
        pytest.skip("InteractiveAssessmentFlow.jsx no está en este árbol")
    return _FLOW.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def rama_contador(flow) -> str:
    i = flow.index("const _trackingSteps = [")
    # El array cierra con `].filter(Boolean);`, no con `];` — el `.filter` está ahí porque los
    # steps gateados por knob se insertan como spreads que pueden quedar vacíos. Cortar por el
    # cierre equivocado daba `ValueError: substring not found` y los tres casos ni llegaban a
    # ejecutarse: un fixture que revienta no informa, sólo hace ruido de otro color.
    j = flow.index("].filter(Boolean);", i)
    return flow[i:j]


# ── La promesa del step ─────────────────────────────────────────────────────────────────────────

def test_la_rama_contador_pregunta_el_pais(rama_contador):
    """Si dejara de preguntarlo, este P-fix sobra — y las tres superficies de abajo se quedarían
    con el default. Que falle aquí obliga a mirar, no a borrar el test."""
    assert "_byField('country')" in rama_contador


def _subtitulo_del_pais(rama_contador: str) -> str:
    """El VALOR del `subtitle:` del step de país en la rama contador.

    La primera versión escaneaba una ventana de 900 chars alrededor del step, y ahí dentro caía mi
    PROPIO comentario, que cita el copy viejo («…los precios del súper») para explicar por qué se
    sobreescribe. El guard acusaba a mi comentario. Es la enésima vez que un comentario derrota a
    un guard en este repo, y van varias mías en esta ola: el remedio, siempre el mismo, es anclar
    en la forma que SÓLO el código tiene — aquí, la llamada `t('…')` del propio campo."""
    i = rama_contador.index("_byField('country')")
    m = re.search(r"subtitle:\s*t\((['\"])(.+?)\1", rama_contador[i:i + 600], re.S)
    assert m, "el step del país en la rama contador no sobreescribe el subtítulo"
    return m.group(2)


def test_el_subtitulo_no_promete_lo_que_esa_rama_no_tiene(rama_contador):
    """En la rama contador no hay platos generados, ni lista de la compra, ni precios. El copy del
    step real promete las tres cosas."""
    sub = _subtitulo_del_pais(rama_contador)
    for promesa in ("precios", "súper", "platos"):
        assert promesa not in sub.lower(), (
            f"el subtítulo de la rama contador promete «{promesa}», que ahí no existe: {sub!r}"
        )


def test_el_subtitulo_dice_lo_que_el_pais_SI_hace_ahi(rama_contador):
    """Un copy honesto no es sólo uno que no miente: también tiene que decirle al usuario por qué
    merece la pena contestar. Si no gobernara nada, la pregunta sobraría."""
    assert "coach" in _subtitulo_del_pais(rama_contador).lower(), (
        "el subtítulo no dice para qué sirve el país en esta rama"
    )


# ── El país ya no es inerte: las tres superficies ───────────────────────────────────────────────

def test_el_coach_filtra_sus_sugerencias_por_pais():
    """P2-SUGGEST-FOODS-COUNTRY. El coach está disponible en modo contador."""
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8", errors="replace")
    i = src.index("def suggest_foods_for_nutrient")
    j = src.index("\n@tool", i + 1)
    assert "country_for_form_data" in src[i:j], (
        "`suggest_foods_for_nutrient` dejó de derivar el país: vuelve a ofrecerle comida de otro "
        "país a un usuario de modo contador"
    )


def test_el_escaner_avisa_de_su_calibracion():
    """P2-VISION-COUNTRY-COPY. El diario del modo contador se llena escaneando."""
    modal = (_BACKEND.parent / "frontend" / "src" / "components" / "dashboard"
             / "ScanMealModal.jsx")
    if not modal.is_file():
        pytest.skip("ScanMealModal.jsx no está en este árbol")
    assert "coerceCountry" in modal.read_text(encoding="utf-8", errors="replace")


def test_el_dia_del_diario_se_corta_con_el_huso_del_usuario():
    """P2-LOCAL-DATE-STR-UTC4. El país no fija el huso directamente, pero es la misma pregunta de
    «dónde vives» y el mismo síntoma: un día cortado donde no toca."""
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8", errors="replace")
    i = src.index("def _local_date_str_for_user")
    j = src.index("\n@tool", i)
    assert "user_tz_offset_min" in src[i:j], (
        "el helper de fecha local volvió a un huso fijo: el diario y el agua del modo contador "
        "cortarían el día en la hora de otro"
    )


# ── Lo que deliberadamente sigue sin país ───────────────────────────────────────────────────────

@pytest.mark.parametrize("modulo", ["food_search.py", "routers/diary.py"])
def test_el_catalogo_del_diario_sigue_sin_pais_a_proposito(modulo):
    """No es deuda pendiente: es `P2-DIARY-CATALOG-COUNTRY`. El diario es RETROSPECTIVO, así que
    filtrar por país le quitaría a un dominicano en España el plato que se acaba de comer. Se ancla
    para que una lectura futura de «0 menciones de country» no lo confunda con un olvido."""
    src = (_BACKEND / modulo).read_text(encoding="utf-8", errors="replace")
    assert "country_for_form_data" not in src, (
        f"{modulo} empezó a filtrar por país — ver P2-DIARY-CATALOG-COUNTRY antes de darlo por "
        f"bueno: el diario registra lo que YA comiste"
    )
