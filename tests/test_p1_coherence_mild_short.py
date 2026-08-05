"""[P1-COHERENCE-MILD-SHORT · 2026-08-05] La banda 0.5-0.9 deja de ser `unknown`.

POR QUÉ EXISTE. `unknown` es la etiqueta de "no sé qué pasó aquí". Medido sobre el
historial persistido de **25 planes / 228 evaluaciones**: era el **28,2% de TODAS
las hipótesis** (202 de 717), el segundo bucket más grande. Y su forma no era una
nube dispersa: de las incógnitas con ratio registrado, **128 de 130 (98,5%) caían
en la banda 0.5-0.9** — el hueco entre el umbral de overdeduct (0.5) y la
tolerancia (~0.9), donde ninguna rama de la cascada llegaba.

`_bucket_unknown_magnitude_ratios` (P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY) existe
justo para exigir esa evidencia antes de inventar categorías; el código advierte
"NO añadir categorías sin ver la FORMA de esos ratios". Aquí está la forma.

Mismo linaje que `P1-COHERENCE-UNQUANTIFIED-LABEL`, que rebautizó 831 de 879
divergencias sin tocar comportamiento: **es SOLO una etiqueta**.

⚠️ NO es accionable. Comprar un 20% por debajo es ruido de envase y redondeo — meterlo
en el banner inundaría al usuario de avisos que no puede accionar, que es justo lo
que `P1-COHERENCE-BANNER-NOISE` cerró.

tooltip-anchor: P1-COHERENCE-MILD-SHORT
"""
import pytest

from shopping_calculator import _classify_divergence_hypothesis


def _clasificar(exp, act, **kw):
    """Solo magnitud: sin fricción de unidades (mismas units a ambos lados)."""
    units = {"g": exp}
    return _classify_divergence_hypothesis(
        exp_qty=exp, act_qty=act, exp_units=units, act_units={"g": act},
        food=kw.get("food", "Pepino"),
        pantry_deduction_applied=kw.get("pantry_deduction_applied", True),
    )


@pytest.mark.parametrize("ratio", [0.55, 0.7, 0.85, 0.89])
def test_la_banda_media_ya_tiene_nombre(ratio):
    """0.5-0.9: por debajo de la receta pero lejos del sub-suministro severo."""
    assert _clasificar(1000.0, 1000.0 * ratio) == "magnitude_mild_short"


@pytest.mark.parametrize("ratio", [0.15, 0.25, 0.45])
def test_el_caso_severo_no_cambia(ratio):
    """<50% sigue siendo lo que era — esta etiqueta no le roba casos.

    Se evitan 0.30-0.40 a propósito: ahí manda `yield_uncovered` (caso 3, ratio
    típico de proteína cocida), que tiene precedencia sobre el caso 4. Mapeado
    empíricamente antes de escribir el test — mi primera versión asumía 0.30 y
    fallaba por la razón equivocada.
    """
    assert _clasificar(1000.0, 1000.0 * ratio) == "pantry_overdeduct"
    assert _clasificar(1000.0, 1000.0 * ratio,
                       pantry_deduction_applied=False) == "magnitude_undersupply"


def test_el_borde_del_umbral_pertenece_al_severo():
    """En 0.5 manda el caso 4; por encima, la banda nueva."""
    assert _clasificar(1000.0, 490.0) == "pantry_overdeduct"
    assert _clasificar(1000.0, 510.0) == "magnitude_mild_short"


def test_comprar_de_mas_no_cae_aqui():
    """La etiqueta es de compra CORTA. Comprar de más sigue sin nombre.

    Deliberado: de las 130 incógnitas con ratio medido, solo 2 estaban por
    encima de 1. Nombrar esa cola con dos casos sería justo lo que el código
    advierte que no se haga — inventar una categoría sin ver su forma.
    """
    assert _clasificar(1000.0, 1500.0) != "magnitude_mild_short"


def test_no_es_accionable():
    """No puede aparecer en el banner del usuario.

    El set accionable vive en `summarize_divergences_for_ui`; si alguien mete
    esta hipótesis ahí, el usuario recibe un aviso por cada envase redondeado.
    """
    import io
    from pathlib import Path
    src = io.open(Path(__file__).resolve().parents[1] / "shopping_calculator.py", encoding="utf-8").read()
    ini = src.index("_ACTIONABLE_HYPOTHESES = {")
    fin = src.index("}", ini)
    bloque = src[ini:fin]
    assert "magnitude_mild_short" not in bloque, (
        "`magnitude_mild_short` entró en el set accionable: comprar un 20% por debajo "
        "es ruido de envase, no algo que el usuario deba corregir a mano."
    )


def test_el_frontend_tiene_su_etiqueta():
    """Sin label es-DO, el test de deriva cross-language rompe."""
    import io
    from pathlib import Path
    js = Path(__file__).resolve().parents[2] / "frontend" / "src" / "utils" / "coherenceLabels.js"
    if not js.exists():
        pytest.skip("repo frontend no disponible en este checkout")
    assert "magnitude_mild_short:" in io.open(js, encoding="utf-8").read()


def test_hereda_la_exencion_de_unknown():
    """No puede escalar a bloqueo: viene de `unknown`, que es exento.

    Sin la exencion el reetiquetado NO seria "solo una etiqueta": esas
    divergencias pasarian de exentas a candidatas a forzar retry. Hoy no
    escalarian por un margen fino -la banda produce |delta| <= 0.49 y el check
    exige > 0.50- pero eso es coincidencia aritmetica: bajar
    MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD a 0.3 ensancha la banda hasta
    |delta| 0.7 y empezaria a escalar. Seria el modo de fallo que
    P1-COHERENCE-SEVERE-NO-NOISE cerro.
    """
    import io as _io
    from pathlib import Path
    src = _io.open(Path(__file__).resolve().parents[1] / "shopping_calculator.py",
                   encoding="utf-8").read()
    ini = src.index("_exempt_hypotheses = [")
    fin = src.index("]", ini)
    assert "magnitude_mild_short" in src[ini:fin], (
        "`magnitude_mild_short` no esta exento: heredo el bucket de `unknown` pero "
        "no su exencion, asi que podria forzar retries."
    )
