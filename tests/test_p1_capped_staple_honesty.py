"""[P1-CAPPED-STAPLE-HONESTY · 2026-07-26] La lista decía "compra una sola vez" sobre 6 días de atún.

Medido en el plan vivo `1070ceb1`:

    [SHOPPING MATH] days_len=3 base_scale=2.333 raw_mult=4.0 eff_mult=9.333   ← escala bien a ~28 días
    [P6-CANNED-PROTEIN-CAP] 'Atún en agua' 3547g → 736g (≈4 latas; person_weeks=4.0)

El agregador calculó **3.547 g** de atún para el ciclo de 30 días — correcto, casi exacto a la
necesidad real de las recetas (yo medí 3.705 g por otra vía). Y un cap de realismo lo recortó a 736 g
porque nadie compra 22 latas de una vez. **El recorte es razonable.**

Lo que no lo es: esas 4 latas se muestran bajo el encabezado *«DESPENSA DEL MES — ESTABLES (COMPRA
UNA SOLA VEZ) · Cantidad calculada para todo el periodo — cómpralos una sola vez»*, y cubren **5,5
días**. El usuario compra la lista, cree que está abastecido el mes, y se queda sin la proteína de su
almuerzo en menos de una semana. Sin un solo aviso.

⚠️ Este fix NO sube la cantidad. Comprar 22 latas de atún es peor consejo que comprar 4 y recomprar.
Lo que se arregla es la **mentira**, no el número.

## Por qué el dato ya existía y nadie lo veía

`_record_cap_applied` guarda cada cap en `_CAPS_APPLIED_LAST_RUN` desde P1-CAPS-COHERENCE-RECONCILE,
pero el único consumidor era el coherence guard (para filtrar divergencias legítimas de magnitud). No
llegaba al item de la lista, así que ni la UI ni el PDF podían saberlo. Se adjunta al item y se
sufija en `display_qty`/`display_string`, que frontend y PDF renderean tal cual — honestidad sin
tocar el frontend.

## Cómo apareció

Persiguiendo una sospecha MÍA que resultó equivocada: creí que el agregador no escalaba los estables
a 30 días. Escala bien (`eff_mult=9.333`). El defecto estaba una capa más allá, en lo que la pantalla
afirma sobre un número que sí es correcto.
"""
import pytest

import shopping_calculator as sc


@pytest.fixture(autouse=True)
def _limpio():
    sc.reset_caps_applied_last_run()
    yield
    sc.reset_caps_applied_last_run()


# ───────────── 1. el caso vivo ─────────────

def test_la_fraccion_del_caso_real():
    """3.547 g → 736 g = 20,8% del ciclo ≈ 6 de 30 días. Coincide con los 5,5 días medidos por la
    otra vía (necesidad de recetas ÷ gramos comprados)."""
    frac = 736.0 / 3546.7
    assert 0.19 < frac < 0.22
    assert max(1, int(round(frac * 30))) == 6


def test_el_cap_se_registra_con_su_razon():
    sc._record_cap_applied("Atún en agua", 3546.7, 736.0, "P6-CANNED-PROTEIN-CAP")
    caps = sc.get_caps_applied_last_run()
    assert len(caps) == 1
    assert caps[0]["reason"] == "P6-CANNED-PROTEIN-CAP"
    assert caps[0]["post_value"] < caps[0]["pre_value"]


# ───────────── 2. el item lo dice ─────────────

def _fuente():
    from pathlib import Path
    return (Path(sc.__file__).resolve()).read_text(encoding="utf-8")


def _bloque_del_item():
    """Bloque que adjunta el cap AL ITEM. Se ancla en `_cap_hit = None`, no en el marker: el marker
    aparece antes en el comentario del knob y una ventana desde ahí leía la región equivocada (le
    pasó a la primera versión de este test)."""
    src = _fuente()
    i = src.index("_cap_hit = None")
    return src[i:src.index("if sku_label:", i)]


def test_el_item_lleva_la_metadata_del_cap():
    bloque = _bloque_del_item()
    for campo in ('"capped_by"', '"capped_pre"', '"capped_post"', '"capped_cycle_fraction"'):
        assert campo in bloque, campo


def test_el_sufijo_va_en_los_dos_strings_que_se_renderean():
    """`display_qty` lo usa la UI web y `display_string` el PDF. Sufijar sólo uno deja la mitad
    mintiendo."""
    bloque = _bloque_del_item()
    assert 'result["display_qty"]' in bloque
    assert 'result["display_string"]' in bloque
    assert bloque.count("de 30 días — recompra") == 2


def test_no_sufija_cuando_el_cap_es_marginal():
    """Un recorte del 5% no cambia la decisión de compra; avisar de eso sería ruido."""
    assert "_frac < 0.9" in _bloque_del_item(), "debe haber umbral, no avisar de todo cap"


def test_no_sube_la_cantidad():
    """La corrección es de copy. Si alguien 'arregla' esto subiendo market_qty, el usuario acaba con
    22 latas de atún."""
    bloque = _bloque_del_item()
    assert "market_qty" not in bloque.split("result = {")[0], \
        "el bloque del cap no debe tocar la cantidad"


# ───────────── 3. knob y fail-safe ─────────────

def test_knob_de_rollback():
    assert isinstance(sc.CAPPED_STAPLE_HONESTY, bool)
    src = _fuente()
    assert "MEALFIT_CAPPED_STAPLE_HONESTY" in src
    # El comentario del knob va partido en dos líneas Y la segunda empieza con `#`, así que un
    # `" ".join(src.split())` deja "…con el # knob" y no casa. Se compara una frase contigua real.
    assert "el número NO cambia" in " ".join(src.split()), \
        "el knob apaga el AVISO, no el cap: debe quedar dicho"


def test_registro_de_cap_es_fail_safe():
    """Best-effort: un cap con datos inválidos no debe romper la cadena del agregador."""
    sc._record_cap_applied(None, "no-numero", None, "X")   # no lanza
    sc._record_cap_applied("Atún", 100.0, 50.0, "Y")
    assert any(c.get("reason") == "Y" for c in sc.get_caps_applied_last_run())

# ───────────── 4. EFECTO OBSERVABLE (lo que faltaba) ─────────────
#
# ⚠️ La primera versión de este archivo pasaba entera con el código ROTO. Verificaba que el bloque
# existiera en el fuente y que la aritmética 736/3547 diera "6 de 30 días" — las dos cosas ciertas —
# mientras `strip_accents` lanzaba NameError, un `except` lo tragaba y el sufijo NUNCA llegaba al
# item. Medido con un spy sobre un plan real: `apply_smart_market_units` corría 50 veces y la
# lectura de caps 0.
#
# Estos tests llaman a la función y miran el DICT que devuelve. Un test que comprueba que el código
# está escrito no comprueba que el código haga algo.

def test_el_item_devuelto_LLEVA_el_sufijo():
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Atún en agua", 3546.7, 736.0, "P6-CANNED-PROTEIN-CAP")
    out = sc.apply_smart_market_units("Atún en agua", 736.0 / 453.6, "g", 736.0)
    assert isinstance(out, dict)
    assert out.get("capped_by") == "P6-CANNED-PROTEIN-CAP"
    assert "de 30 días — recompra" in out["display_qty"]
    assert "de 30 días — recompra" in out["display_string"]


def test_el_acento_del_nombre_no_rompe_el_cruce():
    """La causa raíz fue de acentos: el cap guarda 'atún en agua' y el item se llama 'Atún en agua'.
    `strip_accents` tiene que estar DISPONIBLE — en este módulo se importa dentro de cada función,
    no a nivel de módulo, y usarlo como global lanzaba NameError."""
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Limón", 2125.0, 804.0, "P6-CITRUS-CAP")
    out = sc.apply_smart_market_units("Limón", 804.0 / 453.6, "g", 804.0)
    assert out.get("capped_by") == "P6-CITRUS-CAP"


def test_sin_cap_no_hay_sufijo():
    sc.reset_caps_applied_last_run()
    out = sc.apply_smart_market_units("Avena", 2.0, "g", 907.0)
    assert out.get("capped_by") is None
    assert "recompra" not in out["display_qty"]


def test_cap_marginal_no_avisa():
    """96% del ciclo: se registra la metadata pero NO se molesta al usuario."""
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Avena", 1000.0, 960.0, "TEST-CAP")
    out = sc.apply_smart_market_units("Avena", 960.0 / 453.6, "g", 960.0)
    assert out.get("capped_by") == "TEST-CAP"
    assert "recompra" not in out["display_qty"]


def test_idempotente():
    """La lista se recalcula al cambiar household/duración: el sufijo no debe acumularse."""
    sc.reset_caps_applied_last_run()
    sc._record_cap_applied("Atún en agua", 3546.7, 736.0, "P6-CANNED-PROTEIN-CAP")
    for _ in range(3):
        out = sc.apply_smart_market_units("Atún en agua", 736.0 / 453.6, "g", 736.0)
        assert out["display_qty"].count("recompra") == 1


def test_el_fallo_del_lookup_se_LOGUEA_no_se_traga():
    """El `except: pass` original convirtió un NameError en un no-op invisible y me hizo reportar
    como arreglado algo que no hacía nada."""
    src = _fuente()
    i = src.index("_cap_hit = None")
    bloque = src[i:src.index("result = {", i)]
    assert "except Exception as" in bloque
    assert "logging.warning" in bloque
    # el `except` mudo original, sin la variable de la excepción
    assert "except Exception:" not in bloque
