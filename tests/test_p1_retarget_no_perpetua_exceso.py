"""[P1-RETARGET-NO-PERPETUA-EXCESO · 2026-08-05] Regenerar un día no debe fijar su exceso
como objetivo.

EL CASO REAL. El dueño preguntó si sus problemas de macros venían de tener pocos alimentos
en la Nevera. Medido: NO. Al regenerar un día, `/regenerate-day` calculaba

    day_target = suma_del_día_actual
    day_target = max(day_target, meta_real)     # "nunca bajar de la meta"

El `max` protege en UNA dirección. Con su meta de 123 g de proteína y un día que traía
184 g (+50%), el objetivo quedó fijado en 184. El día regenerado salió en 159 g —MÁS CERCA
de su meta real— y el aviso lo declaró «por debajo de tu objetivo», culpó a la Nevera y le
mandó a comprar comida que no necesitaba (`159 < 0.90 * 184`). Contra su meta real, 159
sigue estando un 29% POR ENCIMA.

El techo es `MEALFIT_BAND_SCORE_UPPER` (1.12), la misma banda que el resto del sistema ya
usa para decidir si un día está en objetivo: perpetuar un valor que otro gate considera
fuera de banda es incoherente.

⚠️ Lo que NO cambia: el suelo. La intención original del retarget (P1-REGEN-DAY-RETARGET)
era impedir que un día degradado perpetuara su DÉFICIT, y eso sigue igual.

tooltip-anchor: P1-RETARGET-NO-PERPETUA-EXCESO

[P2-REGEN-DAY-RETARGET-TO-META · 2026-09-03] Secuela: el techo acotaba el OBJETIVO pero no el
RESULTADO, y dentro de la banda el `max` seguía fijando el exceso: 8 actualizaciones seguidas
subieron un día de 2390 a 2688 kcal (meta 2500). Ahora el objetivo ES la meta en ambas
direcciones; el `max` + techo sobreviven solo bajo `MEALFIT_REGEN_DAY_RETARGET_TO_META=0`.
Los asserts de abajo que hablaban del techo se ejecutan en ese modo legacy.
"""
import os
from unittest import mock

import pytest

import routers.plans as rp


# Los números del caso real del owner.
META_PROT = 123.0
DIA_PREVIO_PROT = 184.0


# ⚠️ Se llama a PRODUCCIÓN, no a una réplica.
#
# La primera versión de este fichero definía aquí su propia copia de la aritmética. Al
# mutar el código de producción (borrar el techo), 8 de estos 9 tests seguían en VERDE:
# verificaban mi reimplementación, no el sistema. Lo único que cazó la mutación fue el
# parser-test, que es la clase más débil de verificación.
#
# Por eso `_retarget_macro_target` vive en `routers/plans.py` y no aquí.
_objetivo = rp._retarget_macro_target


# ------------------------------------------------- el techo

def test_un_dia_excedido_no_fija_su_exceso_como_objetivo():
    """EL test de este P-fix, con los números medidos en producción."""
    obj = _objetivo(DIA_PREVIO_PROT, META_PROT)
    assert obj < DIA_PREVIO_PROT, (
        "el objetivo sigue siendo el exceso del día previo: el motor volverá a "
        "reproducirlo y el aviso volverá a culpar a la Nevera"
    )
    assert obj == pytest.approx(META_PROT)  # [P2-REGEN-DAY-RETARGET-TO-META] la meta, ya no el techo
    with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_TO_META": "0"}):
        assert _objetivo(DIA_PREVIO_PROT, META_PROT) == pytest.approx(META_PROT * 1.12, rel=1e-6)  # legacy


def test_el_dia_que_el_owner_vio_ya_no_dispara_el_aviso():
    """El aviso salta con `nuevo < 0.90 * objetivo`. Con 159 g generados debe callarse."""
    obj = _objetivo(DIA_PREVIO_PROT, META_PROT)
    generado = 159.0
    assert generado >= 0.90 * obj, (
        "159 g siguen contando como fracaso pese a estar MÁS CERCA de la meta que los 184 "
        "de partida"
    )


# ------------------------------------------------- el suelo NO cambia

def test_un_dia_deficitario_sigue_subiendo_a_la_meta():
    """⚠️ La razón por la que el retarget existe (P1-REGEN-DAY-RETARGET) no se toca.

    Un día 'gain_muscle clavado a ~100 g' debe seguir regenerándose contra la meta, no
    contra su propio déficit.
    """
    assert _objetivo(100.0, META_PROT) == pytest.approx(META_PROT)
    assert _objetivo(0.0, META_PROT) == pytest.approx(META_PROT)


def test_un_dia_ya_en_banda_apunta_a_la_meta():
    """[P2-REGEN-DAY-RETARGET-TO-META] Dentro de la banda TAMBIÉN se apunta a la meta:
    perpetuar un exceso «pequeño» era exactamente la deriva (cada actualización adoptaba
    el exceso de la anterior hasta el techo). En legacy, idéntico al de antes."""
    for suma in (123.0, 130.0, 137.0):
        assert _objetivo(suma, META_PROT) == pytest.approx(META_PROT)
    with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_TO_META": "0"}):
        for suma in (123.0, 130.0, 137.0):
            assert _objetivo(suma, META_PROT) == pytest.approx(max(suma, META_PROT))


# ------------------------------------------------- el knob

def test_el_default_es_la_banda_del_sistema():
    entorno = {k: v for k, v in os.environ.items()
               if k not in ("MEALFIT_REGEN_DAY_RETARGET_BAND_CEILING", "MEALFIT_BAND_SCORE_UPPER")}
    with mock.patch.dict(os.environ, entorno, clear=True):
        assert rp._retarget_band_ceiling() == pytest.approx(1.12)


def test_rollback_a_cero_restaura_el_max_puro():
    with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_BAND_CEILING": "0",
                                      "MEALFIT_REGEN_DAY_RETARGET_TO_META": "0"}):
        assert rp._retarget_band_ceiling() == 0.0
        assert _objetivo(DIA_PREVIO_PROT, META_PROT) == pytest.approx(DIA_PREVIO_PROT)


def test_el_techo_nunca_puede_recortar_por_debajo_de_la_meta():
    """⚠️ Un techo < 1.0 convertiría este arreglo en el bug simétrico: recortaría el
    objetivo POR DEBAJO de la meta, que es exactamente lo que el retarget impide."""
    for basura in ("0.5", "0.99", "-3"):
        with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_BAND_CEILING": basura}):
            c = rp._retarget_band_ceiling()
            assert c == 0.0 or c >= 1.0, basura


def test_un_knob_ilegible_no_revienta_la_regeneracion():
    for basura in ("abc", "", "  "):
        with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_BAND_CEILING": basura}):
            assert rp._retarget_band_ceiling() >= 1.0


# ------------------------------------------------- el call site

def test_el_bucle_del_retarget_usa_el_helper():
    """Lo ÚNICO que un parser puede aportar aquí: que el endpoint esté cableado al helper.

    El suelo y el techo ya se verifican EJECUTANDO `_retarget_macro_target` en los tests
    de arriba. Este solo cierra el hueco que quedaría si alguien dejara el helper intacto
    y volviera a poner un `max` a mano dentro del bucle.
    """
    import inspect
    src = inspect.getsource(rp)
    fin = src.index("day_target[_kk] = _t")
    ini = src.rindex("for _kk in day_target:", 0, fin)
    ventana = src[ini:fin]
    assert "_retarget_macro_target(" in ventana, (
        "el bucle del retarget dejó de usar el helper: el techo puede estar sin aplicarse "
        "en producción aunque los tests de arriba sigan verdes"
    )


def test_el_helper_conserva_el_suelo_Y_el_techo():
    """Ambos lados, ejecutados. El suelo es la razón por la que el retarget existe."""
    assert _objetivo(50.0, META_PROT) == pytest.approx(META_PROT)          # suelo
    assert _objetivo(999.0, META_PROT) == pytest.approx(META_PROT)         # [TO-META] la meta
    with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_TO_META": "0"}):
        assert _objetivo(999.0, META_PROT) == pytest.approx(META_PROT * 1.12)  # techo (legacy)
