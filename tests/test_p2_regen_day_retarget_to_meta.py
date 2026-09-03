"""[P2-REGEN-DAY-RETARGET-TO-META · 2026-09-03] Al regenerar un día, el objetivo ES la meta
del perfil, en ambas direcciones.

EL CASO REAL (journal + plan persistido, 2026-09-03). El dueño pulsó «Actualizar platos del
día» 8 veces seguidas. El objetivo de cada macro era `max(suma_actual, meta)` con techo
1.12×meta (P1-RETARGET-NO-PERPETUA-EXCESO). Mientras el día venía por DEBAJO de la meta
apuntaba a 2500 kcal; desde la 5ª corrida el día quedó por ENCIMA y se convirtió en su
propio objetivo, y cada corrida adoptó el pequeño exceso de la anterior:

    objetivo kcal  2500 2500 2500 2500 → 2536 → 2534 → 2622 → 2704
    resultado kcal 2390 2394 2370 2379 → 2536 → 2534 → 2622 → 2704 → 2688
    carbos (meta 334 g): 321 → 387 g   (+16 %, POR ENCIMA del techo: éste acota el
                                        objetivo, no el resultado)

Y el band score (`compute_clinical_band_score` contra `day_target`) reportó 1.0 en todas:
se medía contra el objetivo desplazado. La regla nueva vive en `_retarget_macro_target`
(SSOT ejecutable): meta > 0 ⇒ meta; sin meta ⇒ suma actual (fail-open previo al retarget).
Knob `MEALFIT_REGEN_DAY_RETARGET_TO_META` (default on); en off vuelve `max` + techo.
tooltip-anchor: P2-REGEN-DAY-RETARGET-TO-META
"""
import os
import re
from pathlib import Path
from unittest import mock

import pytest

import routers.plans as rp

_BACKEND = Path(__file__).resolve().parents[1]
_objetivo = rp._retarget_macro_target

META = {"kcal": 2500.0, "protein_g": 135.0, "carbs_g": 334.0, "fats_g": 69.0}
DIA_DERIVADO = {"kcal": 2688.0, "protein_g": 141.0, "carbs_g": 387.0, "fats_g": 68.0}
DIA_DEFICITARIO = {"kcal": 2390.0, "protein_g": 122.0, "carbs_g": 321.0, "fats_g": 71.0}


def test_el_dia_derivado_del_owner_vuelve_a_apuntar_a_su_meta():
    """Los números persistidos el 2026-09-03: 2688 kcal / 387 g de carbos contra 2500 / 334."""
    for k, meta in META.items():
        assert _objetivo(DIA_DERIVADO[k], meta) == pytest.approx(meta), k


def test_el_suelo_de_p1_regen_day_retarget_se_conserva():
    """Un día deficitario sigue subiendo a la meta: la razón por la que el retarget existe."""
    for k, meta in META.items():
        assert _objetivo(DIA_DEFICITARIO[k], meta) == pytest.approx(meta), k
    assert _objetivo(0.0, 123.0) == pytest.approx(123.0)


def test_sin_meta_real_se_queda_con_la_suma_actual():
    """Fail-open idéntico al previo: sin biométricos no hay meta y el día se regenera contra sí mismo."""
    assert _objetivo(2688.0, 0.0) == pytest.approx(2688.0)
    assert _objetivo(2688.0, None) == pytest.approx(2688.0)
    assert _objetivo(None, None) == 0.0


def test_la_deriva_no_puede_repetirse_con_la_regla_nueva():
    """Simula 8 actualizaciones donde el motor entrega un 3 % por encima del objetivo (lo que
    pasó hoy): con `max` + techo el objetivo trepa hasta el techo; con la meta, se queda."""
    def _corrida(objetivo):
        return objetivo * 1.03

    dia = 2379.0
    for _ in range(8):
        dia = _corrida(_objetivo(dia, META["kcal"]))
    assert dia == pytest.approx(META["kcal"] * 1.03)

    with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_TO_META": "0"}):
        dia_legacy = 2379.0
        objetivos = []
        for _ in range(8):
            obj = _objetivo(dia_legacy, META["kcal"])
            objetivos.append(obj)
            dia_legacy = _corrida(obj)
        assert objetivos[-1] == pytest.approx(META["kcal"] * 1.12), "legacy: trepa hasta el techo"
        assert dia_legacy > META["kcal"] * 1.12, "legacy: el techo acota el objetivo, no el resultado"


def test_knob_off_restaura_el_max_con_techo():
    with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_TO_META": "0"}):
        assert _objetivo(2688.0, 2500.0) == pytest.approx(2688.0)          # exceso dentro del techo
        assert _objetivo(387.0, 334.0) == pytest.approx(334.0 * 1.12)      # exceso acotado
        assert _objetivo(2390.0, 2500.0) == pytest.approx(2500.0)          # suelo
    for v in ("1", "true", "yes", "on", "TRUE"):
        with mock.patch.dict(os.environ, {"MEALFIT_REGEN_DAY_RETARGET_TO_META": v}):
            assert _objetivo(2688.0, 2500.0) == pytest.approx(2500.0)


def test_el_band_score_del_endpoint_se_mide_contra_el_objetivo_del_retarget():
    """El «1.0» que se reportaba durante la deriva se medía contra `day_target`. Con el retarget
    a la meta, ese mismo cableado pasa a medir contra la meta: no hace falta un segundo target,
    y este test impide que alguien vuelva a pasarle la suma actual del día."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.index("[P2-REGEN-DAY-BAND-SCORE] day band_score=")
    ventana = src[max(0, i - 900):i]
    assert "compute_clinical_band_score(" in ventana
    assert '"calories": day_target.get("kcal")' in ventana
    assert '"protein": day_target.get("protein_g")' in ventana
    assert "_sum_current" not in ventana, "el band score no puede medirse contra la suma actual"
    fin = src.index("day_target[_kk] = _t")
    ini = src.rindex("for _kk in day_target:", 0, fin)
    assert "_retarget_macro_target(" in src[ini:fin], "el bucle del retarget dejó de usar el helper"


def test_marker_y_knob_documentados():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert re.search(r'_LAST_KNOWN_PFIX = "P2-REGEN-DAY-RETARGET-TO-META · 2026-09-03"', app) or \
        "P2-REGEN-DAY-RETARGET-TO-META · 2026-09-03" in app, "marker bumpeado o huella durable en app.py"
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert 'os.environ.get("MEALFIT_REGEN_DAY_RETARGET_TO_META", "true")' in src
