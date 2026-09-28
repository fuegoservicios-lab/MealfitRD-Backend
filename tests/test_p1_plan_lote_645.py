# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-645 · 2026-09-27] Los avisos push que salían en español, traducidos de verdad.

El guard `test_p1_i18n_push_cron_espanol.py` ahora ve el alias del import y los textos en variables (25 que faltaban).
Pero `_build_zero_log_push_payload` arma el cuerpo por PARTES («…de tus comidas. » + «Abre el diario…»), y la
traducción busca el texto ENTERO que sale: la pieza sola en el catálogo no traduciría nada. Aquí se construyen las
cuatro variantes reales y se exige que cada una salga traducida en los cuatro idiomas.

tooltip-anchor: P1-PLAN-LOTE-645
"""
import pytest

LOCALES = ("en-US", "pt-BR", "fr-FR", "it-IT")


@pytest.mark.parametrize("ceros", [0, 3])
@pytest.mark.parametrize("preferencia", ["manual", "auto_proxy"])
def test_las_cuatro_variantes_del_aviso_sin_registros_salen_traducidas(ceros, preferencia):
    from cron_tasks import _build_zero_log_push_payload
    from push_i18n import translate_push_text
    carga = _build_zero_log_push_payload(ceros, preferencia)
    for loc in LOCALES:
        for campo in ("title", "body"):
            original = carga[campo]
            traducido = translate_push_text(original, loc)
            assert traducido != original, (loc, campo, original)


def test_tu_plan_necesita_una_revision_sale_traducido():
    from push_i18n import translate_push_text
    assert translate_push_text("Tu plan necesita una revisión", "en-US") == "Your plan needs a review"
    assert translate_push_text("Tu plan necesita una revisión", "es-DO") == "Tu plan necesita una revisión"
