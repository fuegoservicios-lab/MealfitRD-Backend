"""[P2-DISPLAY-RETENCION-LOCALES · 2026-08-21] `_display` retenía una copia por cada
idioma visitado y nada la evacuaba jamás.

El mutator hacía `disp_map[locale] = display` sobre el mapa existente y el disparador
del cambio de idioma sólo AÑADE: no había ni una ruta que borrara un idioma abandonado.
Sin poda, sin TTL, sin knob. Un plan de 30 días visitado en los cinco idiomas guardaba
cinco copias completas del texto dentro de `plan_data` — el mismo jsonb que el
comentario de `user_data.py` ya describe como «de cientos de KB a MB con 30 días de
recetas expandidas». A ojo: ~500 B por comida y locale, ×4 comidas ×30 días ≈ 60 KB por
idioma.

POR QUÉ UN TOPE Y NO UN DESALOJO, que era lo que apuntaba el plan: desalojar el idioma
anterior haría re-pagar la traducción ENTERA cada vez que alguien alterna entre dos, y
eso es exactamente lo que `P2-DISPLAY-REDESPACHO-SIN-FILTRO` acaba de evitar en el
mismo módulo. Los dos arreglos tiran en direcciones opuestas; 2 es donde se cruzan:
cubre el ir y venir real (el idioma nuevo y el de antes) y acota el crecimiento a 2×
en vez de 5×.

El activo NUNCA se poda. Eso es lo que hace que la poda sea segura: pase lo que pase,
el idioma que el usuario está leyendo sobrevive.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pdi = importlib.import_module("plan_display_i18n")

_MARKER = "P2-DISPLAY-RETENCION-LOCALES"


def _e(nombre):
    return {"name": nombre, "description": "d", "recipe": ["r"], "ingredients": ["i"]}


def test_por_debajo_del_tope_no_se_toca_nada() -> None:
    m = {"en-US": _e("a"), "fr-FR": _e("b")}
    assert pdi._podar_locales(dict(m), "fr-FR") == m


def test_al_pasar_del_tope_se_conserva_el_activo_y_el_mas_reciente() -> None:
    m = {"en-US": _e("a"), "pt-BR": _e("b"), "fr-FR": _e("c")}
    out = pdi._podar_locales(dict(m), "fr-FR")
    assert set(out) == {"pt-BR", "fr-FR"}, (
        f"se esperaba conservar el activo y el anterior; salió {sorted(out)} [{_MARKER}]"
    )


def test_el_activo_nunca_se_poda() -> None:
    """La propiedad que hace segura la poda: pase lo que pase, el idioma que el
    usuario está leyendo sobrevive."""
    m = {"en-US": _e("a"), "pt-BR": _e("b"), "it-IT": _e("c"), "fr-FR": _e("d")}
    for activo in m:
        out = pdi._podar_locales(dict(m), activo)
        assert activo in out, f"se podó el idioma ACTIVO {activo} [{_MARKER}]"
        assert len(out) <= 2, f"el tope no se respetó para {activo}: {sorted(out)}"


def test_un_activo_que_no_estaba_en_el_mapa_no_rompe() -> None:
    """Defensivo: el mutator llama a esto justo después de insertar el activo, así
    que en producción siempre está. Pero una función que revienta con una entrada
    razonable acaba envuelta en un try/except que se traga otras cosas."""
    m = {"en-US": _e("a"), "pt-BR": _e("b"), "it-IT": _e("c")}
    out = pdi._podar_locales(dict(m), "fr-FR")
    assert len(out) <= 2


def test_no_es_destructivo_con_entradas_raras() -> None:
    assert pdi._podar_locales(None, "en-US") is None
    assert pdi._podar_locales({}, "en-US") == {}
