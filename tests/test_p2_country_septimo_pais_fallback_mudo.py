"""[P2-COUNTRY-SEPTIMO-PAIS-FALLBACK-MUDO · 2026-08-23] Dar de alta el 7.º país deja CUATRO tablas
en fallback silencioso, y ninguna tenía guard de paridad.

LO MEDIDO dando de alta `COUNTRY_PROFILES['AR']` contra el código y el catálogo VIVOS:

    Funciona solo (vive DENTRO del SSOT):
      canonicalize_country('AR')→'AR' · country_for_form_data→'AR'
      pricing_mode_for_country→'beta_no_prices' · unit_system_for_country→'metric'
    Degrada EN SILENCIO:
      (a) pools deterministas dominicanos (el seeder le asigna víveres criollos)
      (b) biblioteca de inspiración dominicana de 87 plantillas
      (c) bloque «USA EXCLUSIVAMENTE» que le autoriza a la vez jamón serrano y sobrasada (ES),
          chile habanero y epazote (MX), sofrito (PR), guascas (CO) y bagels (US)
      (d) piso de presupuesto en pesos dominicanos

ESTE FICHERO NO CONVIERTE LOS FALLBACKS EN EXCEPCIÓN, Y ES DELIBERADO. El fail-open está declarado
por escrito en los cuatro sitios y es la conducta correcta para un país a medio curar: perder la
sección entera es peor que heredarla. Lo que faltaba es que la AUSENCIA no pueda ser SILENCIOSA.
Aquí la paridad se recorre desde `COUNTRY_PROFILES` —el SSOT que se amplía—, así que el alta
número siete pone rojo este fichero el mismo día que se escribe, no el día que un usuario
argentino recibe mangú.

CÓMO SE EXIME UN PAÍS. Con una entrada en `_EXENCIONES` y su razón escrita. Vacío hoy: los seis
países del sistema cumplen las cuatro. Una exención es una decisión, no un olvido — que es
exactamente la diferencia que este fichero existe para forzar.

El último test es el que impide que todo esto nazca inerte: da de alta un país de mentira y exige
que las cuatro comprobaciones lo rechacen.
"""
from __future__ import annotations

import os

import pytest

from constants import COUNTRY_POOLS, COUNTRY_PROFILES

#: {codigo_pais: {clave_de_superficie: "razón escrita"}}. Vacío = ningún país beta está exento.
_EXENCIONES: dict = {}

_LISTAS_POOL = ("proteins", "carbs", "veggies_fats", "fruits")


def _beta() -> list:
    return sorted(cc for cc, p in COUNTRY_PROFILES.items() if p.get("is_beta"))


def _exento(cc, superficie) -> bool:
    razon = (_EXENCIONES.get(cc) or {}).get(superficie)
    return bool(razon and str(razon).strip())


# ── (a) pools deterministas ─────────────────────────────────────────────────────────────────────

def _falta_pool(cc) -> str:
    pools = COUNTRY_POOLS.get(cc)
    if not pools:
        return f"{cc} no tiene entrada en COUNTRY_POOLS: el seeder le asignará víveres criollos"
    vacias = [k for k in _LISTAS_POOL if not (pools.get(k) or [])]
    return f"{cc}: listas de pool ausentes o vacías: {vacias}" if vacias else ""


@pytest.mark.parametrize("cc", _beta())
def test_todo_pais_beta_tiene_pool_determinista_propio(cc):
    if _exento(cc, "pools"):
        pytest.skip(_EXENCIONES[cc]["pools"])
    assert not _falta_pool(cc), _falta_pool(cc)


# ── (b) biblioteca de inspiración ───────────────────────────────────────────────────────────────

def _falta_biblioteca(cc) -> str:
    from graph_orchestrator import _DO_DISH_TEMPLATES_PATH, _dish_templates_path_for_country
    ruta = _dish_templates_path_for_country(cc)
    if ruta == _DO_DISH_TEMPLATES_PATH:
        return (f"{cc} cae a la biblioteca dominicana: no hay dish_templates_{cc.lower()}.json "
                "cableado en _dish_templates_path_for_country")
    if not os.path.isfile(ruta):
        return f"{cc} apunta a {ruta}, que no existe"
    return ""


@pytest.mark.parametrize("cc", _beta())
def test_todo_pais_beta_tiene_biblioteca_de_platos_propia(cc):
    if _exento(cc, "biblioteca"):
        pytest.skip(_EXENCIONES[cc]["biblioteca"])
    assert not _falta_biblioteca(cc), _falta_biblioteca(cc)


# ── (c) bloque del catálogo verificado ──────────────────────────────────────────────────────────

def _falta_catalogo(cc) -> str:
    from shopping_calculator import _COUNTRY_CATALOG_UNPRICED_BY_COUNTRY as _mapa
    if not _mapa.get(cc):
        return (f"{cc} no reclama ninguna fila del catálogo: el bloque «USA EXCLUSIVAMENTE» le "
                "ofrecerá la unión de las cocinas ajenas")
    return ""


@pytest.mark.parametrize("cc", _beta())
def test_todo_pais_beta_reclama_filas_del_catalogo(cc):
    if _exento(cc, "catalogo"):
        pytest.skip(_EXENCIONES[cc]["catalogo"])
    assert not _falta_catalogo(cc), _falta_catalogo(cc)


# ── (d) piso de presupuesto ─────────────────────────────────────────────────────────────────────

def _falta_piso(cc) -> str:
    from nutrition_calculator import _BUDGET_CYCLE_FLOOR_DEFAULTS_BY_CURRENCY as _pisos
    moneda = (COUNTRY_PROFILES.get(cc) or {}).get("currency")
    if moneda not in _pisos:
        return (f"{cc}/{moneda} no tiene piso propio: su presupuesto se compara contra la cesta "
                "DOMINICANA convertida por tipo de cambio")
    return ""


@pytest.mark.parametrize("cc", _beta())
def test_todo_pais_beta_tiene_piso_de_presupuesto_en_su_moneda(cc):
    if _exento(cc, "piso"):
        pytest.skip(_EXENCIONES[cc]["piso"])
    assert not _falta_piso(cc), _falta_piso(cc)


# ── El país nativo no entra en la exigencia (y eso también se ancla) ────────────────────────────

def test_el_pais_nativo_no_es_beta_y_por_eso_no_se_le_exige_pool_propio():
    """DO es la base de la que los demás heredan: exigirle 'pool propio' invertiría el diseño."""
    assert COUNTRY_PROFILES["DO"]["is_beta"] is False
    assert "DO" not in COUNTRY_POOLS


# ── Lo que impide que este fichero nazca inerte ─────────────────────────────────────────────────

def test_un_septimo_pais_sin_curar_pone_rojo_este_fichero(monkeypatch):
    """La comprobación de que el guard puede fallar. Se da de alta un país de mentira con el mismo
    shape que los seis reales y se exige que las CUATRO superficies lo denuncien. Si alguien
    'simplifica' cualquiera de los cuatro helpers a un `return ""`, este test lo caza."""
    falso = dict(COUNTRY_PROFILES)
    falso["ZZ"] = {"name_es": "País de prueba", "currency": "ZZD", "is_beta": True,
                   "has_native_prices": False, "default_tz_offset_min": 0,
                   "unit_system": "metric"}
    monkeypatch.setattr("constants.COUNTRY_PROFILES", falso)
    import constants as _c
    monkeypatch.setattr(_c, "COUNTRY_PROFILES", falso)
    quejas = [f("ZZ") for f in (_falta_pool, _falta_biblioteca, _falta_catalogo, _falta_piso)]
    mudas = [i for i, q in enumerate(quejas) if not q]
    assert not mudas, (
        f"las superficies {mudas} aceptaron un país sin curar en silencio: {quejas}")
    assert "ZZ" in [cc for cc, p in falso.items() if p.get("is_beta")], (
        "el recorrido tiene que salir de COUNTRY_PROFILES, no de una lista a mano")
