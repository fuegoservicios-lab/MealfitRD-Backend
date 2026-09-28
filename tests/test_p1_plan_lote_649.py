# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-649 · 2026-09-27] Un español lee «guineo (plátano)», no «guineo» a secas.

El catálogo del motor es dominicano en los seis países (es el identificador de la Nevera, del guard de coherencia y
del backstop de alergias) y el modelo copia ese identificador: en un plan de España salía «1 guineo mediano» y en uno
de México «habichuelas negras». Renombrar rompería el motor; sustituir en la frase rompería la concordancia (batata→
boniato cambia de género). Lo que se hace es GLOSAR al leer: `nombre_por_pais` dice cómo lo llaman allí y la app lo
pone entre paréntesis tras el identificador, solo con la app en español y solo en los países beta hispanos.

La tabla vive aquí (SSOT) y el frontend la lleva en espejo (`src/data/nombresPorPais.json`): este test los compara.

tooltip-anchor: P1-PLAN-LOTE-649
"""
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ESPEJO = _BACKEND.parent / "frontend" / "src" / "data" / "nombresPorPais.json"


def test_solo_paises_beta_hispanos_y_solo_lo_que_difiere():
    from constants import COUNTRY_PROFILES
    from food_names_i18n import nombres_por_pais, nombres
    t = nombres_por_pais()
    assert set(t) == {"ES", "MX", "CO", "PR"}
    assert set(t) <= set(COUNTRY_PROFILES)
    for pais, filas in t.items():
        for canon, local in filas.items():
            assert canon in nombres(), (pais, canon)          # el identificador existe en el catálogo
            assert local.lower() != canon.lower(), (pais, canon)  # solo lo que difiere


def test_el_platano_de_espana_y_mexico_es_el_guineo():
    from food_names_i18n import nombres_por_pais
    t = nombres_por_pais()
    assert t["ES"]["Guineo"] == "plátano" and t["MX"]["Guineo"] == "plátano"
    assert t["MX"]["Habichuelas negras"] == "frijoles negros"
    assert "Guineo" not in t["PR"]            # en Puerto Rico también se dice guineo


def test_el_espejo_del_frontend_es_identico():
    if not _ESPEJO.exists():
        pytest.skip("el frontend de este checkout aún no trae el espejo (el CI del backend clona su `main`)")
    from food_names_i18n import nombres_por_pais
    assert json.loads(_ESPEJO.read_text(encoding="utf-8")) == nombres_por_pais()
