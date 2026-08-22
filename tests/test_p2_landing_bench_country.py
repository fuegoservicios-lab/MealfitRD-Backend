"""[P2-LANDING-BENCH-COUNTRY · 2026-08-21] El único harness que corre el pipeline real no tenía eje
de país.

`landing_benchmarks.py` es el banco cuyo output alimenta las cifras públicas del landing y guía la
mejora del motor. Sus 25 perfiles se evaluaban **todos como dominicanos**, y el docstring de
`_perfil` afirma que el payload tiene «la MISMA forma que emite el wizard» — afirmación que dejó de
ser cierta el día del flip, porque el wizard emite `country` desde Fase 0 y el banco no.

LO QUE ESO COSTÓ, en concreto: **ninguna de las regresiones de esta auditoría habría movido una sola
cifra del banco.** `P1-VERIFIED-CATALOG-COUNTRY` (el bloque «USA EXCLUSIVAMENTE ESTOS ALIMENTOS»
byte-idéntico entre España y RD) y `P1-COUNTRY-CATALOG-BY-COUNTRY` (los cinco catálogos beta
idénticos entre sí) son defectos que se ven a simple vista **contando caracteres**, y el banco tenía
el sitio perfecto para contarlos.

El modo `structural` es GRATIS —deriva hechos del código, sin LLM ni DB— así que el eje entra ahí:
por cada país, cuánto mide su catálogo verificado y cuánto su contexto temporal (y si ese
contexto le habla del Caribe a quien no vive ahí). Con eso, ambos gaps habrían salido como dos
columnas iguales en una tabla.

Cubre:
  A. El payload vuelve a tener la forma del wizard.
  B. El modo estructural mide por país.
  C. Las dos regresiones de esta ola habrían salido.
  D. Byte-identidad: el default sigue siendo dominicano.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def lb():
    import landing_benchmarks as _lb
    return _lb


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


# ── A. La forma del payload ─────────────────────────────────────────────────────────────────────

def test_el_payload_lleva_country_como_el_wizard(lb):
    """El docstring promete «la MISMA forma que emite el wizard». Desde Fase 0 el wizard emite
    `country`, así que sin este campo la promesa era falsa — y un banco que no reproduce la entrada
    real mide otra cosa."""
    for p in lb.build_landing_profiles():
        assert "country" in p, f"perfil {p.get('_label')} sin `country`"


def test_el_default_sigue_siendo_dominicano(lb):
    """Byte-identidad: la matriz de siempre es la dominicana."""
    assert {p["country"] for p in lb.build_landing_profiles()} == {"DO"}


def test_la_matriz_se_puede_pedir_en_otro_pais(lb):
    for p in lb.build_landing_profiles(country="ES"):
        assert p["country"] == "ES"


def test_un_pais_basura_cae_al_fail_safe(lb):
    """Mismo criterio que `canonicalize_country`: lo desconocido se comporta como RD."""
    for basura in (None, "", "ZZ", "basura"):
        assert {p["country"] for p in lb.build_landing_profiles(country=basura)} == {"DO"}


# ── B. El eje de país en el modo estructural ────────────────────────────────────────────────────

def test_los_hechos_estructurales_tienen_eje_de_pais(lb, knob_on):
    facts = lb.structural_facts()
    por_pais = facts.get("por_pais")
    assert isinstance(por_pais, dict) and por_pais, (
        "`structural_facts` sigue sin eje de país: el banco no puede acusar una regresión de país"
    )
    assert set(por_pais) == {"DO", "ES", "US", "MX", "PR", "CO"}


def test_cada_pais_reporta_lo_que_lo_distingue(lb, knob_on):
    """Las dos magnitudes que habrían acusado los dos gaps de esta ola."""
    for cc, datos in lb.structural_facts()["por_pais"].items():
        assert "catalogo_verificado_chars" in datos, f"{cc}: falta el tamaño del catálogo"
        assert "contexto_temporal_chars" in datos, f"{cc}: falta el tamaño del contexto temporal"
        assert "contexto_temporal_habla_del_caribe" in datos, f"{cc}: falta la marca del Caribe"


# ── C. Las regresiones de esta ola habrían salido ───────────────────────────────────────────────

def test_habria_acusado_el_catalogo_identico_entre_paises(lb, knob_on, monkeypatch):
    """`P1-VERIFIED-CATALOG-COUNTRY` + `P1-COUNTRY-CATALOG-BY-COUNTRY`: el bloque del catálogo era
    byte-idéntico entre España y RD (3824 chars), y luego idéntico entre los cinco beta (5777). Dos
    columnas iguales en esta tabla lo habrían enseñado sin abrir un plan."""
    # Catálogo sintético en vez de `pytest.skip` sin DB: la primera versión saltaba, y un guard que
    # siempre salta es una coartada — ya me pasó hoy con tres anclas que no corrían nunca en CI.
    # Una fila PRECIADA (universal) y una sin precio por país: si el filtro por país se rompe, los
    # seis tamaños vuelven a coincidir.
    _falso = [{"name": "Arroz blanco", "price_per_lb": 35.0, "price_per_unit": 0}]
    _falso += [{"name": n, "price_per_lb": 0, "price_per_unit": 0} for n in
               ("Jamón serrano", "Huitlacoche", "Chontaduro", "Pernil", "Pretzels")]
    import shopping_calculator as _sc
    import graph_orchestrator as _go
    monkeypatch.setattr(_sc, "get_master_ingredients", lambda *a, **k: list(_falso))
    monkeypatch.setattr(_sc, "_verified_ingredients_only_enabled", lambda *a, **k: True)
    _go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    tam = {cc: d["catalogo_verificado_chars"]
           for cc, d in lb.structural_facts()["por_pais"].items()}
    _go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    assert len(set(tam.values())) > 1, (
        f"todos los países reportan el mismo tamaño de catálogo: {tam}. O volvió la regresión, o "
        f"el hecho no se está derivando de verdad"
    )


def test_habria_acusado_el_contexto_temporal_ciego(lb, knob_on):
    """`P1-TIME-CONTEXT-COUNTRY`: antes del arreglo, a un ESPAÑOL se le decía en el prompt que
    «hace MUCHO calor en el Caribe» y que es «Temporada Caribeña».

    Medido al escribir este test: el contexto temporal no NOMBRA el país —mi primera versión
    asumía que sí y falló contra código correcto—. Lo que hace es incluir el bloque caribeño sólo
    para RD y omitirlo en beta, en vez de inventarle a España un equivalente climático. Así que el
    hecho que lo acusa es el Caribe, no el nombre."""
    por_pais = lb.structural_facts()["por_pais"]
    assert por_pais["DO"]["contexto_temporal_habla_del_caribe"] is True, (
        "el contexto dominicano perdió su bloque caribeño"
    )
    for cc in ("ES", "MX", "CO"):
        assert por_pais[cc]["contexto_temporal_habla_del_caribe"] is False, (
            f"a un usuario de {cc} se le vuelve a hablar del Caribe"
        )
    assert por_pais["DO"]["contexto_temporal_chars"] > por_pais["ES"]["contexto_temporal_chars"]


# ── D. Lo que ya medía sigue midiendo ───────────────────────────────────────────────────────────

def test_no_se_pierde_ningun_hecho_de_los_que_ya_habia(lb):
    """El landing consume estas claves (`systemFacts.js`): perder una rompe una cifra pública."""
    facts = lb.structural_facts()
    for clave in ("micronutrientes_dri", "reglas_condicion_backend",
                  "condiciones_chips_formulario", "condiciones_solo_backend",
                  "reglas_medicacion_backend", "alergias_chips_formulario",
                  "dietas_formulario"):
        assert clave in facts, f"desapareció el hecho estructural {clave!r}"


def test_la_matriz_no_encoge_y_su_docstring_no_miente(lb):
    """Son 25, no 20. El docstring decía «20 perfiles» y la matriz había crecido sin que nadie
    actualizara la prosa — deriva de documentación encontrada al escribir este test, no rotura.
    Se ancla la CIFRA contra el propio docstring para que la próxima vez que crezca, falle aquí."""
    import re
    perfiles = lb.build_landing_profiles()
    assert len(perfiles) >= 25, f"la matriz encogió a {len(perfiles)}"
    m = re.search(r"matriz:\s*(\d+)\s*perfiles", lb.build_landing_profiles.__doc__ or "")
    assert m, "el docstring de la matriz ya no declara cuántos perfiles tiene"
    assert int(m.group(1)) == len(perfiles), (
        f"el docstring dice {m.group(1)} perfiles y hay {len(perfiles)}"
    )
