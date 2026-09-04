"""[P1-COHERENCE-MIRROR-KEEP · 2026-08-21] La costura (a), re-diagnosticada: no era léxico
DO-tuned. Era un espejo roto.

La doc canónica atribuía los «4 fantasmas» de los planes beta a que «el vocabulario del lado
esperado del guard es DO-tuned y las filas beta de F2 no entraron», y mandaba a ampliar un léxico.
Tres dimensiones de la auditoría del 2026-08-20 lo re-diagnosticaron por separado y las tres
llegaron al mismo sitio, que no tiene nada que ver con vocabulario:

    lado ESPERADO (recetas)   expected_raw se filtra con `_is_verified_for_shopping`,
                              que exige PRECIO > 0                     → las filas beta SALEN
    lado AGREGADO (lista)     el agregador tiene DOS ramas `keep` que conservan sin precio
                              (baking staples y catálogo-país)         → las filas beta ENTRAN

Toda fila sin precio queda, por construcción, «en la lista y ausente de las recetas» = `unknown`.
No es un fallo de reconocimiento: es que el guard se compara contra una versión de sí mismo a la
que le faltan dos ramas. Los conteos casan 1:1 en producción:

    plan ES 6a4321f5   4 ítems sin precio  ↔  los «4 fantasmas» documentados
    plan US 2245eb45   3 ítems sin precio  ↔  `_shopping_coherence_block_history` registra HOY
                                              {'unknown': 3, 'recipe_unquantified': 3},
                                              13 entradas consecutivas del 2026-08-20

Y el diagnóstico equivocado tenía un costo propio: mandaba a trabajar en un léxico durante días,
cuando el arreglo es una función de seis líneas.

EL SSOT. En vez de repetir el `if/elif/else` del agregador en el filtro (que es lo que ya se hizo
una vez y produjo esta divergencia), nace `_survives_shopping_list(name)`: la pregunta «¿este
nombre sobrevive a la lista de compras?» pasa a tener UNA respuesta, y los dos lados la hacen.

ORDEN. Este P-fix va DESPUÉS de P1-COUNTRY-KEEP-RESPECT-QTY, y no es una preferencia: al conservar
estas filas en el lado esperado aparecen divergencias de MAGNITUD contra la cantidad que el
agregador les asigne. Mientras esa cantidad fuera 150 g fijos, este arreglo habría convertido 4
avisos inocuos en bloqueos con retry GARANTIZADO-FÚTIL — ningún reintento elimina una divergencia
estructural.

LO QUE SALE GRATIS. El detector `[VERIFIED-ONLY-GUARD-BLIND]` —la única señal que existe para «la
lista salió incompleta sin aviso», el miedo explícito del dueño— era 100% falsos positivos en
país beta: acusaba al LLM de desobedecer con alimentos que estaban perfectamente en la lista. Con
el espejo arreglado vuelve a reportar sólo lo que de verdad se cayó.

Cubre:
  A. El SSOT existe y responde por las tres ramas.
  B. Una fila de catálogo-país deja de ser fantasma en el guard.
  C. Un staple de horneado tampoco es fantasma (la otra rama sin espejo).
  D. Un ingrediente off-catálogo GENUINO sigue filtrándose (no se abrió la puerta).
  E. El detector guard-blind deja de acusar a las filas beta.
  F. Los knobs de keep gobiernan las dos orillas a la vez.
  G. Parser-based: el filtro no volvió a llamar a `_is_verified_for_shopping` a secas.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] Este módulo necesita el catálogo/la base de datos o el
# .env local (pasa en el checkout del dueño; en el CI sin NEON_DATABASE_URL se salta con motivo).
pytestmark = pytest.mark.needs_local_data

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SC_PATH = _BACKEND_ROOT / "shopping_calculator.py"


@pytest.fixture(scope="module")
def sc():
    import shopping_calculator as _sc
    return _sc


@pytest.fixture(autouse=True)
def verified_only(monkeypatch):
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")


# ── A. El SSOT ──────────────────────────────────────────────────────────────────────────────────

def test_el_ssot_responde_por_las_tres_ramas(sc):
    """«¿Sobrevive este nombre a la lista de compras?» pasa a tener UNA respuesta. Las tres ramas
    del agregador —precio, staple de horneado, catálogo-país— la comparten."""
    assert sc._survives_shopping_list("Pollo"), "una fila con precio debe sobrevivir"
    assert sc._survives_shopping_list("Levadura"), "un staple de horneado se CONSERVA, no se dropea"
    assert sc._survives_shopping_list("Almejas"), "una fila de catálogo-país se CONSERVA"


def test_el_ssot_sigue_rechazando_lo_que_de_verdad_se_dropea(sc):
    """El control que impide que el arreglo se convierta en un pase libre: un nombre que ningún
    tier resuelve (garble) sigue sin sobrevivir, y por tanto sigue fuera de los dos lados."""
    assert not sc._survives_shopping_list("Zzqx inventado que no existe")


# ── B/C. Las dos ramas sin espejo dejan de producir fantasmas ───────────────────────────────────

@pytest.mark.parametrize("nombre", ["Almejas", "Acelgas", "Membrillo"])
def test_una_fila_de_catalogo_pais_ya_no_es_fantasma(sc, nombre):
    """RED pre-fix: el lado esperado la borraba, el agregado la conservaba, y la diferencia se
    reportaba como `unknown`/`aggregated_only`. Se comprueba sobre el filtro real del guard."""
    expected = {nombre: {"g": 500.0}, "Pollo": {"g": 400.0}}
    filtrado = sc._filter_expected_to_shopping_survivors(expected)
    assert nombre in filtrado, f"'{nombre}' sigue saliendo del lado esperado del guard"
    assert "Pollo" in filtrado


def test_un_staple_de_horneado_tampoco_es_fantasma(sc):
    """La rama hermana, que llevaba sin espejo desde P1-BAKING-STAPLES (2026-07-01) — es decir
    que el defecto es más viejo que el sistema de países y afectaba también a planes dominicanos
    con panqueques."""
    filtrado = sc._filter_expected_to_shopping_survivors({"Levadura": {"g": 10.0}})
    assert "Levadura" in filtrado


# ── D. La puerta no se abrió ────────────────────────────────────────────────────────────────────

def test_un_ingrediente_off_catalogo_genuino_sigue_filtrandose(sc):
    """La razón de existir del filtro: un ingrediente que el LLM inventó y que NINGÚN tier
    resuelve se dropea de la lista, así que dejarlo en el lado esperado forzaría un retry por algo
    que el sistema no puede comprar. Eso no cambia."""
    filtrado = sc._filter_expected_to_shopping_survivors(
        {"Zzqx inventado que no existe": {"g": 5.0}, "Pollo": {"g": 400.0}})
    assert "Zzqx inventado que no existe" not in filtrado
    assert "Pollo" in filtrado


# ── E. El detector guard-blind recupera su significado ──────────────────────────────────────────

def test_el_detector_guard_blind_deja_de_acusar_a_las_filas_beta(sc, caplog):
    """`[VERIFIED-ONLY-GUARD-BLIND]` es la ÚNICA señal para «la lista salió incompleta sin aviso».
    En país beta era 100% falsos positivos: acusaba al LLM de desobedecer con alimentos que
    estaban en la lista. Un detector que grita siempre se apaga en una semana, y entonces la
    amputación REAL pasa desapercibida."""
    import logging
    with caplog.at_level(logging.WARNING):
        sc._filter_expected_to_shopping_survivors({"Almejas": {"g": 500.0}}, emit_blind_warning=True)
    assert "Almejas" not in caplog.text


def test_el_detector_guard_blind_sigue_acusando_lo_que_si_desaparece(sc, caplog):
    """Control del anterior: la señal sigue viva para el caso que la justifica."""
    import logging
    with caplog.at_level(logging.WARNING):
        sc._filter_expected_to_shopping_survivors(
            {"Zzqx inventado que no existe": {"g": 5.0}}, emit_blind_warning=True)
    assert "VERIFIED-ONLY-GUARD-BLIND" in caplog.text


# ── F. Los knobs gobiernan las dos orillas ──────────────────────────────────────────────────────

def test_apagar_el_keep_de_catalogo_pais_cierra_las_dos_orillas_a_la_vez(sc, monkeypatch):
    """El rollback parcial documentado (`MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP=false`) devuelve el
    DROP de la fila. El espejo tiene que seguirlo: si el agregador la dropea y el lado esperado la
    conserva, aparece la divergencia SIMÉTRICA (`expected_only`) — el mismo bug con el signo
    cambiado. Un espejo que sólo funciona con el knob encendido no es un espejo."""
    monkeypatch.setenv("MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP", "false")
    assert not sc._survives_shopping_list("Almejas")
    assert "Almejas" not in sc._filter_expected_to_shopping_survivors({"Almejas": {"g": 500.0}})


# ── G. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_filtro_del_guard_no_vuelve_a_llamar_al_predicado_a_secas():
    """El defecto era literalmente `if _is_verified_for_shopping(k)` en el filtro del lado
    esperado. Este guard impide que vuelva: el filtro debe preguntar por el SSOT."""
    src = _SC_PATH.read_text(encoding="utf-8", errors="replace")
    assert "P1-COHERENCE-MIRROR-KEEP" in src
    assert "_survives_shopping_list" in src
    assert "expected_raw.items() if _is_verified_for_shopping(k)" not in src, (
        "el filtro del lado esperado volvió a usar el predicado de precio a secas"
    )


def test_el_ssot_replica_las_mismas_tres_ramas_del_agregador():
    """Anti-drift estructural: el helper debe nombrar las mismas tres condiciones que el
    `if/elif/else` del agregador. Si alguien añade una cuarta rama de keep allí y la olvida aquí,
    el espejo se vuelve a romper — y esta vez con un nombre que dice de quién es la culpa."""
    src = _SC_PATH.read_text(encoding="utf-8", errors="replace")
    i = src.find("def _survives_shopping_list")
    assert i > 0
    # El cuerpo corta en el PRÓXIMO `def`, no en un número mágico de caracteres: la primera
    # versión de este assert usaba una ventana de 1200 y el docstring —que explica el defecto—
    # ya la agotaba, así que medía la prosa en vez del código (la lección del ancla GAP-08).
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    for pieza in ("_is_verified_for_shopping", "is_baking_pantry_staple",
                  "is_country_catalog_unpriced_item"):
        assert pieza in cuerpo, f"el SSOT no contempla la rama '{pieza}'"
