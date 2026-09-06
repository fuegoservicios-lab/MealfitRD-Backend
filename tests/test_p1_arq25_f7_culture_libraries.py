"""[P1-ARQ25-F7-CULTURE · 2026-09-05] Fase 7 (subfase D): las seis bibliotecas de platos cumplen la barra de
cobertura del roadmap (§13.4) para que ningún perfil de cocina quede «delgado»: ≥80 plantillas por biblioteca,
mínimos por franja, ≥10 familias de proteína, ≥12 técnicas; vocabulario unificado (las de PR/US nacieron con
'ninguna'/'lacteo'/'mixto'/'a la plancha' que el muestreador del brief NUNCA casaba con el pool: plantillas
muertas); 100 % de constituyentes resueltos en los snapshots compilados; y sin snacks sin valor nutricional.
"""
import json
from pathlib import Path

import pytest

import dish_registry as dr

_BACKEND = Path(__file__).resolve().parents[1]
_DATA = _BACKEND / "data"

BAR = {"total": 80, "desayuno": 18, "almuerzo": 28, "cena": 22, "merienda": 16, "proteins": 10, "techniques": 12}
# [ARQ27-P1-03 · 2026-09-06] `tofu` entra al vocabulario: Tofu firme era una fila del catálogo con
# CERO usos como constituyente mientras el pool vegano programaba una familia `Tofu` que ninguna
# plantilla podía servir. Nombra un alimento —no una clase, como `legumbre`—, así que el selector la
# resuelve por etiqueta y no necesita puente.
VOCAB_PROTEIN = {"none", "huevo", "pollo", "res", "cerdo", "pescado", "camarones", "atun", "pavo",
                 "chivo", "queso", "legumbre", "mixta", "tofu"}
SLOTS = {"desayuno", "almuerzo", "cena", "merienda"}
DROPPED_US = {"Malvaviscos con galletas Graham y cacao", "Pretzels con mostaza", "Miel con nueces pecanas",
              "Bolitas de papa con salsa barbacoa", "Papas ralladas fritas con kétchup"}


def _templates(lib: str) -> list:
    fname = dr.LIBRARIES[lib][0]
    doc = json.loads((_DATA / fname).read_text(encoding="utf-8"))
    return doc["templates"] if isinstance(doc, dict) else doc


@pytest.mark.parametrize("lib", sorted(dr.LIBRARIES))
def test_a_barra_de_cobertura_por_biblioteca(lib):
    ts = _templates(lib)
    slots = {s: 0 for s in SLOTS}
    for t in ts:
        for s in t.get("slots") or []:
            slots[s] = slots.get(s, 0) + 1
    assert len(ts) >= BAR["total"], f"[{lib}] {len(ts)} plantillas < {BAR['total']}"
    for s in ("desayuno", "almuerzo", "cena", "merienda"):
        assert slots[s] >= BAR[s], f"[{lib}] {s}: {slots[s]} < {BAR[s]}"
    assert len({str(t.get('protein')).lower() for t in ts}) >= BAR["proteins"], f"[{lib}] pocas familias de proteína"
    assert len({str(t.get('technique')).lower() for t in ts}) >= BAR["techniques"], f"[{lib}] pocas técnicas"


@pytest.mark.parametrize("lib", sorted(dr.LIBRARIES))
def test_b_vocabulario_unificado(lib):
    for t in _templates(lib):
        assert set(t.get("slots") or []) <= SLOTS and t.get("slots"), t["name"]
        assert str(t.get("protein")).lower() in VOCAB_PROTEIN, f"[{lib}] proteína fuera de vocabulario: {t['name']} → {t.get('protein')}"
        tech = str(t.get("technique") or "").lower()
        assert tech and tech not in ("ninguna", "ninguno", "a la plancha"), f"[{lib}] técnica muerta en {t['name']}: {tech!r}"
        assert str(t.get("base") or "").lower() not in ("ninguno", ""), f"[{lib}] base vacía en {t['name']}"
        if lib != "do":
            assert t.get("constituents"), f"[{lib}] sin constituyentes: {t['name']}"


@pytest.mark.parametrize("lib", sorted(dr.LIBRARIES))
def test_c_snapshots_compilados_con_todo_resuelto(lib):
    p = Path(dr.snapshot_path(lib))
    if not p.exists():
        pytest.skip("snapshot no compilado")
    snap = json.loads(p.read_text(encoding="utf-8"))
    st = snap["stats"]
    assert st["templates"] == len(_templates(lib)), f"[{lib}] snapshot desfasado respecto a la biblioteca: recompila"
    assert st["excluded"] == 0, f"[{lib}] plantillas excluidas en el snapshot"
    # [ARQ27-P0-02 · 2026-09-06] Las 4 de DO que este test ya nombraba abajo («solo los 4 declarados
    # sin resolver») figuraban `ok` teniendo exclusiones dentro: el compilador solo miraba
    # `not_in_catalog` para el estado. Ahora son `partial`, que es lo que siempre fueron — la Batida
    # de zapote no lleva zapote. El test decía la verdad en su comentario y la contradecía en su assert.
    esperados_parciales = 4 if lib == "do" else 0
    assert st["ok"] == st["templates"] - esperados_parciales, f"[{lib}] plantillas no-ok en el snapshot"
    assert st["partial"] == esperados_parciales, f"[{lib}] parciales: {st['partial']}"
    if lib == "do":
        assert st["resolution_pct"] >= 99.0, "DO: solo los 4 declarados sin resolver (Menta, Zapote, Chillo, Salami de pavo)"
    else:
        assert st["resolution_pct"] == 100.0, f"[{lib}] {st['resolution_pct']} % resueltos"


def test_d_los_snacks_sin_valor_salieron_de_us():
    names = {t["name"] for t in _templates("us")}
    assert not (names & DROPPED_US), names & DROPPED_US
    # y la cena de US ya no cuelga de una sola técnica ('ninguna' era el 31 % de la biblioteca)
    techs = [str(t.get("technique")).lower() for t in _templates("us")]
    assert techs.count("frío") / len(techs) < 0.35


def test_e_cada_biblioteca_ofrece_candidatos_para_cada_franja_y_familia_principal():
    """Gate del roadmap: cobertura por franja/cultura vía el registry (lo que el allocator ve de verdad)."""
    if not Path(dr.snapshot_path("es")).exists():
        pytest.skip("snapshots no compilados")
    for lib, (_, cc, _c) in dr.LIBRARIES.items():
        for slot in ("desayuno", "almuerzo", "cena", "merienda"):
            assert dr.template_candidates(cc, slot, None, k=3), f"[{lib}] sin candidatos en {slot}"
        for fam in ("pollo", "pescado", "res", "huevo"):
            assert dr.template_candidates(cc, "almuerzo" if fam != "huevo" else "desayuno", fam, k=2), f"[{lib}] sin candidatos de {fam}"
