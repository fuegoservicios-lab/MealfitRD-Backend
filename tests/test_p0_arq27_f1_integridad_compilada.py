# -*- coding: utf-8 -*-
"""[ARQ27-P0-02 + ARQ27-P0-03 · 2026-09-06] Una plantilla incompleta figuraba ÍNTEGRA, y un dato
ausente se contaba como cero.

**P0-02.** `compile_template` marcaba `status='ok'` si había constituyentes resueltos y ninguna
exclusión por `not_in_catalog`. `declared_unresolved` y `no_grams` no contaban para el estado, así
que cuatro plantillas DO figuraban íntegras teniendo exclusiones dentro:

  · «Batida de zapote ligera» — sin zapote.
  · «Chillo al horno» — el chillo se compone con filete de pescado blanco.
  · «Mangú con salami de pavo» — el salami se compone con jamón de pavo.
  · «Frutas picadas con limón y menta» — sin menta.

Las cuatro salían al pool con su nombre prometiendo un ingrediente que la receta compilada no tiene.
«Cada ingrediente resuelto o excluido» cuadra un contador; no certifica que el plato esté completo. Y
un plato SUSTITUIDO no es el plato del título: necesita identidad propia, no un rótulo heredado.

**P0-03.** `_f(None)` devuelve `0.0`, así que la suma de nutrientes incorporaba el hueco como si
fuera un dato y `derive_risk_attributes` emitía `sodium_high=False`, `phosphorus_high=False`. Medido
sobre el catálogo VIVO el 06-sep: 5 de 347 filas sin `phosphorus_mg` (Hoja santa, Chontaduro,
Champús, Borojó, Achiote — el lote beta de CO/MX), que sostienen 7 constituyentes ya compilados en
MX y CO. Los otros nueve nutrientes están completos hoy.

O sea: a un perfil renal se le presentaba «Arroz con pollo colombiano» como fósforo bajo cuando la
verdad era que nadie lo había medido. Cero medido y cero por ausencia no son el mismo número.
Contrato I20.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import dish_registry as DR  # noqa: E402

PAISES = ["DO", "PR", "MX", "CO", "ES", "US"]

_CATALOGO = [
    {"name": "Arroz blanco", "category": "Víveres", "kcal_per_100g": 130, "protein_g_per_100g": 2.7,
     "carbs_g_per_100g": 28, "fats_g_per_100g": 0.3, "fiber_g_per_100g": 0.4, "sodium_mg_per_100g": 1,
     "potassium_mg_per_100g": 35, "phosphorus_mg_per_100g": 43, "saturated_fat_g_per_100g": 0.1,
     "sugars_g_per_100g": 0.1},
    # La fila del lote beta: existe, se resuelve, y NO trae fósforo.
    {"name": "Achiote", "category": "Despensa", "kcal_per_100g": 300, "protein_g_per_100g": 3,
     "carbs_g_per_100g": 60, "fats_g_per_100g": 5, "fiber_g_per_100g": 20, "sodium_mg_per_100g": 20,
     "potassium_mg_per_100g": 100, "phosphorus_mg_per_100g": None, "saturated_fat_g_per_100g": 1,
     "sugars_g_per_100g": 0.5},
]


def _idx():
    return DR.build_catalog_index(_CATALOGO)


def _compilar(constituyentes, declared=(), **extra):
    t = {"template_id": "tpl_x", "name": "Plato de prueba", "slots": ["almuerzo"],
         "base": "viveres", "protein": "none", "technique": "hervido", **extra}
    return DR.compile_template(t, _idx(), library="do", constituents=constituyentes,
                               declared_unresolved=list(declared))


# ── P0-02: qué puede llamarse íntegro ─────────────────────────────────────────────────────────
def test_todo_resuelto_es_ok():
    c = _compilar([{"name": "Arroz blanco", "grams": 100}])
    assert c["status"] == "ok" and c["excluded"] == []


@pytest.mark.parametrize("caso,cons,decl", [
    ("declared_unresolved", [{"name": "Arroz blanco", "grams": 100}], ["Zapote"]),
    ("no_grams", [{"name": "Arroz blanco", "grams": 100}, {"name": "Achiote", "grams": 0}], []),
    ("not_in_catalog", [{"name": "Arroz blanco", "grams": 100}, {"name": "Zapote", "grams": 80}], []),
])
def test_las_tres_exclusiones_impiden_llamarlo_integro(caso, cons, decl):
    """Antes solo `not_in_catalog` degradaba el estado. Las tres dejan la receta incompleta."""
    c = _compilar(cons, decl)
    assert c["status"] == "partial", f"{caso} siguió compilando como íntegro"
    assert any(e["reason"] == caso for e in c["excluded"])


def test_un_opcional_curado_si_puede_quedar_integro():
    """La única salida: que la FUENTE marque el constituyente `optional`. Una excepción visible, no
    un silencio — y sigue apareciendo en `excluded` para que el revisor la vea."""
    c = _compilar([{"name": "Arroz blanco", "grams": 100},
                   {"name": "Perejil de adorno", "grams": 2, "optional": True}])
    assert c["status"] == "ok"
    assert [e["reason"] for e in c["excluded"]] == ["not_in_catalog"]
    assert c["excluded"][0]["optional"] is True


def test_sin_nada_resuelto_es_excluded():
    assert _compilar([{"name": "Zapote", "grams": 80}])["status"] == "excluded"


def test_las_cuatro_plantillas_do_ya_no_figuran_integras():
    """El caso vivo. Ninguna plantilla `ok` de ninguna biblioteca puede llevar dentro una exclusión
    bloqueante — que es lo que hacían las cuatro de DO."""
    malas = []
    for c in PAISES:
        for t in (DR.load_registry(c) or {}).get("templates") or []:
            if t.get("status") != "ok":
                continue
            bloq = [e for e in (t.get("excluded") or [])
                    if e.get("reason") in DR._BLOCKING_EXCLUSIONS and not e.get("optional")]
            if bloq:
                malas.append((c, t["name"], [e["reason"] for e in bloq]))
    assert not malas, f"plantillas íntegras con exclusiones dentro: {malas}"


def test_el_zapote_no_produce_una_batida_de_zapote_integra():
    """Nombrado explícitamente en el criterio de cierre del gap."""
    do = DR.load_registry("DO") or {}
    zapote = [t for t in do.get("templates") or [] if "zapote" in str(t.get("name", "")).lower()]
    assert zapote, "desapareció la plantilla del zapote: revisa si el caso sigue siendo el mismo"
    for t in zapote:
        tiene = any("zapote" in str(c.get("canonical", "")).lower() for c in (t.get("constituents") or []))
        assert tiene or t["status"] != "ok", f"«{t['name']}» figura íntegra sin zapote"


def test_las_stats_distinguen_plantillas_de_lineas():
    """`excluded` (plantillas con status excluded) y `constituents_excluded` (LÍNEAS de ingrediente)
    se leían como la misma cifra. La identidad obliga a que cuadren."""
    for c in PAISES:
        st = (DR.load_registry(c) or {}).get("stats") or {}
        assert st["constituents"] == st["resolved"] + st["constituents_excluded"], (c, st)
        assert st["templates"] == st["ok"] + st["partial"] + st["excluded"], (c, st)


# ── P0-03: un dato ausente no es un cero ──────────────────────────────────────────────────────
def test_un_nutriente_ausente_no_suma_cero_ni_certifica_nada():
    c = _compilar([{"name": "Arroz blanco", "grams": 100}, {"name": "Achiote", "grams": 300}])
    assert c["nutrition_unknown"] == {"phosphorus_mg": ["Achiote"]}
    ira = c["intrinsic_risk_attributes"]
    assert ira["phosphorus_high"] is None, "el fósforo desconocido se certificó como no-alto"
    # los nutrientes que SÍ están siguen siendo booleanos de verdad
    assert ira["sodium_high"] is False and ira["energy_dense"] in (True, False)


def test_cero_medido_no_se_confunde_con_ausente():
    """La otra mitad: un cero REAL en el catálogo es un dato y se comporta como tal."""
    cat = [dict(_CATALOGO[1], name="Sal", phosphorus_mg_per_100g=0)]
    t = {"template_id": "t", "name": "x", "slots": ["almuerzo"], "base": "b", "protein": "none"}
    c = DR.compile_template(t, DR.build_catalog_index(cat), library="do",
                            constituents=[{"name": "Sal", "grams": 5}], declared_unresolved=[])
    assert c["nutrition_unknown"] == {}
    assert c["intrinsic_risk_attributes"]["phosphorus_high"] is False


def test_sin_huecos_no_se_emite_el_campo():
    """El snapshot no engorda: `nutrition_unknown` es `{}` en el 99 % de las plantillas."""
    c = _compilar([{"name": "Arroz blanco", "grams": 100}])
    assert c["nutrition_unknown"] == {}
    assert all(isinstance(c["intrinsic_risk_attributes"][k], bool)
               for k in ("sodium_high", "potassium_high", "phosphorus_high"))


def test_los_snapshots_vivos_marcan_sus_siete_huecos():
    """Las 7 plantillas de MX/CO apoyadas en Achiote/Champús. Si el catálogo gana el dato, el número
    baja y este test lo dice — no falla por bajar, falla si alguien vuelve a certificar el hueco."""
    con_hueco = []
    for c in PAISES:
        for t in (DR.load_registry(c) or {}).get("templates") or []:
            u = t.get("nutrition_unknown") or {}
            if not u:
                continue
            con_hueco.append((c, t["name"]))
            for nutr in u:
                for flag, fuentes in DR._RISK_SOURCES.items():
                    if nutr in fuentes:
                        assert t["intrinsic_risk_attributes"][flag] is None, \
                            f"{c}/{t['name']}: {flag} se certificó con {nutr} desconocido"
    assert len(con_hueco) <= 7, f"crecieron los huecos de datos: {con_hueco}"


def test_el_selector_bloquea_solo_a_quien_exige_ese_dato():
    """El criterio: un faltante de fósforo bloquea al perfil renal, no a todo el mundo."""
    libre = DR.template_candidates("CO", "almuerzo", None, k=99)
    renal = DR.template_candidates("CO", "almuerzo", None, k=99,
                                   require_known_nutrients=("phosphorus_mg",))
    assert len(renal) < len(libre), "exigir fósforo conocido no descartó nada en CO"
    for cd in renal:
        t = next(x for x in DR.load_registry("CO")["templates"] if x["template_id"] == cd["template_id"])
        assert "phosphorus_mg" not in (t.get("nutrition_unknown") or {})
    # y a quien no lo exige no se le recorta nada
    assert len(DR.template_candidates("CO", "almuerzo", None, k=99,
                                      require_known_nutrients=("sodium_mg",))) == len(libre)


def test_la_condicion_clinica_decide_que_nutriente_se_exige():
    """El cableado: `renal` exige fósforo y potasio, `hta` exige sodio, y un perfil sin condiciones no
    exige nada. Resuelto con el registry SSOT `condition_rules`, no con una tabla de términos nueva."""
    import horizon as H
    assert H.required_nutrients(None) == ()
    assert H.required_nutrients({"clinical": {"conditions": []}}) == ()
    assert H.required_nutrients({"clinical": {"conditions": ["enfermedad renal crónica"]}}) == \
        ("phosphorus_mg", "potassium_mg")
    assert "sodium_mg" in H.required_nutrients({"clinical": {"conditions": ["hipertensión arterial"]}})
