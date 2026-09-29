# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Dos instrumentos de variedad que no veían lo que había en el plato.

(1) Pescados de los países beta. `_MAIN_PROTEIN_ALIASES['pescado']` (el mapa del contador cross-día, del gate
same-day, de `reeleccion_dia` y del armador determinista) no conocía trucha, sardina, mojarra, bagre,
huachinango, lenguado, caballa ni boquerón, y SÍ contaba «dorado», que en el plato es casi siempre un adjetivo:
G24 CO D1 contaba pescado por «Plátano Maduro Dorado», no por sus sardinas, y la trucha de D1 y D3 no existía.
Las especies salen del vocabulario de pescado del escáner (`_ALLERGEN_SYNONYMS['pescado']`, que ya incluye
`vocabulario_mar.PESCADOS_EXTRA`), no de otra tabla. Knob `MEALFIT_BETA_FISH_SPECIES_COUNT`.

(2) `_count_staple_repetitions` casaba los básicos por subcadena: «pina» dentro de «espinaca» (G24 DO: `pina: 2`
sin una piña). Ahora con el resolvedor del gate same-day (`culinary_context._name_has_token`), y su espejo del
armador determinista (`deterministic_day._basicos_de`) con él. Knob `MEALFIT_STAPLE_TOKEN_MATCH`.
"""
from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent


def _plan(textos_por_dia):
    """Un plan de N días, una comida por día cuyo nombre e ingrediente es el texto dado."""
    return [{"day": i + 1, "meals": [{"meal": "Almuerzo", "name": t, "ingredients": [f"120 g de {t}"]}]}
            for i, t in enumerate(textos_por_dia)]


# ── (1) pescados ─────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("especie", ["Trucha", "Sardinas", "Mojarra", "Bagre", "Huachinango", "Lenguado",
                                     "Caballa", "Boquerones", "Filete de dorada"])
def test_las_especies_beta_son_pescado(especie):
    assert "pescado" in go._protein_gate_labels_in_text(f"150 g de {especie}"), especie


def test_trucha_y_sardina_cuentan_entre_dias():
    dias = _plan(["Trucha a la plancha con patacones", "Ensalada estilo ceviche con sardinas",
                  "Trucha al horno con limón"])
    assert go._count_cross_day_heavy_protein_repetition(dias).get("pescado") == 3
    assert go.build_variety_report({"days": dias})["cross_day_proteins"].get("pescado") == 3


def test_dorado_adjetivo_no_es_pescado():
    """G24 CO D1: el «pescado» salía del plátano, no de las sardinas."""
    dias = _plan(["Plátano maduro dorado con queso", "Tostadas doradas con aguacate", "Papas doradas al horno"])
    assert "pescado" not in go._count_cross_day_heavy_protein_repetition(dias)
    assert "pescado" not in go._protein_gate_labels_in_text("Plátano Maduro Dorado")
    assert "pescado" in go._protein_gate_labels_in_text("150 g de filete de dorado"), "la frase inequívoca se queda"


def test_sardinas_en_el_almuerzo_y_trucha_en_la_cena_el_mismo_dia():
    dia = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada fresca con sardinas y garbanzos",
         "ingredients": ["120 g de Sardinas en lata", "80 g de Garbanzos"]},
        {"meal": "Cena", "name": "Arepa asada con ensalada fresca y trucha",
         "ingredients": ["1 Arepa de maíz", "100 g de Trucha"]},
    ]}
    assert go.build_variety_report({"days": [dia]})["same_day_protein_repeats"] >= 1
    assert go._days_with_same_day_protein_repeat({"days": [dia]}) == [1]


@pytest.mark.parametrize("texto", ["Ensalada César con pollo", "1 taza de fumet", "1 cda de salsa inglesa",
                                   "Carpaccio de remolacha", "Arroz con atún"])
def test_lo_que_no_es_un_pez_de_plato_no_cuenta(texto):
    """Del escáner salen sólo las especies: el aderezo César, el fondo, la salsa inglesa y los homónimos
    («carpa» ⊂ «carpaccio») se quedan fuera; el atún conserva su etiqueta."""
    assert "pescado" not in go._protein_gate_labels_in_text(texto), texto


def test_el_armador_determinista_ve_las_mismas_especies():
    import deterministic_day as dd
    assert "pescado" in dd._pesadas_de({"name": "Trucha al horno", "ingredients": ["150 g de Trucha"]})
    assert "pescado" not in dd._pesadas_de({"name": "Plátano maduro dorado", "ingredients": ["1 Plátano maduro"]})


def test_knob_de_especies_apagado_deja_el_mapa_como_estaba(monkeypatch):
    import pescado_especies as pe
    base = {"pescado": ["pescado", "tilapia", "dorado"], "atun": ["atun"], "pollo": ["pollo"]}
    vocab = ["pescado", "atun", "trucha", "sardina", "dorado", "cesar"]
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    apagado = copy.deepcopy(base)
    assert pe.extender_pescado(apagado, vocab) == ([], [])
    assert apagado == base
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", True)
    encendido = copy.deepcopy(base)
    anadidos, quitados = pe.extender_pescado(encendido, vocab)
    assert anadidos == ["trucha", "sardina", "filete de dorada"] and quitados == ["dorado"]
    assert encendido["pescado"] == ["pescado", "tilapia", "trucha", "sardina", "filete de dorada"]
    assert encendido["atun"] == ["atun"], "el atún no se muda de etiqueta"


# ── (2) básicos entre días ───────────────────────────────────────────────────────────────────────────────


def test_espinaca_no_es_pina():
    dias = _plan(["Ensalada de espinacas con uva", "Revoltillo con espinaca"])
    assert "pina" not in go._count_staple_repetitions(dias)


def test_pina_si_cuenta():
    dias = _plan(["Piña picada con yogurt", "Batido de piña"])
    assert go._count_staple_repetitions(dias).get("pina") == 2


def test_espejo_del_armador_determinista():
    import deterministic_day as dd
    assert "pina" not in dd._basicos_de({"name": "Ensalada de espinacas", "ingredients": ["2 tazas de Espinacas"]})
    assert "pina" in dd._basicos_de({"name": "Piña con yogurt", "ingredients": ["1 taza de Piña"]})


def test_knob_de_basicos_apagado_vuelve_a_la_subcadena(monkeypatch):
    import basicos_por_token as bpt
    dias = _plan(["Ensalada de espinacas con uva", "Revoltillo con espinaca"])
    monkeypatch.setattr(bpt, "STAPLE_TOKEN_MATCH", False)
    assert go._count_staple_repetitions(dias).get("pina") == 2      # la conducta previa, medida
    monkeypatch.setattr(bpt, "STAPLE_TOKEN_MATCH", True)
    assert "pina" not in go._count_staple_repetitions(dias)


# ── ronda 4 · (1) el pez en conserva se reescribe ENTERO ─────────────────────────────────────────────────


def _dia_dos_pescados(nombre, linea, pasos):
    return [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
         "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"],
         "recipe": ["Hornea la tilapia 20 minutos a 200 °C."]},
        {"meal": "Cena", "name": nombre, "ingredients": [linea, "1/2 Aguacate"],
         "ingredients_raw": [linea, "1/2 Aguacate"], "recipe": list(pasos)},
    ]}]


def _textos(meal):
    return [meal["name"], *meal["ingredients"], *meal["ingredients_raw"], *meal["recipe"]]


_JUNTO_AL_NUEVO = re.compile(
    r"latas?\s+de\s+pechuga|pechuga de pollo\s+(?:en\s+lata|enlatad|en\s+aceite|en\s+(?:salsa\s+de\s+)?tomate|"
    r"en\s+vinagre|en\s+conserva|ahumad|salad)", re.IGNORECASE)

_CONSERVAS = [
    ("Ensalada de sardinas en lata con aguacate", "1 lata de Sardinas en lata (120 g)",
     ["Escurre las sardinas en lata y mézclalas con el aguacate.",
      "Añade un pellizco del líquido de la lata de sardinas si quieres darle sabor, y sirve."]),
    ("Arroz con sardinas", "2 latas de sardinas (250 g)", ["Abre las latas de sardinas y mézclalas con el arroz."]),
    ("Sardinas en aceite con casabe", "120 g de Sardinas en aceite", ["Escurre las sardinas en aceite y sírvelas."]),
    ("Sardinas guisadas con papa", "120 g de Sardinas en salsa de tomate",
     ["Calienta las sardinas en salsa de tomate con la papa 5 minutos."]),
    ("Ensalada de caballa", "1 lata de caballa en aceite de oliva (110 g)", ["Desmenuza la caballa en aceite de oliva."]),
    ("Tosta de boquerones en vinagre", "80 g de Boquerones en vinagre", ["Coloca los boquerones en vinagre sobre el pan."]),
    ("Ensalada de trucha ahumada", "100 g de Trucha ahumada", ["Corta la trucha ahumada en tiras."]),
]


@pytest.mark.parametrize("nombre,linea,pasos", _CONSERVAS, ids=[c[1] for c in _CONSERVAS])
def test_el_pez_en_conserva_se_reescribe_entero(nombre, linea, pasos):
    """Revisión r3: tilapia + «Ensalada de sardinas en lata» el mismo día se reescribía a «Pechuga de pollo en lata» /
    «1 lata de pechuga de pollo en lata» — el fallo que P2-PROTEIN-LADDER-GAPS cerró para el atún."""
    dias = _dia_dos_pescados(nombre, linea, pasos)
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "DO"}, None) == 1
    cena = dias[0]["meals"][1]
    assert cena.get("_protein_autofix_applied") == "pescado->pollo"
    assert "pechuga de pollo" in cena["ingredients"][0].lower()
    for texto in _textos(cena):
        assert not _JUNTO_AL_NUEVO.search(texto), texto
        assert not re.search(r"\blatas?\b|enlatad", texto, re.IGNORECASE), texto


def test_el_liquido_de_la_lata_se_va_con_la_lata():
    dias = _dia_dos_pescados(*_CONSERVAS[0])
    go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None)
    assert not any("líquido" in p or "liquido" in p for p in dias[0]["meals"][1]["recipe"])


def test_el_pez_fresco_en_salsa_sigue_como_antes():
    """«en salsa de tomate» sólo es la lata en los peces que se venden así: una merluza en salsa es una preparación."""
    dias = _dia_dos_pescados("Merluza en salsa de tomate", "150 g de Merluza",
                             ["Cocina la merluza en salsa de tomate 10 minutos."])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["name"] == "Pechuga de pollo en salsa de tomate"


def test_los_compuestos_salen_de_pescado_especies():
    import pescado_especies as pe
    comp = set(go._PROTEIN_SOURCE_COMPOUNDS["pescado"])
    for c in ("sardinas en lata", "lata de sardina", "lata de sardinas en lata", "sardinas en aceite",
              "caballa en salsa de tomate", "boquerones en vinagre", "trucha ahumada"):
        assert c in comp, c
    assert "merluza en salsa de tomate" not in comp and "tilapia en aceite" not in comp
    assert set(pe.compuestos_de_conserva(["sardina", "trucha"])) <= comp


def test_knob_apagado_sin_compuestos_ni_filtro(monkeypatch):
    import pescado_especies as pe
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    assert pe.compuestos_de_conserva(["sardina"]) == ()
    als = ("pescado", "tilapia", "mero")
    assert pe.preparar_reescritura(als, {"name": "Tilapia"}) == als


def test_el_filtro_de_presentes_conserva_el_primer_alias():
    """La concordancia de los pasos lee el alimento viejo del PRIMER alias: el filtro no puede cambiarlo."""
    import pescado_especies as pe
    als = ("pescado", "tilapia", "mero", "sardinas en lata")
    assert pe.preparar_reescritura(als, {"name": "Mero al horno", "ingredients": ["150 g de mero"]}) == ("pescado", "mero")


# ── ronda 4 · (2) la escalera respeta la dieta ───────────────────────────────────────────────────────────


_PESCETARIANOS = ("pescetariano", "Pescetariana", "pescatarian")


@pytest.mark.parametrize("dieta", _PESCETARIANOS)
@pytest.mark.parametrize("cena", [
    ("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero a la plancha 4 minutos por lado."]),
    _CONSERVAS[0],
], ids=["mero", "sardinas-en-lata"])
def test_pescetariano_nunca_recibe_carne(dieta, cena):
    """Base: tilapia + mero con dieta pescetariana → «Pechuga de pollo a la plancha». El pescado sólo se cambia por
    otra proteína del mar; si la escalera no tiene ninguna, no se reescribe (decide el gate)."""
    dias = _dia_dos_pescados(*cena)
    antes = copy.deepcopy(dias)
    assert go._protein_repeat_autofix(dias, {"dietType": dieta, "country": "DO"}, None) == 0
    assert dias == antes
    assert go._scan_diet_violations({"days": dias}, dieta) == []


def test_pescetariano_atun_repetido_no_pasa_a_carne_ni_a_legumbre():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada de atún", "ingredients": ["120 g de Atún en agua", "1 taza de Lechuga"]},
        {"meal": "Cena", "name": "Arroz con vegetales", "ingredients": ["120 g de atún en agua", "1 taza de Arroz"]},
    ]}]
    go._protein_repeat_autofix(dias, {"dietType": "pescetariano"}, None)
    assert go._scan_diet_violations({"days": dias}, "pescetariano") == []
    assert "atún" in dias[0]["meals"][1]["ingredients"][0].lower()


def test_pescetariano_marisco_repetido_si_pasa_a_pescado():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Camarones al ajillo", "ingredients": ["150 g de Camarones", "1 taza de Arroz"]},
        {"meal": "Cena", "name": "Ensalada con camarones", "ingredients": ["120 g de Camarones", "1 taza de Lechuga"]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "pescetariano"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "camarones->pescado"


def test_omnivoro_sigue_como_antes():
    dias = _dia_dos_pescados("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero."])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["name"] == "Pechuga de pollo a la plancha con ensalada"


def test_la_dieta_sale_del_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _protein_repeat_autofix", 1)[1].split("\ndef ", 1)[0]
    assert "destino_apto_para_la_dieta" in cuerpo and "_diet_pool_item_banned" in cuerpo
    pe_src = (_BACKEND / "pescado_especies.py").read_text(encoding="utf-8")
    assert "canonicalize_diet_type" in pe_src


# ── ronda 4 · (3) las anchoas ────────────────────────────────────────────────────────────────────────────


def test_anchoas_son_pescado():
    """Datos: la única comida del corpus con anchoas (ES, «Coca de vegetales con anchoas») lleva 40 g — la mitad de la
    proteína del plato — y el catálogo la tiene en «Proteínas». Anchoas y boquerones el mismo día son el mismo pez."""
    assert "pescado" in go._protein_gate_labels_in_text("Coca de vegetales con anchoas")
    dia = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Coca de vegetales con anchoas", "ingredients": ["40 g de Anchoas"]},
        {"meal": "Cena", "name": "Boquerones fritos", "ingredients": ["150 g de Boquerones"]},
    ]}
    assert go._days_with_same_day_protein_repeat({"days": [dia]}) == [1]
    assert "atun" not in go._protein_gate_labels_in_text("40 g de Anchoas")

# ── contrato ─────────────────────────────────────────────────────────────────────────────────────────────


def test_knobs_documentados():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH"):
        assert knob in doc, knob
    assert "P1-PLAN-LOTE-857" in doc


def test_knobs_en_el_inventario_del_arranque():
    """Ronda 4: `basicos_por_token` se importaba en la primera llamada, así que su knob faltaba en el
    `[KNOBS/INVENTORY]` que `graph_orchestrator` emite al terminar de importarse. Se mide en un proceso limpio y sin
    importar antes los módulos del lote (importarlos primero es justo lo que escondía el fallo)."""
    codigo = (
        "import logging, os, sys\n"
        "os.environ['MEALFIT_DISABLE_SEMANTIC_CACHE'] = 'true'\n"
        "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')\n"
        "import graph_orchestrator\n"
    )
    out = subprocess.run([sys.executable, "-c", codigo], cwd=_BACKEND, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    inventario = [ln for ln in out.stdout.splitlines() if "[KNOBS/INVENTORY]" in ln]
    assert inventario, "el arranque no emitió el inventario de knobs"
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH"):
        assert knob + "=" in inventario[-1], knob


def test_marcador_y_anclas():
    for mod in ("pescado_especies.py", "basicos_por_token.py"):
        src = (_BACKEND / mod).read_text(encoding="utf-8")
        assert "[P1-PLAN-LOTE-857 · 2026-09-29]" in src and "tooltip-anchor: P1-PLAN-LOTE-857" in src, mod
    src_go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src_go.split("def _count_staple_repetitions", 1)[1].split("\ndef ", 1)[0]
    assert "basicos_por_token" in cuerpo and "any(a in text_norm for a in alias_list)" not in cuerpo
    assert 'pescado_especies").extender_pescado(_MAIN_PROTEIN_ALIASES, _ALLERGEN_SYNONYMS["pescado"])' in src_go
    src_dd = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    assert "basicos_por_token" in src_dd.split("def _basicos_de", 1)[1].split("\ndef ", 1)[0]
