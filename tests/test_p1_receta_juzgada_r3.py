# -*- coding: utf-8 -*-
"""[P1-RECETA-JUZGADA-R3 · 2026-09-10] Las 19 recetas, reescritas contra lo que el dueño escribió.

Tercera ronda de juicio humano: 19 de 19 «cambiar». Este P-fix aplica las correcciones que tocan
la RECETA (las de etiqueta van aparte). La más importante no es de estilo:

## La de seguridad

«*usar yuca dulce pelada, hervida hasta ablandar y escurrida; desechar el agua y majar antes de
mezclar. **No confiar en rallar y dorar para eliminar sus compuestos tóxicos**.*»

Tenía razón. La yuca cruda lleva glucósidos cianogénicos; lo que los arrastra es hervir y **botar
el agua**, no rallar y pasar por el sartén. La receta anterior rallaba la yuca CRUDA, la exprimía y
la freía — el paso que él señaló.

## Las otras que este test ancla

- **La sal, después de escurrir.** En los platos que hierven y botan el agua, salar el hervor manda
  la sal al fregadero y la ficha la cuenta igual. Es el mismo hallazgo de contabilidad de
  `P1-SODIO-DEL-DIA-DETERMINISTA`, dicho ahora en el paso.
- **El vinagre del mangú, con cantidad definida.** «Sin él la cebolla no queda encurtida». La
  cantidad va en la LISTA de ingredientes, no en el paso: la regla de sus rondas anteriores prohíbe
  números de ingrediente dentro de los pasos.
- **El solape declarado.** «Hornear el pescado mientras hierve la yuca» — y como
  `P1-MINUTOS-DE-LA-RECETA` descuenta lo que corre dentro de otro paso, decirlo bajó la tilapia de
  45 a 30 minutos sin tocar un número.
"""
import json
import pathlib

import pytest

import dish_registry as dr

RECETAS = json.loads(
    (pathlib.Path(dr.REGISTRY_DIR) / "recipe_library_do_v1.json").read_text(encoding="utf-8"))["por_id"]
REGISTRO = json.loads(
    (pathlib.Path(dr.REGISTRY_DIR) / "dish_registry_do_v1.json").read_text(encoding="utf-8"))
POR_ID = {t["template_id"]: t for t in REGISTRO["templates"]}


def pasos(tid):
    return " ".join((RECETAS.get(tid) or {}).get("pasos") or [])


# ── la corrección de seguridad ───────────────────────────────────────────────
def test_la_yuca_se_hierve_y_se_bota_el_agua():
    """Rallar en crudo y dorar NO neutraliza los glucósidos cianogénicos. Hervir y botar, sí."""
    t = pasos("tpl_86da7bdca661").lower()
    assert "hierve" in t or "hiérvela" in t, "la yuca volvió a ir cruda al sartén"
    assert "bota el agua" in t, "no se bota el agua de la cocción, que es lo que arrastra lo amargo"
    assert "yuca dulce" in t, "no dice qué yuca: la amarga necesita otro proceso"
    assert "rállala" not in t, "sigue rallando la yuca cruda como método principal"


def test_el_paso_explica_por_que_no_basta_rallar():
    """Un paso que manda sin decir por qué invita a volver al método viejo cuando estorbe."""
    t = pasos("tpl_86da7bdca661").lower()
    assert "rallarla cruda" in t and "no hace ese trabajo" in t


# ── la sal, después de escurrir ──────────────────────────────────────────────
@pytest.mark.parametrize("tid,plato", [
    ("tpl_36409c4726eb", "Mangú de plátano verde con atún y cebolla encurtida"),
    ("tpl_8b37ecfd46e1", "Tilapia al horno con yuca y cebolla"),
])
def test_la_sal_no_va_al_agua_que_se_bota(tid, plato):
    t = pasos(tid).lower()
    assert "sin sal" in t, f"«{plato}» sigue salando el agua de hervor"
    assert "sazona" in t or "sazónala" in t, "la sal desapareció del plato en vez de moverse"


# ── el vinagre: cantidad en la LISTA, no en el paso ──────────────────────────
def test_el_vinagre_del_mangu_existe_como_ingrediente():
    t = POR_ID["tpl_36409c4726eb"]
    nombres = [c["canonical"] for c in t["constituents"]]
    assert any("vinagre" in n.lower() for n in nombres), (
        f"la receta nombra el vinagre y no está en la lista: {nombres}")
    assert t["status"] == "ok", "el vinagre no resolvió contra el catálogo"


def test_el_paso_del_vinagre_no_lleva_numeros():
    """La regla que salió de sus seis rondas: sin cantidades de ingrediente dentro de los pasos."""
    paso = next(p for p in RECETAS["tpl_36409c4726eb"]["pasos"] if "vinagre" in p.lower())
    import re
    assert not re.search(r"\d+\s*(ml|g\b|gramos|cucharada)", paso, re.I), (
        f"volvió una cantidad de ingrediente al paso: {paso!r}")


# ── el solape declarado ──────────────────────────────────────────────────────
def test_la_tilapia_declara_que_hornea_mientras_hierve():
    t = pasos("tpl_8b37ecfd46e1").lower()
    assert "mientras la yuca hierve" in t, "el solape que pidió el dueño se perdió"
    assert POR_ID["tpl_8b37ecfd46e1"]["logistics"]["prep_minutes_est"] <= 35, (
        "el solape está escrito y el tiempo no lo refleja: revisa `minutos_de_los_pasos`")


# ── lo que NO puede romperse al reescribir ───────────────────────────────────
JUZGADAS = ["tpl_7f0b6fc54350", "tpl_6a5089265418", "tpl_cc64fe57c76f", "tpl_b991edb7ec87",
            "tpl_86da7bdca661", "tpl_4e48570063d2", "tpl_e1265f14a8b4", "tpl_36409c4726eb",
            "tpl_3a0517b6dc80", "tpl_b29321fc8917", "tpl_53d21a09b17f", "tpl_612f9a6638b2",
            "tpl_385a17ed7bed", "tpl_af7f6c3725a6", "tpl_7ec10278eed6", "tpl_6a5fe80136a7",
            "tpl_02895f974ed3", "tpl_b83a2e9e039a", "tpl_8b37ecfd46e1"]


@pytest.mark.parametrize("tid", JUZGADAS)
def test_las_19_siguen_teniendo_receta_y_plantilla(tid):
    assert tid in RECETAS, f"{tid} perdió su receta al reescribirla"
    assert tid in POR_ID, f"{tid} quedó huérfana: su plantilla no está en el registro"
    assert 3 <= len(RECETAS[tid]["pasos"]) <= 5, "la regla de 3-5 pasos"


@pytest.mark.parametrize("tid", JUZGADAS)
def test_ninguna_receta_mete_mililitros(tid):
    """El agua se mide EN PALABRAS. La regla es suya, de la cuarta ronda."""
    import re
    for p in RECETAS[tid]["pasos"]:
        assert not re.search(r"\d+\s*ml", p, re.I), f"{tid}: {p!r}"


def test_el_pollo_conserva_su_temperatura_interna():
    """Sin sustitutos del termómetro: es la regla que salió de la ronda de la temperatura."""
    assert "74 °C" in pasos("tpl_53d21a09b17f")
    assert "63 °C" in pasos("tpl_8b37ecfd46e1")
