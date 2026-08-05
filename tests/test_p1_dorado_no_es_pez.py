"""[P1-DORADO-NO-ES-PEZ · 2026-08-05] "dorado" es el adjetivo, no el pez.

DOS SÍNTOMAS, UNA PALABRA. El dueño reportó a la vez (a) un aviso amarillo «el nombre
puede no reflejar la proteína real» sobre «Pastel **Dorado** de Batata y Queso», y (b) que
«Arreglar este día» no hacía nada. Eran el mismo bug:

    ⚠️ RECETA MENCIONA INGREDIENTES NO LISTADOS: en los pasos escribiste `dorado`
    (que cuenta como pescado), pero el array `ingredients` NO los contiene

La receta decía «hornea hasta que esté dorado». El guard leyó el pez, dedujo que la receta
usa un ingrediente no comprable, y rechazó el plato TRES veces hasta agotar los reintentos.

MEDIDO antes de tocar nada (7 días de journal de producción):
  · `dorado` = 18 divergencias de coherencia, la causa NÚMERO UNO
  · `pescado` (la palabra real) = 1
  · `master_ingredients` con "dorad" = **0 filas**

Ese último dato es el que cierra la decisión: el pez no se puede comprar en esta app, así
que el LLM nunca podrá listarlo. Un alias que no puede acertar solo puede fallar.

TERCERA vez que esta palabra quema reintentos; las dos anteriores trataron el síntoma
(P3-SWAP-RETRY-COHERENCE-HINT le pidió al LLM auto-revisarse; P1-SWAP-COHERENCE-REPAIR
intentó añadir la línea que falta — imposible, no hay nada que añadir).

tooltip-anchor: P1-DORADO-NO-ES-PEZ
"""
import pytest

from constants import PROTEIN_SYNONYMS
import graph_orchestrator as go


def test_dorado_suelto_no_es_un_alias_de_pescado():
    assert "dorado" not in PROTEIN_SYNONYMS["pescado"]
    assert "dorado" not in go._PHANTOM_PROTEIN_SYNS["pescado"]


def test_las_frases_inequivocas_se_conservan():
    """«filete de dorado» SÍ nombra al pez: ahí el sustantivo no es ambiguo.

    Quitar el alias suelto no puede convertirse en 'dejamos de detectar el pez'.
    """
    assert "filete de dorado" in PROTEIN_SYNONYMS["pescado"]


def test_los_pescados_reales_siguen_detectandose():
    """El arreglo no puede vaciar la categoría: los que SÍ existen en el catálogo
    (Bacalao, Filete de pescado blanco, Mero, Salmón, Sardinas, Tilapia) siguen."""
    for pez in ("pescado", "tilapia", "mero", "bacalao", "salmon"):
        assert pez in PROTEIN_SYNONYMS["pescado"], pez
        assert pez in go._PHANTOM_PROTEIN_SYNS["pescado"], pez


@pytest.mark.parametrize("nombre", [
    "Pastel Dorado de Batata y Queso con Crema Crujiente",  # el caso real reportado
    "Pollo Dorado al Horno",
    "Arroz Dorado con Vegetales",
])
def test_un_plato_dorado_no_promete_pescado(nombre):
    """Ningún alias de pescado debe casar con estos nombres."""
    bajo = nombre.lower()
    casan = [a for a in PROTEIN_SYNONYMS["pescado"] if a in bajo]
    assert not casan, "%r casó con %s" % (nombre, casan)


def test_el_paso_de_receta_mas_comun_del_espanol_no_inventa_un_ingrediente():
    """El caso que rompió «Arreglar este día», en su forma mínima."""
    for paso in ("Hornea hasta que esté dorado y cremoso.",
                 "Dora la cebolla a fuego medio.",
                 "Saltea hasta dorar ligeramente."):
        bajo = paso.lower()
        casan = [a for a in PROTEIN_SYNONYMS["pescado"] if a in bajo]
        assert not casan, "%r casó con %s" % (paso, casan)


def test_las_listas_de_alergenos_NO_se_tocaron():
    """⚠️ Decisión deliberada, no un olvido.

    `dorado` sigue en las listas de alérgenos/mariscos de graph_orchestrator y
    condition_rules. Para una alergia, marcar de más es la dirección SEGURA: un aviso de
    sobra no daña a nadie, uno de menos sí. Este test existe para que un futuro «limpiemos
    dorado de todas partes» tenga que leer esta razón primero.
    """
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    # Las listas de mariscos/alérgenos conservan el alias.
    assert src.count('"dorado"') >= 3, (
        "alguien quitó `dorado` de las listas de alérgenos; eso NO es lo que hizo "
        "P1-DORADO-NO-ES-PEZ y afloja un guard de seguridad"
    )
