"""[P1-SCAN-ALIASES · 2026-08-10] El catálogo traía la respuesta y nadie la pedía.

`master_ingredients` tiene una columna `aliases` con **816 sinónimos escritos a
mano**: «pollo»→Pechuga de pollo, «baking powder»→Polvo de hornear, «harina pan»→
Harina de maíz precocida, «pan»→Pan blanco familiar. Es exactamente la pregunta
que hace el escáner de fotos («el modelo leyó este texto libre, ¿qué alimento
es?»), contestada a mano por quien armó el catálogo.

El escáner no la leía. Su `SELECT` ni siquiera pedía la columna. Medido contra el
catálogo real: **484 de los 816 aliases no llevaban a su alimento**; con la
columna conectada, 9 — y los 9 son los que apuntan a varios alimentos a la vez
(«nueces» → Almendras fileteadas y Nueces mixtas; «mariscos» → Pulpo, Calamar y
Mejillones), donde negarse es la respuesta correcta.

Efecto para quien usa la app: «pan», «pollo», «aceite», «harina», «pescado»,
«crackers» y 470 más pasaron de «sin match en el catálogo» (no se podía agregar) a
resolver a su alimento.

Hermano [P1-SCAN-BREAD-GENERIC]: el dueño fotografió pan de agua y el modelo leyó
«pan de hot dog». El catálogo no tiene ninguno de los dos, así que un tipo de pan
inventado se queda sin match — mientras que «pan» a secas SÍ es un alias que entra.
El prompt ahora nombra los panes de mesa dominicanos y pide el genérico cuando el
tipo no es seguro.

La causa raíz del mapeo (un conector como prueba de identidad) vive en
`test_p1_scan_catalog_match.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from constants import resolve_scanned_food

_BACKEND = Path(__file__).resolve().parent.parent
_UD_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")

# Espejo del catálogo real con las trampas que importan: un alias que choca con el
# NOMBRE de otro alimento, y un alias con dos dueños.
ALIASES = {
    "Pan blanco familiar": ["pan blanco", "pan de molde", "pan de caja", "pan"],
    "Pan integral familiar": ["pan integral", "pan de trigo integral"],
    "Pechuga de pollo": ["pollo", "pechuga"],
    "Polvo de hornear": ["baking powder", "royal"],
    "Atún en agua": ["atún", "atún enlatado", "atún en lata"],
    "Repollo": ["repollo morado"],           # choca con el NOMBRE de otro alimento
    "Repollo morado": [],
    "Maní": ["nueces"],                      # ambiguo a propósito
    "Mantequilla de maní": [],
    "Guineo": ["nueces"],                    # 2º dueño del mismo alias
}


@pytest.mark.parametrize("detectado,esperado", [
    # Lo que antes no se podía agregar y ahora sí.
    ("pan", "Pan blanco familiar"),
    ("pan de molde", "Pan blanco familiar"),
    ("pan integral", "Pan integral familiar"),   # empate roto por el sinónimo curado
    ("pollo", "Pechuga de pollo"),
    ("baking powder", "Polvo de hornear"),
    # [corregido 2026-08-10] Lo etiqueté como ambiguo («en agua» vs «en aceite») y
    # me equivoqué: el catálogo YA resolvió esa duda declarando «atún en lata» como
    # sinónimo de «Atún en agua». Un juicio a priori no manda sobre una decisión que
    # el curador del catálogo ya tomó explícitamente.
    ("atún en lata", "Atún en agua"),
    # El NOMBRE gana sobre el alias ajeno: si no, «repollo morado» se resolvería a
    # «Repollo» y el alimento propio quedaría inalcanzable.
    ("repollo morado", "Repollo morado"),
    # Un alias con dos dueños no decide nada.
    ("nueces", None),
])
def test_aliases(detectado, esperado):
    assert resolve_scanned_food(detectado, list(ALIASES), ALIASES) == esperado


def test_sin_aliases_el_resolutor_sigue_funcionando():
    """El parámetro es opcional: los callers viejos no cambian de comportamiento."""
    assert resolve_scanned_food("pollo", list(ALIASES)) is None
    assert resolve_scanned_food("pan blanco familiar", list(ALIASES)) == "Pan blanco familiar"


def test_el_endpoint_pide_los_aliases():
    """No basta con saber usarlos: la consulta tiene que traerlos.

    Ese era literalmente el estado anterior — la respuesta curada estaba en la
    tabla, a una columna de distancia, y el SELECT no la pedía."""
    i = _UD_SRC.find("def _match_against_catalog")
    cuerpo = _UD_SRC[i:i + 2500]
    assert re.search(r"SELECT[^\"]*\baliases\b[^\"]*FROM master_ingredients", cuerpo), (
        "el SELECT del escáner debe traer la columna `aliases`"
    )
    i2 = _UD_SRC.find("def _match_catalog(")
    assert "aliases" in _UD_SRC[i2:i2 + 1800], (
        "el matcher debe pasarle los aliases al resolutor"
    )


def test_el_prompt_prefiere_el_pan_generico():
    """[P1-SCAN-BREAD-GENERIC] Preferir el genérico no es rendirse: es que el
    genérico SÍ existe en el catálogo y el tipo específico inventado no."""
    i = _UD_SRC.find("_VISION_PROMPT = (")
    prompt = _UD_SRC[i:i + 3000]
    assert "pan de agua" in prompt and "sobao" in prompt, (
        "el prompt debe nombrar los panes de mesa dominicanos"
    )
    assert re.search(r"escribe simplemente 'pan'", prompt), (
        "el prompt debe pedir el nombre genérico cuando el tipo no es seguro"
    )


def test_marker_bumpeado():
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', app_src)
    assert m, "falta _LAST_KNOWN_PFIX en app.py"
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-08-10", f"marker anterior a este P-fix: {m.group(1)!r}"
