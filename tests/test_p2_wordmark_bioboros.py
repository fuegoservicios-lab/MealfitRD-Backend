"""[P2-WORDMARK-BIOBOROS · 2026-07-30 · ampliado 2026-07-31] El wordmark tiene UN dueño.

Historia corta: la marca estaba escrita a mano en 12 sitios. El rebrand a "Bioboros"
buscaba los tres fragmentos `Mealfit<span>R</span><span>D</span>` ADYACENTES, así que
no alcanzó los archivos donde estaban en líneas separadas. Cayeron dos:

  - `Logo.jsx` (el compartido) → el usuario vio "Mealfit" en la app ya rebrandeada.
  - `Plan.jsx`, la pantalla "Diseñando tu plan" → sobrevivió hasta el 31 de julio
    mostrando "Bioboros" + una "R" indigo con puntito + una "D" rosa. El sufijo "RD"
    venía de "MealfitRD", donde era el país y separarlo SIGNIFICABA algo; sobre
    "Bioboros" quedaba un sufijo sin referente, y encima bicolor — el recurso que el
    owner ya había descartado dos veces (ver el comentario de `Wordmark.jsx`).

Los dos fallaron por la MISMA razón, con dos meses de diferencia. Por eso el guard no
persigue la cadena "RD": persigue el patrón estructural que la produce.
"""
import re
from pathlib import Path

import pytest

_FRONT = Path(__file__).resolve().parent.parent.parent / "frontend" / "src"
_WORDMARK = _FRONT / "components" / "common" / "Wordmark.jsx"


def _jsx_files():
    return sorted(p for p in _FRONT.rglob("*.jsx"))


# ------------------------------------------------- el patrón que produjo el bug

# Un <span> que fija un color y cuyo ÚNICO contenido es una letra suelta. Es la firma
# del "acento por letra": no depende de que la letra sea R o D, ni de que esté pegada
# a la marca — dos anclajes que caducarían al primer cambio de branding.
_LETRA_ACENTUADA = re.compile(
    r"<span[^>]*\bcolor\s*:\s*['\"]?#[0-9A-Fa-f]{3,8}[^>]*>\s*([A-ZÁÉÍÓÚÑ])\s*</span>",
    re.DOTALL,
)


@pytest.mark.parametrize("ruta", _jsx_files(), ids=lambda p: p.name)
def test_ninguna_letra_suelta_coloreada_como_acento_de_marca(ruta: Path):
    """Ninguna letra individual lleva color propio.

    `Wordmark.jsx` documenta que la marca es MONOCROMO por decisión de producto, tomada
    tras descartar en vivo dos versiones con acento (bicolor indigo+rosa, y las tres
    "o" en verde). Un acento por letra reintroduce justo eso, y además se salta el SSOT.
    """
    src = ruta.read_text(encoding="utf-8")
    hits = _LETRA_ACENTUADA.findall(src)
    assert not hits, (
        f"P2-WORDMARK-BIOBOROS regresión en {ruta.name}: letra(s) sueltas con color "
        f"propio {hits!r}. El wordmark es monocromo por decisión de producto y se "
        f"renderiza SOLO vía `<Wordmark/>`. Si de verdad quieres un acento, cámbialo "
        f"en Wordmark.jsx y habla la decisión — es la tercera versión y las dos "
        f"anteriores se descartaron por esto."
    )


# ------------------------------------------------------- el SSOT sigue siendo uno

def test_wordmark_es_monocromo():
    """El propio componente no puede llevar color hardcodeado."""
    src = _WORDMARK.read_text(encoding="utf-8")
    cuerpo = src[src.index("export const Wordmark"):]
    assert not re.search(r"color\s*:\s*['\"]?#", cuerpo), (
        "P2-WORDMARK-BIOBOROS regresión: `Wordmark.jsx` fija un color hex. Debe heredar "
        "la tinta del contexto (`--text-main`) para funcionar igual en claro y oscuro."
    )
    assert "Bioboros" in cuerpo, "el wordmark dejó de renderizar la marca"


def test_la_pantalla_de_carga_del_plan_usa_el_componente():
    """`Plan.jsx` fue el último sitio con la marca a mano; que no vuelva a escribirla.

    Anclado al render, no a una ventana de bytes: exige el import Y el uso.
    """
    plan = (_FRONT / "pages" / "Plan.jsx").read_text(encoding="utf-8")
    assert re.search(r"^import\s+Wordmark\s+from\s+['\"].*Wordmark['\"]", plan, re.M), (
        "P2-WORDMARK-BIOBOROS regresión: Plan.jsx ya no importa `Wordmark`."
    )
    assert "<Wordmark />" in plan or "<Wordmark/>" in plan, (
        "P2-WORDMARK-BIOBOROS regresión: Plan.jsx importa `Wordmark` pero no lo renderiza "
        "— la pantalla 'Diseñando tu plan' volvería a dibujar la marca a mano."
    )
