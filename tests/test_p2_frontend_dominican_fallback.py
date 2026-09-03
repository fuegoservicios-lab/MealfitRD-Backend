"""[P2-FRONTEND-DOMINICAN-FALLBACK · 2026-08-21] Si el swap fallaba, al español le caía mangú.

`AssessmentContext.jsx` lleva una base local de recetas cableada —`DOMINICAN_MEALS`: Mangú con
Huevo, La Bandera, Sancocho Light, Moro de Guandules, Crema de Auyama— y la usa como red cuando el
swap por IA falla: sustituye el plato, lo escribe en `planData` y devuelve el nombre nuevo.

Para un dominicano es exactamente lo que debe hacer. Para un usuario beta es tres cosas malas a la
vez:

  1. **Le da un plato que no puede cocinar.** Los víveres del mangú no están en su lista de la
     compra ni en su catálogo de país.
  2. **Lo escribe en el plan.** No es un aviso: el plato ajeno queda persistido, así que la lista de
     la compra y la receta dejan de coincidir — la incoherencia que el guard receta↔lista existe
     para cazar, introducida por el propio frontend.
  3. **Lo hace en silencio.** El usuario ve un plato nuevo y no tiene forma de saber que la IA
     falló; parece una decisión del sistema.

EL ARREGLO NO INVENTA RECETAS. La tentación es curar `SPANISH_MEALS`, `MEXICAN_MEALS`… — cinco
tablas de platos con sus calorías y sus pasos, escritas de memoria. Eso es fabricar datos
nutricionales, que es lo que costó la auditoría de procedencia del catálogo. En beta se hace lo
honesto: **no sustituir**, avisar de que el cambio no salió y dejar el plato original intacto.

Y el patrón ya existe DIEZ LÍNEAS MÁS ARRIBA: la rama `swap_llm_retries_exhausted` (422) muestra un
toast y devuelve `null` sin tocar el plan. Esto reusa esa forma en vez de inventar otra.

Byte-identidad dominicana: con `country='DO'` —o sin país, o con el knob de UI apagado— la red local
funciona exactamente como siempre.

Lo que queda abierto y se dice: un usuario beta cuyo swap falla se queda sin alternativa. Curar
plantillas por país es trabajo de contenido, hermano de `P1-BETA-FRAGMENT-DEPTH`, y se cierra
AÑADIENDO.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CTX = (Path(__file__).resolve().parent.parent.parent
        / "frontend" / "src" / "context" / "AssessmentContext.jsx")


@pytest.fixture(scope="module")
def src() -> str:
    if not _CTX.is_file():
        pytest.skip("AssessmentContext.jsx no está en este árbol")
    return _CTX.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def rama_fallback(src) -> str:
    """El bloque que va desde el comentario del fallback local hasta el `return` que cierra la
    rama. Se corta por marcadores del código, no por un número de líneas mágico."""
    i = src.index("// Fallback Local")
    j = src.index("return localFallback.name", i) + len("return localFallback.name")
    return src[i:j]


def test_el_fallback_local_pregunta_por_el_pais(rama_fallback):
    """Sin esta pregunta, la red dominicana se le aplica a los seis países."""
    assert re.search(r"country", rama_fallback), (
        "la rama del fallback local no mira el país: a un español se le sigue insertando mangú "
        "cuando el swap falla"
    )


def test_en_beta_no_se_sustituye_el_plato(rama_fallback):
    """La decisión: no inventar recetas. Se avisa y se deja el plato original — que es lo que hace
    la rama del 422 diez líneas más arriba."""
    i = rama_fallback.index("country")
    guarda = rama_fallback[max(0, i - 400):i + 700]
    assert "return null" in guarda, (
        "la rama beta no devuelve null: si sustituye igualmente, el plato ajeno acaba PERSISTIDO "
        "en el plan y la lista de la compra deja de coincidir con la receta"
    )


def test_el_usuario_se_entera(rama_fallback):
    """Fallar en silencio sería el tercer defecto de los tres: el usuario vería un plato nuevo (o
    ninguno) sin saber que la IA no pudo."""
    i = rama_fallback.index("country")
    guarda = rama_fallback[max(0, i - 400):i + 700]
    assert "toast" in guarda, "la rama beta no avisa: el usuario no sabe que el cambio no salió"


def test_reusa_el_ssot_de_paises_del_frontend(src, rama_fallback):
    """`coerceCountry`/`COUNTRIES` de `config/countries.js` son el espejo del backend con test de
    paridad. Comparar contra `'DO'` a mano aquí sería la tabla que P1-DIET-CANON-SSOT prohíbe."""
    assert re.search(r"from\s+'\.\./config/countries'", src), (
        "AssessmentContext dejó de importar el SSOT de países"
    )
    assert re.search(r"coerceCountry|DEFAULT_COUNTRY", rama_fallback), (
        "la rama compara el país a mano en vez de usar el SSOT del frontend"
    )


def test_la_red_dominicana_sigue_intacta(src):
    """Byte-identidad: la tabla local no se toca ni se vacía. Un dominicano cuyo swap falla sigue
    recibiendo su alternativa."""
    assert "const DOMINICAN_MEALS" in src
    for plato in ("Mangú con Huevo", "La Bandera", "Sancocho Light"):
        assert plato in src, f"desapareció {plato!r} de la red local dominicana"


def test_no_se_inventaron_tablas_de_platos_por_pais(src):
    """El error que este arreglo NO comete. Cinco tablas de recetas con sus calorías escritas de
    memoria son datos nutricionales fabricados — la clase que costó la auditoría de procedencia.
    Si algún día se curan con fuente, este test se actualiza a propósito."""
    for inventada in ("SPANISH_MEALS", "MEXICAN_MEALS", "COLOMBIAN_MEALS", "AMERICAN_MEALS"):
        assert inventada not in src, (
            f"apareció {inventada}: si son recetas curadas con procedencia, actualiza este test; "
            f"si son de memoria, son datos fabricados"
        )
