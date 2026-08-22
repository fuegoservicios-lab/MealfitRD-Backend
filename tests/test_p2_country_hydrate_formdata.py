"""[P2-COUNTRY-HYDRATE-FORMDATA · 2026-08-21] El país persistido nunca hidrataba el formulario.

`_hydrateFieldQualifies` (`AssessmentContext.jsx`) decide qué campo del `health_profile` puede
sobrescribir el estado local. Su regla general es «hidrata si lo local está VACÍO», y el propio
comentario del fichero documenta que esa regla falla para los campos con **default truthy**:
`targetWeightAuto`, `includeSupplements` y `budgetCurrency` ya tienen su rama especial, esta última
porque «un presupuesto declarado en USD se re-leía como DOP en otro dispositivo».

`country` es el TERCER caso de esa misma clase. Entró después de aquel audit (F0, 2026-08-16) y
nadie lo revisó contra la regla: su default es `'DO'`, que es truthy, así que la rama genérica nunca
califica y el país persistido **no hidrata jamás**.

POR QUÉ IMPORTA MÁS QUE UN CAMPO MAL PRESELECCIONADO. No es que se vea mal: es que **un default
sembrado es indistinguible de una elección**. El usuario que eligió España abre la app en otro
dispositivo, o empieza un segundo plan, y ve «República Dominicana» ya marcada — no como un hueco
que rellenar sino como algo que él parece haber contestado. Avanza sin tocarlo y el plan sale
dominicano sin que nadie haya decidido eso. Es la lección que este proyecto ya tiene registrada del
país pisado por la renovación.

EL ARREGLO ES EL ESPEJO EXACTO DE `budgetCurrency`, y esa forma es deliberada: mismo tipo (string
con default truthy), misma semántica (el DB refleja la última elección persistida) y misma
protección de la edición viva (`editedFieldsRef`). Un criterio propio para `country` sería una
segunda regla para el mismo problema — justo lo que el comentario de ese fichero existe para evitar.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CTX = (Path(__file__).resolve().parent.parent.parent
        / "frontend" / "src" / "context" / "AssessmentContext.jsx")


@pytest.fixture(scope="module")
def predicado() -> str:
    if not _CTX.is_file():
        pytest.skip("AssessmentContext.jsx no está en este árbol")
    src = _CTX.read_text(encoding="utf-8", errors="replace")
    i = src.index("const _hydrateFieldQualifies")
    j = src.index("\nconst ", i + 1)
    return src[i:j]


def test_el_pais_tiene_su_rama_en_el_ssot_de_hidratacion(predicado):
    """Sin rama propia, el default `'DO'` (truthy) hace que la regla genérica —«hidrata si lo local
    está vacío»— no califique nunca."""
    assert re.search(r"k\s*===\s*'country'", predicado), (
        "`country` no tiene rama en `_hydrateFieldQualifies`: con default 'DO' truthy, el país "
        "persistido no hidrata NUNCA y el usuario vuelve a ver República Dominicana"
    )


def test_la_rama_reconoce_DO_como_hueco_no_como_eleccion(predicado):
    """El corazón del arreglo: `'DO'` tiene que contar como «todavía no ha elegido», igual que
    `'DOP'` cuenta para la moneda. Si la rama existe pero no menciona 'DO', no hace nada."""
    i = predicado.index("'country'")
    rama = predicado[i:i + 320]
    assert "'DO'" in rama, (
        "la rama de `country` no trata 'DO' como el default sembrado: sigue sin hidratar"
    )


def test_es_el_espejo_de_budgetCurrency(predicado):
    """Misma forma para el mismo problema. Si alguien inventa un criterio distinto para `country`,
    hay dos reglas para la misma clase de bug — que es lo que el comentario de ese fichero existe
    para impedir."""
    def _forma(clave, default):
        i = predicado.index(f"'{clave}'")
        rama = predicado[i:i + 320]
        return (f"'{default}'" in rama
                and "null" in rama and "undefined" in rama
                and "typeof v === 'string'" in rama
                and "v !== cur" in rama)
    assert _forma("budgetCurrency", "DOP"), "cambió la rama de budgetCurrency: revisa el espejo"
    assert _forma("country", "DO"), (
        "la rama de `country` no es el espejo de la de `budgetCurrency` — mismo tipo, mismo "
        "default truthy, misma semántica: debería ser la misma forma"
    )


def test_no_hidrata_encima_de_una_eleccion_real(predicado):
    """El error opuesto: si la rama aceptara cualquier valor local, el perfil pisaría al usuario
    que acaba de elegir México en este dispositivo. `v !== cur` y el chequeo del default son lo
    que lo impide."""
    i = predicado.index("'country'")
    rama = predicado[i:i + 320]
    assert "v !== cur" in rama, "la rama de `country` pisaría una elección viva del usuario"


def test_el_default_sembrado_sigue_siendo_DO(predicado):
    """Ancla cruzada: si alguien cambia el default del formulario y no la rama, la rama deja de
    reconocer el hueco y el bug vuelve en silencio. Los dos valores tienen que moverse juntos."""
    src = _CTX.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"country:\s*'DO'", src), (
        "el default de `country` en `initialFormData` ya no es 'DO': la rama de "
        "`_hydrateFieldQualifies` lo sigue asumiendo y dejaría de hidratar"
    )
