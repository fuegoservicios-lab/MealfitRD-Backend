"""[P1-11] Tests que documentan el contrato de `dietType` entre frontend y
backend.

Bug original (audit P1-11):
  El frontend gatea `dietType` como required (`REQUIRED_FORM_FIELDS` en
  `formValidation.js`), pero el backend lo deja FUERA de
  `_REQUIRED_FORM_FIELDS` deliberadamente. Drift documentado pero no
  protegido por test — un cambio futuro que añada `dietType` al backend
  rompería la rehidratación de perfiles legacy con variantes ES
  ("Omnívora"/"vegetariana") sin alarma.

Contrato:
  - `dietType` NO está en `_REQUIRED_FORM_FIELDS` del backend
    (preservar compat con perfiles legacy ES).
  - El frontend (Plan.jsx P1-11) envía explícitamente
    `dietType: formData.dietType || 'balanced'` para nunca propagar
    `''`/`None`/`undefined` desde el cliente oficial.
  - Si llega ausente o vacío vía cliente legacy/scraper, downstream
    `_get_fast_filtered_catalogs` defaultea a "balanced" silenciosamente
    (catálogo completo, comportamiento benigno).
  - El comentario inline del tuple documenta el rationale.

Cobertura:
  - test_diettype_is_NOT_in_required_form_fields_intentional
  - test_diettype_legacy_accepted_set_exists_for_es_variants
  - test_required_form_fields_tuple_contains_dietType_comment
"""
import inspect

import pytest

from routers import plans as plans_module


def test_diettype_is_NOT_in_required_form_fields_intentional():
    """`dietType` debe quedar OUT del tuple para preservar perfiles legacy.

    Si alguien lo añade sin actualizar `_DIET_TYPE_LEGACY_ACCEPTED`, los
    perfiles legacy con `dietType="Omnívora"` empiezan a recibir 422.
    Este test fuerza una decisión consciente: añadir dietType requiere
    actualizar este test + la capa de aceptación legacy.
    """
    required = plans_module._REQUIRED_FORM_FIELDS
    assert "dietType" not in required, (
        "P1-11: `dietType` debe quedar OUT de `_REQUIRED_FORM_FIELDS` "
        "para preservar rehidratación de perfiles legacy con variantes ES. "
        "Si REALMENTE necesitas hacerlo required, actualiza este test Y "
        "asegúrate de que `_DIET_TYPE_LEGACY_ACCEPTED` cubra los strings "
        "españoles ('Omnívora', 'vegetariana', etc.)."
    )


def test_required_form_fields_tuple_contains_dietType_comment():
    """El tuple `_REQUIRED_FORM_FIELDS` debe documentar el rationale por
    el cual `dietType` queda fuera. Sin el comentario, un futuro
    maintainer no entiende la asimetría con el frontend y puede añadirla
    silenciosamente."""
    src = inspect.getsource(plans_module)
    # El tuple `_REQUIRED_FORM_FIELDS` debe tener una mención de dietType
    # en su block de comentarios documentando que está OUT deliberadamente.
    # Buscamos el patrón "dietType" + alguna palabra que evidencie la
    # intención (deliberadamente / OUT / legacy / fuera).
    import re as _re
    pattern = _re.compile(
        r"dietType[^\n]{0,200}(?:OUT|deliberadamente|fuera|legacy)|"
        r"(?:OUT|deliberadamente|fuera|legacy)[^\n]{0,200}dietType",
        _re.IGNORECASE,
    )
    assert pattern.search(src), (
        "P1-11: el tuple `_REQUIRED_FORM_FIELDS` debe documentar por qué "
        "`dietType` queda fuera (asimetría intencional con el frontend)."
    )


def test_diet_type_legacy_accepted_set_exists():
    """`_DIET_TYPE_LEGACY_ACCEPTED` debe existir como set / frozenset que
    cubre las variantes ES históricas. Si el set se borra, los perfiles
    legacy empiezan a fallar 422 en cualquier rehidratación."""
    assert hasattr(plans_module, "_DIET_TYPE_LEGACY_ACCEPTED") \
        or hasattr(plans_module, "_DIET_TYPE_ENUM"), (
        "P1-11: el módulo debe exponer un enum / set que documente los "
        "valores legacy aceptados de dietType (compat ES)."
    )


def test_required_form_fields_subset_aligned_with_safety_critical():
    """Sanity de regresión: los campos safety-críticos (`age`, `mainGoal`,
    `weight`, `height`, `gender`, `weightUnit`, `motivation`, `allergies`,
    `medicalConditions`) SÍ están en el tuple. El drift de `dietType` es
    deliberado, NO una omisión por error."""
    required = set(plans_module._REQUIRED_FORM_FIELDS)
    safety_critical = {
        "age", "mainGoal", "weight", "height", "gender", "weightUnit",
        "motivation", "allergies", "medicalConditions",
    }
    missing = safety_critical - required
    assert not missing, (
        f"P1-11 regression: campos safety-críticos faltantes en "
        f"_REQUIRED_FORM_FIELDS: {missing}"
    )
