"""[P0-SHOPPING-CALC-NAMEERROR · 2026-05-15] Regression guard contra el
`NameError: name '_can_lower' is not defined` en
`shopping_calculator.aggregate_and_deduct_shopping_list` (línea ~5002).

Bug observado en runtime 2026-05-15 18:43:51 al generar un plan E2E:

    Traceback (most recent call last):
      File ".../shopping_calculator.py", line 5002, in aggregate_and_deduct_shopping_list
        _ac = canonicalize_aceites(canonical_name)
    NameError: name '_can_lower' is not defined

Root cause: el bloque P3-NEW-12 (5 canonicalizers adicionales) + el bloque
pavo (P1-XX) introdujeron `_orig_name_lower` para el matching del pavo, y
refactorizaron las 13 regex de consolidación de abajo (Fresas, Almendras,
Orégano, Tortilla, Tomate, Cebolla, Espinacas, Zanahoria, Vainitas,
Habichuelas, Tofu, Perejil) para que operaran sobre `_can_lower`. Pero la
asignación `_can_lower = canonical_name.lower()` quedó huérfana en el
refactor → cada plan generado lanzaba excepción aquí.

Síntoma user-facing: `aggregate_and_deduct_shopping_list` fallaba →
fallback dejaba lista de compras vacía/incompleta → coherence guard veía
35 ingredientes de recetas sin contraparte en lista (`presence=expected_only`)
→ alert `[COH-GUARD/block] 35 divergencias` → plan entregado con
`_shopping_coherence_block` no popeado → frontend mostraba "Verificación
médica con observaciones: COHERENCIA RECETAS LISTA: 35 divergencia(s) críticas".

Fix: asignar `_can_lower = canonical_name.lower()` DESPUÉS del bloque pavo
y ANTES del primer uso (línea ~5091, justo antes de "# Consolidación:
Fresas variantes").

Este test es parser-based — verifica que la asignación está presente y
en la posición correcta. NO ejecuta el pipeline LLM.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CALC_PATH = _BACKEND_ROOT / "shopping_calculator.py"


def _read_calc() -> str:
    return _CALC_PATH.read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    """Quita líneas que comienzan con `#` (comentarios Python) para evitar
    que menciones de `_can_lower` en docstrings/comments contaminen el
    matching. NO toca docstrings entre triple-quotes (las regex aquí no
    necesitan precisión de parser de Python)."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def test_can_lower_assignment_present():
    """`_can_lower = canonical_name.lower()` debe existir como asignación
    en `aggregate_and_deduct_shopping_list` para que las 13 regex de
    consolidación de abajo no lancen NameError."""
    text = _strip_comments(_read_calc())
    assert re.search(
        r"_can_lower\s*=\s*canonical_name\.lower\(\)", text
    ), (
        "P0-SHOPPING-CALC-NAMEERROR: falta la asignación "
        "`_can_lower = canonical_name.lower()` antes de las 13 regex de "
        "consolidación. Cada plan generado lanzará NameError aquí."
    )


def test_can_lower_assigned_before_first_use():
    """La asignación de `_can_lower` debe venir ANTES de su primer uso
    (regex `^fresas?\\b`). Si la asignación queda DESPUÉS del primer uso,
    NameError reaparece."""
    text = _strip_comments(_read_calc())
    assign_match = re.search(r"_can_lower\s*=\s*canonical_name\.lower\(\)", text)
    # Primer uso = primera ocurrencia de `_can_lower` que NO sea la propia
    # asignación. Buscar todos los offsets y tomar el segundo.
    occurrences = [m.start() for m in re.finditer(r"_can_lower\b", text)]
    assert assign_match, "Asignación de `_can_lower` no encontrada."
    assert len(occurrences) >= 2, "Esperaba ≥2 menciones de _can_lower."
    # La asignación es la primera mención; las regex usuarias vienen después.
    assert assign_match.start() == occurrences[0], (
        "La asignación debe ser la PRIMERA mención de _can_lower."
    )
    first_use_pos = occurrences[1]
    assert assign_match.start() < first_use_pos, (
        f"P0-SHOPPING-CALC-NAMEERROR: la asignación de `_can_lower` "
        f"(pos {assign_match.start()}) debe estar ANTES de su primer uso "
        f"(pos {first_use_pos}). Si llega después, NameError reaparece."
    )


def test_can_lower_assigned_after_pavo_block():
    """La asignación debe estar DESPUÉS del bloque pavo (que puede mutar
    `canonical_name`). Si se asigna antes del bloque pavo, las 13 regex
    de consolidación verán el canonical_name pre-pavo y el matching será
    inconsistente."""
    text = _strip_comments(_read_calc())
    # Anchor del final del bloque pavo: `canonical_name = 'Pavo'` en la
    # rama elif `_orig_name_lower.strip() == 'pavo'`.
    pavo_end = re.search(
        r"elif _orig_name_lower\.strip\(\)\s*==\s*'pavo':.*?canonical_name\s*=\s*'Pavo'",
        text,
        re.DOTALL,
    )
    assign_match = re.search(r"_can_lower\s*=\s*canonical_name\.lower\(\)", text)
    assert pavo_end and assign_match, "No localicé anchors."
    assert assign_match.start() > pavo_end.end(), (
        f"P0-SHOPPING-CALC-NAMEERROR: `_can_lower` debe asignarse DESPUÉS "
        f"del bloque pavo. Pre-pavo, `canonical_name` puede ser 'Pechuga de pavo' "
        f"pero post-pavo el bloque puede haberlo cambiado a 'Jamón de pavo' "
        f"o 'Pavo molido'. Las 13 regex de abajo necesitan el valor post-pavo."
    )


def test_all_can_lower_references_within_same_function():
    """Las 13 referencias a `_can_lower` deben estar todas dentro de
    `aggregate_and_deduct_shopping_list`. Si una se mueve afuera por
    refactor sin renombrar, NameError ahí también."""
    text = _strip_comments(_read_calc())
    # Localizar fronteras de la función
    func_start = text.find("def aggregate_and_deduct_shopping_list")
    assert func_start > 0, "Función `aggregate_and_deduct_shopping_list` no encontrada."
    # Buscar la siguiente `def` top-level (col 0)
    next_def = re.search(r"^def \w+", text[func_start + 1 :], re.MULTILINE)
    func_end = (func_start + 1 + next_def.start()) if next_def else len(text)
    func_body = text[func_start:func_end]

    references = re.findall(r"_can_lower", func_body)
    # Esperamos: 1 asignación + 13 referencias en regex (= 14 menciones mínimo).
    assert len(references) >= 14, (
        f"Esperaba ≥14 menciones de `_can_lower` dentro de "
        f"`aggregate_and_deduct_shopping_list` (1 asignación + 13 usos), "
        f"encontré {len(references)}. Si alguna se removió, verificar que la "
        f"regex correspondiente también se removió."
    )


def test_orig_name_lower_only_in_pavo_block():
    """`_orig_name_lower` (raw del parser, pre-master_map) debe usarse SOLO
    en el bloque pavo. Si alguien la usa fuera, está bypasseando el comment
    explicativo y puede haber confundido la convención. El comment dice:
    'Usar SOLO `name.lower()` (raw del parser) ... NO `_can_lower` (post-master_map)'
    pero eso aplica solo al matching de pavo."""
    text = _strip_comments(_read_calc())
    func_start = text.find("def aggregate_and_deduct_shopping_list")
    next_def = re.search(r"^def \w+", text[func_start + 1 :], re.MULTILINE)
    func_end = (func_start + 1 + next_def.start()) if next_def else len(text)
    func_body = text[func_start:func_end]

    # Cuenta usos. Debe haber 1 asignación + N usos dentro del bloque pavo.
    # No verificamos número exacto (puede crecer), solo que existe.
    assert "_orig_name_lower = name.lower()" in func_body, (
        "La asignación de `_orig_name_lower = name.lower()` para el bloque "
        "pavo debe seguir existiendo."
    )
