# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1007 · G-20 · 2026-11-14] Validación compartida contra prompt-injection en tools de perfil/ajustes.

Evita lógica duplicada entre `tools.py::update_form_field` y `ajustes_de_la_app.py::cambiar_ajuste`.
Rechaza delimitadores de control/marcadores, longitud abusiva e instrucciones de override sin bloquear alimentos
ni términos cotidianos en español/inglés.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_MARKER = re.compile(r"<<\s*[A-Z_]{3,}|<\s*/?\s*(?:system|instruction|assistant|tool_call)\b|\[INST\]", re.IGNORECASE)
_RE_INJECTION = re.compile(
    r"(?:"
    r"ignora(?:r)?\s+(?:las\s+)?instrucciones|"
    r"ignore\s+(?:all\s+|previous\s+)?instructions|"
    r"system\s*prompt|"
    r"\b(?:act[uú]a\s+como|act\s+as|you\s+are\s+now|bypass\s+all)\b|"
    r"(?:drop\s+table|delete\s+from|insert\s+into)\s+[a-z_]+"
    r")",
    re.IGNORECASE,
)


def contiene_inyeccion_texto(valor: Any, *, max_len: int = 240) -> Optional[str]:
    """Devuelve código corto del motivo si `valor` parece inyección/abuso, o `None` si es dato limpio."""
    if valor is None or isinstance(valor, bool):
        return None
    s = str(valor)
    if len(s) > max_len:
        return "too_long"
    if _RE_CONTROL.search(s):
        return "control_chars"
    if _RE_MARKER.search(s):
        return "marker_injection"
    if _RE_INJECTION.search(s):
        return "instruction_injection"
    return None
