"""[P1-PLAN-LOTE-1021 · A-24] Logs sin datos de salud ni texto del usuario: recuento y huella, no contenido.

Barrido del journal de producción (24 h, 15-nov): el router de memoria escribía la frase del usuario y su reescritura
(«alergias o intolerancias a ingredientes de panes y alimentos con gluten»), el caché de embeddings la repetía, y en el
código `[G-20 SOURCE-INTENT-GUARD]` escribía los argumentos de la herramienta y el texto del usuario, y
`[P0-CHAT-ALLERGY-MERGE]` las listas de alergias/condiciones completas. Son datos de salud en los logs del servidor.

`huella(texto)` da lo necesario para correlacionar dos líneas del mismo turno sin guardar el contenido.
"""
from __future__ import annotations

import hashlib


def huella(texto) -> str:
    """`len=NN h=xxxxxxxx` (sha256 truncado; el mismo texto da la misma huella)."""
    try:
        t = str(texto or "")
        return f"len={len(t)} h={hashlib.sha256(t.encode('utf-8', 'replace')).hexdigest()[:8]}"
    except Exception:
        return "len=? h=?"


def claves(args) -> str:
    """Solo los NOMBRES de los argumentos de una herramienta (`field`, `new_value`…), nunca sus valores."""
    try:
        return ",".join(sorted(str(k) for k in args)) if isinstance(args, dict) else type(args).__name__
    except Exception:
        return "?"
