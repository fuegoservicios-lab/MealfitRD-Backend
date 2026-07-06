"""[P1-RECALC-SERIALIZE · 2026-07-06] withRecalcLock serializa (cola de promesas).

Carrera reportada por el owner: eligió marca Campos→Wala en el picker y "la lista
debió actualizarse en tiempo real, pero no lo hizo" — el PDF quedó en Campos (y
con el laurel pre-fix RD$701). Causa: `withRecalcLock` era solo un FLAG con
try/finally (consultado por restoreSessionData) — dos recalcs corrían AMBOS en
paralelo. El auto-refresh del Dashboard (P2-SHOPLIST-AUTO-REFRESH, ~30s en planes
grandes) seguía in-flight cuando el usuario eligió Wala; el recalc de la marca
terminó primero y el del auto-refresh aterrizó DE ÚLTIMO → pisó lista/DB con el
costeo viejo.

Fix: cadena de promesas en withRecalcLock — todo recalc espera al anterior; el
último disparado (la elección del usuario) SIEMPRE gana, en UI y en DB. Con
escape de 90s si un op se cuelga (jamás un Dashboard congelado).
"""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CTX = (BACKEND.parent / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")


def test_recalc_lock_serializes_via_promise_chain():
    assert "P1-RECALC-SERIALIZE" in CTX
    assert "recalcChainRef" in CTX, "la cola vive en un ref (sobrevive re-renders)"
    i = CTX.index("const withRecalcLock")
    body = CTX[i:i + 1200]
    assert "recalcChainRef.current.then(_run, _run)" in body, (
        "cada op se encola tras la anterior — éxito O fallo (la cadena no muere)"
    )
    assert "Promise.race" in body and "90000" in body, (
        "escape de 90s: un op colgado no congela los recalcs siguientes"
    )


def test_lock_flag_semantics_preserved():
    """El flag sigue existiendo para consumidores read-only (restoreSessionData
    omite sync mientras hay recalc in-flight) — la cola es ADITIVA."""
    i = CTX.index("const withRecalcLock")
    body = CTX[i:i + 1200]
    assert "setRecalcLock(true)" in body and "setRecalcLock(false)" in body
    assert re.search(r"finally\s*\{\s*\n?\s*setRecalcLock\(false\)", CTX[i:i + 1200]), (
        "release garantizado en finally (P0-B2 intacto)"
    )


def test_return_value_still_propagates():
    i = CTX.index("const withRecalcLock")
    body = CTX[i:i + 1200]
    assert "return p;" in body, "el caller sigue recibiendo el valor de asyncFn (contrato P0-B2)"
