"""[P1-PAUSE-AGE-TRUE-CLOCK · 2026-08-07] La edad de una pausa no es
`NOW() - updated_at`: es el reloj que la propia pausa reinicia.

Un chunk pausado por nevera entra en bucle — pausa → TTL 12h → escala a modo
flexible → vuelve a violar la restricción → se re-pausa — y en cada vuelta, más
los recordatorios de cada 4h, alguien escribe `updated_at = NOW()`. El TTL que
debía rescatarlo mide justo esa columna, así que el chunk **nunca cumple 12h**.

Medido en producción 2026-08-07:

    chunk            NOW()-updated_at    edad REAL de la pausa
    76a6836d wk2           7,0 h                19,0 h
    9cf5e313 wk2           5,3 h                17,3 h

`_mode_history` de esos planes muestra la MISMA escalada repetida el 05, el 06 y
el 07: tres días dando vueltas sin entregar los días al usuario.

`_pantry_pause_started_at` sobrevive a las vueltas (se escribe con `setdefault`),
así que es el ancla honesta. El regex del CASE evita que un valor corrupto
reviente el cast y con él el cron entero: sin match, degrada al reloj viejo.

Este test es parser-based a propósito: `cron_tasks` no se puede importar sin el
stack LLM completo, y el contrato que hay que defender es "el fragmento existe,
lo usa el recovery, y nadie vuelve a medir desde `updated_at` ahí".
"""
from __future__ import annotations

import re
from pathlib import Path

_CRON = Path(__file__).resolve().parents[1] / "cron_tasks.py"
_SRC = _CRON.read_text(encoding="utf-8")


def _cuerpo(nombre: str) -> str:
    ini = _SRC.index(f"def {nombre}(")
    sig = re.search(r"\ndef [a-zA-Z_]", _SRC[ini + 10:])
    return _SRC[ini: ini + 10 + (sig.start() if sig else len(_SRC))]


def test_el_fragmento_ancla_en_pantry_pause_started_at():
    frag = _SRC[_SRC.index("_PAUSE_AGE_SECONDS_SQL = "):][:900]
    assert "_pantry_pause_started_at" in frag, (
        "El fragmento dejó de anclar en el inicio real de la pausa."
    )
    assert "ELSE" in frag and "updated_at" in frag, (
        "Falta el fallback a `updated_at`: una pausa de otro tipo (sin la clave "
        "de nevera) quedaría sin edad y el cron la trataría como recién creada."
    )


def test_el_cast_esta_protegido_por_regex():
    """Sin el guard, un `_pantry_pause_started_at` corrupto revienta el cast y
    con él la consulta ENTERA del cron — no una fila, el tick completo."""
    frag = _SRC[_SRC.index("_PAUSE_AGE_SECONDS_SQL = "):][:900]
    assert "~ '^" in frag and "timestamptz" in frag


def test_el_fragmento_respeta_el_alias():
    """El recovery consulta sin alias; otros call sites usan `q.`. Si el
    fragmento ignorara el alias, el SQL no compilaría en el segundo caso."""
    frag = _SRC[_SRC.index("_PAUSE_AGE_SECONDS_SQL = "):][:900]
    assert frag.count("{alias}") >= 3


def test_el_recovery_usa_el_fragmento_y_no_el_reloj_viejo():
    cuerpo = _cuerpo("_recover_pantry_paused_chunks")
    assert "pause_age_seconds_sql()" in cuerpo, (
        "`_recover_pantry_paused_chunks` volvió a medir por su cuenta. El TTL de "
        "12h no se cumplirá nunca mientras los recordatorios refresquen updated_at."
    )
    assert "EXTRACT(EPOCH FROM (NOW() - updated_at))" not in cuerpo, (
        "Quedó el cálculo viejo: mide el reloj que la propia pausa reinicia."
    )
    assert "P1-PAUSE-AGE-TRUE-CLOCK" in cuerpo


def test_la_llamada_va_dentro_de_una_f_string():
    """Trampa real de esta edición: estos SQL viven en `\"\"\"...\"\"\"`, donde
    `\" + f() + \"` es TEXTO, no concatenación. La primera versión de este fix
    dejó literalmente `\" + pause_age_seconds_sql() + \"` dentro del SELECT."""
    cuerpo = _cuerpo("_recover_pantry_paused_chunks")
    assert '" + pause_age_seconds_sql' not in cuerpo, (
        "El fragmento quedó como texto literal dentro del SQL."
    )
    assert 'f"""' in cuerpo, "El bloque SQL dejó de ser f-string: la interpolación no ocurre."
