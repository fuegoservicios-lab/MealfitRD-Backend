"""[P2-ROLLBACK-QUERY-CIEGA-AL-NULL · 2026-08-23] G43: la consulta del runbook de rollback
devolvía 0 SIEMPRE, y ese 0 se leía como «rollback limpio, puedes apagar el knob».

MEDIDO contra producción (2026-08-23):

    chunks vivos ......................... 93
    de ellos con country NULL ............ 86
    la consulta DEL RUNBOOK devuelve ..... 0
    la consulta CORREGIDA devuelve ....... 0   ← el mismo número, por razones distintas

Y en SQL: `SELECT (NULL NOT IN ('DO')) IS NULL` → **true**. `NOT IN` sobre un campo NULL no es
ni verdadero ni falso, así que la fila se omite EN SILENCIO. Los 86 no salían — y con ellos no
habría salido ningún chunk beta.

POR QUÉ ERA INVISIBLE: el 0 de hoy es CORRECTO, porque hoy no hay chunks beta. La consulta
acertaba por casualidad, y habría dado 0 igual con la cola llena de españoles. *Un verificador
que acierta por casualidad no se distingue de uno que funciona hasta el día que importa.*

Y medía la fuente equivocada: al hacer pickup el worker sobreescribe `form_data` con el perfil
VIVO y `country` no está en `_protected_keys`, así que el país que decide de verdad es el del
perfil cuando el snapshot no lo trae. `COALESCE(snapshot, perfil, 'DO')` reproduce esa misma
precedencia.

Comprobado tras el cambio: agrupando los 93 chunks vivos por la expresión nueva salen 93
clasificados, ninguno fuera.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_RUNBOOK = _BACKEND / "docs" / "country_system_f1.md"


def _sql_del_runbook() -> str:
    """Sólo las líneas de SQL del bloque de rollback: los comentarios explican el defecto y
    nombran `NOT IN` a propósito, así que un guard textual ingenuo se acusaría a sí mismo."""
    src = io.open(_RUNBOOK, encoding="utf-8").read()
    i = src.index("P2-ROLLBACK-QUERY-CIEGA-AL-NULL")
    bloque = src[i:src.index("> ```", i)]
    return "\n".join(l for l in bloque.split("\n") if not l.lstrip("> ").startswith("--"))


def test_ninguna_consulta_del_rollback_compara_con_not_in_sobre_un_campo_nullable():
    """EL defecto: `NULL NOT IN ('DO')` es NULL, o sea que la fila se omite en silencio."""
    sql = _sql_del_runbook()
    assert "NOT IN ('DO')" not in sql, (
        "volvió el NOT IN sobre el país: con 86 de 93 chunks a NULL, la consulta devuelve 0 "
        "aunque la cola esté llena de chunks beta"
    )


def test_las_tres_consultas_resuelven_el_pais_por_coalesce():
    """Las TRES (contar, congelar, convertir): si una se queda atrás, el operador cuenta bien y
    actúa mal — que es peor que no contar."""
    sql = _sql_del_runbook()
    assert sql.count("COALESCE(q.pipeline_snapshot->'form_data'->>'country',") >= 3, (
        "alguna de las tres consultas del rollback dejó de resolver el país por COALESCE"
    )


def test_la_precedencia_es_la_del_worker_y_no_otra():
    """El worker sobreescribe `form_data` con el perfil vivo al hacer pickup, así que el orden
    correcto es snapshot → perfil → 'DO'. Invertirlo daría un país distinto del que el motor
    va a usar de verdad."""
    sql = _sql_del_runbook()
    m = re.search(r"COALESCE\(q\.pipeline_snapshot->'form_data'->>'country',\s*\n?\s*>?\s*"
                  r"up\.health_profile->>'country',\s*'DO'\)", sql)
    assert m, "la precedencia snapshot → perfil → 'DO' ya no es la del runbook"


def test_las_mutaciones_unen_user_profiles():
    """Un UPDATE que filtra por el perfil necesita el JOIN, o el WHERE no compila."""
    sql = _sql_del_runbook()
    assert sql.count("FROM user_profiles up") >= 2, (
        "los dos UPDATE del rollback perdieron el JOIN con user_profiles"
    )
    assert "LEFT JOIN user_profiles up ON up.id = q.user_id" in sql, (
        "el SELECT de conteo debe usar LEFT JOIN: un chunk cuyo perfil falte no puede "
        "desaparecer del recuento — es justo el caso que este gap cierra"
    )


def test_el_porque_queda_escrito_con_su_medicion():
    """El siguiente que lea esto tiene que poder distinguir «0 porque está limpio» de «0 porque
    la consulta es ciega». Sin la medición al lado, el número no dice cuál de las dos."""
    src = io.open(_RUNBOOK, encoding="utf-8").read()
    for pista in ("93", "86", "NULL NOT IN ('DO')", "_protected_keys"):
        assert pista in src, f"el runbook perdió la evidencia: falta «{pista}»"
