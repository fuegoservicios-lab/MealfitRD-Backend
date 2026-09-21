# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-146 · 2026-09-20] «Continuar con Apple» NATIVO — la parte del backend, APAGADA por knob.

El dueño preguntó por qué el iPhone no ofrece Google: el OAuth por redirección no vuelve a la app, y Apple (4.8)
exige su propio botón si se ofrece Google. Neon Auth no tiene Apple como proveedor ⇒ flujo nativo: el binario pide
la credencial, `apple_auth` verifica el JWT y `apple_identity` lo resuelve a un usuario de `neon_auth`.
Los casos funcionales viven en `test_p1_plan_lote_146_apple_verify.py` y `…_apple_identity.py`; aquí, las anclas.
(+ lote 145, solo frontend: el modal de cerrar sesión cae al correo del PERFIL.)"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_el_algoritmo_es_fijo_y_el_login_nace_apagado():
    v = _src("apple_auth.py")
    assert 'algorithms=["RS256"]' in v and 'return _env_bool("MEALFIT_APPLE_SIGNIN", False)' in v
    assert "hmac.compare_digest" in v, "el nonce se compara en tiempo constante"


def test_el_enlace_vive_en_las_tablas_de_better_auth_sin_ddl():
    i = _src("apple_identity.py")
    assert 'INSERT INTO neon_auth.account ("accountId", "providerId", "userId", "updatedAt")' in i
    assert "CREATE TABLE" not in i.upper().replace("CREATE TABLE`", "")
    k = i.index('if not identidad.get("email_verified"):')
    assert k < i.index("fila = _por_correo(email)"), "jamás se busca por correo antes de exigirlo verificado"


def test_los_tests_que_abren_conexion_cruda_tienen_plazo_duro():
    """Dos gates de la misma noche murieron colgados 15 min en `_connect_gen`: `connect_timeout` no cubre eso."""
    for rel in ("tests/test_food_db_population_coverage.py", "tests/test_fix_round_2026_07_29_bad_aliases.py"):
        assert "hilo.join(25)" in _src(rel), rel


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', _src("app.py"))
    assert m and int(m.group(1)) >= 146
