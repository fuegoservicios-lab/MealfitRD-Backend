"""[P1-NEON-AUTH-MIGRATION · 2026-06-13] Anclas de la migración de Auth
Supabase → Neon Auth (Better Auth). Cierra el último acoplamiento a Supabase:
el frontend y el backend quedan 100% sobre Neon (datos + auth).

Contratos cubiertos (parser-based, sin red):
  1. `neon_auth.py` valida JWTs vía JWKS con algoritmo FIJO EdDSA (sin
     algorithm-confusion) y es fail-secure (retorna None ante cualquier fallo).
  2. `auth.py` usa `verify_neon_jwt` y NO `supabase.auth.get_user` (P0-AUDIT-1
     preservado bajo el nuevo proveedor).
  3. `db_core.py` NO importa el paquete `supabase` ni crea un cliente
     (`_storage_client = None` — placeholder del object storage de visual_diary,
     pendiente de migrar a un provider nuevo; vision está disabled).
  4. `requirements.txt` no lista `supabase`; sí `PyJWT` + `cryptography`.
  5. Cero `from supabase import` / `import supabase` en código prod.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (_BACKEND_ROOT / rel).read_text(encoding="utf-8")


def test_neon_auth_module_fixed_algorithm_and_fail_secure():
    src = _read("neon_auth.py")
    # Algoritmo FIJO EdDSA — sin esto, algorithm-confusion / none.
    assert re.search(r'algorithms\s*=\s*\[\s*["\']EdDSA["\']\s*\]', src), (
        "neon_auth.py debe fijar algorithms=['EdDSA'] en jwt.decode "
        "(defensa contra algorithm-confusion / alg=none)."
    )
    # Valida issuer Y audience.
    assert "issuer=" in src and "audience=" in src, (
        "neon_auth.py debe validar issuer + audience del token."
    )
    # Fail-secure: la función pública retorna None ante excepción.
    m = re.search(r"def verify_neon_jwt\(.*?(?=\ndef |\Z)", src, re.DOTALL)
    assert m and "return None" in m.group(0), (
        "verify_neon_jwt debe retornar None ante fallo (fail-secure)."
    )


def test_auth_uses_neon_not_supabase():
    src = _read("auth.py")
    assert "verify_neon_jwt" in src, "auth.py debe usar verify_neon_jwt."
    assert "supabase.auth.get_user(" not in src, (
        "auth.py NO debe invocar supabase.auth.get_user (Supabase Auth eliminado)."
    )


def test_db_core_no_supabase_client():
    src = _read("db_core.py")
    assert not re.search(r"^\s*from supabase import", src, re.MULTILINE), (
        "db_core.py NO debe importar el paquete supabase."
    )
    assert not re.search(r"create_client\(", src), (
        "db_core.py NO debe crear un cliente supabase."
    )
    assert re.search(r"^_storage_client = None", src, re.MULTILINE), (
        "db_core.py debe exponer `_storage_client = None` (placeholder del "
        "object storage de visual_diary, pendiente de migrar a provider nuevo)."
    )


def test_requirements_swapped():
    src = _read("requirements.txt")
    assert not re.search(r"(?mi)^supabase==", src), (
        "requirements.txt NO debe listar supabase."
    )
    assert re.search(r"(?mi)^PyJWT==", src), "requirements.txt debe listar PyJWT."
    assert re.search(r"(?mi)^cryptography==", src), (
        "requirements.txt debe listar cryptography (verificación Ed25519)."
    )


def test_no_supabase_package_import_in_prod():
    """Ningún módulo prod (no test/script/scratch) importa el paquete supabase."""
    offenders = []
    skip_dirs = {"tests", "scripts", "scratch", "venv-test", "venv", ".venv"}
    for p in _BACKEND_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from supabase import|import supabase)\b", line):
                offenders.append(f"{p.relative_to(_BACKEND_ROOT)}:{i}")
    assert not offenders, (
        "Imports del paquete supabase en código prod (prohibido post "
        "P1-NEON-AUTH-MIGRATION):\n" + "\n".join(offenders)
    )
