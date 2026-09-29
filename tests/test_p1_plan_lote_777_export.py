"""[P1-PLAN-LOTE-777 · 2026-09-28] Los regalos de la cuenta (`account_grants`) aparecen en la
exportación de datos del usuario (`GET /api/account/export`) y los ids del PERSONAL que los
otorgó o revocó nunca salen en ese JSON.

Por qué las dos mitades: la Política de Privacidad §2 promete «aparece en la exportación de sus
datos» — omitir la tabla sería incumplirla. Pero `account_grants.granted_by`/`revoked_by` son el
`user_id` del ADMIN que actuó, no del titular de la cuenta: exportarlos filtraría la identidad de
alguien del equipo hacia un usuario cualquiera. `reason` (el motivo del regalo) SÍ es información
sobre la PERSONA — sobre por qué recibió el regalo, no sobre quién lo dio — y debe seguir
saliendo.

Parser-based (regex sobre source) — no levanta el stack (pytest local → Neon cuelga), mismo
patrón que test_p2_privacy_settings.py (que ya ancla el resto del contrato del endpoint:
throttle, no-IDOR, quota-exempt, strip de `embedding`).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_APP = _BACKEND / "app.py"
_DOC = _BACKEND / "docs" / "regalos_cuenta.md"


def _app_src() -> str:
    assert _APP.exists(), f"No existe {_APP}"
    return _APP.read_text(encoding="utf-8")


def _tables_tuple_src(src: str) -> str:
    m = re.search(r"_ACCOUNT_EXPORT_TABLES\s*=\s*\((.*?)\n\)", src, re.DOTALL)
    assert m, "Falta la tupla _ACCOUNT_EXPORT_TABLES en app.py."
    return m.group(1)


def _stripped_tuple_src(src: str) -> str:
    m = re.search(r"_ACCOUNT_EXPORT_STRIPPED_KEYS\s*=\s*\((.*?)\)", src)
    assert m, "Falta la tupla _ACCOUNT_EXPORT_STRIPPED_KEYS en app.py."
    return m.group(1)


def test_account_grants_esta_en_la_tabla_del_export():
    tables_src = _tables_tuple_src(_app_src())
    assert re.search(r'\("account_grants",\s*"user_id",\s*\d+\)', tables_src), (
        "account_grants debe estar en _ACCOUNT_EXPORT_TABLES con cap por filas: la Política de "
        "Privacidad §2 promete que los regalos de la cuenta aparecen en la exportación."
    )


def test_embedding_sigue_primero_en_los_stripped_keys():
    """Contrato compartido con test_p2_privacy_settings.py::test_export_payload_hygiene — ese
    test exige `_ACCOUNT_EXPORT_STRIPPED_KEYS = ("embedding"` literal. Si esta suite reordenara
    la tupla al añadir los ids del personal, tumbaría esa regresión sin tocarla."""
    src = _app_src()
    assert re.search(r'_ACCOUNT_EXPORT_STRIPPED_KEYS\s*=\s*\(\s*"embedding"', src), (
        "`embedding` debe seguir siendo el primer elemento de _ACCOUNT_EXPORT_STRIPPED_KEYS."
    )


def test_ids_del_personal_se_excluyen_del_export():
    stripped_src = _stripped_tuple_src(_app_src())
    for campo in ("granted_by", "revoked_by"):
        assert f'"{campo}"' in stripped_src, (
            f"{campo} identifica al ADMIN que otorgó/revocó el regalo, no al titular de la "
            "cuenta — no debe salir en la exportación de datos del usuario."
        )


def test_el_motivo_del_regalo_si_se_exporta():
    """`reason` es información SOBRE la persona (por qué recibió el regalo), no del personal —
    la invariante contraria a la del test anterior. Si alguien la stripeara "por si acaso" junto
    a granted_by/revoked_by, la Política de Privacidad §2 dejaría de cumplirse en silencio."""
    stripped_src = _stripped_tuple_src(_app_src())
    assert '"reason"' not in stripped_src, (
        "`reason` (el motivo del regalo) es información sobre la persona, no del personal: "
        "debe seguir exportándose. No pertenece a _ACCOUNT_EXPORT_STRIPPED_KEYS."
    )


def test_doc_de_regalos_declara_la_exportacion_y_la_higiene():
    doc = _DOC.read_text(encoding="utf-8")
    for trozo in ("exportación", "granted_by", "revoked_by"):
        assert trozo in doc, (
            f"backend/docs/regalos_cuenta.md debe mencionar «{trozo}»: documenta que los "
            "regalos salen en la exportación sin los ids del personal."
        )
