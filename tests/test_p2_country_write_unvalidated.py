"""G45: las cuatro puertas de health_profile.country comparten una validación."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_services_rechaza_antes_de_leer_o_crear_perfil(monkeypatch) -> None:
    import services

    monkeypatch.setattr(
        services,
        "get_user_profile",
        lambda _uid: (_ for _ in ()).throw(AssertionError("leyó DB antes de validar")),
    )
    with pytest.raises(HTTPException) as caught:
        services.merge_form_data_with_profile("user-1", {"country": "España"})
    assert caught.value.status_code == 400


def test_services_acepta_codigo_y_ausencia_sin_coercer(monkeypatch) -> None:
    import services

    monkeypatch.setattr(services, "get_user_profile", lambda _uid: None)
    monkeypatch.setattr(services, "upsert_user_profile", lambda *_args, **_kwargs: None)
    assert services.merge_form_data_with_profile("guest", {"country": "ES"})["country"] == "ES"
    assert "country" not in services.merge_form_data_with_profile("guest", {"age": 30})


def test_chat_pasa_el_codigo_resuelto_por_el_helper_central(monkeypatch) -> None:
    import constants
    import tools

    seen = []
    original = constants.assert_supported_country

    def _spy(raw):
        seen.append(raw)
        return original(raw)

    monkeypatch.setattr(constants, "assert_supported_country", _spy)
    ok, value = tools._valor_de_campo_para_perfil("country", "España")
    assert (ok, value) == (True, "ES")
    assert seen == ["ES"]


def test_chat_rechaza_pais_desconocido_sin_escribir_do() -> None:
    import tools

    assert tools._valor_de_campo_para_perfil("country", "Marte") == (False, None)


def test_todas_las_puertas_reusan_assert_supported_country() -> None:
    targets = {
        "PATCH profile": BACKEND_ROOT / "routers" / "user_data.py",
        "analyze": BACKEND_ROOT / "routers" / "plans.py",
        "services": BACKEND_ROOT / "services.py",
        "chat tool": BACKEND_ROOT / "tools.py",
    }
    missing = [label for label, path in targets.items() if "assert_supported_country" not in path.read_text(encoding="utf-8")]
    assert not missing, f"puertas sin helper central: {missing}"


def test_helper_central_no_canonicaliza_ni_hace_strip() -> None:
    source = (BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
    start = source.index("def assert_supported_country")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    assert "canonicalize_country(raw)" not in body
    assert "return canonicalize_country" not in body
    assert ".strip(" not in body
    assert "raw not in COUNTRY_PROFILES" in body


def test_pfix_marker_cierra_g45() -> None:
    services = (BACKEND_ROOT / "services.py").read_text(encoding="utf-8")
    assert "P2-COUNTRY-WRITE-UNVALIDATED" in services
