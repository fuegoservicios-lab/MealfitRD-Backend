# -*- coding: utf-8 -*-
"""[P1-TARGETS-NAMEERROR · 2026-08-12] El contador nacía muerto: 500 en producción.

`/api/nutrition/targets` usaba `get_user_profile` SIN importarlo (el idioma de
user_data.py es import lazy por-endpoint) y el fetch del perfil vivía FUERA del
try → NameError en runtime → 500 crudo → el dashboard del modo seguimiento
mostraba «Reintenta» para siempre. Se escapó porque:
  1. NameError es runtime-only: `import routers.user_data` pasa limpio.
  2. Los tests del endpoint eran estructurales/de rama guest — NADIE llamaba el
     happy path. «No está verificado hasta verlo fallar contra el código roto»:
     este archivo llama el endpoint DE VERDAD con el perfil real del incidente.

Dos contratos:
  A. Happy path e2e (mock solo en db.get_user_profile, la aritmética es real).
  B. Fail-closed: si la DB explota, el cliente recibe {ok:false} — jamás 500.
"""
import asyncio
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]

# El perfil EXACTO del incidente (user 5cb8d0d6, 2026-08-12 03:41 UTC).
_PERFIL_INCIDENTE = {
    "age": "23", "gender": "male", "height": "168", "weight": "123",
    "dietType": "balanced", "goalPace": "gradual", "mainGoal": "gain_muscle",
    "allergies": ["Ninguna"], "weightUnit": "lb", "activityLevel": "sedentary",
    "medicalConditions": ["Ninguna"],
}


def _llamar(monkeypatch, perfil_fn):
    """Invoca api_nutrition_targets con db.get_user_profile parcheado EN db —
    el endpoint hace `from db import get_user_profile` lazy, así que parchear
    routers.user_data crearía el atributo y ENMASCARARÍA el NameError original."""
    import db
    monkeypatch.setattr(db, "get_user_profile", perfil_fn)
    from routers.user_data import api_nutrition_targets
    return asyncio.run(api_nutrition_targets(verified_user_id="user-incidente"))


def test_happy_path_llama_el_endpoint_de_verdad(monkeypatch):
    """La regresión del incidente: con perfil completo, ok:true y la forma
    idéntica a plan_data (calories int + macros strings con 'g')."""
    out = _llamar(monkeypatch, lambda uid: {"health_profile": dict(_PERFIL_INCIDENTE)})
    assert out.get("ok") is True, out
    assert isinstance(out.get("calories"), int) and out["calories"] > 0
    for k in ("protein", "carbs", "fats"):
        v = out["macros"][k]
        assert isinstance(v, str) and v.endswith("g") and int(v[:-1]) > 0, (k, v)


def test_perfil_incompleto_declara_missing_fields(monkeypatch):
    hp = {k: v for k, v in _PERFIL_INCIDENTE.items() if k != "weight"}
    out = _llamar(monkeypatch, lambda uid: {"health_profile": hp})
    assert out.get("ok") is False
    assert out.get("missing_fields") == ["weight"]


def test_db_explotando_devuelve_json_honesto_no_500(monkeypatch):
    """Contrato B: el endpoint es fail-closed — una excepción de DB retorna
    {ok:false, reason:calc_error}, jamás burbujea (el 500 del incidente)."""
    def _boom(uid):
        raise RuntimeError("db caída simulada")
    out = _llamar(monkeypatch, _boom)
    assert out.get("ok") is False
    assert out.get("reason") == "calc_error"


def test_fetch_del_perfil_vive_dentro_del_try():
    """Parser-anchor del endurecimiento: el `from db import get_user_profile` y
    el to_thread del perfil van DENTRO del try (después del `try:` del cuerpo).
    Fuera del try, cualquier hipo de DB vuelve a ser un 500 crudo."""
    src = (BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    m = re.search(r"async def api_nutrition_targets\(.*?(?=\n@router|\nasync def |\ndef )", src, re.DOTALL)
    assert m, "api_nutrition_targets no encontrada"
    body = m.group(0)
    try_idx = body.index("try:")
    import_idx = body.index("from db import get_user_profile")
    fetch_idx = body.index("asyncio.to_thread(get_user_profile")
    assert try_idx < import_idx < fetch_idx, (
        "el import lazy y el fetch del perfil deben vivir DENTRO del try — "
        "fuera, una excepción de DB es un 500 crudo al cliente (el incidente)."
    )
