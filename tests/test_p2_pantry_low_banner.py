"""[P2-PANTRY-LOW-BANNER · 2026-06-21] Aviso inmediato de nevera baja en "Mi Nevera".

Decisión del owner (Fase 4): cuando con el tiempo el usuario borra/agota los alimentos de su
nevera y gestiona su inventario manualmente, debe ver un aviso INMEDIATO en Mi Nevera (no esperar
días al push del próximo chunk de mantenimiento).

Diseño cero-drift: el endpoint `GET /api/plans/pantry-status` expone EXACTAMENTE el mismo conteo
que usa el guard de mantenimiento server-side — `_count_meaningful_pantry_items(get_user_inventory_net(uid))`
comparado contra `CHUNK_MIN_FRESH_PANTRY_ITEMS`. Así el banner del frontend dice exactamente lo que
el backend haría al preparar la próxima lista, sin reimplementar la lógica de "fresco" en JS.

Estos tests anclan ese contrato (parser-based sobre el source de prod).
"""
import routers.plans as plans_mod


def _src():
    return open(plans_mod.__file__, encoding="utf-8").read()


def test_endpoint_existe():
    src = _src()
    assert '@router.get("/pantry-status")' in src
    assert "P2-PANTRY-LOW-BANNER" in src


def test_usa_el_mismo_conteo_que_el_guard_de_mantenimiento():
    src = _src()
    # Cero drift: reusa las MISMAS funciones del servidor, no reimplementa "fresco".
    assert "get_user_inventory_net" in src
    assert "_count_meaningful_pantry_items" in src
    assert "CHUNK_MIN_FRESH_PANTRY_ITEMS" in src
    # Devuelve la señal que consume el banner.
    assert "is_below" in src and "meaningful_count" in src and "min_required" in src


def test_read_only_sin_quota():
    # El endpoint es polling read-only (cero costo LLM) → get_verified_user_id, NO verify_api_quota
    # (mismo criterio que los GET del Historial, P1-AUDIT-3). Aislamos la firma del endpoint.
    src = _src()
    idx = src.find('@router.get("/pantry-status")')
    assert idx > -1
    sig = src[idx: idx + 320]
    assert "Depends(get_verified_user_id)" in sig
    assert "verify_api_quota" not in sig
