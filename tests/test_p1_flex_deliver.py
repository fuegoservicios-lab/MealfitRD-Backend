# [P1-FLEX-DELIVER · 2026-08-08] El modo flexible TTL-escalado ENTREGA con Compra Urgente,
# no re-pausa. Evidencia (journal 2026-08-08, plan f380821a del owner + 9cf5e313):
#   17:42 chunk expira pending_user_action → re-encola FLEXIBLE
#   17:46 P1-PANTRY-EXIST-WAIVER omite la validación pre-gen («la lista de compras define
#         qué comprar») → GENERA los 4 días → la validación FINAL vs live inventory los
#         rechaza por 26 faltantes → pausa (reason=flexible_live_unreachable, mentirosa:
#         el inventario SÍ respondió) → TTL 2h → flexible → genera → ... ciclo infinito.
# Dos guardas sobre la misma condición OSCILAN (clase conocida) y cada vuelta quema una
# generación completa de API (~cada 2h por plan atascado). El flexible existe PORQUE el
# usuario no repuso a tiempo: exigirle la nevera llena después de generar contradice al
# waiver desplegado el mismo día. La rama stale_snapshot conserva la pausa (ahí el
# faltante puede ser un espejismo del snapshot viejo, no una decisión de producto).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_CT = open(os.path.join(os.path.dirname(__file__), "..", "cron_tasks.py"), encoding="utf-8").read()


def _bloque_validacion_final():
    # Ventana ESTRUCTURAL, no de chars fijos (esa clase caducó 6 veces en memoria): del log-line
    # ancla hasta el comentario del except que cierra el bloque ([P0-1/FAIL-CLOSED]).
    i = _CT.find("generado con flexible_mode falló validación vs live inventory")
    assert i > 0, "el bloque de validación final flexible desapareció"
    j = _CT.find("[P0-1/FAIL-CLOSED]", i)
    assert j > i, "el cierre del bloque ([P0-1/FAIL-CLOSED]) desapareció"
    return _CT[i - 500: j]


def test_flexible_ttl_entrega_no_pausa():
    blk = _bloque_validacion_final()
    assert "P1-FLEX-DELIVER" in blk, (
        "falta la rama de entrega: el modo flexible TTL-escalado debe ENTREGAR con "
        "Compra Urgente persistida en vez de re-pausar (ciclo infinito medido)")
    # la rama de entrega decide por _pantry_flexible_mode (TTL-escalado), no por _is_flex
    # (que incluye stale_snapshot — esa rama SÍ conserva la pausa):
    assert '_pantry_flexible_mode' in blk


def test_maquinaria_de_honestidad_se_conserva():
    # La entrega flexible NO es silenciosa: supplement persistido + push + días marcados
    # deben seguir en el bloque (son la promesa de producto de la Compra Urgente).
    blk = _bloque_validacion_final()
    assert "_persist_pantry_supplement_to_plan_data" in blk
    assert "_dispatch_push_notification" in blk
    assert "emergency_pantry_unsafe" in blk


def test_post_merge_tercera_guarda_tambien_omite_en_flexible():
    """La guarda dura post-merge (`_PantryViolationPostMerge`) re-imponía strict en flexible y
    revivía el ciclo con OTRA reason (`pantry_violation_post_merge`) — medido: lista 85→191.

    [P3-FLEX-GUARD-ANCHOR-SSOT · 2026-08-22] Este caso anclaba en `_p04_flex_skip` y en la lectura
    a pelo de `_pantry_flexible_mode`. `P1-POSTMERGE-WAIVER-SSOT` (2026-08-22) sustituyó ese
    mecanismo por una delegación a `_pantry_gate_waiver_reason` — y lo hizo **justamente porque**
    leer el flag a mano honraba 1 de los 4 waivers. O sea: el guard exigía la forma concreta que
    resultó ser el defecto.

    Se reconvierte, no se borra. La propiedad sigue siendo la misma —esta guarda no puede tumbar
    un chunk ya perdonado— y ahora se ancla en el SSOT, que es lo que la hace cierta para los
    cuatro waivers y no sólo para uno.
    """
    i = _CT.find("HARD VALIDATION POST-MERGE")
    assert i > 0, "el bloque post-merge desapareció"
    blk = _CT[i: _CT.find("Recalcular contadores absolutos", i)]

    assert "_pantry_gate_waiver_reason(" in blk, (
        "la guarda dura post-merge dejó de delegar en `_pantry_gate_waiver_reason`. Decidir aquí "
        "con lecturas sueltas de form_data es exactamente lo que hacía que honrara 1 de los 4 "
        "waivers"
    )
    assert "not _p04_advisory_skip and not _p04_waiver" in blk, (
        "el waiver debe COMPONERSE con el advisory, no reemplazarlo: son dos condiciones "
        "distintas y quitar cualquiera resucita una cabeza del ciclo"
    )
    assert "_PantryViolationPostMerge(" in blk, "el raise sigue vivo para el modo estricto"


def test_post_merge_no_vuelve_a_decidir_por_su_cuenta():
    """La otra mitad, en negativo — y el caso que este fichero no tenía.

    Lo que rompió la cuarta guarda no fue que le faltara una condición: fue que se la leyó ELLA
    MISMA. Un guard que sólo exige «llama al SSOT» se satisface con una llamada decorativa junto a
    la lectura a mano de siempre. Así que además se prohíbe la lectura suelta del flag dentro del
    bloque."""
    i = _CT.find("HARD VALIDATION POST-MERGE")
    blk = _CT[i: _CT.find("Recalcular contadores absolutos", i)]
    codigo = "\n".join(l for l in blk.splitlines() if not l.strip().startswith("#"))
    assert 'form_data.get("_pantry_flexible_mode")' not in codigo, (
        "la guarda dura post-merge vuelve a leer `_pantry_flexible_mode` por su cuenta. Ése es "
        "el mecanismo que honraba 1 de los 4 waivers: la decisión es de "
        "`_pantry_gate_waiver_reason`, no de este bloque"
    )


def test_pausa_sigue_viva_para_stale_snapshot():
    # El return False de la pausa NO desaparece del bloque: la rama no-TTL (stale_snapshot)
    # lo conserva — quitar la pausa por completo sería el error simétrico.
    blk = _bloque_validacion_final()
    assert "_pause_chunk_for_final_inventory_validation(" in blk
    assert "return False" in blk
