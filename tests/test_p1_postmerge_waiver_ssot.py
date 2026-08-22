"""[P1-POSTMERGE-WAIVER-SSOT · 2026-08-22] La CUARTA guarda de nevera decidía sola.

## El incidente

Plan real `2245eb45`, chunk 2. Los 7 chunks son `chunk_kind='initial_plan'`, sin
`_pantry_flexible_mode` ni `_pantry_advisory_only`, y **`attempts = 0`**.

Secuencia medida en la cola:

1. La guarda PRE-pipeline consultó `_pantry_gate_waiver_reason` → `initial_plan_autonomy`
   → dejó correr el chunk.
2. El bucle de validación existencial (`P1-PANTRY-EXIST-WAIVER`, la TERCERA guarda) vio
   el mismo waiver e hizo `break` en la iteración 0 — por eso `attempts = 0`: los 2
   reintentos de `CHUNK_PANTRY_MAX_RETRIES` **nunca se gastaron** y al LLM jamás se le
   dijo que había un problema de nevera.
3. La CUARTA guarda —el hard-guard post-merge (`P0-4`)— decidía con **dos lecturas
   sueltas** de `form_data` (`_pantry_flexible_mode`, `_pantry_quantity_violations`) en
   vez de con la SSOT. No vio `initial_plan_autonomy`, levantó
   `_PantryViolationPostMerge` dentro de `conn.transaction()` → **ROLLBACK** del merge
   → `pending_user_action`.

Resultado: se pagó el LLM entero, no se le dio ninguna oportunidad de corregir, y el
resultado se tiró por una condición que las tres guardas anteriores ya habían perdonado.
El usuario se quedó con el Dashboard vacío.

## Por qué el blanket no lo cazó

`_pantry_gate_waiver_reason` avisa en su docstring: «⚠️ No añadas una tercera lectura
suelta de `_pantry_flexible_mode`: llama a esta función. `test_p1_pantry_gate_ssot.py`
falla si aparece una guarda que decide sola.»

Pero ese test recorre una tupla de **UN SOLO elemento** (`_should_pause_for_empty_pantry`).
La promesa del nombre —«ninguna guarda lee el flag por su cuenta»— era mucho más ancha
que lo que el test miraba. Un guard que ya no puede fallar es peor que no tenerlo: da
por cubierto lo que no cubre. `P1-PANTRY-EXIST-WAIVER` (la tercera) lo dice con todas
las letras en su comentario: «no leía flag alguno, por eso el blanket del SSOT nunca la
vio».

Este archivo cierra la cuarta cabeza Y hace auditable el inventario, para que la quinta
falle un test en vez de descubrirse en producción.

## El contrato

Cuando la SSOT concede un waiver, el post-merge **anota y entrega** en lugar de tumbar
el bloque: marca las comidas ofensoras (`_pantry_violated`) y persiste
`_pantry_supplement_required` para que salga la categoría «🚨 Compra Urgente». Es el
mismo contrato que el modo flexible ya tenía («la entrega marcada es el contrato») y que
el camino síncrono del chunk 1 aplica desde P0-5: entregar un menú con lo que falta
señalado es estrictamente mejor que no entregar nada.
"""
import ast
import io
import os
import re

import pytest

_CRON_PATH = os.path.join(os.path.dirname(__file__), "..", "cron_tasks.py")
_CRON = io.open(_CRON_PATH, encoding="utf-8").read()
_LINEAS = _CRON.splitlines()


def _bloque_post_merge() -> str:
    """Región del hard-guard post-merge, anclada por su tooltip.

    Ventana ESTRECHA a propósito. La primera versión de este helper tomaba el rango
    entre la primera y la última línea con el prefijo `_p04_`, y como `_p04_pause_snap`
    vive ~2.000 líneas más abajo, el "bloque" abarcaba media función: la aserción
    «llama a la SSOT» pasaba porque encontraba una llamada de OTRA guarda. Un test que
    mira demasiado no puede fallar — exactamente el defecto que este archivo cierra.
    Anclamos por tooltip (convención del repo: renombrarlo rompe el test antes que
    producción).
    """
    idx = [i for i, l in enumerate(_LINEAS) if "P1-POSTMERGE-WAIVER-SSOT" in l]
    assert idx, (
        "no se encontró el tooltip-anchor `P1-POSTMERGE-WAIVER-SSOT` en cron_tasks.py — "
        "el guard post-merge debe llevarlo para que este test sepa qué mira"
    )
    ini, fin = min(idx), max(idx)
    # Hacia atrás lo justo para incluir `_p04_advisory_skip` (la exención de CANTIDAD,
    # que se calcula justo encima y debe seguir siendo independiente del waiver).
    return "\n".join(_LINEAS[max(0, ini - 20): fin + 45])


class TestLaCuartaGuardaDelegaEnLaSSOT:
    def test_llama_a_la_ssot(self):
        bloque = _bloque_post_merge()
        assert "_pantry_gate_waiver_reason" in bloque, (
            "el hard-guard post-merge debe preguntarle a la SSOT si la nevera puede "
            "bloquear a este chunk; decidir solo es como murió el plan 2245eb45"
        )

    def test_ya_no_lee_el_flag_por_su_cuenta(self):
        bloque = _bloque_post_merge()
        assert "_p04_flex_skip" not in bloque, (
            "`_p04_flex_skip` era la lectura suelta de `_pantry_flexible_mode`: honraba "
            "1 de los 4 waivers. Debe salir del waiver de la SSOT."
        )

    def test_el_advisory_de_cantidad_sigue_siendo_su_propia_exencion(self):
        """`_pantry_quantity_violations` NO es un waiver de nevera.

        Es una violación de CANTIDAD aceptada deliberadamente aguas arriba. Colapsarla
        dentro de la SSOT mezclaría dos preguntas distintas («¿existe el alimento?» vs
        «¿alcanza la cantidad?») y volvería a desincronizar las guardas.
        """
        bloque = _bloque_post_merge()
        assert "_pantry_quantity_violations" in bloque

    def test_cuando_hay_waiver_anota_en_vez_de_tumbar(self):
        bloque = _bloque_post_merge()
        assert "_mark_meals_violating_pantry" in bloque, (
            "con waiver activo el chunk debe ENTREGAR marcando las comidas ofensoras "
            "(contrato P0-5 / flexible), no hacer rollback del merge"
        )

    def test_marker_anclado_en_el_fuente(self):
        assert "P1-POSTMERGE-WAIVER-SSOT" in _CRON


class TestElInventarioDeLecturasSueltasEsAuditable:
    """El blanket original prometía «ninguna guarda» y miraba UNA.

    Aquí el inventario de sitios que leen el flag dentro de `_chunk_worker` es
    EXPLÍCITO. No todos son bugs — varios son telemetría o propagación legítima — pero
    ninguno puede aparecer sin que alguien lo declare aquí, que es justo lo que faltaba.
    """

    # Nombres de variable / patrón de cada lectura conocida dentro de `_chunk_worker`.
    # Añadir una entrada NUEVA obliga a justificarla en el comentario de al lado.
    LECTURAS_DECLARADAS = {
        "_learning_flexible_mode",   # telemetría del learning, no decide pausas
        "_is_flex",                  # elige el copy del log / la rama de entrega flexible
        "_p02_flexible",             # gate de reservas (ya consulta la SSOT aparte)
        "_p02_advisory_only",        # ídem
    }

    def _lecturas_en_chunk_worker(self) -> set:
        tree = ast.parse(_CRON)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_chunk_worker"
        )
        cuerpo = ast.get_source_segment(_CRON, fn) or ""
        halladas = set()
        for linea in cuerpo.splitlines():
            limpia = linea.strip()
            if limpia.startswith("#"):
                continue  # los comentarios narran el historial; no son lecturas
            if "_pantry_flexible_mode" not in limpia and "_pantry_advisory_only" not in limpia:
                continue
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", limpia)
            if m:
                halladas.add(m.group(1))
            elif limpia.startswith("if "):
                halladas.add(limpia[:60])
        return halladas

    def test_no_aparecieron_lecturas_nuevas_sin_declarar(self):
        halladas = self._lecturas_en_chunk_worker()
        nombradas = {h for h in halladas if not h.startswith("if ")}
        nuevas = nombradas - self.LECTURAS_DECLARADAS
        assert not nuevas, (
            f"lecturas nuevas de `_pantry_flexible_mode`/`_pantry_advisory_only` sin "
            f"declarar en LECTURAS_DECLARADAS: {sorted(nuevas)}. Si la lectura TOMA UNA "
            f"DECISIÓN de pausar/gatear, debe delegar en `_pantry_gate_waiver_reason`; "
            f"si es telemetría, decláralo aquí con su motivo."
        )

    def test_la_lectura_del_guard_post_merge_ya_no_esta(self):
        """Regresión directa: `_p04_flex_skip` fue la que mató al plan 2245eb45."""
        assert "_p04_flex_skip" not in self._lecturas_en_chunk_worker()


class TestLasViolacionesDejanDeSerEscrituraMuerta:
    """`_p0_4_violations` se escribía al pausar y **nadie la leía jamás**.

    Un grep sobre todo `backend/` devolvía UNA sola aparición: la escritura. El detalle
    de por qué murió el chunk —qué ingredientes faltaban, en qué día— quedaba guardado
    en el snapshot y se descartaba en silencio. Al reanudar, el modelo recibía el MISMO
    prompt que ya había fallado, así que la reanudación era un reintento a ciegas.

    Ahora `_resolve_pantry_pause_markers` —la SSOT que cierra toda pausa de nevera, y
    por tanto el único punto por el que pasan las 7 rutas de reanudación— promueve esas
    violaciones a `form_data['_pantry_correction']`, que es lo que
    `build_pantry_correction_context` convierte en el bloque «CORRECCIÓN OBLIGATORIA».
    """

    def test_la_violacion_guardada_llega_al_proximo_prompt(self):
        import cron_tasks
        snap = {
            "form_data": {"user_id": "u1"},
            "_p0_4_violations": [
                {"day": 1, "error": "ERRORES DE DESPENSA: INEXISTENTES: 1 cebolla, 40 g de arroz."},
            ],
        }
        cron_tasks._resolve_pantry_pause_markers(snap, "reanudado_por_test")
        correccion = snap["form_data"].get("_pantry_correction")
        assert correccion, "las violaciones guardadas deben viajar al reintento"
        assert "cebolla" in correccion

    def test_sin_violaciones_no_inventa_correccion(self):
        import cron_tasks
        snap = {"form_data": {}}
        cron_tasks._resolve_pantry_pause_markers(snap, "otra_cosa")
        assert "_pantry_correction" not in snap["form_data"]

    def test_no_pisa_una_correccion_ya_presente(self):
        """Si el worker ya puso una corrección más fresca, gana la suya."""
        import cron_tasks
        snap = {
            "form_data": {"_pantry_correction": "la fresca"},
            "_p0_4_violations": [{"day": 1, "error": "la vieja"}],
        }
        cron_tasks._resolve_pantry_pause_markers(snap, "x")
        assert snap["form_data"]["_pantry_correction"] == "la fresca"

    def test_es_fail_safe_ante_snapshot_raro(self):
        import cron_tasks
        for basura in ({"_p0_4_violations": "no soy lista"}, {"_p0_4_violations": []}, {}):
            cron_tasks._resolve_pantry_pause_markers(dict(basura), "x")  # no debe reventar


class TestLaSSOTSigueSiendoLaSSOT:
    def test_el_waiver_de_autonomia_inicial_existe_y_es_gateable(self):
        import cron_tasks
        assert cron_tasks._pantry_gate_waiver_reason(chunk_kind="initial_plan") == "initial_plan_autonomy"

    def test_rolling_refill_sin_flags_sigue_gateado(self):
        import cron_tasks
        assert cron_tasks._pantry_gate_waiver_reason(chunk_kind="rolling_refill") is None

    @pytest.mark.parametrize("flag,esperado", [
        ("_pantry_flexible_mode", "flexible_mode"),
        ("_pantry_advisory_only", "advisory_only"),
    ])
    def test_los_cuatro_motivos_siguen_distinguibles(self, flag, esperado):
        import cron_tasks
        assert cron_tasks._pantry_gate_waiver_reason(form_data={flag: True}) == esperado
