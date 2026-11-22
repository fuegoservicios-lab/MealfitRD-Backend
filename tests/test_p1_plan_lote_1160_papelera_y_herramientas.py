"""[P1-PLAN-LOTE-1160 · 2026-11-21] La conversación del dueño del 21-nov, turno a turno.

«Borrar todos» se llevó los tres potes de su Despensa con su etiqueta (DELETE duro). Al pedirle al coach que los
recuperara: dijo «Recuperé…» sin recuperar nada, anotó los suplementos como TOMADOS con la única herramienta que le
quedaba (`log_consumed_meal`) y terminó en «no puedo desde aquí». Causas y defensas:
  1. el gating de herramientas miraba solo el ÚLTIMO mensaje («Los 3 habían» no trae palabras clave) → ventana;
  2. no había de dónde recuperar → papelera de 30 días (`papelera_nevera`) en los tres borrados del usuario;
  3. «Borrar todos» vaciaba también los potes (el aviso dice «todos los alimentos») → solo alimentos;
  4. el coach no tenía herramienta → `recuperar_de_la_nevera` + regla en los dos manuales;
  5. título «Galleta de koko կրկ» → fuera las letras de otros alfabetos.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


# ── 1. la ventana del gating ───────────────────────────────────────────────────────────────────────────────────────

class _Tool:
    def __init__(self, name):
        self.name = name


def _catalogo():
    import tools
    return [_Tool(getattr(t, "name", "")) for t in tools.agent_tools]


def _conversacion_del_dueno():
    from langchain_core.messages import AIMessage, HumanMessage
    return [
        HumanMessage(content="Recupera los alimentos que borre sin querer"),
        AIMessage(content="..."),
        HumanMessage(content="Quise decir los suplementos de la despensa"),
        AIMessage(content="..."),
        HumanMessage(content="Los 3 habían"),
        AIMessage(content="..."),
        HumanMessage(content="Agrega esos 3 y ya"),
        AIMessage(content="..."),
        HumanMessage(content="Digo, agrega esos dos y ya"),
    ]


def test_una_respuesta_corta_sigue_teniendo_las_herramientas_de_la_despensa(monkeypatch):
    import agent
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_ENABLED", raising=False)
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_VENTANA", raising=False)
    nombres = {t.name for t in agent._select_tools_for_turn(_catalogo(), {"messages": _conversacion_del_dueno()})}
    for falta in ("guardar_suplemento", "recuperar_de_la_nevera", "modify_pantry_inventory", "check_current_pantry"):
        assert falta in nombres, falta


def test_con_ventana_1_vuelve_el_fallo_del_21_nov(monkeypatch):
    """Ver el test fallar contra la conducta vieja: solo el último mensaje ⇒ sin herramientas de la Despensa."""
    import agent
    monkeypatch.setenv("MEALFIT_CHAT_TOOL_GATING_VENTANA", "1")
    nombres = {t.name for t in agent._select_tools_for_turn(_catalogo(), {"messages": _conversacion_del_dueno()})}
    assert "guardar_suplemento" not in nombres and "recuperar_de_la_nevera" not in nombres


def test_las_notas_del_sistema_no_cuentan_como_lo_que_dijo_el_usuario(monkeypatch):
    import agent
    from langchain_core.messages import HumanMessage
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_VENTANA", raising=False)
    msgs = [HumanMessage(content="[Instrucción del sistema] revisa la despensa y los suplementos"),
            HumanMessage(content="desayuné dos huevos")]
    nombres = {t.name for t in agent._select_tools_for_turn(_catalogo(), {"messages": msgs})}
    assert "guardar_suplemento" not in nombres and "modify_pantry_inventory" not in nombres


def test_pedir_recuperar_liga_la_herramienta(monkeypatch):
    import agent
    monkeypatch.delenv("MEALFIT_CHAT_TOOL_GATING_VENTANA", raising=False)
    st = {"messages": [MagicMock(type="user", content="recupera lo que borré sin querer")]}
    assert "recuperar_de_la_nevera" in {t.name for t in agent._select_tools_for_turn(_catalogo(), st)}


# ── 2. la papelera ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_el_borrado_y_su_copia_van_en_la_misma_sentencia(monkeypatch):
    import papelera_nevera as pn
    vistas = []
    monkeypatch.setattr(pn, "_write", lambda sql, params: vistas.append((sql, params)) or [
        {"papelera_id": 7, "original_id": 52134, "ingredient_name": "DESIGNS FOR SPORT MULTI + PHYTO", "kind": "supplement"}])
    monkeypatch.setattr(pn, "_purgar", lambda uid: None)
    out = pn.borrar_y_archivar("u1", ids=[52134], via="uno")
    sql, params = vistas[0]
    assert "DELETE FROM public.user_inventory WHERE" in sql and "INSERT INTO public.user_inventory_papelera" in sql
    assert "to_jsonb(b)" in sql and "user_id = %s" in sql and params[0] == "u1"
    assert out[0]["papelera_id"] == 7


def test_borrar_todos_deja_los_potes(monkeypatch):
    import papelera_nevera as pn
    vistas = []
    monkeypatch.setattr(pn, "_write", lambda sql, params: vistas.append(sql) or [])
    monkeypatch.setattr(pn, "_purgar", lambda uid: None)
    pn.borrar_y_archivar("u1", solo_alimentos=True, via="todos")
    assert "kind = 'food'" in vistas[0]
    src = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    i = src.index('@router.delete("/inventory/items")\nasync def api_delete_all_inventory')
    assert 'borrar_y_archivar, uid, solo_alimentos=True, via="todos"' in src[i:i + 1200]


def test_sin_la_tabla_se_borra_igual(monkeypatch):
    """Deploy antes que la migración: no poder recuperar es malo, no poder borrar es peor."""
    import papelera_nevera as pn
    llamadas = []

    def _w(sql, params):
        llamadas.append(sql)
        if "user_inventory_papelera" in sql:
            raise RuntimeError('relation "public.user_inventory_papelera" does not exist')
        return [{"papelera_id": None, "original_id": 1, "ingredient_name": "Pollo", "kind": "food"}]

    monkeypatch.setattr(pn, "_write", _w)
    assert pn.borrar_y_archivar("u1", ids=[1])[0]["original_id"] == 1
    assert len(llamadas) == 2 and "user_inventory_papelera" not in llamadas[1]


def test_otro_error_no_se_traga(monkeypatch):
    import papelera_nevera as pn
    monkeypatch.setattr(pn, "_write", lambda sql, params: (_ for _ in ()).throw(RuntimeError("lock timeout")))
    with pytest.raises(RuntimeError):
        pn.borrar_y_archivar("u1", ids=[1])


def test_restaurar_reinserta_con_etiqueta_y_suma_si_ya_existe():
    import papelera_nevera as pn
    import inspect
    sql = inspect.getsource(pn.restaurar)
    assert "jsonb_populate_record(NULL::public.user_inventory" in sql
    assert "ON CONFLICT (user_id, ingredient_name, unit) DO UPDATE" in sql
    assert "COALESCE(public.user_inventory.serving_label, EXCLUDED.serving_label)" in sql
    assert "DISTINCT ON (fila->>'ingredient_name', fila->>'unit')" in sql   # dos copias del mismo nombre no chocan


def test_buscar_por_nombres_acepta_un_trozo_del_nombre(monkeypatch):
    import papelera_nevera as pn
    monkeypatch.setattr(pn, "listar", lambda uid: [
        {"papelera_id": 1, "ingredient_name": "DESIGNS FOR SPORT MULTI + PHYTO", "kind": "supplement"},
        {"papelera_id": 2, "ingredient_name": "Creatine 5000 Creatine Monohydrate Capsules", "kind": "supplement"},
        {"papelera_id": 3, "ingredient_name": "Atlas Gainer Advanced Mass Vanilla", "kind": "supplement"}])
    enc, faltan = pn.buscar_por_nombres("u1", ["multi", "creatine", "whey"])
    assert [e["papelera_id"] for e in enc] == [1, 2] and faltan == ["whey"]


def test_la_migracion_esta_en_los_dos_directorios():
    a = (_BACKEND / "migrations" / "p1_plan_lote_1160_papelera_nevera.sql").read_text(encoding="utf-8")
    b = (_BACKEND.parent / "migrations" / "p1_plan_lote_1160_papelera_nevera.sql")
    if b.exists():
        assert b.read_text(encoding="utf-8") == a
    assert "ON DELETE CASCADE" in a and "CREATE TABLE IF NOT EXISTS public.user_inventory_papelera" in a


def test_la_papelera_se_va_con_la_cuenta_y_sale_en_el_export():
    import db_profiles
    assert "user_inventory_papelera" in db_profiles._USER_SCOPED_TABLES_USERID
    assert '("user_inventory_papelera", "user_id"' in (_BACKEND / "app.py").read_text(encoding="utf-8")


# ── 3. la herramienta del coach ────────────────────────────────────────────────────────────────────────────────────

_TRES = [
    {"papelera_id": 1, "ingredient_name": "Atlas Gainer Advanced Mass Vanilla", "kind": "supplement", "con_etiqueta": False},
    {"papelera_id": 2, "ingredient_name": "Creatine 5000 Creatine Monohydrate Capsules", "kind": "supplement", "con_etiqueta": True},
    {"papelera_id": 3, "ingredient_name": "DESIGNS FOR SPORT MULTI + PHYTO", "kind": "supplement", "con_etiqueta": True},
]


def test_sin_nombres_solo_lista(monkeypatch):
    import papelera_nevera as pn
    import tools
    monkeypatch.setattr(pn, "listar", lambda uid: list(_TRES))
    monkeypatch.setattr(pn, "restaurar", lambda *a, **k: pytest.fail("no debe restaurar sin que lo pida"))
    out = tools.recuperar_de_la_nevera.func("u1")
    assert "Atlas Gainer" in out and "nada se ha recuperado" in out and "REFRESH_INVENTORY" not in out


def test_los_3_recupera_todo(monkeypatch):
    import papelera_nevera as pn
    import tools
    monkeypatch.setattr(pn, "listar", lambda uid: list(_TRES))
    pedidos = []

    def _restaurar(uid, papelera_ids=None, original_ids=None):
        pedidos.extend(papelera_ids)
        return [{"id": 10 + i, "ingredient_name": f["ingredient_name"]} for i, f in enumerate(_TRES)]

    monkeypatch.setattr(pn, "restaurar", _restaurar)
    out = tools.recuperar_de_la_nevera.func("u1", todo=True)
    assert pedidos == [1, 2, 3] and out.startswith("Recuperado en su Nevera") and "[UI_ACTION: REFRESH_INVENTORY]" in out
    assert "no lo anotes en su diario" in out


def test_papelera_vacia_no_finge(monkeypatch):
    import papelera_nevera as pn
    import tools
    monkeypatch.setattr(pn, "listar", lambda uid: [])
    out = tools.recuperar_de_la_nevera.func("u1", nombres=["creatina"])
    assert "NO digas que recuperaste" in out and "REFRESH_INVENTORY" not in out


def test_invitado_no(monkeypatch):
    import tools
    assert "iniciar sesión" in tools.recuperar_de_la_nevera.func("guest-123", todo=True)


def test_esta_en_el_catalogo_y_en_la_tabla_de_ids():
    import tools
    assert "recuperar_de_la_nevera" in {getattr(t, "name", "") for t in tools.agent_tools}
    doc = (_BACKEND / "docs" / "agent_tools_user_id_table.md").read_text(encoding="utf-8")
    assert "`recuperar_de_la_nevera`" in doc


# ── 4. los manuales ────────────────────────────────────────────────────────────────────────────────────────────────

def test_los_dos_manuales_prohiben_anotar_lo_recuperado():
    from prompts.coach_compact import build_coach_static_manual
    compacto = build_coach_static_manual(country="DO")
    largo = (_BACKEND / "prompts" / "chat_agent.py").read_text(encoding="utf-8")
    for texto in (compacto, largo):
        assert "recuperar_de_la_nevera" in texto
        assert "JAMÁS uses `log_consumed_meal` para eso" in texto
        assert "Nunca digas «recuperé»" in texto
    assert "SU DESPENSA" in compacto and "Alacena" not in compacto


def test_la_regla_larga_viaja_con_su_herramienta():
    """El manual largo tiene tope de tokens: la regla S2 sale solo cuando el turno liga `recuperar_de_la_nevera`."""
    from prompts.coach_lean import podar_extras_no_ligados
    texto = "S. otra regla\nS2. RECUPERAR LO BORRADO [P1-PLAN-LOTE-1160]: si pide recuperar...\nT. LA HORA\n"
    assert "S2. RECUPERAR" not in podar_extras_no_ligados(texto, ["log_consumed_meal"])
    assert "S2. RECUPERAR" in podar_extras_no_ligados(texto, ["log_consumed_meal", "recuperar_de_la_nevera"])


def test_el_bloque_de_suplementos_se_llama_despensa():
    import suplementos
    import inspect
    assert '" SU DESPENSA: "' in inspect.getsource(suplementos)


# ── 5. el título ───────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("crudo,limpio", [
    ("Galleta de koko կրկ", "Galleta de koko"),
    ("Pan con aguacate 🥑", "Pan con aguacate 🥑"),
    ("Mangú con huevo y salami", "Mangú con huevo y salami"),
    ("Proteína 2× al día", "Proteína 2× al día"),
    ("Café и чай", "Café"),
])
def test_el_titulo_pierde_las_letras_de_otros_alfabetos(crudo, limpio):
    from agent import _limpiar_titulo_de_otros_alfabetos
    assert _limpiar_titulo_de_otros_alfabetos(crudo) == limpio
