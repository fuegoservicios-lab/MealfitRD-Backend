"""[P1-SEEDER-CURED-MEAT-BETA · 2026-08-23]

Guard de conducta para el sesgo de carnes curadas del seeder multinacional.
No ancla una lista de nombres: observa las proteínas que el prompt público
asigna realmente con semillas determinísticas.
"""
import random
import re

import ai_helpers as ah


_ASSIGNED_PROTEIN_RX = re.compile(
    r"OPCIÓN [A-Z]+ .*?DEBE incluir obligatoriamente: (.*?) \+", re.I)
# Oráculo de auditoría independiente del vocabulario que se está probando. No
# comprueba que producción "contenga" estas grafías: clasifica la SALIDA real
# para que retirar un término del SSOT no vuelva ciego al propio test.
_AUDIT_CURED_TOKENS = (
    "bacalao", "arenque", "salami", "salchichon", "pepperoni", "mortadela",
    "tocino", "panceta", "longaniza", "chorizo", "salchicha", "embutido",
    "jamon", "tocineta", "morcilla", "embuchado", "chistorra", "sobrasada",
    "butifarra", "cecina",
)
_BETA_COUNTRIES = ("ES", "MX", "CO", "PR", "US")
_SEEDS = range(80)
_MARGIN = 0.03


def _assigned_cured_fraction(country: str) -> float:
    cured = total = 0
    for seed in _SEEDS:
        random.seed(seed)
        prompt = ah.get_deterministic_variety_prompt(
            "", {"country": country, "mainGoal": "lose_fat"}, days_count=3)
        assigned = _ASSIGNED_PROTEIN_RX.findall(prompt)
        assert len(assigned) == 3, "el guard dejó de observar las tres asignaciones del seeder"
        total += len(assigned)
        cured += sum(
            ah._token_matches_wb(name, _AUDIT_CURED_TOKENS)
            for name in assigned
        )
    return cured / total


def test_curados_beta_no_superan_do_mas_margen(monkeypatch):
    """Cada pool beta queda en paridad clínica con DO bajo `lose_fat`.

    La mutación de retirar `morcilla` de las dos fuentes hace que ES vuelva a
    escogerla sin penalty y exceda el margen.
    """
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    do_fraction = _assigned_cured_fraction("DO")
    measured = {cc: _assigned_cured_fraction(cc) for cc in _BETA_COUNTRIES}
    assert all(fraction <= do_fraction + _MARGIN for fraction in measured.values()), (
        f"curados asignados fuera de paridad: DO={do_fraction:.3%}, beta={measured!r}"
    )


def test_hardener_beta_excluye_curados_con_match_de_palabra(monkeypatch):
    """El backstop estructural comparte vocabulario y no muerde subcadenas."""
    import graph_orchestrator as go

    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_SALTCURED_MAIN", True)
    skeleton = {"days": [{
        "day": 1,
        "protein_pool": ["Morcilla", "Lomo embuchado", "Pollo", "Jamoncillo de leche"],
        "carb_pool": [],
        "fruit_pool": [],
    }]}
    counts = go.harden_day_pools(skeleton, {}, None)
    assert skeleton["days"][0]["protein_pool"] == ["Pollo", "Jamoncillo de leche"]
    assert counts["saltcured_removed"] == 2


def test_scrub_del_planner_deriva_ambas_listas_del_ssot():
    """Evita que `_SKELETON_RESTRICTED`/`_EMBUTIDO_KEYS` vuelvan a copias DO."""
    source = open("graph_orchestrator.py", encoding="utf-8").read()
    planner = source.index("async def plan_skeleton_node")
    start = source.index("P1-SEEDER-CURED-MEAT-BETA", planner)
    block = source[start:source.index("# [P1-FORM-AUDIT-BATCH", start)]
    assert "_SKELETON_RESTRICTED = ('atún', 'atun', *_CURADOS_RESTRICTED)" in block
    assert "_EMBUTIDO_KEYS = _CURADOS_RESTRICTED" in block
    assert "_curado_matches_wb" in block
