"""[P4-CONSTRAINT-ABC · 2026-06-14] `ClinicalConstraint` ABC + `ClinicalConstraintEngine`.

Generaliza el patrón YA PROBADO del cap renal (las 3 capas: ajuste-en-fuente → enforce-per-comida
solver-independiente → red-de-seguridad en el punto único de salida) a una jerarquía declarativa, SIN
reescribir la matemática validada. Cada constraint es un *shell de despacho* que DELEGA a las funciones
existentes de `graph_orchestrator.py` — NUNCA reimplementa el cap (ni el multiply por RENAL_PROTEIN_
GKG_CEILING, ni la reasignación de kcal, ni el trim). El refactor es behavior-preserving (opción (b) del
blueprint): el engine NO reordena los 8 guards de la capa clínica; solo encapsula los *cuerpos* renal
(Guard 1), la sustitución (Guard 3), el cap-en-fuente y la red de salida en objetos constraint.

Precedencia = `condition_rules.ConditionRule.precedence` (renal=10 primero — seguridad; subs=30 después
→ reproduce "trim-de-magnitud-renal ANTES de swap-de-identidad" que es el orden actual Guard1→Guard3).

Import lazy de `graph_orchestrator` DENTRO de los métodos para romper el ciclo (graph_orchestrator
importa este módulo a nivel-módulo; este módulo solo lo toca en runtime).
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClinicalContext:
    """Locals per-run que los guards recomputan, pasados al hook enforce_on_plan para no recomputarlos
    por constraint. `protein_g` es el target diario YA capeado (=`_pg`); `db` la instancia compartida."""
    db: object = None             # IngredientNutritionDB (compartida)
    daily_cals: float = 0.0       # `_daily_cals`
    protein_g: float = 0.0        # `_pg` (target de proteína per-día, ya capeado en la fuente)
    active_macros: Optional[dict] = None


class ClinicalConstraint(ABC):
    """Wrapper declarativo sobre un guard validado existente. Los 3 hooks son OPCIONALES (default no-op)
    → un constraint solo-sustitución implementa solo `enforce_on_plan`; el renal implementa los 3."""

    id: str = ""
    precedence: int = 100  # menor = se aplica primero (seguridad primero), igual que condition_rules

    def applies(self, form_data: dict) -> bool:
        return False

    # ── Capa 1: FUENTE — reescribe los targets de macros sobre nutrition/active_macros ──
    def adjust_targets(self, nutrition: dict, form_data: dict) -> None:
        return None

    # ── Capa 2: PER-COMIDA — muta las comidas/ingredientes/macros del plan ensamblado ──
    def enforce_on_plan(self, plan: dict, form_data: dict, nutrition: dict, ctx: ClinicalContext) -> None:
        return None

    # ── Capa 3: SALIDA — red de seguridad en el punto único (gate/metadata/derivación) ──
    def safety_net(self, plan: dict, form_data: dict, nutrition: dict) -> None:
        return None


class RenalProteinCapConstraint(ClinicalConstraint):
    """ERC: cap de proteína 0.8 g/kg (KDIGO). Implementa las 3 capas, cada una DELEGANDO a la función
    validada existente. NINGUNA aritmética del cap se reimplementa aquí (tooltip-anchor: P4-CONSTRAINT-ABC
    delegation — un rewrite que deje de delegar falla el test de equivalencia)."""

    id = "renal"
    precedence = 10  # la más alta (seguridad primero), igual que condition_rules renal=10

    def applies(self, form_data: dict) -> bool:
        import graph_orchestrator as go
        return bool(go.CONDITION_RULES_ENABLED and go._is_renal_condition(form_data))

    def adjust_targets(self, nutrition: dict, form_data: dict) -> None:
        import graph_orchestrator as go
        go._apply_renal_cap_to_nutrition(nutrition, form_data)   # def graph_orchestrator — SIN CAMBIOS

    def enforce_on_plan(self, plan: dict, form_data: dict, nutrition: dict, ctx: ClinicalContext) -> None:
        import graph_orchestrator as go
        go._enforce_renal_per_meal(plan, ctx.protein_g, ctx.daily_cals, ctx.db)  # cuerpo verbatim Guard 1

    def safety_net(self, plan: dict, form_data: dict, nutrition: dict) -> None:
        import graph_orchestrator as go
        go._renal_exit_safety_net(plan, nutrition, form_data)    # cuerpo verbatim red de salida


class SubstitutionEngineConstraint(ClinicalConstraint):
    """TODAS las condiciones con sustitución (DM2 azúcar / HTA sodio / dislipidemia satfat) en UN solo
    pase. NO se parte por-condición: `collect_substitutions` ya las mergea en orden de precedencia y
    `_apply_condition_substitutions` corre el pase único first-match-wins + el delta quirúrgico de macros
    (partirlo cambiaría el comportamiento — tokens solapados entre condiciones + delta una-vez-por-comida).
    El registro `CONDITION_RULES` sigue siendo el SSOT del CONTENIDO; este constraint es la unidad de
    EJECUCIÓN."""

    id = "substitutions"
    precedence = 30  # corre DESPUÉS del trim renal (10) — preserva el orden Guard1→Guard3 actual

    def applies(self, form_data: dict) -> bool:
        import graph_orchestrator as go
        if not go.CONDITION_RULES_ENABLED:
            return False
        try:
            from condition_rules import collect_substitutions
            return bool(collect_substitutions(form_data))
        except Exception:
            return False

    def enforce_on_plan(self, plan: dict, form_data: dict, nutrition: dict, ctx: ClinicalContext):
        import graph_orchestrator as go
        return go._apply_condition_substitutions(plan, form_data)  # pase único — SIN CAMBIOS; retorna #comidas


class ClinicalConstraintEngine:
    """Construye los constraints activos para el perfil (en orden de precedencia) y despacha los hooks.
    El orden de los guards de la capa clínica se PRESERVA: el engine se invoca por-slot (`enforce_one`),
    no en un pase atómico que reordenaría (food-safety se mantiene físicamente entre el slot renal y el
    de sustitución). Ver blueprint §4-5."""

    _REGISTRY = (RenalProteinCapConstraint, SubstitutionEngineConstraint)

    def __init__(self, form_data: dict):
        self.form_data = form_data if isinstance(form_data, dict) else {}
        self.active = sorted(
            (c for c in (cls() for cls in self._REGISTRY) if c.applies(self.form_data)),
            key=lambda c: c.precedence,
        )

    def _get(self, constraint_id: str):
        for c in self.active:
            if c.id == constraint_id:
                return c
        return None

    def run_adjust_targets(self, nutrition: dict) -> None:
        for c in self.active:
            c.adjust_targets(nutrition, self.form_data)

    def enforce_one(self, constraint_id: str, plan: dict, nutrition: dict, ctx: ClinicalContext):
        """Despacha el hook enforce_on_plan de UN constraint (preserva la posición del guard) y retorna
        su resultado (p.ej. #comidas sustituidas). No-op (None) si ese constraint no está activo."""
        c = self._get(constraint_id)
        if c is not None:
            return c.enforce_on_plan(plan, self.form_data, nutrition, ctx)
        return None

    def run_safety_net(self, plan: dict, nutrition: dict) -> None:
        for c in self.active:
            c.safety_net(plan, self.form_data, nutrition)
