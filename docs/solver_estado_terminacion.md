# El solver: en qué estado termina, medido

> Cierre del gap P3 «factibilidad conjunta y estado de terminación» del roadmap 2.6. El auditor
> pedía «distinguir *no converge* de *no hay solución* y de *se acabó el tiempo*» y lo estimaba en
> ≈1 semana. **Dos de los tres estados ya existían y están distinguidos por macro**; del tercero no
> hay evidencia de que ocurra. Lo que faltaba no era código: era la medición que dice si vale la
> pena invertir esa semana.

## Los estados que el solver ya marca

| Marca en el plato | Qué significa | Dónde |
|---|---|---|
| `_solver_not_converged` + `_solver_failed_macros` | el escalado por línea no clavó el target, y **qué macro** quedó fuera | `P2-SOLVER-METHOD-OBS`, `P3-SOLVER-CONVERGED-BAND` |
| `_solver_infeasible` + `_solver_residuals` | **no hay solución** con estos alimentos dentro del clamp: falta o sobra un *portador*, ninguna re-escala lo arregla | `P3-SOLVER-FEASIBILITY` |
| `_solver_greedy_fallback` | el LSQ lanzó excepción o está apagado y actuó el greedy | `P2-SOLVER-METHOD-OBS` |
| `_solver_frozen_lines` | cuántas líneas quedaron pinneadas y no se pudieron mover | `P2-SOLVER-PIN-FROZEN` |

«No converge» y «no hay solución» **no son lo mismo y el código ya lo sabe**: la segunda trae
además la dirección por macro (`{'protein': 'high'}` = hace falta más proteína de la que estos
alimentos pueden dar), que es justo lo que la reparación aguas abajo necesita para decidir si
añadir un portador o re-escalar.

## Lo medido (1.174 platos de 200 planes vivos, 06-sep)

| Estado | Platos | % |
|---|---|---|
| `_solver_not_converged` | 485 | **41 %** |
| `_solver_frozen_lines` | 258 | 22 % |
| `_solver_infeasible` | 86 | **7 %** |
| `_solver_abstained_coverage` | 3 | 0 % |
| `_solver_greedy_fallback` | 0 | **0 %** |

Macro que no converge: grasas 431 · proteína 353 · carbos 92.
Infactibilidades por dirección: `protein:high` 34 · `fats:low` 20 · `carbs:high` 16 ·
`fats:high` 16 · `carbs:low` 7 · `kcal:high` 2 · `kcal:low` 1.

## Qué dicen esos números

1. **No converger es la norma, no la excepción** (41 %). El clamp por línea acota cuánto puede
   moverse cada ingrediente, y con esa cota el target casi nunca se clava exacto: los closers y el
   rebalance de después terminan el trabajo. Leer ese 41 % como «el solver falla mucho» sería
   confundir *no clavó el target él solo* con *el plato salió mal*.
2. **El LSQ no revienta nunca** (0 greedy en 1.174 platos). La red de seguridad existe y no se usa.
3. **El 7 % infactible es la señal accionable**, y ya viene con dirección: `protein:high` en 34
   platos significa que el plato no tiene un portador de proteína suficiente y ninguna re-escala
   lo va a arreglar. Ese es el mismo hueco que persiguen el cerrador de proteína y el piso de
   franja — o sea que hay **dos capas trabajando sobre la misma señal sin compartirla**.
4. **«Se acabó el tiempo» no aparece**, y el solver no tiene por qué producirlo: es un mínimo
   cuadrados sobre unas pocas variables, no una búsqueda. Añadir un estado de timeout sería
   modelar una situación de la que no hay ni un caso.

## Recomendación

**No invertir la semana en factibilidad conjunta todavía.** Lo que la medición señala como valioso
es más barato: que el cerrador de proteína y el piso de franja **consuman `_solver_infeasible`**
en vez de volver a deducir el mismo hueco por su cuenta. Son 34 platos con la respuesta ya escrita
en el propio plato.

Anclado por [`test_p3_solver_termination_states.py`](../tests/test_p3_solver_termination_states.py).
