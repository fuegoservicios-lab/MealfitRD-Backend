# Validación clínica — [P0-CLINICAL-VALIDATION · 2026-06-14]

Cierra **la parte implementable en código** del gap P0 del audit: _"precisión autoafirmada, sin
validación externa ni humana"_. La validación tiene tres capas; aquí cubrimos las dos automatizables.

## 1. Harness de precisión (`scripts/benchmark_macro_compliance.py`)

Mide, sobre un set FIJO y diverso de 20 perfiles held-out, qué tan cerca queda el plan ENTREGADO de
su propio TARGET (kcal/P/C/F), end-to-end (skeleton → LLM → solver → assemble → review). Separa REAL
vs FALLBACK. **Fix [P0-CLINICAL-VALIDATION]:** ahora abre los pools de Neon antes de generar (se
creaban con `open=False`; sin abrirlos, todo caía a `PoolClosed`→fallback y el harness medía el plan
matemático, no el path real). Uso: `python scripts/benchmark_macro_compliance.py [N]`.

## 2. Export de validación (`scripts/clinical_validation_export.py`)

Sobre planes REALES persistidos en Neon (no generaciones nuevas → costo cero), hace dos cosas:

- **Check de integridad automático (validación "externa" determinista):** para cada comida recomputa
  los macros DESDE LOS INGREDIENTES vía el catálogo `master_ingredients` (ground-truth independiente
  del LLM) y los compara con los macros que el LLM AFIRMÓ en el header. Divergencia grande = el plato
  no aporta lo que el header dice. Es lo más cercano a validación externa sin un dataset público.
  - **Confound conocido:** la recomputación es un LÍMITE INFERIOR cuando el catálogo no resuelve todos
    los ingredientes (los no resueltos aportan 0). La columna `res_pct` lo expone; el agregado de
    integridad solo cuenta días con `res_pct >= 60`.
- **Export para nutricionista (habilita la validación humana):** CSV (`utf-8-sig`, abre en Excel) con
  una fila por día: `target` vs `claim`(LLM) vs `recomp`(catálogo) por macro + `band_score` + `res_pct`
  + columnas en blanco `nutricionista_aprobado`/`notas`. Un profesional revisa la muestra y marca.

Uso: `python scripts/clinical_validation_export.py --n 15 --days 45 --out /tmp/clinical_review.csv`.

### Hallazgo medido (2026-06-14, 15 planes reales / 43 días)

- **Integridad claim-vs-recomputado: ~63%** dentro de ±15% (el resto = LLM sobre-afirma macros y/o
  catálogo no resuelve). Mezcla de dos causas; `res_pct` distingue.
- **El macro roto es la PROTEÍNA:** target ~154g pero entregado frecuentemente 47–110g (30–70% del
  target) — déficit sistémico. Las kcal quedan mucho más cerca del target. Esto es el insumo directo
  para la mejora de precisión (gap P0-3).

## 3. Lo que NO es código (honesto)

- **Benchmark público externo (estilo NutriBench):** requiere un dataset de ground-truth nutricional
  (idealmente de comida latina/dominicana) que NO está integrado — no hay acceso al dataset y el
  público existente es comida general en inglés. El harness #2 (recompute desde catálogo propio) es el
  proxy determinista disponible hasta que exista un dataset.
- **Revisión por nutricionista certificado:** es un paso de PROCESO, no de código. El export #2 lo
  habilita (genera la muestra revisable); la revisión la hace un humano.

Tests: [`test_p0_clinical_validation_export.py`](../tests/test_p0_clinical_validation_export.py).
