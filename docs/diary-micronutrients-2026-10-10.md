# Daily nutrient contract v1 · 2026-10-10

17 daily indicators, including fiber and sodium. New fields: magnesium_mg,
zinc_mg, b12_mcg, folate_mcg (DFE), vit_e_mg (alpha-tocopherol), vit_k_mcg,
selenium_mcg, vit_b6_mg, iodine_mcg. Existing eight keys remain unchanged.

Missing composition is null; reported zero remains zero. Negative, boolean or
nonfinite values are rejected. Each meal includes ingredient coverage; the day
includes meal coverage per nutrient. A complete day requires every registered
meal to have complete ingredient composition for that nutrient. Old payloads
without per-nutrient evidence are partial, never assumed complete.

All surfaces use the same calculation. Only consumed meals count; photos and
custom macros without ingredients are not backfilled from invented estimates.
The existing calories-versus-ingredients guard remains active.

The adult reference registry requires known age (19+) and sex; pregnancy and
lactation differ. RDA/AI are intake references, not toxicity limits. No universal
140% health ceiling is inferred. Sodium retains WHO's recommended maximum;
additional UL markers are omitted because this food diary does not establish
supplement intake, chemical forms, clinical applicability, or jurisdiction.
Mg's supplement-only UL, folate's folic-acid/DFE distinction and B12's absent UL
are explained in the UI. Reference source/scope/date accompany each target.

Catalog migration is additive and nullable. The importer uses only audited
catalog/FDC identity pairs; it validates nutrient ID and unit and never falls
back from total folate to DFE. Missing imported values remain null. New daily
indicators require reported provenance; old fill/proxy values without that
evidence are not counted. Source, food identifier, retrieval date, version and
description are returned with each meal and daily coverage. Iodine matches
unique exact Standard Reference identifiers in USDA/FDA/NIH Release 4 (2024);
unmatched foods are unknown. US composition estimates may vary locally.

History recalculates existing stored ingredient descriptions against the current
catalog; old photos remain unknown. Catalog corrections can change past totals,
as disclosed in the UI. Import backups preserve prior composition and provenance.
Responses identify the version/source actually used for each computation; they
are not snapshots of what a person previously saw. No per-nutrient network
request is introduced. Catalog is cached for five minutes and deployment
restarts its process after the transactional import to invalidate that cache.

Validation: Python unit/integration contracts, JS optimistic/share parity, web
and native builds, translations, type checking, changed-file lint, isolated real
component layout at 320/390/430/1280px and 200% text in Chromium/Firefox/WebKit.
Run frontend visual fixture with `playwright test --config=playwright.nutrients.config.js`.
Cross-repo Python tests support MEALFIT_FRONTEND_TEST_DIR for the exact paired checkout.
The visual fixture stays outside production assets. App Store build 41 remains
in review; this compatible backend/web/OTA release does not replace that submission.

## Production verification on 2026-10-10

The transactional import applied 184 audited food matches in a 354-row catalog.
Reported coverage: Mg 182, Zn 182, B12 160, folate DFE 157, E 153, K 155,
selenium 167, B6 175, iodine 54. Remaining missing/unverified values stay unknown.
The original seven extension columns had a NOT NULL constraint; migration
removes that constraint to preserve absence instead of coercing it to zero.
The initial failed import rolled back before any food rows were changed.

Real-catalog portion check (100 g egg, 200 g milk, 100 g white rice): B6 0.300 mg
with three known ingredients; iodine 116.2 mcg with two known ingredients and
partial coverage. Adding an unknown photo keeps the daily coverage partial.
The previous code and new code used the same resolver/catalog for the compute
comparison: ten meals, 100 repetitions, medians 5.246 and 6.029 ms; new p95
6.072 ms. This measures calculation, not end-to-end API/network latency. The
synthetic ten-meal response is 106,059 bytes or 1,938 bytes gzip; repeated foods
compress unusually well. No per-nutrient query or network request was added.

Validation passed: 57 Python contracts, 74 frontend tests, 30 visual checks across
Chromium/Firefox/WebKit, eight compiled-web smoke checks in Chromium/WebKit,
TypeScript, changed-file ESLint, translations and bundle gates. Initial JS gzip
147.0 KiB / 148 KiB; Terser uses safe defaults. Native OTA builds and its package
checks passed. This is not a physical-device confirmation of the new feature.

Runtime rollback copies and before/after catalog evidence are preserved in
`/opt/mealfit/rollback-micros-20261010/`. Older hashed web assets and OTA packages
must be retained during the atomic frontend publication for already-open clients.
