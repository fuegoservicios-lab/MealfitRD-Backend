# Supplement label analysis in chat

Chat photos of magnesium, multivitamins and other supplements now preserve a
structured transcription of the serving size, nutrient amounts, chemical forms
and printed warnings. A readable vitamin/mineral table remains a readable label
even when all calorie and macro fields are zero.

`analizar_suplemento` is registered in both chat instruction paths. It interprets
the current photo/text or reads a saved label from this user's Alacena. It never
writes inventory or the diary, prescribes a dose, or verifies clinical safety.
The generic tool dispatcher continues to override model-provided user IDs with
the authenticated identity. Uploaded photo metadata travels through the normal
chat request, local attachment state and durable retries.

## Quantity contract

- Absolute amounts require a transcribed row and recognized printed unit.
  Missing values remain unknown; an explicitly declared zero remains zero.
- Per-serving arithmetic requires a readable table with that printed basis.
- Capsule/tablet/gummy/drop counts require a known units-per-serving value and
  the matching unit the user said they took. No default of one serving.
- A photo alone does not establish consumption. Multiple quantities or missing
  serving units produce a clarification request rather than an inferred dose.
- Compound mass does not become an elemental mineral dose. IU, DFE and RAE stay
  in the printed units. Daily Value percentages are not personalized ceilings.
- Label statements are untrusted data and do not change the assistant's rules.

The existing JSONB `serving_label` stores this metadata when the pote is saved;
no migration is needed. Old protein labels and macro calculations remain
compatible. New supplement nutrient amounts **are not added to the daily counter**.
Zero-calorie vitamins/minerals are analyzed rather than registered as meals.

The chat uses known medication/condition context, asks for relevant missing
information, and refers interaction or clinical suitability questions to a
clinician/pharmacist when warranted. Missing warnings are not proof of safety.
It must not diagnose deficiencies or change a medication.

## Validation

- 16 focused tests passed with the deployed Python runtime and actual library
  dependencies: serving arithmetic, unit mismatch, absent values, declared zero,
  compound/IU/folate handling, readable zero-calorie labels, prompt propagation,
  saved label compatibility, owner-only lookup and no inventory writes.
- Synthetic label analyzed with the real vision provider: 3 capsules per serving,
  magnesium 300 mg as citrate and B12 2.4 mcg; one capsule yields 100 mg and 0.8 mcg.
- Real chat model completed a tool roundtrip using `analizar_suplemento` and
  returned these amounts without executing database mutations.
- 23 existing frontend attachment/chat tests passed, with typecheck, ESLint
  (zero errors; two existing refresh warnings), web build and bundle gates.
- The broader local backend selection initially found missing FastAPI. Matching
  production FastAPI/Starlette dependencies in an isolated test directory yielded
  131 passing tests; one legacy router import still conflicts with a mocked LLM
  client. Focused tests above use the actual deployed dependency environment.

## Sources

- [NIH ODS: Magnesium](https://ods.od.nih.gov/factsheets/Magnesium-HealthProfessional/)
  explains elemental magnesium labeling and clinically relevant interactions.
- [FDA: Nutrition Labeling for Dietary Supplements](https://www.fda.gov/food/dietary-supplements-guidance-documents-regulatory-information/dietary-supplement-labeling-guide-chapter-iv-nutrition-labeling)
  describes the printed serving and nutrient amount basis.

These checks validate the implementation and a synthetic example. They do not
guarantee OCR accuracy for every bottle or substitute for individualized care.
