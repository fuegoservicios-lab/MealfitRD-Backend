# Food label nutrients and portions

Food-label scans now retain declared micronutrients, their serving description and explicit consumed servings. Chat tools and the scanner persist one snapshot atomically with the diary row. Reading/repeating a meal preserves that evidence; quantity/food corrections invalidate previous evidence unless replaced. Unknown nutrients remain null, not zero. Label values take precedence over generic ingredient estimates without double counting.

Deployment requires `migrations/p1_food_label_nutrients_2026_10_12.sql` before the Python code. The nullable column is additive; rollback can retain it. Existing meals are not backfilled using guessed labels. Runtime-only food-intent guards and chat-photo recovery must be retained when publishing from this checkout.

Packaged-cookie ambiguity is clarified before writes. A calorie correction cannot manufacture a packet or gram weight. Explicit piece counts/weights and requested estimates remain available.

Voice opening prepares local SDP and microphone concurrently after a user gesture. The server reuses its HTTP connection and skips the 30-day spend query only when the optional budget is unlimited; positive caps remain enforced. No model/API migration. Reference: https://developers.openai.com/api/docs/guides/voice-webrtc . Device/network latency still needs measurement on the phone.

Validation: 40 targeted Python tests plus 10 parameterized subtests; 16 frontend tests; typecheck; ESLint (two pre-existing AgentPage Fast Refresh warnings); web/native builds and size budgets. Real vision read a synthetic label (three cookies per serving), retained fiber/sodium/calcium/iron and multiplied two servings correctly; absent magnesium remained null. This fixture is not a claim about any branded product. No user diary was modified by that test.
