# Free iOS access — 9 October 2026

iOS is a free offering with no subscription entitlement and no purchase or payment requirement. Web billing and Android entitlements remain separate.

- Base monthly allowance: 1,000 coach messages and 100 generation credits, designed to support approximately 14 days of frequent use. Actual duration depends on usage.
- All current app functions remain available, including manual logging, photos, pantry, plans and voice. Existing global voice safeguards remain applicable; live voice has no separate paid tier.
- Admin → Accounts → open an account → **iPhone · Gratis → Añadir recarga gratuita** adds 1,000 messages and 100 generation credits for 14 days. This is an additive free grant and can be repeated. It never changes PayPal or the web subscription.
- Base allowances renew on the first day of each month (UTC). Unused gifts expire after 14 days, independently of the monthly renewal.
- The recharge requires the administrator allowlist, the existing CSRF header, a reason and a UUID idempotency key. Both meters are granted in one transaction, with audit logging before the write. The history supports revocation of each gift.

`MEALFIT_IOS_FREE_ENABLED=true` enables this offering. Run `migrations/ios_free_allowance_2026_10_09.sql` before deploying the code. It adds scope columns with `web` defaults to existing records; it does not delete usage, modify subscriptions or rewrite historical gifts.

The API derives the iOS scope from the native WebView origin `capacitor://localhost`. Authenticated identity checks still apply. Profile reads, quotas, model routing, usage logs and server-owned plan snapshots use the scope; no client-provided tier controls the allowance. Scope is preserved through detached chat, live voice and asynchronous plan workers.

Validation: backend quota/billing/admin/generation tests, client tests for cached paid profiles and recharge retry, and PostgreSQL migration/retry/revocation checks using rolled-back test grants.

App Store: prepare a new binary containing this model. Physical iPhone testing and updated review information are required before resubmission. This change does not guarantee approval.
