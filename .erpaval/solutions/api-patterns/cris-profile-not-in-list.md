---
name: Global CRIS profile may not enumerate in ListInferenceProfiles but still invokes
description: `global.anthropic.claude-sonnet-4-6` invokes successfully but does not appear in the list returned by `bedrock.list_inference_profiles`. Verify invocation, not listing.
type: reference
---

`bedrock.list_inference_profiles(typeEquals='SYSTEM_DEFINED')` does not
always return all active global CRIS profiles for an account, and the
`APPLICATION` type returns nothing on accounts without custom
application profiles. This gap can cause a false "this profile
doesn't exist" conclusion when the profile is actually callable.

Verification procedure:
  1. Do **not** rely on `list_inference_profiles` as proof of access.
  2. Call `invoke_model(modelId='global.<vendor>.<model>', body=...)` with a minimal payload and check the response.
  3. If invocation succeeds with the expected `model` field, the CRIS resolution works.

Confirmed working IDs on my current AWS account (2026-05):
  - `global.cohere.embed-v4:0` (embed)
  - `global.anthropic.claude-sonnet-4-6` (classification)
  - `global.anthropic.claude-haiku-4-5-20251001-v1:0` (Haiku eval)

The `ListInferenceProfiles` filtering appears to be a permissions-related
quirk on `SYSTEM_DEFINED` vs `APPLICATION`; invocation uses a different
RBAC path that resolves the profile regardless.

**When you hit this**: `ListInferenceProfiles` returns few rows but you
know the profile is in the region. Test with an actual `invoke_model`
ping before concluding the profile is unavailable.
