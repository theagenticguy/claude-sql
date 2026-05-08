---
name: Use Anthropic's canonical XML tag names for Claude system prompts
description: Prefer `<instructions>`, `<context>`, `<examples><example>...</example></examples>`, `<input>`, `<anti_patterns>`; avoid invented tags like `<task>` or `<output_format>`.
type: best-practice
---

Anthropic's prompt-engineering style guide (2026-05,
platform.claude.com/docs/en/docs/build-with-claude/prompt-engineering/use-xml-tags)
prescribes specific tag names. Consistency across prompts in a codebase
matters — Claude has been trained on these names.

**Canonical tags**:
  - `<instructions>` — task description, rules, output directives. Use instead of `<task>`.
  - `<context>` — background / situational framing. Use for label definitions and semantic rules.
  - `<examples>` wrapping multiple `<example>` — few-shot cases. Plural wrapper with singular children; nested `<input>` + `<output>` inside each is consistent with their variable-content conventions.
  - `<input>` — variable input slot (the thing being processed). Top-level, not an example sub-tag per Anthropic's reference, though using it inside `<example>` is fine.
  - Custom descriptive tags for output behavior: e.g. `<calibration>`, `<anti_patterns>`, `<output_rules>`, `<quality_bar>`.

**Anti-patterns they explicitly flag**:
  - Inconsistent tag names across prompts in the same codebase.
  - Flat structure when content has natural hierarchy — nest when you can.
  - Mixing markdown bullets with XML for the same content — pick one style.
  - Telling Claude what NOT to do without saying what TO do.

**Interactions with caching + structured output** (AWS Bedrock docs, 2026-05):
  - XML is transparent to prompt caching (it's just text tokens).
  - Toggling `output_config.format` invalidates both system and messages cache — don't flip structured output on/off between calls.
  - `output_config.format` injects its own system prompt describing the JSON schema; your XML-tagged system prompt coexists with it.
  - No hard XML-vs-JSON-schema conflicts, but token count rises when both are in play.

Canonical template:
```
<instructions>
Task + rules + output format directives.
</instructions>
<context>
Background, label definitions, semantic rules.
</context>
<calibration>
Decision rules, priors, confidence thresholds.
</calibration>
<examples>
<example><input>...</input><output>...</output></example>
<example><input>...</input><output>...</output></example>
</examples>
<anti_patterns>
Things NOT to do, with positive counter-examples.
</anti_patterns>
```

Anthropic recommends 3-5 examples; more rarely pays off once the
semantics are covered.

**When you hit this**: rewriting a markdown-style system prompt for
Claude. Replace headings (`## Calibration`) with `<calibration>`. Keep
tag names consistent across your four or five pipeline prompts so the
model sees the same structure every time.
