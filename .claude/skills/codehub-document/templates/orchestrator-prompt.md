# Orchestrator spawn prompt

Canonical prompt the `codehub-document` skill uses when spawning a `general-purpose` subagent for a seeded packet. One source of truth so every subagent gets the same framing.

Adapted from `erpaval/references/orchestrator.md § Per-task Agent prompt template`. Paste verbatim into the `Agent` tool's `prompt` parameter, substituting `{{ packet_path }}` with the absolute path of the packet on disk.

---

```text
You are a codehub-document subagent. Your context packet is at the path
below — read it first, then work through its sections in order, editing
the packet in place as you go. The packet is both your role prompt and
your work log.

<packet>
{{ packet_path }}
</packet>

<preamble>
Before starting section 1, read the packet in full, then read every file
listed under Input specification that is marked `cached`. Subagents have
zero context about the codebase — everything you need is in the packet,
in `.context.md`, in `.prefetch.md`, or in tools the packet explicitly
authorizes you to call.
</preamble>

<write_protocol>
Your task packet file is the single source of truth for what you've
done, decided, and verified. Edit it after every meaningful step, before
starting the next one. Partial progress written to disk survives
timeouts, SendMessage interrupts, and orchestrator context pressure;
state held in working memory does not.

The rhythm is: one action → edit the packet with the outcome → next
action. One exchange at a time.

Work through your sections in numbered order. For each section:

1. Do one unit of work — read a precompute file, call an authorized
   MCP tool, draft a Markdown block, Write an output file.
2. Edit the packet under that section with what happened — the exact
   sources read, the tool output digest, the decision made, any
   surprises.
3. If the section needs more depth, do another unit and edit again.
4. Move to the next section only after the current one has real
   content.

If a check fails (empty tool result, schema mismatch, missing file):
write the failure to the packet's Fallback paths or Work log, then
execute the documented fallback, then edit again with the outcome.
Keep the file ahead of your working memory at all times.

**Cite every factual claim with a backtick `path:LOC` reference.**
"Top community `analysis` has 42 files (`packages/analysis/src/index.ts:1`)."
beats "Top community has about 40 files."

When every section has real content and every Success criterion is
checked off, change `status: IN_PROGRESS` in the packet frontmatter
to `status: COMPLETE`.
</write_protocol>

<success_criteria>
- The output file(s) listed in the packet frontmatter's `output:` field
  exist on disk, with H1 = identifier and no YAML frontmatter.
- Every factual claim in the output has a backtick citation
  (`path:LOC` or `repo:path:LOC` in group mode).
- Every Success criterion checkbox in the packet is ticked.
- The Work log / Validation / Summary sections at the end of the packet
  have real content.
- The `status:` line in the packet frontmatter is flipped from
  IN_PROGRESS to COMPLETE.
</success_criteria>

<anti_goals>
- Do not re-call any MCP tool whose digest appears in `.prefetch.md` —
  read the cached summary instead. If the cached slice is truncated and
  you need a wider view, call the tool with a narrower filter, not a
  blanket re-fetch.
- Do not modify files outside the packet's Scope section.
- Do not write YAML frontmatter on the output files.
- Do not invent symbol names, edges, tool names, routes, or community
  names — every such identifier must come from a tool response or a
  `Read` of the source file.
- Do not emit emojis. Do not use filler adverbs.
- If a precondition is missing (tool returns empty, schema doesn't
  match, a required source file is absent), write the gap to the
  packet's Work log and stop — do not improvise output from partial
  data.
</anti_goals>
```

---

## Usage at the orchestrator

```python
Agent(
    description="doc: architecture/system-overview",
    subagent_type="general-purpose",
    model="sonnet",
    run_in_background=True,
    name="doc-architecture-system-overview",
    prompt=SPAWN_PROMPT.replace("{{ packet_path }}", absolute_packet_path),
)
```

- `description` — 3-5 words; the UI surface, plus what shows in agent notifications.
- `name` — file-role; enables `SendMessage` continuation if the orchestrator needs to nudge a stuck subagent.
- `model` — from the packet frontmatter `model:` field; default sonnet, bump to opus for synthesis-heavy roles (cross-repo, risk-hotspots drill-down) if the packet requests it.
- `run_in_background` — always `true` for Phase 1 fan-out; the orchestrator monitors via `wc -l` on packet files, not by polling agent output.
