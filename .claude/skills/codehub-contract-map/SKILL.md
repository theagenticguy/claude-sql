---
name: codehub-contract-map
description: "Use when the user asks for a cross-repo contract map, an API-consumer matrix, or a service-interaction diagram across a repo group. Examples: \"map the HTTP contracts between services\", \"which services call the billing API\", \"show the contract matrix for the platform group\". GROUP MODE ONLY — requires a named group. DO NOT use on a single repo (use `codehub-document` with `reference/public-api.md`). DO NOT use if `mcp__codehub__group_list` does not include the group."
allowed-tools: "Read, Write, mcp__codehub__group_list, mcp__codehub__group_status, mcp__codehub__group_contracts, mcp__codehub__group_query, mcp__codehub__route_map, mcp__codehub__list_repos"
argument-hint: "<group-name> [--output <path>] [--committed]"
color: magenta
model: sonnet
---

# codehub-contract-map

Standalone group-only skill. Renders `group_contracts` into a Markdown + Mermaid artifact. Fires on direct invocations ("map the contracts") without needing the full `codehub-document` orchestration.

## Preconditions

1. A `<group-name>` positional argument is required. If missing or if `mcp__codehub__group_list` does not return the name, refuse with:
   `Contract map requires a named group — run 'codehub group list' to see registered groups.`
2. `mcp__codehub__group_status({group})` must return `fresh: true` for every member. If any member is stale, abort and name each stale repo.

## Arguments

- `<group-name>` (required positional) — the group to map.
- `--output <path>` (optional) — override the output path.
- `--committed` (optional) — write to a committed path instead of `.codehub/`.

Default output path:
- without `--committed`: `.codehub/groups/<name>/contracts.md` (gitignored)
- with `--committed`: `docs/<group>/contracts.md`

## Process

1. Run the preconditions. Refuse on missing/unknown group.
2. `mcp__codehub__group_list` — confirm `<group-name>` exists; read member list.
3. `mcp__codehub__group_status({group})` — confirm freshness per member. Abort with named stale repos otherwise.
4. `mcp__codehub__group_contracts({group})` — the spine. Returns `{consumerRepo, consumerRepoUri, consumerSymbol, producerRepo, producerRepoUri, producerRoute, method, path}` per row (legacy `consumerRepo`/`producerRepo` are the registry names; the `*RepoUri` siblings are the Sourcegraph-style cross-repo handle and are the preferred handle going forward).
5. If `group_contracts` returns `[]` (zero inter-repo contracts): still write the artifact with a `No inter-repo contracts detected` banner and an empty matrix. Do not error.
6. `mcp__codehub__group_query({group, text: "api handlers"})` — disambiguate producer-side locations.
7. For each member repo: `mcp__codehub__route_map({repo})` for handler-path citations.
8. Build the consumer/producer matrix: rows = producers, columns = consumers, cell = contract count.
9. Build the Mermaid `flowchart LR` showing inter-repo edges, labeled with contract counts.
10. Assemble the output using the template below.
11. `Write` to the resolved output path.

## Output template

### Normal case (contracts exist)

```markdown
# <group> · Contract map

*Generated <ISO-8601>. Members: <list>. Graph hashes: <list>.*

## Contracts matrix

Rows = producers; columns = consumers. Cell = number of contracts.

|       | billing | core | web |
|-------|---------|------|-----|
| billing | —     | 3    | 5   |
| core  | —       | —    | 12  |
| web   | —       | —    | —   |

## Flow

```mermaid
flowchart LR
  web --> billing : 5
  web --> core : 12
  billing --> core : 3
```

## Notable contracts

- **`web:packages/checkout/src/api.ts:22` → `billing:packages/api/src/handlers/invoice.ts:45`**
  - Method: `POST /v1/invoices`
  - Shape: `{amount, userId, idempotencyKey}`

- ... (top 10 contracts with direction, method, path, both-ends citations, shape summary)

## See also (other repos in group)

- [billing docs →](../billing/.codehub/docs/README.md)
- [core docs →](../core/.codehub/docs/README.md)
- [web docs →](../web/.codehub/docs/README.md)
```

### Empty case (zero contracts)

```markdown
# <group> · Contract map

*Generated <ISO-8601>. Members: <list>.*

**No inter-repo contracts detected.** The group graph does not currently encode cross-repo edges between these repos.

This can mean:
1. The repos genuinely do not interact (check whether that's expected).
2. `group_sync` has not yet run for this group — try `codehub group sync <name>`.
3. The contract surface is not yet captured by scanners (e.g., pub-sub channels that the graph does not model).

## Members

| Repo | Graph hash | Last indexed |
|---|---|---|
| billing | sha256:… | 2026-04-27T18:12:04Z |
| core | sha256:… | 2026-04-27T18:11:02Z |
| web | sha256:… | 2026-04-27T17:58:41Z |

## Empty matrix

|       | billing | core | web |
|-------|---------|------|-----|
| billing | —     | 0    | 0   |
| core  | —       | —    | 0   |
| web   | —       | —    | —   |
```

## Document format rules

- H1 = "{{group}} · Contract map".
- **Every citation MUST use the group-qualified form**: `` `<repo>:<path>:<LOC>` ``.
- The Mermaid diagram appears only when there is ≥ 1 inter-repo contract.
- Matrix table always rendered as an N×N grid, even when most cells are zero.
- Each member-repo link uses a relative path rooted at the group directory.
- No YAML frontmatter on the output.
- No emojis.

## Fallback paths

- If `group_contracts` times out: emit a partial matrix with `*partial — timed out*` in the affected rows; do not error. Record the timeout in a trailing `## Known limitations` section.
- If `group_query` returns nothing for `"api handlers"`: try `"http route"`, `"mcp tool"`, `"message consumer"` in order.
- If `route_map` errors for a single member: fall back to citing just `repo:package/path` without the `:LOC` suffix for that member; mark inline as `*route_map unavailable*`.

## Quality checklist

- [ ] `<group-name>` was required; refused if missing.
- [ ] `group_list` validated the name.
- [ ] Every member repo was `fresh` per `group_status`; otherwise aborted with named stale repos.
- [ ] Every citation uses the `repo:path:LOC` form.
- [ ] Matrix renders as a full N×N grid.
- [ ] Mermaid diagram appears iff ≥ 1 contract.
- [ ] Empty case produces an artifact (not an error) with the "No inter-repo contracts detected" banner.
- [ ] Output path respects `--committed` and `--output`.
- [ ] "See also (other repos in group)" footer lists every member repo's docs root.
