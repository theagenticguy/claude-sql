/-
  Invariant: the transcript render sort key is a TOTAL order on distinct rows,
  and the uuid tiebreak is what makes it so (the determinism claim).

  Python surface (verified against source, branch feat/v2-hexagonal):
    src/claude_sql/domain/transcript.py:317-327 (render_turn_text)
      turns.append((ts, _collapse_kind_rank(row), str(row.uuid or ""), line))
      ...
      turns.sort(key=lambda t: (t[0], t[1], t[2]))     # (ts, kind_rank, uuid)
    docstring (transcript.py:306-308): "Ordering is (ts, kind_rank, uuid) ...
      with the message uuid as the final tiebreak."

  `render_turn_text` promises to be "Pure and deterministic: the same input
  always renders the same bytes." That promise is FALSE unless the sort key is a
  total order — a Python list `.sort` on a key with ties leaves tied elements in
  input order, and input order here is the (nondeterministic) DuckDB `read_json`
  scan order. So determinism reduces to: any two DISTINCT rows are strictly
  ordered by the key. Since a session's rows carry distinct message uuids, the
  uuid is exactly the component that guarantees this.

  We model a row as its sort-key triple `(ts, rank, uuid)` (strings ↦ Nat, which
  is order-faithful for the lexicographic comparison the proof is about) and:
    * `key_total` — with distinct uuids, the full `(ts,rank,uuid)` order is
      connex: one row is strictly below the other. This IS determinism.
    * `partial_not_total` — the `(ts,rank)` key WITHOUT the uuid is NOT connex:
      two distinct rows tied on `(ts,rank)` are unordered. This is the proof
      earning its keep — it pins WHY the uuid tiebreak is load-bearing.

  Core Lean only — Nat lexicographic order, `omega` discharges the arithmetic.
-/

namespace ClaudeSql.TurnSort

/-- A render row reduced to its sort-key components. `ts` and `uuid` are modeled
    as `Nat` (order-isomorphic to the ISO-timestamp / uuid strings for the
    lexicographic comparison at issue); `rank` is the `_collapse_kind_rank`. -/
structure Row where
  ts : Nat
  rank : Nat
  uuid : Nat

/-- The full sort key order used by `turns.sort`: lexicographic on
    `(ts, rank, uuid)`. -/
def lt (a b : Row) : Prop :=
  a.ts < b.ts ∨ (a.ts = b.ts ∧ (a.rank < b.rank ∨ (a.rank = b.rank ∧ a.uuid < b.uuid)))

/-- The key with the uuid tiebreak DROPPED: lexicographic on `(ts, rank)` only. -/
def ltPartial (a b : Row) : Prop :=
  a.ts < b.ts ∨ (a.ts = b.ts ∧ a.rank < b.rank)

/-- Determinism: distinct rows (distinct uuids) are STRICTLY ordered by the full
    key — the order is connex on distinct rows, so `sort` has no ties to resolve
    by (nondeterministic) input order. -/
theorem key_total (a b : Row) (h : a.uuid ≠ b.uuid) : lt a b ∨ lt b a := by
  unfold lt
  omega

/-- The full key is also irreflexive — no row is strictly below itself — so it
    is a genuine strict order, not merely connex. -/
theorem key_irrefl (a : Row) : ¬ lt a a := by
  unfold lt
  omega

/-- The uuid tiebreak earns its keep: the `(ts, rank)`-only key is NOT connex.
    Here are two DISTINCT rows (distinct uuids, same `ts` and `rank`) that the
    partial key leaves unordered — exactly the tie where `sort` would fall back
    to input order and break the determinism promise. -/
theorem partial_not_total :
    ∃ a b : Row, a.uuid ≠ b.uuid ∧ ¬ ltPartial a b ∧ ¬ ltPartial b a := by
  refine ⟨⟨0, 0, 0⟩, ⟨0, 0, 1⟩, ?_, ?_, ?_⟩
  · decide
  · unfold ltPartial; simp
  · unfold ltPartial; simp

end ClaudeSql.TurnSort
