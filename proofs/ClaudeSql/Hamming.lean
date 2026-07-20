/-
  Invariant: 64-bit Hamming distance is a metric-shaped signal.

  Python surface (verified against source, branch feat/v2-hexagonal):
    src/claude_sql/domain/dedup.py:179-188
      def hamming_distance_64(a: int, b: int) -> int:
          a &= _U64_MASK
          b &= _U64_MASK
          return bin(a ^ b).count("1")      # popcount of the xor

  The near-dup gate (`NEAR_DUP_HAMMING_THRESHOLD = 3`, dedup.py:86) compares
  this distance against a small constant, so the ordering it induces must be
  well-behaved: identical signatures must read as distance 0, the distance must
  not depend on argument order, and it can never exceed the bit width (64) — a
  distance above the width would be a modelling bug that could never trip, or,
  worse, silently mis-rank.

  We model the two u64 signatures as bit lists (`List Bool`) and the Python
  `bin(a ^ b).count("1")` as position-wise popcount of the xor. Proving over a
  bit list rather than `Nat.xor` keeps the popcount recursion in plain sight and
  the proofs short; the 64-bit width enters as the `length = 64` hypothesis.

  Core Lean only — no mathlib.
-/

namespace ClaudeSql.Hamming

/-- Position-wise popcount of the xor of two bit vectors: the count of positions
    where the bits differ. This is exactly `bin(a ^ b).count("1")` over the bit
    lists (`Bool.xor x y` is `true` iff bit `x` differs from bit `y`). Extra bits
    past the shorter vector contribute nothing (the trailing `_ , _ => 0` arms),
    matching the fixed-width masking the Python does before the xor. -/
def hamming : List Bool → List Bool → Nat
  | x :: xs, y :: ys => (if Bool.xor x y then 1 else 0) + hamming xs ys
  | _, _ => 0

/-- Reflexivity: a signature has distance 0 from itself (`d(x,x) = 0`). -/
theorem hamming_refl : ∀ x : List Bool, hamming x x = 0
  | [] => rfl
  | b :: bs => by
      simp only [hamming]
      rw [hamming_refl bs]
      cases b <;> rfl

/-- Symmetry: distance is independent of argument order (`d(x,y) = d(y,x)`). -/
theorem hamming_symm : ∀ x y : List Bool, hamming x y = hamming y x
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by
      simp only [hamming]
      rw [hamming_symm as bs]
      cases a <;> cases b <;> rfl

/-- The popcount never exceeds the width of the first vector. -/
theorem hamming_le_length : ∀ x y : List Bool, hamming x y ≤ x.length
  | [], _ => Nat.le_refl 0
  | _ :: _, [] => Nat.zero_le _
  | a :: as, b :: bs => by
      simp only [hamming, List.length_cons]
      have ih := hamming_le_length as bs
      split <;> omega

/-- The 64-bit bound: for 64-wide signatures the distance is at most 64
    (`d(x,y) ≤ 64`), so a `< 3` near-dup test is always meaningful. -/
theorem hamming_le_64 (x y : List Bool) (h : x.length = 64) : hamming x y ≤ 64 := by
  have := hamming_le_length x y
  omega

end ClaudeSql.Hamming
