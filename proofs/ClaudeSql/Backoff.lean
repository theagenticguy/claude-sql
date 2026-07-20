/-
  Invariant: retry-queue exponential backoff is monotone and capped.

  Python surface (verified against source, branch feat/v2-hexagonal):
    src/claude_sql/infrastructure/sqlite_state/retry_queue.py:79-82
      def _backoff_delta(attempts: int) -> timedelta:
          minutes = min(2**attempts, _BACKOFF_CAP_MIN)   # _BACKOFF_CAP_MIN = 60 (line 39)
          return timedelta(minutes=minutes)

  We model the minute count `min(2^attempts, 60)` as a pure Nat function and
  prove the two properties the drain/enqueue logic silently relies on:
    * cap:  the backoff never exceeds 60 minutes, for ANY attempt count
            (so a runaway attempts counter can't schedule a retry years out);
    * mono: more attempts never schedule an EARLIER retry
            (backoff is non-decreasing in `attempts`).

  Core Lean only — pure Nat arithmetic, no mathlib.
-/

namespace ClaudeSql.Backoff

/-- Backoff minutes for a given attempt count: `min (2^attempts) 60`. -/
def backoffMinutes (attempts : Nat) : Nat := min (2 ^ attempts) 60

/-- The 60-minute cap: the delay never exceeds 60, whatever the attempt count. -/
theorem backoff_cap (n : Nat) : backoffMinutes n ≤ 60 :=
  Nat.min_le_right _ _

/-- Monotone in `attempts`: more attempts never yield a smaller delay. -/
theorem backoff_mono {a b : Nat} (h : a ≤ b) : backoffMinutes a ≤ backoffMinutes b :=
  Nat.le_min.mpr
    ⟨Nat.le_trans (Nat.min_le_left _ _) (Nat.pow_le_pow_right (by decide) h),
     Nat.min_le_right _ _⟩

/-- Once the cap is reached (2^a ≥ 60) the delay is pinned at exactly 60. -/
theorem backoff_saturates {a : Nat} (h : 60 ≤ 2 ^ a) : backoffMinutes a = 60 :=
  Nat.min_eq_right h

end ClaudeSql.Backoff
