/-
  Invariant: the watch debounce cannot starve a continuously-written file.

  Python surface (verified against source, branch feat/realtime-refresh-and-watch):
    src/claude_sql/domain/watch.py:due_paths
      due = [
          path
          for path, entry in pending.items()
          if now_ns - entry.last_event_ns >= quiet_period_ns
          or (max_wait_ns > 0 and now_ns - entry.first_event_ns >= max_wait_ns)
      ]
    src/claude_sql/domain/watch.py:note_events
      # a path already pending KEEPS its first_event_ns and advances last_event_ns

  The debounce refreshes a transcript once it has been quiet for
  `quiet_period_ns`. That rule alone is not live: a session appended to more
  often than the quiet period never satisfies it, so its rows would never reach
  the snapshot. The implementation adds a second, disjunctive bound measured
  from the FIRST unflushed event, which `note_events` is careful never to move.

  Modeled over Nat with `now`, `first`, `last`, `quiet`, `maxWait`, and the two
  facts the caller guarantees: `first ≤ last` (the first event is not after the
  last) and `last ≤ now` (events are not from the future).

  Proven here:
    * dueAt_of_quiet     — the idle rule fires as specified;
    * dueAt_of_starved   — the starvation guard fires as specified;
    * no_starvation      — THE POINT. With maxWait > 0, a file is due once
                           maxWait has elapsed since its first event, for ANY
                           amount of subsequent write activity (any `last`).
    * mono_in_now        — once due, still due later (time never un-dues a file);
    * maxWait_zero_is_pure_idle — maxWait = 0 disables the guard exactly, so the
                           documented "0 restores the pure idle rule" holds.

  Core Lean only — pure Nat arithmetic, no mathlib.
-/

namespace ClaudeSql.Debounce

/-- One pending file's debounce state, in monotonic-clock nanoseconds. -/
structure Pending where
  first : Nat
  last  : Nat

/-- The quiet-period disjunct: the file has been untouched for `quiet`. -/
def idle (p : Pending) (now quiet : Nat) : Bool :=
  quiet ≤ now - p.last

/-- The starvation-guard disjunct: `maxWait` elapsed since the FIRST event. -/
def starved (p : Pending) (now maxWait : Nat) : Bool :=
  0 < maxWait && maxWait ≤ now - p.first

/-- `due_paths`' membership test for one pending file. -/
def dueAt (p : Pending) (now quiet maxWait : Nat) : Bool :=
  idle p now quiet || starved p now maxWait

/-- The idle rule fires exactly when the quiet period has elapsed. -/
theorem dueAt_of_quiet {p : Pending} {now quiet maxWait : Nat}
    (h : quiet ≤ now - p.last) : dueAt p now quiet maxWait = true := by
  simp [dueAt, idle, h]

/-- The starvation guard fires exactly when `maxWait` has elapsed. -/
theorem dueAt_of_starved {p : Pending} {now quiet maxWait : Nat}
    (hpos : 0 < maxWait) (h : maxWait ≤ now - p.first) :
    dueAt p now quiet maxWait = true := by
  simp [dueAt, starved, hpos, h]

/--
  NO STARVATION. Given an armed guard (`0 < maxWait`) and enough elapsed time
  since the file's first unflushed event, the file is due — regardless of
  `p.last`, i.e. regardless of how much writing has happened since.

  This is the property a pure quiet-period debounce lacks: there, a `last` that
  keeps advancing keeps `now - last` below `quiet` forever.
-/
theorem no_starvation {p : Pending} {now quiet maxWait : Nat}
    (hpos : 0 < maxWait) (helapsed : maxWait ≤ now - p.first) :
    dueAt p now quiet maxWait = true :=
  dueAt_of_starved hpos helapsed

/--
  Concretely: a file written to on every single tick still becomes due. Take
  `last = now` (written this very instant, so the idle rule is maximally
  unsatisfied) and a first event `maxWait` ago.
-/
theorem busy_file_still_due {quiet maxWait k : Nat} (hpos : 0 < maxWait) :
    dueAt { first := k, last := k + maxWait } (k + maxWait) quiet maxWait = true := by
  refine dueAt_of_starved hpos ?_
  simp

/-- Monotone in `now`: once due, always due. -/
theorem mono_in_now {p : Pending} {now now' quiet maxWait : Nat}
    (hle : now ≤ now') (h : dueAt p now quiet maxWait = true) :
    dueAt p now' quiet maxWait = true := by
  unfold dueAt idle starved at h ⊢
  rcases Bool.or_eq_true _ _ |>.mp h with hidle | hstarved
  · have : quiet ≤ now' - p.last :=
      Nat.le_trans (of_decide_eq_true hidle) (Nat.sub_le_sub_right hle _)
    simp [this]
  · have hand := Bool.and_eq_true _ _ |>.mp hstarved
    have hpos : 0 < maxWait := of_decide_eq_true hand.left
    have : maxWait ≤ now' - p.first :=
      Nat.le_trans (of_decide_eq_true hand.right) (Nat.sub_le_sub_right hle _)
    simp [hpos, this]

/-- `maxWait = 0` disables the guard exactly: dueness reduces to the idle rule. -/
theorem maxWait_zero_is_pure_idle (p : Pending) (now quiet : Nat) :
    dueAt p now quiet 0 = idle p now quiet := by
  simp [dueAt, starved]

/-- A zero quiet period makes every pending file due at once (the flush-now path). -/
theorem quiet_zero_is_always_due (p : Pending) (now maxWait : Nat) :
    dueAt p now 0 maxWait = true := by
  simp [dueAt, idle]

end ClaudeSql.Debounce
