/-
  ClaudeSql.Basic — toolchain smoke test.

  Proves the Lake project builds and Lean can discharge a trivial goal from
  commit one, so a red `proofs` gate always means a *real* proof regression,
  never a broken toolchain.
-/

theorem toolchain_ok : True := trivial
