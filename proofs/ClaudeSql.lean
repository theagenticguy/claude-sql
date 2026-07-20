/-
  ClaudeSql — proof-layer root.

  Imports every invariant module so `lake build` (the `proofs` mise task)
  checks them all. Each import below corresponds to one pure/algebraic domain
  invariant proven against its cited Python surface.
-/

import ClaudeSql.Basic
import ClaudeSql.Backoff
import ClaudeSql.Hamming
import ClaudeSql.TurnSort
