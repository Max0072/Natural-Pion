# Early drafts, outside the method

Copied 2026-11-04 from the author's desktop, unchanged. Neither file describes
the method as it stands (`ngd_pion/natural_pion.py`, `docs/paper/FINAL.md`);
both are kept as a record of ideas considered.

* `ngd_pion.md` -- an intermediate write-up of the algorithm: measured `S`,
  separate EMA coefficients, the trust region dropped. Superseded by
  [`../ALGORITHM.md`](../ALGORITHM.md).
* `natural_gradient_pion_addendum.md` -- asymptotic speed-ups on top of the
  original `natural_gradient_pion.md` (not in this repository): low-rank
  sketches of `A` and `S` (Oja, Frequent Directions, Nystrom), structured
  Cholesky, a Woodbury Cayley, alternating sides, CG for `F(X) = G`, msign as
  a retraction. None was implemented; `ALGORITHM.md` keeps the dense exact path
  and excludes the addendum. It uses the older notation (Cholesky of `A`,
  `F(X) = 2(AXS' + S'XA)`), not the generalised eigenbasis of the paper.
