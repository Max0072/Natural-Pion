# Journal index

Generated from `docs/JOURNAL.md` by `scripts/journal_index.py` -- 53 entries, 4880 lines. Do not edit by hand; re-run the script.

Line numbers are for `sed -n 'Np,+40p' docs/JOURNAL.md`.

| date | entry | line |
|---|---|---|
| 2026-08-24 | evening — the two anchor pairs | 11 |
| 2026-08-24 | 20:45 — what the runs actually cost, and an NGD number that changes the plan | 173 |
| 2026-08-24 | 22:10 — the b200 pair started, and step 1 finally has its second number | 262 |
| 2026-08-24 | 22:30 — fairshare, what a GPU costs, and the dirty marker fixed | 323 |
| 2026-08-24 | 22:50 — the fix committed, the running pair left alone | 409 |
| 2026-08-25 | ~05:30 — the anchor missed, and the gap missed by more | 447 |
| 2026-08-25 | the six differences, read from the code rather than listed | 528 |
| 2026-08-25 | their code read. Three real differences found, two entries closed | 585 |
| 2026-08-25 | all four changes made, `980e18c` | 669 |
| 2026-08-25 | 09:45 — the anchor re-run is in flight | 727 |
| 2026-08-25 | ~10:50 — `genoa` access arrived mid-morning | 778 |
| 2026-08-25 | the corpus is the same corpus, and the entry is gone | 812 |
| 2026-08-25 | 15:30 — the re-run: gap 4.5x -> 1.8x, criterion missed by 0.0011 | 850 |
| 2026-08-25 | the setting, not the optimizer: two differences on 56.5% of the model | 923 |
| 2026-08-25 | 16:35 — tidying, and what was deliberately left alone | 983 |
| 2026-08-25 | the 158x, measured rather than argued | 1027 |
| 2026-08-25 | the reference stays the reference | 1119 |
| 2026-08-25 | `ngd-pion` and `ngd-pion-ref` | 1193 |
| 2026-08-25 | 246531: the algorithm is runnable | 1247 |
| 2026-08-25 | 246607: I compared the wrong two things | 1335 |
|  | 246607: micro-batch 512 fits, for free | 1363 |
| 2026-08-25 | 246613: the missing 0.42 s is warm-up, not the allocator | 1387 |
| 2026-08-25 | settled: micro-batch 512 with `expandable_segments` | 1490 |
| 2026-08-25 | Newton-Schulz: measured, then dropped | 1530 |
| 2026-08-25 | 246662/246663: the smoke test earned its keep on the first try | 1604 |
| 2026-08-25 | the default configuration is now the anchor, minus the optimizer | 1642 |
| 2026-08-25 | first learning-rate probes, and a hypothesis cleanly killed | 1687 |
| 2026-08-26 | anchor round 3, and the decision to accept it | 1765 |
| 2026-08-26 | where the operator comes from, and what alpha actually measures | 1817 |
| 2026-08-26 | jobs 253057 / 253058: what `alpha == 1` actually is | 2178 |
| 2026-08-26 | the learning-rate sweep never varied only `eta` | 2459 |
| 2026-08-26 | evening -- the headline result, and what the literature already says | 2723 |
| 2026-08-27 | the full-length result, and why concurrent runs must share a seed | 2895 |
|  | The true Fisher does not rescue `eta = 2` -- the step is noise, not curvature | 3103 |
|  | Not degeneracy: the step tracks the scale of the curvature, not its spread | 3200 |
| 2026-08-27 | the measured `S` was never given its own `eta`, and `alpha` is not vestigial | 3316 |
| 2026-08-27 | correction: `rho` is clean, the clip is the confound, and `kappa` falls out | 3458 |
| 2026-08-27 | the measured S wins by 0.26, and the curvature gap is the independence assumption | 3541 |
| 2026-08-27 | `curv_exact`: the diagnostic, and a toy-scale first reading | 3665 |
| 2026-08-27 | evening -- 274008 and 274907: the Fisher line closes | 3738 |
| 2026-08-27 | evening -- Shampoo on so(n): the preconditioner changes, the geometry does not | 3822 |
| 2026-08-27 | night -- 276154: the first GPU reading, and prediction 1 lands | 3919 |
| 2026-08-27 | night -- 278352 / 278521: `eta` bracketed, and the step cost is real | 4023 |
| 2026-08-27 | night -- the control nobody had run: at 150 steps the rotation mostly hurts | 4126 |
| 2026-08-27 | night -- the 150-step ranking does not merely fail to predict; it inverts | 4240 |
| 2026-08-27 | night -- 279222/279617: the crossover, and a correction to this morning's reading | 4329 |
| 2026-08-28 | the step is 96% noise, and the memory we have is on the wrong side | 4422 |
| 2026-08-28 | 296244: momentum helps, and the `eta* ~ 2 SNR^2` account is refuted | 4520 |
| 2026-08-28 | read the horizontal gap, not the vertical one | 4607 |
| 2026-08-28 | 297391: the step is mostly the retraction, not the factorisation | 4656 |
| 2026-08-28 | their `pion_msign` read: it is our Shampoo-on-so(n) without the accumulator | 4709 |
| 2026-08-28 | the generator carries the half now, and it moved `pred_drop` with it | 4770 |
| 2026-08-28 | the trust region fires zero times in 338 steps, and `eta` is doing its job | 4821 |
