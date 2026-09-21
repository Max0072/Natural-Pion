# ADR 0004 -- the corpus is their corpus and the budget is 9.6B tokens

- **Status:** Accepted
- **Date:** 2026-08-25. Recorded retroactively on 2026-09-22 from `AGENTS.md`, `ALGORITHM.md` and the journal; the original decision is older, and the journal is the source for what was measured at the time.
- **Decided by:** project

## Context

The comparison is only meaningful on the data the Pion paper used, and the
number of steps in their released code disagreed with their paper.

## Decision

C4 with the T5 vocabulary, 9.6B tokens: 73 242 steps of 131 072 tokens
(B = 512 sequences of 256). The rest of C4 is not tokenised. The 37 500-step
reading of their shell script is rejected as a defect of their released code, and
`RunConfig.train_steps` follows the paper.

## Consequences

Both runs consume about 6% of C4 (their 9.6B out of roughly 156B). The shards
are a deterministic partition of an already shuffled crawl, so our first 64 are
exchangeable with any others.

## Evidence

Tokenising the remainder would cost about 156 core-hours, 312 GB and days of
fairshare for a different draw from the same distribution. `TokenCorpus` samples
without an epoch, so an undersized corpus repeats tokens silently, which is why
the size matters (`harness/data.py`).
