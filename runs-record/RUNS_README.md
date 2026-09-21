# What is in here

One directory per investigation, named for the question rather than the date.
Every directory keeps `log.jsonl`, `diagnostics.jsonl` and `manifest.json` for
each run; `manifest.json` records the full configuration and the git commit, so
any of these is reproducible from its own contents.

**Checkpoints were deleted everywhere except `headline`, `rtx`, `b200`,
`spectrum` and `anchor`.** They are 780 MB each, they regenerate by re-running,
and keeping them for throwaway 150-step sweeps cost 66 GB. The ones kept are
the full-length runs and the two whose weights an analysis still reads.

## Baselines, full length (73 242 steps)

    anchor/     two Pion anchor runs, written directly under runs/ by an early
                launch and moved here. Same config hash as two of the b200
                ones but different runs: val 3.4414 and 3.4059 against 3.4432
                and 3.4062. Same seed, same configuration, different hardware
                scheduling -- which makes them a measurement of full-length
                non-determinism, about 0.002.
    b200/       four Pion anchor runs on B200. val 3.3937 3.4432 3.4080 3.4062
    rtx/        two Pion anchor runs on RTX.  val 3.3719 3.3866
    headline/   the first full-length NGD-Pion runs. eta 1.0 completed at
                val 3.6728; eta 0.5 crashed at step 41 500 on a non-converging
                eigh inside basis_congruence.

## The covariance bug and its consequences

    smoke/      the first NGD-Pion runs, before any of the fixes
    qoc/  diag/ the quad-over-curv and skew-ratio diagnostics that led to it
    floor2/     the spectral floor probe that found curv going negative
    fp32cov/    verification that disabling autocast around the gram fixes it

## Throughput, which turned out to be page cache rather than code

    timing/ timing2/ ctl-idle/   per-window s/step traces on loaded and idle
                                 nodes. See docs/JOURNAL.md, 2026-08-26.

## Hyperparameters, with AdamW pinned at 1e-3

    split/      eta swept for NGD-Pion. Optimum near 3; 3e-3 was AdamW's.
    ablated/    eta swept for pion_ablated. Breaks between 0.5 and 3.
    tfac/ tfac2/  refactorisation cadence. 100 is bad, everything below 25 is
                  the same, and the saving stops there.
    seeds/      four seeds at one configuration: sd 0.024. The noise floor
                that a single 150-step run can resolve is about 0.05.
    a2/         alpha 2 against alpha 3 at eta 1, four seeds each: 0.8 se,
                not resolved, and not resolvable without ~75 seeds per arm.

## Initialisation

    htinit/     first attempt, scaled by Frobenius norm. **Void**: that made
                the operator norm grow with the spikiness, so the losses ranked
                by the scaling error rather than by the tail.
    htinit2/    rescaled to the feature-learning spectral norm, but still a
                pure power law across the whole spectrum rather than a bulk
                with a tail.
    initgrid/   eta swept per initialisation, which is the only fair way to
                compare them.
    tail/       the current construction: Marchenko-Pastur bulk with the top
                20% replaced by a power-law tail.
    spectrum/   2000 steps of plain AdamW, kept with its checkpoint, so the
                singular spectra of trained weights can be compared against
                their initialisation.

## Other

    anglecap/   a cap on the rotation angle. Made things worse; the divergence
                it was meant to prevent was AdamW's, not the rotation's.
    delta2/     BackwardProbe measuring ||delta|| per layer, which confirmed
                that S = I distributes the step across layers with the wrong
                sign.
