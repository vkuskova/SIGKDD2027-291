# When Do Hierarchical Models Work in Panel Time Series?
## A Noise-Corrected Diagnostic and Validation Benchmark

Reproducibility repository — KDD 2027 Datasets & Benchmarks
submission. Code is MIT-licensed (see LICENSE). Every number in
the paper traces to a named file in `artifacts/`, produced by a
single canonical run of one script in `code/`.

## Layout
```
code/             seven self-contained scripts (run order below)
frozen_inputs/    frozen upstream artifacts (provenance below)
artifacts/        canonical outputs, one folder per study
```

## Artifacts → paper map
```
recalibration/     Sec 5, Fig 1, Table 1; thresholds JSON
vdem_diagnostics/  Sec 6, Fig 2, Tables 3-4 inputs
baselines/         Table 2 (Task 1 leaderboard); Task 2 CV
                   selector (Sec 7.2)
tcbv_calibration/  Sec 7.1-7.2; Fig 3 (appendix); split audit
studyM/            Sec 7.3-7.4; Fig 4 + per-arm table (appendix)
paper_figures/     column-sized Figs 3-4
```

## Run order and runtimes (Google Colab; set RECAL_BASE to run locally against this repo's layout)
1. `recal_bv_nc.py`         CPU ~5 min   -> recalibration/
2. `vdem_rediag.py`         CPU ~2 min   -> vdem_diagnostics/
3. `tcbv_calibration.py`    CPU ~3 min   -> tcbv_calibration/
   (regenerates every sweep panel from its condition seed and
   VERIFIES draw-exactness against frozen_inputs/
   study7_v3_results.csv before any new computation; aborts on
   mismatch)
4. `studyM_mixed_panel.py`  GPU (T4) ~40 min -> studyM/
   (resume-safe per (arm, rep))
5. `paper_figures.py`       CPU seconds  -> paper_figures/
6. `task2_split_eval.py`    CPU seconds  -> tcbv_calibration/
   task2_split_audit.json   (split-half selection audit, Sec 7.2)
7. `external_baselines.py`  CPU ~15 min  -> baselines/
   (Task 1 leaderboard: weighted DerSimonian-Laird and REML;
   Task 2 CV-linear selector. Verifies its regenerated grid
   row-by-row against the canonical recalibration CSVs and the
   sweep against archived values; aborts on any mismatch)

## Using the suite as a benchmark
A candidate diagnostic supplies per-panel statistics plus a
calibration rule. Task 1 (regime recovery): calibrate on the
calibration stream, score 4-class accuracy per held-out
condition. Task 2 (outcome prediction): score AUC / balanced
accuracy of a pre-fit statistic against the frozen sweep's
realized hierarchy outcomes. Entries selecting among candidate
statistics must report a nested or split-half audit
(see task2_split_eval.py). Current entries — Task 1: raw index,
weighted DL, REML, unweighted moment correction (reference);
Task 2: CV-linear selector, pre-fit SNR (reference).

## Frozen inputs and provenance
- `HDL_merged_notdev_selected.csv` — V-Dem-derived panel, 18,604
  rows, 134 eligible countries, 10 indicators (public V-Dem
  data; paper Appendix B documents variables and preprocessing).
- `study7_v3_results.csv` — per-replication estimator outcomes of
  the frozen length-heterogeneity sweep (120 replications).
  Reused, never rerun; regeneration is verified against this
  file's archived index values.
- `vdem/seed_{42,123,7}/democracy_results.csv` — frozen per-seed
  V-Dem estimator results; regional outcomes (paper Table 4) are
  recomputed from these files.
- `hnavar_joint_final.py` — frozen H-NAVAR implementation used
  unmodified by the mixed-panel study.

## Determinism and verification
All synthetic generation is seeded (documented condition-seed
formulas); the mixed-panel study seeds torch and is reproducible
across processes. Scripts 3 and 7 embed hard verification gates
against frozen/canonical artifacts and abort on mismatch.
`sha256_manifest.txt` lists checksums for every file present at
assembly.
