# When Do Hierarchical Models Work in Panel Time Series?
# A Noise-Corrected Diagnostic and Validation Benchmark
Reproducibility bundle for KDD 2027 Datasets & Benchmarks submission, Paper #291.

Every number in the paper traces to a named file in `artifacts/`,
each produced by a single canonical run of one script in `code/`.

## Layout
```
code/                       five self-contained scripts (run order below)
frozen_inputs/              frozen upstream artifacts (provenance below)
artifacts/recalibration/    Study outputs: noise-corrected diagnostic
                            validation (Sec 5; Fig 1; Tables 1, 7)
artifacts/vdem_diagnostics/ V-Dem re-diagnostics (Sec 6; Fig 2; Table 2)
artifacts/tcbv_calibration/ sweep regeneration + sufficiency statistic
                            (Sec 7.1-7.2; Fig 3)
artifacts/studyM/           mixed-panel shared-prior study (Sec 7.3-7.4;
                            Fig 4; Table 8)
artifacts/paper_figures/    column-sized Figures 3 and 4
```

## Run order and runtimes (Google Colab)
1. `recal_bv_nc.py`        — CPU, ~5 min  -> artifacts/recalibration/
2. `vdem_rediag.py`        — CPU, ~2 min  -> artifacts/vdem_diagnostics/
   (reads frozen_inputs/HDL_merged_notdev_selected.csv and the
   thresholds JSON from step 1)
3. `tcbv_calibration.py`   — CPU, ~3 min  -> artifacts/tcbv_calibration/
   (regenerates every sweep panel from its condition seed and VERIFIES
   draw-exactness against the archived index values in
   frozen_inputs/study7_v3_results.csv before any new computation;
   aborts on any mismatch)
4. `studyM_mixed_panel.py` — GPU (T4), ~40 min -> artifacts/studyM/
   (resume-safe per (arm, rep))
5. `paper_figures.py`      — CPU, seconds -> artifacts/paper_figures/

Scripts read/write Google Drive paths under `/KDD_HNAVAR/` when run
in Colab; set `RECAL_BASE` to run against a local copy of this
bundle's layout instead.

## Frozen inputs and provenance
- `HDL_merged_notdev_selected.csv` — V-Dem-derived panel, 18,604 rows,
  134 eligible countries, 10 indicators (public V-Dem data; see paper
  Sec 6 for variable list and licensing).
- `study7_v3_results.csv` — per-replication estimator outcomes of the
  frozen length-heterogeneity sweep (120 replications). Reused, never
  rerun; `tcbv_calibration.py` verifies exact panel regeneration
  against this file's archived BV values.
- `vdem/seed_{42,123,7}/democracy_results.csv` — frozen per-seed
  V-Dem estimator results; regional outcomes in the paper (Table 3)
  are recomputed from these files, never quoted.
- `hnavar_joint_final.py` — frozen H-NAVAR implementation used
  unmodified by the mixed-panel study.

## Determinism
All synthetic generation is seeded (`numpy.random.default_rng` with
documented condition-seed formulas); the mixed-panel study seeds
torch as well and is reproducible across processes. Diagnostic
computations are deterministic given inputs.

## Verification
`sha256_manifest.txt` lists checksums for every file present at
bundle assembly. Files produced by re-running the scripts should be
compared against the corresponding artifact CSVs.
