# ============================================================
# RECALIBRATION STUDY — Noise-Corrected Diagnostics (KDD version)
# ============================================================
# Replaces Study 5's threshold derivation. Three phases:
#
#   PHASE 1 — Validation against known truth.
#     In cells B/D the DGP perturbs nonzero pool entries with
#     N(0, HET_SCALE). True per-entry slope dispersion is therefore
#     HET_SCALE on the support and 0 elsewhere, so the ground-truth
#     BV target is computable per replication:
#         BV_true = mean_ij( HET_SCALE * mask_ij )
#     We show raw BV inflates as T_c shrinks (estimation noise)
#     while noise-corrected BV_nc tracks BV_true across T_c.
#
#   PHASE 2 — Threshold derivation on a CALIBRATION set only.
#     Median split on BV_nc and CRC_nc across the four base cells,
#     using seeds disjoint from every validation condition.
#
#   PHASE 3 — HELD-OUT validation of the thresholds:
#     (a) fresh replications, same geometry (new seeds);
#     (b) geometry shifts: T_c in {15, 25, 80}, C in {25, 100}, N=8;
#     (c) functional-form shift: strong nonlinearity (NL_SCALE=0.6
#         vs frozen 0.15);
#     (d) intermediate regimes: HET_SCALE in {0.05, 0.125} and
#         half-strength factor loadings — no "correct" cell exists,
#         so we report where the diagnostic places them.
#     Classification accuracy is reported for BOTH raw-BV thresholds
#     and BV_nc thresholds on every held-out condition, so the paper
#     can show what the correction buys.
#
# CANONICAL DIAGNOSTIC RECIPE (applies to synthetic AND V-Dem):
#   1. Global z-normalisation of the stacked panel (frozen
#      normalise()).
#   2. Per-unit OLS at K_DIAG = 1 regardless of the model's K,
#      WITH a per-unit intercept (units keep their own means under
#      global normalisation; omitting the intercept biases slopes).
#   3. Unit inclusion floor: residual dof = T_eff - p - 1 >= DOF_MIN
#      (p = N*K_DIAG slope regressors). Shorter units are excluded
#      from the diagnostics (they carry no slope information).
#   4. BV_nc: DerSimonian–Laird-style moment correction per entry:
#         tau2_ij = max(0, Var_c(betahat_ij) - mean_c(se2_c_ij))
#         BV_nc   = mean_ij sqrt(tau2_ij)
#      se2 from per-equation sigma2_hat * [(X'X)^{-1}]_jj.
#   5. CRC_nc: mean over pairs/vars of max(0, |r| - E0|r|) with
#      E0|r| = sqrt(2 / (pi * (T_ij - 2))) — first-order null bias
#      of |corr| under independence. Raw CRC reported alongside.
#
# Differences from the frozen NeurIPS recipe (documented for the
# paper): raw-vs-normalised inconsistency removed, ridge removed,
# K unified at 1, intercept added, sampling-noise correction added.
#
# Runtime: numpy only, no GPU. Full grid ~3-6 min on Colab CPU.
# Artifacts -> Drive: /content/drive/MyDrive/KDD_HNAVAR/
# ============================================================

# ── Cell 1: Drive mount and paths ─────────────────────────
import os, sys, json, math, time
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/KDD_HNAVAR'
else:
    BASE = os.environ.get('RECAL_BASE', './KDD_HNAVAR_local')
OUT_DIR = os.path.join(BASE, 'recalibration')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[path] OUT_DIR = {OUT_DIR}")

# ── Cell 2: Configuration ──────────────────────────────────
SEED_CAL   = 42      # calibration stream
SEED_VAL   = 90210   # held-out stream (disjoint by construction)
R_CAL      = 50      # calibration reps per cell
R_VAL      = 50      # held-out reps per condition
DOF_MIN    = 5
K_DIAG     = 1

# Frozen base geometry (Study 5)
BASE_N, BASE_C, BASE_T = 5, 50, 40
SIGMA2, SPARSITY       = 0.25, 0.40
HET_BASE, NL_BASE      = 0.25, 0.15

if os.environ.get('RECAL_SMOKE') == '1':
    R_CAL = R_VAL = 4
    print("[SMOKE MODE] Reduced reps; results not scientifically meaningful.")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Cell 3: DGP (frozen generate_dgp, parameterised) ──────
# Verbatim from study5_v5_nonlinear.py except HET_SCALE, NL_SCALE,
# FACTOR_SCALE are explicit arguments; defaults reproduce the frozen
# DGP exactly.
def sample_pool_graph(N, K, sparsity, rng, sr_max=0.50):
    while True:
        A = np.zeros((N, N, K))
        for i in range(N): A[i, i, 0] = rng.uniform(0.15, 0.30)
        for i in range(N):
            for j in range(N):
                if i != j and rng.random() < sparsity:
                    A[i, j, 0] = rng.choice([-1., 1.]) * rng.uniform(0.05, 0.20)
        if np.max(np.abs(np.linalg.eigvals(A[:, :, 0]))) <= sr_max:
            return A

def generate_dgp(dgp, N, C, T_c, K, sigma2, sparsity, rng,
                 het_scale=HET_BASE, nl_scale=NL_BASE, factor_scale=1.0):
    A_pool = sample_pool_graph(N, K, sparsity, rng)
    hetero = dgp in ('B', 'D')
    factor = dgp in ('C', 'D')
    if factor:
        F_series = np.zeros(T_c + K)
        F_series[0] = rng.normal(0, 1)
        for t in range(1, T_c + K):
            F_series[t] = 0.7 * F_series[t-1] + rng.normal(0, 0.5)
        loadings = rng.uniform(0.5, 1.0, (C, N)) * factor_scale
    else:
        F_series, loadings = None, None
    mask = (A_pool != 0).astype(float)
    Y_parts = []
    for c in range(C):
        if hetero:
            A_c = A_pool + rng.normal(0, het_scale, A_pool.shape) * mask
        else:
            A_c = A_pool.copy()
        buf = np.zeros((T_c + K, N))
        buf[:K] = rng.normal(0, 0.1, (K, N))
        eps = rng.normal(0, np.sqrt(sigma2), (T_c, N))
        for t in range(K, T_c + K):
            lags   = buf[t-1, :]
            raw    = A_c[:, :, 0].T @ lags
            raw_nl = raw + nl_scale * np.tanh(raw)
            shock  = eps[t - K]
            if factor:
                shock = shock + loadings[c] * F_series[t]
            buf[t] = np.clip(raw_nl + shock, -8, 8)
        Y_parts.append(buf[K:])
    # Ground-truth BV target: het dispersion on support, 0 off it,
    # averaged over ALL entries (matches BV's averaging convention).
    bv_true = float((het_scale if hetero else 0.0) * mask.mean())
    return np.vstack(Y_parts).astype(np.float32), [T_c] * C, bv_true

# ── Cell 4: Canonical diagnostics ─────────────────────────
def normalise(Y):                      # frozen (hnavar_joint_final)
    mu = Y.mean(0)
    sd = np.where(Y.std(0) < 1e-10, 1.0, Y.std(0))
    return (Y - mu) / sd

def unit_design(Yc, K):
    """Lagged design with intercept column appended."""
    L = len(Yc)
    Xl = np.hstack([Yc[K-1-k:L-1-k] for k in range(K)])
    Yo = Yc[K:]
    Xl = np.hstack([Xl, np.ones((len(Xl), 1))])
    return Xl, Yo

def diagnostics(Y, split_spec, K=K_DIAG, dof_min=DOF_MIN):
    """Return dict with bv_raw, bv_nc, crc_raw, crc_nc, n_units_used.

    bv_raw : mean over slope entries of cross-unit std of betahat
             (old definition, on the canonical recipe).
    bv_nc  : DL moment correction, see header.
    crc_*  : mean |resid corr| over pairs x vars, raw and
             null-debiased.
    """
    Yn = normalise(Y)
    N = Yn.shape[1]
    p = N * K                          # slope regressors (excl. intercept)
    betas, se2s, resids = [], [], []
    start = 0
    for L in split_spec:
        Yc = Yn[start:start+L]; start += L
        T_eff = L - K
        dof = T_eff - p - 1            # minus intercept
        if dof < dof_min:
            continue
        Xl, Yo = unit_design(Yc, K)
        XtX = Xl.T @ Xl
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        B = XtX_inv @ Xl.T @ Yo        # (p+1, N) — last row intercepts
        E = Yo - Xl @ B
        sig2 = (E**2).sum(0) / dof     # (N,) per-equation sigma2_hat
        diagXtX = np.diag(XtX_inv)[:p] # slope entries only
        se2 = np.outer(diagXtX, sig2)  # (p, N): se2 of betahat_{j,i}
        betas.append(B[:p].flatten())  # slopes only
        se2s.append(se2.flatten())
        resids.append(E)
    n_used = len(betas)
    if n_used < 2:
        return dict(bv_raw=np.nan, bv_nc=np.nan, crc_raw=np.nan,
                    crc_nc=np.nan, n_units_used=n_used)
    Bmat  = np.array(betas)            # (n_used, p*N)
    S2mat = np.array(se2s)
    var_obs  = Bmat.var(axis=0, ddof=1)
    mean_se2 = S2mat.mean(axis=0)
    tau2  = np.maximum(0.0, var_obs - mean_se2)
    bv_raw = float(np.sqrt(var_obs).mean())
    bv_nc  = float(np.sqrt(tau2).mean())
    tot_r, tot_nc, cnt = 0.0, 0.0, 0
    lens = {len(r) for r in resids}
    if len(lens) == 1 and lens.pop() >= 3:
        # Vectorised path: equal lengths -> one corr matrix per var.
        T_eq = len(resids[0])
        e0 = math.sqrt(2.0 / (math.pi * (T_eq - 2)))
        iu = np.triu_indices(n_used, k=1)
        for n in range(N):
            M = np.stack([r[:, n] for r in resids])   # (n_used, T)
            ok = M.std(axis=1) > 1e-10
            Rm = np.corrcoef(M)
            pair_ok = ok[iu[0]] & ok[iu[1]]
            r_abs = np.abs(Rm[iu])[pair_ok]
            tot_r  += float(r_abs.sum())
            tot_nc += float(np.maximum(0.0, r_abs - e0).sum())
            cnt    += int(pair_ok.sum())
    else:
        # Fallback: unequal lengths (real panels, e.g. V-Dem).
        for a in range(n_used):
            for b in range(a+1, n_used):
                ri, rj = resids[a], resids[b]
                T_ij = min(len(ri), len(rj))
                if T_ij < 3: continue
                e0 = math.sqrt(2.0 / (math.pi * (T_ij - 2)))
                for n in range(N):
                    si, sj = ri[:T_ij, n].std(), rj[:T_ij, n].std()
                    if si > 1e-10 and sj > 1e-10:
                        r = abs(float(np.corrcoef(ri[:T_ij, n],
                                                  rj[:T_ij, n])[0, 1]))
                        tot_r  += r
                        tot_nc += max(0.0, r - e0)
                        cnt    += 1
    crc_raw = tot_r / cnt if cnt else 0.0
    crc_nc  = tot_nc / cnt if cnt else 0.0
    return dict(bv_raw=bv_raw, bv_nc=bv_nc, crc_raw=crc_raw,
                crc_nc=crc_nc, n_units_used=n_used)

# ── Cell 5: Runner ────────────────────────────────────────
def run_condition(tag, dgp, N, C, T_c, reps, seed_base,
                  het_scale=HET_BASE, nl_scale=NL_BASE,
                  factor_scale=1.0):
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(
            seed_base + ord(dgp) * 100_000 + rep)
        Y, spec, bv_true = generate_dgp(
            dgp, N, C, T_c, K_DIAG, SIGMA2, SPARSITY, rng,
            het_scale=het_scale, nl_scale=nl_scale,
            factor_scale=factor_scale)
        d = diagnostics(Y, spec)
        d.update(tag=tag, dgp=dgp, rep=rep, N=N, C=C, T_c=T_c,
                 het_scale=het_scale, nl_scale=nl_scale,
                 factor_scale=factor_scale, bv_true=bv_true)
        rows.append(d)
    return rows

t0 = time.time()
all_rows = []

# PHASE 1 — truth tracking across T_c (calibration seed stream)
for T in [15, 25, 40, 80]:
    for dgp in 'ABCD':
        all_rows += run_condition(f'p1_T{T}', dgp, BASE_N, BASE_C, T,
                                  R_CAL, SEED_CAL)
print(f"[phase 1] done  {time.time()-t0:.0f}s")

# PHASE 2 — calibration set = phase-1 base geometry (T_c=40)
df = pd.DataFrame(all_rows)
cal = df[df.tag == 'p1_T40'].copy()
THR = {m: float(cal[m].median())
       for m in ['bv_raw', 'bv_nc', 'crc_raw', 'crc_nc']}
print("[phase 2] thresholds (calibration medians):")
for k, v in THR.items(): print(f"    {k:8s} = {v:.4f}")

def classify(row, bv_col, crc_col):
    hi_bv  = row[bv_col]  > THR[bv_col]
    hi_crc = row[crc_col] > THR[crc_col]
    return {(False, False): 'A', (True, False): 'B',
            (False, True): 'C', (True, True): 'D'}[(hi_bv, hi_crc)]

# PHASE 3 — held-out conditions (disjoint seed stream)
val_rows = []
# (a) fresh reps, base geometry
for dgp in 'ABCD':
    val_rows += run_condition('val_base', dgp, BASE_N, BASE_C, BASE_T,
                              R_VAL, SEED_VAL)
# (b) geometry shifts
for T in [15, 25, 80]:
    for dgp in 'ABCD':
        val_rows += run_condition(f'val_T{T}', dgp, BASE_N, BASE_C, T,
                                  R_VAL, SEED_VAL + 1_000_000)
for C_ in [25, 100]:
    for dgp in 'ABCD':
        val_rows += run_condition(f'val_C{C_}', dgp, BASE_N, C_, BASE_T,
                                  R_VAL, SEED_VAL + 2_000_000)
for dgp in 'ABCD':
    val_rows += run_condition('val_N8', dgp, 8, BASE_C, BASE_T,
                              R_VAL, SEED_VAL + 3_000_000)
# (c) strong nonlinearity
for dgp in 'ABCD':
    val_rows += run_condition('val_strongNL', dgp, BASE_N, BASE_C,
                              BASE_T, R_VAL, SEED_VAL + 4_000_000,
                              nl_scale=0.60)
# (d) intermediate regimes (characterisation only, no true cell)
for hs in [0.05, 0.125]:
    val_rows += run_condition(f'int_het{hs}', 'B', BASE_N, BASE_C,
                              BASE_T, R_VAL, SEED_VAL + 5_000_000,
                              het_scale=hs)
val_rows += run_condition('int_halffactor', 'C', BASE_N, BASE_C,
                          BASE_T, R_VAL, SEED_VAL + 6_000_000,
                          factor_scale=0.5)
dfv = pd.DataFrame(val_rows)
print(f"[phase 3] done  {time.time()-t0:.0f}s")

# ── Cell 6: Scoring and reports ───────────────────────────
dfv['pred_raw'] = dfv.apply(classify, axis=1,
                            args=('bv_raw', 'crc_raw'))
dfv['pred_nc']  = dfv.apply(classify, axis=1,
                            args=('bv_nc', 'crc_nc'))
scored = dfv[~dfv.tag.str.startswith('int_')].copy()
scored['ok_raw'] = scored.pred_raw == scored.dgp
scored['ok_nc']  = scored.pred_nc  == scored.dgp

print("\n== HELD-OUT CLASSIFICATION ACCURACY (raw vs noise-corrected) ==")
acc = scored.groupby('tag')[['ok_raw', 'ok_nc']].mean().round(3)
print(acc.to_string())
print("\nOverall held-out:  raw = {:.3f}   nc = {:.3f}".format(
    scored.ok_raw.mean(), scored.ok_nc.mean()))

print("\n== PHASE 1: BV vs truth across T_c (cells B/D, mean ± sd) ==")
p1 = df[df.dgp.isin(['B', 'D'])]
rep1 = p1.groupby('T_c').agg(
    bv_true=('bv_true', 'mean'),
    bv_raw_m=('bv_raw', 'mean'), bv_raw_s=('bv_raw', 'std'),
    bv_nc_m=('bv_nc', 'mean'),   bv_nc_s=('bv_nc', 'std')).round(4)
print(rep1.to_string())
print("\n== PHASE 1: null cells A/C (should be ~0 after correction) ==")
p0 = df[df.dgp.isin(['A', 'C'])]
rep0 = p0.groupby('T_c')[['bv_raw', 'bv_nc']].mean().round(4)
print(rep0.to_string())

print("\n== INTERMEDIATE REGIMES: placement (share per predicted cell) ==")
inter = dfv[dfv.tag.str.startswith('int_')]
for tag, g in inter.groupby('tag'):
    dist = g.pred_nc.value_counts(normalize=True).round(2).to_dict()
    print(f"  {tag:16s} -> {dist}   "
          f"(bv_nc={g.bv_nc.mean():.3f}, crc_nc={g.crc_nc.mean():.3f})")

# ── Cell 7: Artifacts ─────────────────────────────────────
df.to_csv(os.path.join(OUT_DIR, 'recal_phase1_calibration.csv'),
          index=False)
dfv.to_csv(os.path.join(OUT_DIR, 'recal_phase3_heldout.csv'),
           index=False)
with open(os.path.join(OUT_DIR, 'recal_thresholds.json'), 'w') as f:
    json.dump({'thresholds': THR,
               'recipe': {'K_DIAG': K_DIAG, 'DOF_MIN': DOF_MIN,
                          'normalise': 'global_z', 'ols': 'intercept',
                          'bv_nc': 'DL_moment', 'crc_nc': 'null_debias'},
               'seeds': {'cal': SEED_CAL, 'val': SEED_VAL},
               'R_cal': R_CAL, 'R_val': R_VAL}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
Ts = sorted(p1.T_c.unique())
for ax, cells, ttl in [(axes[0], ['B', 'D'], 'Heterogeneous cells (B, D)'),
                       (axes[1], ['A', 'C'], 'Null cells (A, C)')]:
    sub = df[df.dgp.isin(cells)]
    g = sub.groupby('T_c')
    ax.errorbar(Ts, g.bv_raw.mean(), yerr=g.bv_raw.std(),
                marker='o', label='raw BV', capsize=3)
    ax.errorbar(Ts, g.bv_nc.mean(), yerr=g.bv_nc.std(),
                marker='s', label='noise-corrected BV', capsize=3)
    ax.plot(Ts, g.bv_true.mean(), 'k--', label='ground truth')
    ax.set_xlabel('$T_c$'); ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=8)
axes[0].set_ylabel('BV')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_bv_vs_truth.png'), dpi=200)
print(f"\n[done] artifacts in {OUT_DIR}   total {time.time()-t0:.0f}s")
