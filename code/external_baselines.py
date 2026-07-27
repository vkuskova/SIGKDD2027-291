# ============================================================
# EXTERNAL BASELINES — Task 1 and Task 2 (KDD)
# ============================================================
# Adds the two baselines the external review identified as the
# gap between borderline and clearer accept:
#
# PART A — Task 1 (regime recovery) leaderboard.
#   Two established random-effects heterogeneity estimators are
#   computed per coefficient entry from the SAME per-unit
#   estimates as the paper's unweighted moment correction:
#     dlw  : precision-weighted DerSimonian-Laird
#            Q = sum w_c (b_c - bbar_w)^2, w_c = 1/se2_c;
#            tau2 = max(0, (Q-(k-1)) / (S1 - S2/S1)),
#            S1 = sum w, S2 = sum w^2
#     reml : restricted maximum likelihood, golden-section search
#            on the REML log-likelihood over tau2
#   Each baseline forms BV_x = mean_ij sqrt(tau2_x,ij) and is
#   paired with the paper's CRC_nc for the dependence axis, so the
#   comparison isolates the heterogeneity axis. Thresholds use the
#   same pooled-median calibration rule, fixed on the calibration
#   stream before held-out scoring. Reported per method:
#   per-condition and overall 4-class accuracy (stratified
#   bootstrap CI) and binary heterogeneity-axis accuracy.
#
#   VERIFICATION GATE: the grid regenerates every panel from the
#   recalibration seeds; the paper's bv_raw/bv_nc/crc_nc values
#   recomputed here must match the canonical
#   recal_phase1_calibration.csv / recal_phase3_heldout.csv row
#   by row (atol 1e-5) or the script aborts.
#
# PART B — Task 2 (outcome prediction) baseline.
#   CV-linear selector: per sweep panel, hold out the last 20% of
#   each unit's rows; fit (i) a pooled ridge linear VAR and
#   (ii) per-unit empirical-Bayes-shrunk linear coefficients
#   (per-unit OLS shrunk toward the precision-weighted mean with
#   the dlw tau2); the selector's score is the validation-MSE
#   margin  m = mse_pool - mse_shrunk  (positive: hierarchy
#   predicted to pay). Scored by AUC and best balanced accuracy
#   against the frozen sweep's realized H-NAVAR vs Pool B
#   outcomes, with the same split-half audit as the reference
#   entry. Sweep panels regenerate under the script-3 draw-exact
#   gate.
#
# INPUTS (Drive):
#   /KDD_HNAVAR/recalibration/recal_phase1_calibration.csv
#   /KDD_HNAVAR/recalibration/recal_phase3_heldout.csv
#   /KDD_HNAVAR/frozen/study7_v3_results.csv
# OUTPUTS (Drive):
#   /KDD_HNAVAR/baselines/task1_leaderboard.csv
#   /KDD_HNAVAR/baselines/task1_per_condition.csv
#   /KDD_HNAVAR/baselines/task2_baselines.csv
#   /KDD_HNAVAR/baselines/baselines_summary.json
#
# Runtime: CPU only, ~10-20 min (REML per entry dominates).
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
P1_PATH = os.path.join(BASE, 'recalibration',
                       'recal_phase1_calibration.csv')
P3_PATH = os.path.join(BASE, 'recalibration',
                       'recal_phase3_heldout.csv')
S7_PATH = os.path.join(BASE, 'frozen', 'study7_v3_results.csv')
OUT_DIR = os.path.join(BASE, 'baselines')
os.makedirs(OUT_DIR, exist_ok=True)
for p in (P1_PATH, P3_PATH, S7_PATH):
    assert os.path.exists(p), f"Missing input: {p}"
print(f"[path] OUT_DIR = {OUT_DIR}")

import numpy as np, pandas as pd

# ── Cell 2: Recalibration-grid DGP and diagnostics ────────
# Verbatim from recal_bv_nc.py so regeneration is draw-exact.
SEED_CAL, SEED_VAL = 42, 90210
R_CAL = R_VAL = 50
DOF_MIN, K_DIAG = 5, 1
BASE_N, BASE_C, BASE_T = 5, 50, 40
SIGMA2, SPARSITY = 0.25, 0.40
HET_BASE, NL_BASE = 0.25, 0.15

if os.environ.get('BASE_SMOKE') == '1':
    R_CAL = R_VAL = 4
    print("[SMOKE MODE]")

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
            lags = buf[t-1, :]
            raw = A_c[:, :, 0].T @ lags
            raw_nl = raw + nl_scale * np.tanh(raw)
            shock = eps[t - K]
            if factor:
                shock = shock + loadings[c] * F_series[t]
            buf[t] = np.clip(raw_nl + shock, -8, 8)
        Y_parts.append(buf[K:])
    return np.vstack(Y_parts).astype(np.float32), [T_c] * C

def normalise(Y):
    mu = Y.mean(0)
    sd = np.where(Y.std(0) < 1e-10, 1.0, Y.std(0))
    return (Y - mu) / sd

def unit_design(Yc, K):
    L = len(Yc)
    Xl = np.hstack([Yc[K-1-k:L-1-k] for k in range(K)])
    Yo = Yc[K:]
    Xl = np.hstack([Xl, np.ones((len(Xl), 1))])
    return Xl, Yo

def reml_tau2(b, s2, lo=0.0, hi=10.0, iters=80):
    """Golden-section max of the REML log-likelihood over tau2."""
    def nll(t2):
        w = 1.0 / (s2 + t2)
        mu = (w * b).sum() / w.sum()
        return -(0.5 * np.log(w).sum() - 0.5 * math.log(w.sum())
                 - 0.5 * (w * (b - mu) ** 2).sum())
    g = (math.sqrt(5) - 1) / 2
    a, d = lo, hi
    c_, b_ = d - g * (d - a), a + g * (d - a)
    fc, fb = nll(c_), nll(b_)
    for _ in range(iters):
        if fc < fb:
            d, b_, fb = b_, c_, fc
            c_ = d - g * (d - a); fc = nll(c_)
        else:
            a, c_, fc = c_, b_, fb
            b_ = a + g * (d - a); fb = nll(b_)
    return max(0.0, (a + d) / 2)

def diagnostics_multi(Y, split_spec, K=K_DIAG, dof_min=DOF_MIN):
    """recal_bv_nc.py diagnostics extended with dlw and reml
    heterogeneity estimators on the same betas/se2s."""
    Yn = normalise(Y)
    N = Yn.shape[1]
    p = N * K
    betas, se2s, resids = [], [], []
    start = 0
    for L in split_spec:
        Yc = Yn[start:start+L]; start += L
        T_eff = L - K
        dof = T_eff - p - 1
        if dof < dof_min:
            continue
        Xl, Yo = unit_design(Yc, K)
        XtX = Xl.T @ Xl
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        B = XtX_inv @ Xl.T @ Yo
        E = Yo - Xl @ B
        sig2 = (E**2).sum(0) / dof
        diagXtX = np.diag(XtX_inv)[:p]
        se2 = np.outer(diagXtX, sig2)
        betas.append(B[:p].flatten())
        se2s.append(se2.flatten())
        resids.append(E)
    n_used = len(betas)
    if n_used < 2:
        return None
    Bmat = np.array(betas); S2mat = np.array(se2s)
    var_obs = Bmat.var(axis=0, ddof=1)
    mean_se2 = S2mat.mean(axis=0)
    tau2_ours = np.maximum(0.0, var_obs - mean_se2)
    # weighted DL and REML per entry
    k = n_used
    tau2_dlw = np.zeros(Bmat.shape[1])
    tau2_rml = np.zeros(Bmat.shape[1])
    for j in range(Bmat.shape[1]):
        b = Bmat[:, j]; s2 = np.maximum(S2mat[:, j], 1e-12)
        w = 1.0 / s2
        mu_w = (w * b).sum() / w.sum()
        Q = (w * (b - mu_w) ** 2).sum()
        S1, S2_ = w.sum(), (w ** 2).sum()
        denom = S1 - S2_ / S1
        tau2_dlw[j] = max(0.0, (Q - (k - 1)) / denom) if denom > 0 else 0.0
        tau2_rml[j] = reml_tau2(b, s2)
    # CRC (vectorised equal-length path; matches recal_bv_nc.py)
    tot_r, tot_nc, cnt = 0.0, 0.0, 0
    T_eq = len(resids[0])
    if T_eq >= 3:
        e0 = math.sqrt(2.0 / (math.pi * (T_eq - 2)))
        iu = np.triu_indices(n_used, k=1)
        for n in range(N):
            M = np.stack([r[:, n] for r in resids])
            ok = M.std(axis=1) > 1e-10
            Rm = np.corrcoef(M)
            pair_ok = ok[iu[0]] & ok[iu[1]]
            r_abs = np.abs(Rm[iu])[pair_ok]
            tot_r += float(r_abs.sum())
            tot_nc += float(np.maximum(0.0, r_abs - e0).sum())
            cnt += int(pair_ok.sum())
    return dict(
        bv_raw=float(np.sqrt(var_obs).mean()),
        bv_nc=float(np.sqrt(tau2_ours).mean()),
        bv_dlw=float(np.sqrt(tau2_dlw).mean()),
        bv_reml=float(np.sqrt(tau2_rml).mean()),
        crc_raw=tot_r / cnt if cnt else 0.0,
        crc_nc=tot_nc / cnt if cnt else 0.0)

# ── Cell 3: Regenerate grid, verify vs canonical, collect ─
CONDS = []
for T in [15, 25, 40, 80]:
    for d in 'ABCD':
        CONDS.append((f'p1_T{T}', d, BASE_N, BASE_C, T, R_CAL,
                      SEED_CAL, HET_BASE, NL_BASE, 1.0))
for d in 'ABCD':
    CONDS.append(('val_base', d, BASE_N, BASE_C, BASE_T, R_VAL,
                  SEED_VAL, HET_BASE, NL_BASE, 1.0))
for T in [15, 25, 80]:
    for d in 'ABCD':
        CONDS.append((f'val_T{T}', d, BASE_N, BASE_C, T, R_VAL,
                      SEED_VAL + 1_000_000, HET_BASE, NL_BASE, 1.0))
for C_ in [25, 100]:
    for d in 'ABCD':
        CONDS.append((f'val_C{C_}', d, BASE_N, C_, BASE_T, R_VAL,
                      SEED_VAL + 2_000_000, HET_BASE, NL_BASE, 1.0))
for d in 'ABCD':
    CONDS.append(('val_N8', d, 8, BASE_C, BASE_T, R_VAL,
                  SEED_VAL + 3_000_000, HET_BASE, NL_BASE, 1.0))
for d in 'ABCD':
    CONDS.append(('val_strongNL', d, BASE_N, BASE_C, BASE_T, R_VAL,
                  SEED_VAL + 4_000_000, HET_BASE, 0.60, 1.0))

t0 = time.time()
rows = []
for tag, dgp, N, C, T_c, reps, seed_base, hs, nl, fs in CONDS:
    for rep in range(reps):
        rng = np.random.default_rng(
            seed_base + ord(dgp) * 100_000 + rep)
        Y, spec = generate_dgp(dgp, N, C, T_c, K_DIAG, SIGMA2,
                               SPARSITY, rng, het_scale=hs,
                               nl_scale=nl, factor_scale=fs)
        d = diagnostics_multi(Y, spec)
        d.update(tag=tag, dgp=dgp, rep=rep)
        rows.append(d)
    print(f"[{tag} {dgp}] done  {time.time()-t0:.0f}s", flush=True)
grid = pd.DataFrame(rows)

# Verification gate vs canonical artifacts
canon = pd.concat([pd.read_csv(P1_PATH), pd.read_csv(P3_PATH)],
                  ignore_index=True)
canon = canon[~canon.tag.str.startswith('int_')]
mg = grid.merge(canon, on=['tag', 'dgp', 'rep'],
                suffixes=('', '_canon'), validate='1:1')
assert len(mg) == len(grid), "join mismatch with canonical CSVs"
for col in ['bv_raw', 'bv_nc', 'crc_raw', 'crc_nc']:
    d = (mg[col] - mg[f'{col}_canon']).abs().max()
    assert d < 1e-5, f"VERIFICATION FAILED: {col} max dev {d}"
print(f"[verify] regeneration matches canonical CSVs on all four "
      f"paper statistics ({len(mg)} panels)")

# ── Cell 4: Task 1 leaderboard ────────────────────────────
cal = grid[grid.tag == 'p1_T40']
heldout = grid[grid.tag.str.startswith('val_')]
METHODS = {'raw': 'bv_raw', 'ours': 'bv_nc',
           'dlw': 'bv_dlw', 'reml': 'bv_reml'}
CRC_COL = {'raw': 'crc_raw', 'ours': 'crc_nc',
           'dlw': 'crc_nc', 'reml': 'crc_nc'}
THR = {}
for mname, col in METHODS.items():
    THR[mname] = dict(bv=float(cal[col].median()),
                      crc=float(cal[CRC_COL[mname]].median()))
print("\nThresholds (pooled calibration medians):")
for mname, t in THR.items():
    print(f"  {mname:5s} bv={t['bv']:.4f} crc={t['crc']:.4f}")

def classify4(bv, crc, tb, tc):
    return np.where(bv > tb, np.where(crc > tc, 'D', 'B'),
                    np.where(crc > tc, 'C', 'A'))

rng_b = np.random.default_rng(7)
lead_rows, cond_rows = [], []
for mname, col in METHODS.items():
    tb, tc = THR[mname]['bv'], THR[mname]['crc']
    pred = classify4(heldout[col].to_numpy(),
                     heldout[CRC_COL[mname]].to_numpy(), tb, tc)
    ok = pred == heldout.dgp.to_numpy()
    het_true = heldout.dgp.isin(['B', 'D']).to_numpy()
    het_pred = heldout[col].to_numpy() > tb
    ok2 = het_pred == het_true
    groups = [ok[(heldout.tag == t).to_numpy()]
              for t in sorted(heldout.tag.unique())]
    boots = [np.mean([g[rng_b.integers(0, len(g), len(g))].mean()
                      for g in groups]) for _ in range(2000)]
    for t in sorted(heldout.tag.unique()):
        m_ = (heldout.tag == t).to_numpy()
        cond_rows.append(dict(method=mname, condition=t,
                              acc4=float(ok[m_].mean()),
                              acc_het=float(ok2[m_].mean())))
    lead_rows.append(dict(
        method=mname, acc4=float(ok.mean()),
        ci_lo=float(np.percentile(boots, 2.5)),
        ci_hi=float(np.percentile(boots, 97.5)),
        acc4_T15=float(ok[(heldout.tag == 'val_T15').to_numpy()].mean()),
        acc_het=float(ok2.mean())))
lead = pd.DataFrame(lead_rows)
print("\n== TASK 1 LEADERBOARD (held-out) ==")
print(lead.round(3).to_string(index=False))
pd.DataFrame(cond_rows).to_csv(
    os.path.join(OUT_DIR, 'task1_per_condition.csv'), index=False)
lead.to_csv(os.path.join(OUT_DIR, 'task1_leaderboard.csv'),
            index=False)

# ── Cell 5: Task 2 — CV-linear selector on the sweep ──────
S7_N, S7_C, S7_K = 5, 20, 1

def generate_sweep_panel(T_c, pert, rep):
    cond_seed = 42 + T_c * 10000 + int(pert * 1000) * 100 + rep
    rng = np.random.default_rng(cond_seed)
    A_pool = sample_pool_graph(S7_N, S7_K, SPARSITY, rng)
    Y_parts = []
    for c in range(S7_C):
        mask = (A_pool != 0).astype(float)
        A_c = A_pool + rng.normal(0, pert * 0.25, A_pool.shape) * mask
        if np.max(np.abs(np.linalg.eigvals(A_c[:, :, 0]))) > 0.95:
            for shrink in [0.5, 0.25, 0.1, 0.0]:
                A_try = A_pool + shrink * (A_c - A_pool)
                if np.max(np.abs(np.linalg.eigvals(A_try[:, :, 0]))) <= 0.90:
                    A_c = A_try; break
        buf = np.zeros((T_c + S7_K, S7_N))
        buf[:S7_K] = rng.normal(0, 0.1, (S7_K, S7_N))
        eps = rng.normal(0, np.sqrt(SIGMA2), (T_c, S7_N))
        for t in range(S7_K, T_c + S7_K):
            raw = A_c[:, :, 0].T @ buf[t-1, :]
            buf[t] = np.clip(raw + 0.15 * np.tanh(raw) + eps[t - S7_K],
                             -8, 8)
        Y_parts.append(buf[S7_K:])
    return np.vstack(Y_parts).astype(np.float32), [T_c] * S7_C

def bv_ridge_s7(Y, split_spec, K, alpha=0.01):
    N = Y.shape[1]; start = 0; betas = []
    for L in split_spec:
        Yc = Y[start:start+L]; start += L
        if L <= K + 1: continue
        Xl = np.hstack([Yc[K-1-k:L-1-k] for k in range(K)])
        Yo = Yc[K:]
        if len(Xl) < N*K + 1: continue
        XtX = Xl.T @ Xl
        B = np.linalg.solve(XtX + alpha*np.eye(XtX.shape[0]), Xl.T @ Yo)
        betas.append(B.flatten())
    if len(betas) < 2: return float('nan')
    return float(np.array(betas).std(axis=0).mean())

def cv_linear_selector(Y, split_spec, val_frac=0.20, alpha=1.0):
    """Margin = val MSE(pooled ridge) - val MSE(EB-shrunk per-unit).
    Positive: hierarchy predicted to pay. Deterministic."""
    Yn = normalise(Y)
    N = Yn.shape[1]; p = N * K_DIAG
    tr_X, tr_Y, units = [], [], []
    start = 0
    for L in split_spec:
        Yc = Yn[start:start+L]; start += L
        Xl, Yo = unit_design(Yc, K_DIAG)
        n_val = max(1, int(round(len(Xl) * val_frac)))
        units.append(dict(Xtr=Xl[:-n_val], Ytr=Yo[:-n_val],
                          Xva=Xl[-n_val:], Yva=Yo[-n_val:]))
    Xp = np.vstack([u['Xtr'] for u in units])
    Yp = np.vstack([u['Ytr'] for u in units])
    Bp = np.linalg.solve(Xp.T @ Xp + alpha*np.eye(Xp.shape[1]),
                         Xp.T @ Yp)
    # per-unit OLS + weighted-DL shrinkage toward precision mean
    betas, se2s = [], []
    for u in units:
        Xu, Yu = u['Xtr'], u['Ytr']
        dof = len(Xu) - Xu.shape[1]
        if dof < 3:
            betas.append(None); se2s.append(None); continue
        XtX_inv = np.linalg.pinv(Xu.T @ Xu, rcond=1e-8,
                                 hermitian=True)
        Bu = XtX_inv @ Xu.T @ Yu
        Eu = Yu - Xu @ Bu
        sig2 = (Eu**2).sum(0) / dof
        se2 = np.outer(np.diag(XtX_inv), sig2)
        betas.append(Bu); se2s.append(np.maximum(se2, 1e-10))
    ok_idx = [i for i, b in enumerate(betas) if b is not None]
    Ball = np.stack([betas[i] for i in ok_idx])       # (k, p+1, N)
    S2all = np.stack([se2s[i] for i in ok_idx])
    k = len(ok_idx)
    W = 1.0 / S2all
    mu_w = (W * Ball).sum(0) / W.sum(0)
    Q = (W * (Ball - mu_w) ** 2).sum(0)
    S1, S2_ = W.sum(0), (W ** 2).sum(0)
    denom = np.maximum(S1 - S2_ / S1, 1e-10)
    tau2 = np.maximum(0.0, (Q - (k - 1)) / denom)
    mse_pool, mse_shr, n_tot = 0.0, 0.0, 0
    for i, u in enumerate(units):
        Xv, Yv = u['Xva'], u['Yva']
        if len(Xv) == 0: continue
        mse_pool += float(((Xv @ Bp - Yv) ** 2).sum())
        if betas[i] is None:
            Bs = mu_w
        else:
            lam = tau2 / (tau2 + se2s[i])
            Bs = mu_w + lam * (betas[i] - mu_w)
        mse_shr += float(((Xv @ Bs - Yv) ** 2).sum())
        n_tot += Yv.size
    return (mse_pool - mse_shr) / n_tot

s7 = pd.read_csv(S7_PATH).drop_duplicates(
    subset=['T_c', 'pert_scale', 'rep'], keep='last')
t0 = time.time()
t2_rows, mism = [], 0
for _, r in s7.iterrows():
    T_c, pert, rep = int(r['T_c']), float(r['pert_scale']), int(r['rep'])
    Y, spec = generate_sweep_panel(T_c, pert, rep)
    if not np.isclose(bv_ridge_s7(Y, spec, S7_K), r['bv'],
                      rtol=0, atol=1e-4):
        mism += 1; continue
    t2_rows.append(dict(T_c=T_c, pert_scale=pert, rep=rep,
                        win=int(r['mse_hn_vs_poolB'] < 0),
                        cv_margin=cv_linear_selector(Y, spec)))
assert mism == 0, f"{mism} sweep regeneration mismatches"
t2 = pd.DataFrame(t2_rows)
print(f"\n[verify] all {len(t2)} sweep panels draw-exact "
      f"({time.time()-t0:.0f}s)")

def auc_mw(x, y):
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))

def best_bacc(x, y):
    xs = np.sort(np.unique(x)); cuts = (xs[1:]+xs[:-1])/2
    best = 0.0; P, Nn = y.sum(), (1-y).sum()
    for c in cuts:
        pr = (x > c).astype(int)
        b = ((pr & y).sum()/P + ((1-pr) & (1-y)).sum()/Nn)/2
        best = max(best, b)
    return float(best)

y = t2.win.to_numpy(); x = t2.cv_margin.to_numpy()
res_t2 = dict(auc=round(auc_mw(x, y), 3),
              bal_acc=round(best_bacc(x, y), 3))
splits = []
for sel, ev in [([0,1,2,3],[4,5,6,7]), ([4,5,6,7],[0,1,2,3])]:
    te = t2[t2.rep.isin(ev)]
    splits.append(round(auc_mw(te.cv_margin.to_numpy(),
                               te.win.to_numpy()), 3))
print("\n== TASK 2: CV-linear selector vs realized outcomes ==")
print(f"  AUC={res_t2['auc']}  best bal_acc={res_t2['bal_acc']}  "
      f"split-half AUCs={splits}")
print("  (reference SNR entry: AUC 0.904, bal_acc 0.867, "
      "split AUCs 0.933/0.882)")
t2.to_csv(os.path.join(OUT_DIR, 'task2_baselines.csv'), index=False)

with open(os.path.join(OUT_DIR, 'baselines_summary.json'), 'w') as f:
    json.dump({'task1_thresholds': THR,
               'task1_leaderboard': lead.to_dict(orient='records'),
               'task2_cv_linear': dict(**res_t2,
                                       split_half_aucs=splits)},
              f, indent=2)
print(f"\n[done] artifacts in {OUT_DIR}   total {time.time()-t0:.0f}s")
