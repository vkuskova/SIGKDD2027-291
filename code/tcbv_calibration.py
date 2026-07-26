# ============================================================
# Tc x BV CALIBRATION — Sufficiency Condition for Hierarchy (KDD)
# ============================================================
# Turns the NeurIPS paper's uncalibrated "T_c x BV" proposal into a
# calibrated, validated condition — WITHOUT re-running the neural
# sweep. Study 7's per-condition seeding is fully deterministic
# (cond_seed = SEED + T_c*10^4 + int(pert*10^3)*10^2 + rep), so every
# frozen panel is exactly regenerable. This script:
#
#   1. Regenerates all 120 Study 7 panels from their condition seeds.
#   2. VERIFIES draw-exactness: recomputes Study 7's own BV recipe
#      (raw data + ridge) and requires a float match against the
#      frozen 'bv' column. Aborts on any mismatch.
#   3. Computes CANONICAL diagnostics (bv_nc, noise_floor, crc_nc)
#      on each regenerated panel and joins with the frozen
#      performance results (mse_hn_vs_poolB etc.).
#   4. Calibrates a sufficiency statistic on win = (H-NAVAR beats
#      Pool B on held-out MSE). Candidates:
#        bv_nc            true-heterogeneity level alone
#        tc_x_bvnc        T_c * bv_nc      (paper's proposal, in
#                                           corrected units)
#        snr              bv_nc / noise_floor  — per-unit signal-to-
#                         noise; theory says deltas are learnable
#                         when true dispersion exceeds per-unit
#                         estimation noise, i.e. snr ~ 1 boundary
#        tc_x_snr         T_c * snr
#      For each: best single-threshold balanced accuracy + AUC.
#   5. Places the V-Dem groups (from vdem_diag_groups.csv) on the
#      calibrated statistic and compares the prediction against the
#      EMPIRICAL per-group outcome computed from the frozen per-seed
#      V-Dem results (gap vs Pool B, win rates) — the 96/78/53
#      decomposition, recomputed from artifacts, never quoted.
#
# INPUTS (Drive):
#   /KDD_HNAVAR/frozen/study7_v3_results.csv          (from bundle)
#   /KDD_HNAVAR/frozen/vdem/seed_{42,123,7}/democracy_results.csv
#   /KDD_HNAVAR/vdem_diagnostics/vdem_diag_groups.csv (script 2)
# OUTPUTS (Drive):
#   /KDD_HNAVAR/tcbv_calibration/tcbv_merged.csv
#   /KDD_HNAVAR/tcbv_calibration/tcbv_calibration.json
#   /KDD_HNAVAR/tcbv_calibration/fig_tcbv_plane.png
#
# GEOMETRY CAVEAT (report in paper): Study 7 runs at N=5, C=20,
# balanced panels; V-Dem groups are N=10, C=28-69, unbalanced. The
# statistic transfers at the level of the diagnostic recipe; a
# targeted top-up run at V-Dem-like geometry is the follow-up if the
# boundary placement looks fragile.
#
# Runtime: numpy only, no GPU. ~2-4 min.
# ============================================================

# ── Cell 1: Drive mount and paths ─────────────────────────
import os, sys, json, math, time, warnings
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/KDD_HNAVAR'
else:
    BASE = os.environ.get('RECAL_BASE', './KDD_HNAVAR_local')
S7_PATH   = os.path.join(BASE, 'frozen', 'study7_v3_results.csv')
VDEM_DIR  = os.path.join(BASE, 'frozen', 'vdem')
DIAG_PATH = os.path.join(BASE, 'vdem_diagnostics',
                         'vdem_diag_groups.csv')
OUT_DIR   = os.path.join(BASE, 'tcbv_calibration')
os.makedirs(OUT_DIR, exist_ok=True)
for p in (S7_PATH, DIAG_PATH):
    assert os.path.exists(p), f"Missing input: {p}"
print(f"[path] OUT_DIR = {OUT_DIR}")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── Cell 2: Frozen Study 7 DGP (verbatim) ─────────────────
SEED     = 42
N_vars   = 5
K        = 1
C        = 20
SIGMA2   = 0.25
SPARSITY = 0.40

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

def generate_dgp_B_param(N, C, T_c, K, sigma2, sparsity, pert_scale, rng):
    A_pool = sample_pool_graph(N, K, sparsity, rng)
    Y_parts = []
    for c in range(C):
        mask = (A_pool != 0).astype(float)
        A_c  = A_pool + rng.normal(0, pert_scale * 0.25, A_pool.shape) * mask
        if np.max(np.abs(np.linalg.eigvals(A_c[:, :, 0]))) > 0.95:
            for shrink in [0.5, 0.25, 0.1, 0.0]:
                A_c_try = A_pool + shrink * (A_c - A_pool)
                if np.max(np.abs(np.linalg.eigvals(A_c_try[:, :, 0]))) <= 0.90:
                    A_c = A_c_try; break
        buf = np.zeros((T_c + K, N))
        buf[:K] = rng.normal(0, 0.1, (K, N))
        eps = rng.normal(0, np.sqrt(sigma2), (T_c, N))
        for t in range(K, T_c + K):
            lags = buf[t-1, :]
            raw  = A_c[:, :, 0].T @ lags
            buf[t] = np.clip(raw + 0.15 * np.tanh(raw) + eps[t - K], -8, 8)
        Y_parts.append(buf[K:])
    return np.vstack(Y_parts).astype(np.float32), [T_c] * C

def beta_variance_index_ridge(Y, split_spec, K, alpha=0.01):
    """Study 7's frozen BV recipe — used ONLY to verify regeneration."""
    N = Y.shape[1]; start = 0; betas = []
    for L in split_spec:
        Yc = Y[start:start+L]; start += L
        if L <= K + 1: continue
        Xl = np.hstack([Yc[K-1-k:L-1-k] for k in range(K)])
        Yo = Yc[K:]
        if len(Xl) < N*K + 1: continue
        XtX = Xl.T @ Xl
        B   = np.linalg.solve(XtX + alpha*np.eye(XtX.shape[0]), Xl.T @ Yo)
        betas.append(B.flatten())
    if len(betas) < 2: return float('nan')
    return float(np.array(betas).std(axis=0).mean())

# ── Cell 3: Canonical diagnostics (identical to script 2) ─
K_DIAG, DOF_MIN = 1, 5
SE2_MAX, MIN_UNITS_ENTRY, VAR_TOL = 4.0, 5, 1e-8

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

def canonical_diagnostics(Y, split_spec, K=K_DIAG, dof_min=DOF_MIN):
    Yn = normalise(Y)
    N = Yn.shape[1]
    p = N * K
    betas, se2s, resids = [], [], []
    start = 0
    for L in split_spec:
        Yc = Yn[start:start+L]; start += L
        col_ok = np.tile(Yc.std(0) >= VAR_TOL, K)
        p_used = int(col_ok.sum())
        dof = (L - K) - p_used - 1
        if p_used == 0 or dof < dof_min:
            continue
        Xl_full, Yo = unit_design(Yc, K)
        Xl = np.hstack([Xl_full[:, :p][:, col_ok], Xl_full[:, -1:]])
        XtX = Xl.T @ Xl
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-8, hermitian=True)
        B = XtX_inv @ Xl.T @ Yo
        E = Yo - Xl @ B
        sig2 = (E**2).sum(0) / dof
        se2_red = np.outer(np.diag(XtX_inv)[:p_used], sig2)
        beta_full = np.full((p, N), np.nan)
        se2_full  = np.full((p, N), np.nan)
        beta_full[col_ok] = B[:p_used]
        se2_full[col_ok]  = se2_red
        info_ok = se2_full <= SE2_MAX
        beta_full[~info_ok] = np.nan
        se2_full[~info_ok]  = np.nan
        betas.append(beta_full.flatten())
        se2s.append(se2_full.flatten())
        resids.append(E)
    n_used = len(betas)
    if n_used < 2:
        return None
    Bmat, S2mat = np.array(betas), np.array(se2s)
    n_per_entry = (~np.isnan(Bmat)).sum(axis=0)
    entry_ok = n_per_entry >= MIN_UNITS_ENTRY
    if not entry_ok.any():
        return None
    var_obs  = np.nanvar(Bmat[:, entry_ok], axis=0, ddof=1)
    mean_se2 = np.nanmean(S2mat[:, entry_ok], axis=0)
    tau2 = np.maximum(0.0, var_obs - mean_se2)
    tot_nc, cnt = 0.0, 0
    lens = {len(r) for r in resids}
    if len(lens) == 1 and lens.pop() >= 3:
        T_eq = len(resids[0])
        e0 = math.sqrt(2.0 / (math.pi * (T_eq - 2)))
        iu = np.triu_indices(n_used, k=1)
        for n in range(N):
            M = np.stack([r[:, n] for r in resids])
            ok = M.std(axis=1) > 1e-10
            Rm = np.corrcoef(M)
            pair_ok = ok[iu[0]] & ok[iu[1]]
            r_abs = np.abs(Rm[iu])[pair_ok]
            tot_nc += float(np.maximum(0.0, r_abs - e0).sum())
            cnt    += int(pair_ok.sum())
    return dict(
        bv_raw=float(np.sqrt(var_obs).mean()),
        bv_nc=float(np.sqrt(tau2).mean()),
        noise_floor=float(np.sqrt(mean_se2).mean()),
        crc_nc=tot_nc / cnt if cnt else 0.0)

# ── Cell 4: Regenerate, verify, join ──────────────────────
s7 = pd.read_csv(S7_PATH).drop_duplicates(
    subset=['T_c', 'pert_scale', 'rep'], keep='last')
print(f"Frozen Study 7 rows: {len(s7)}")
t0 = time.time()
rows, mismatches = [], []
for _, r in s7.iterrows():
    T_c, pert, rep = int(r['T_c']), float(r['pert_scale']), int(r['rep'])
    cond_seed = SEED + T_c * 10000 + int(pert * 1000) * 100 + rep
    rng = np.random.default_rng(cond_seed)
    Y, spec = generate_dgp_B_param(
        N_vars, C, T_c, K, SIGMA2, SPARSITY, pert, rng)
    bv_check = beta_variance_index_ridge(Y, spec, K)
    if not np.isclose(bv_check, r['bv'], rtol=0, atol=1e-4):
        mismatches.append((T_c, pert, rep, bv_check, r['bv']))
        continue
    d = canonical_diagnostics(Y, spec)
    d.update(T_c=T_c, pert_scale=pert, rep=rep)
    rows.append(d)
if mismatches:
    print("REGENERATION MISMATCHES — aborting. First 5:")
    for m in mismatches[:5]: print("  ", m)
    raise SystemExit(1)
print(f"[verify] all {len(rows)} panels regenerated draw-exact "
      f"({time.time()-t0:.0f}s)")
diag = pd.DataFrame(rows)
m = s7.merge(diag, on=['T_c', 'pert_scale', 'rep'], validate='1:1')
m['win']       = (m['mse_hn_vs_poolB'] < 0).astype(int)
m['tc_x_bvnc'] = m['T_c'] * m['bv_nc']
m['snr']       = m['bv_nc'] / m['noise_floor']
m['tc_x_snr']  = m['T_c'] * m['snr']

print("\n== Per-condition summary (mean over reps) ==")
summ = m.groupby(['T_c', 'pert_scale']).agg(
    bv_nc=('bv_nc', 'mean'), floor=('noise_floor', 'mean'),
    snr=('snr', 'mean'), gap=('mse_hn_vs_poolB', 'mean'),
    win=('win', 'mean')).round(4)
print(summ.to_string())

# ── Cell 5: Calibrate the sufficiency statistic ───────────
def best_threshold(x, y):
    """Max balanced accuracy over all cut points; returns (thr, bacc)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    cuts = np.unique((xs[1:] + xs[:-1]) / 2)
    best = (np.nan, 0.0)
    P, Nn = ys.sum(), (1 - ys).sum()
    for c in cuts:
        pred = (xs > c).astype(int)
        tpr = (pred & ys).sum() / P if P else 0
        tnr = ((1 - pred) & (1 - ys)).sum() / Nn if Nn else 0
        b = (tpr + tnr) / 2
        if b > best[1]:
            best = (float(c), float(b))
    return best

def auc_mw(x, y):
    """Rank-based AUC (Mann-Whitney)."""
    x1, x0 = x[y == 1], x[y == 0]
    if len(x1) == 0 or len(x0) == 0: return np.nan
    ranks = pd.Series(x).rank().to_numpy()
    r1 = ranks[y == 1].sum()
    return float((r1 - len(x1) * (len(x1) + 1) / 2)
                 / (len(x1) * len(x0)))

y = m['win'].to_numpy()
print("\n== Sufficiency-statistic calibration "
      "(win = H-NAVAR beats Pool B, held-out MSE) ==")
calib = {}
for stat in ['bv_nc', 'tc_x_bvnc', 'snr', 'tc_x_snr']:
    x = m[stat].to_numpy()
    thr, bacc = best_threshold(x, y)
    a = auc_mw(x, y)
    calib[stat] = dict(threshold=thr, balanced_acc=bacc, auc=a)
    print(f"  {stat:10s}  thr={thr:8.4f}  bal_acc={bacc:.3f}  AUC={a:.3f}")
best_stat = max(calib, key=lambda k: calib[k]['balanced_acc'])
print(f"\nSelected statistic: {best_stat}  "
      f"(threshold {calib[best_stat]['threshold']:.4f})")

# ── Cell 6: V-Dem groups — prediction vs empirical outcome ─
OECD_2007 = frozenset({
    "Australia", "Austria", "Belgium", "Canada", "Chile", "Denmark",
    "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Israel", "Italy", "Japan", "South Korea", "Mexico", "Netherlands",
    "New Zealand", "Norway", "Poland", "Portugal", "Spain", "Sweden",
    "Switzerland", "Turkey", "United Kingdom",
    "United States of America"})
SSA = frozenset({
    "Angola", "Benin", "Botswana", "Burundi", "Cameroon",
    "Central African Republic", "Chad", "Ethiopia", "Gabon", "Ghana",
    "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho",
    "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal",
    "Sierra Leone", "South Africa", "Swaziland", "Tanzania",
    "The Gambia", "Togo", "Uganda", "Zambia", "Zimbabwe"})
def assign_group(n):
    return ('OECD' if n in OECD_2007 else
            'SSA' if n in SSA else 'Non-OECD non-SSA')

seed_frames = []
for sd in [42, 123, 7]:
    p = os.path.join(VDEM_DIR, f'seed_{sd}', 'democracy_results.csv')
    assert os.path.exists(p), f"Missing frozen V-Dem results: {p}"
    seed_frames.append(pd.read_csv(p))
vd = pd.concat(seed_frames, ignore_index=True)
vd = vd.dropna(subset=['mse_hnavar', 'mse_pool_B', 'delta_norm', 'T_c'])
vd['group'] = vd['country_name'].map(assign_group)
vd['gap']   = vd['mse_hnavar'] - vd['mse_pool_B']
emp = vd.groupby('group').agg(
    n_obs=('gap', 'size'), gap_pB=('gap', 'mean'),
    hn_wins=('gap', lambda s: (s < 0).mean())).round(4)
print("\n== Empirical V-Dem outcome (recomputed from frozen per-seed "
      "artifacts) ==")
print(emp.to_string())

diag_groups = pd.read_csv(DIAG_PATH)
dg = diag_groups[diag_groups.group != 'Full panel'].copy()
dg['snr']       = dg['bv_nc'] / dg['noise_floor']
dg['tc_x_bvnc'] = dg['mean_Tc'] * dg['bv_nc']
dg['tc_x_snr']  = dg['mean_Tc'] * dg['snr']
thr = calib[best_stat]['threshold']
dg['sufficiency_met'] = dg[best_stat] > thr
print(f"\n== V-Dem groups on the calibrated statistic ({best_stat}, "
      f"thr={thr:.4f}) ==")
print(dg[['group', 'mean_Tc', 'bv_nc', 'noise_floor', 'snr',
          'tc_x_bvnc', 'tc_x_snr', 'sufficiency_met']]
      .round(4).to_string(index=False))

# ── Cell 7: Artifacts ─────────────────────────────────────
m.to_csv(os.path.join(OUT_DIR, 'tcbv_merged.csv'), index=False)
with open(os.path.join(OUT_DIR, 'tcbv_calibration.json'), 'w') as f:
    json.dump({'calibration': calib, 'selected': best_stat,
               'vdem_groups': dg.to_dict(orient='records'),
               'empirical_outcome': emp.reset_index()
                                      .to_dict(orient='records'),
               'geometry_caveat':
                   'Study 7: N=5, C=20, balanced; V-Dem: N=10, '
                   'C=28-69, unbalanced.'}, f, indent=2)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
sc = ax.scatter(m['T_c'] * (1 + 0.03 * np.random.default_rng(0)
                            .standard_normal(len(m))),
                m['bv_nc'], c=m['win'], cmap='RdYlGn', vmin=0, vmax=1,
                s=22, alpha=0.75, edgecolors='none')
for _, r in dg.iterrows():
    ax.scatter(r['mean_Tc'], r['bv_nc'], marker='*', s=260,
               edgecolors='k', linewidths=0.8, zorder=5,
               c='tab:blue')
    ax.annotate(r['group'], (r['mean_Tc'], r['bv_nc']),
                textcoords='offset points', xytext=(6, 6), fontsize=8)
if best_stat == 'tc_x_bvnc':
    ts = np.linspace(m['T_c'].min(), max(m['T_c'].max(),
                                         dg['mean_Tc'].max()), 100)
    ax.plot(ts, thr / ts, 'k--', lw=1,
            label=f'{best_stat} = {thr:.2f}')
    ax.legend(fontsize=8)
ax.set_xlabel('$T_c$'); ax.set_ylabel('BV$_{nc}$')
ax.set_title('H-NAVAR vs Pool B: win (green) / loss (red)',
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_tcbv_plane.png'), dpi=200)
print(f"\n[done] artifacts in {OUT_DIR}")
