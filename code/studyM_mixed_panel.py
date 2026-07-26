# ============================================================
# STUDY M — Mixed-Length Panels and the Shared-Prior Conflict (KDD)
# ============================================================
# Tests the MECHANISM behind the V-Dem SSA failure. Study 7 fits one
# model per homogeneous-T panel and never reproduces SSA's net
# failure (its most adverse cell still nets a win). The V-Dem
# H-NAVAR, by contrast, fits ONE model with ONE shared tau2 across a
# panel of mixed series lengths. Hypothesis:
#
#   In a mixed-T_c panel with EQUAL true heterogeneity across blocks,
#   the shared tau2 calibrates to what the long-series blocks can
#   support (their units contribute more windows, hence more gradient
#   weight). The short block's deltas are then under-shrunk relative
#   to its own data and overfit: its HN-vs-PoolB gap degrades or
#   reverses RELATIVE TO a homogeneous panel of the same T_c — even
#   though the mixed panel contains MORE total data. (The confound
#   direction is conservative: extra pool data should HELP the short
#   block, so degradation cannot be a data-quantity artifact.)
#
# PAIRED DESIGN (per rep)
#   Draw one A_pool and ONE set of 30 unit matrices A_c at
#   pert_scale = 1.0 (equal true heterogeneity by construction,
#   matching the V-Dem finding that regions have comparable BV_nc).
#   Fit four arms on the SAME units:
#     mixed    : units 0-9 at T=80, 10-19 at T=40, 20-29 at T=25;
#                one joint fit, one shared tau2. Evaluate per block.
#     homog80  : all 30 units at T=80   (control)
#     homog40  : all 30 units at T=40   (control)
#     homog25  : all 30 units at T=25   (control)
#   T=25 as the short block because homogeneous T=25 wins decisively
#   in frozen Study 7 (win rate 1.00, gap -0.085 at pert=1.0), so a
#   mixed-panel reversal is cleanly attributable to mixing.
#
# PRIMARY CONTRAST (per rep, paired)
#   gap_T25(mixed) - gap_T25(homog25)     [prediction: > 0]
#   gap_T80(mixed) - gap_T80(homog80)     [prediction: ~ 0]
#   where gap = mse_hnavar - mse_poolB on the external holdout
#   (negative = H-NAVAR wins), per study7's split_train_val
#   convention (last 10% of each unit's windows, never trained on).
#
# SECONDARY OUTCOMES
#   tau2_learned per arm  [prediction: mixed ~ homog80 > homog25]
#   per-block mean |delta| contribution on train windows
#                          [under-shrinkage made visible]
#   per-block canonical BV_nc / noise_floor / SNR on the mixed panel
#                          [ties to the ordinal sufficiency statistic]
#
# ESTIMATORS: H-NAVAR joint (frozen hnavar_joint_final.py) only; the
# contrast is HN vs its own Pool B. No Pool A, no cold-starts.
#
# INPUTS (Drive):
#   /KDD_HNAVAR/frozen/hnavar_joint_final.py   (from bundle
#                                               code/common/)
# OUTPUTS (Drive):
#   /KDD_HNAVAR/studyM/studyM_results.csv      (per rep x arm x block)
#   /KDD_HNAVAR/studyM/studyM_contrasts.csv    (per rep, paired)
#   /KDD_HNAVAR/studyM/studyM_summary.json
#   /KDD_HNAVAR/studyM/fig_studyM_contrast.png
#
# RESUME: per (arm, rep) caching; safe to interrupt and re-run.
# Runtime: 8 reps x 4 joint fits (C=30, EP=500). Roughly 1.5-3 min
# per fit on a T4 => ~1-2.5 h GPU total. Set STUDYM_SMOKE=1 for a
# mechanics-only check.
# ============================================================

# ── Cell 1: Drive mount, paths, frozen module ─────────────
import os, sys, json, math, time, importlib.util, warnings
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/KDD_HNAVAR'
else:
    BASE = os.environ.get('RECAL_BASE', './KDD_HNAVAR_local')
MODULE_PATH = os.path.join(BASE, 'frozen', 'hnavar_joint_final.py')
OUT_DIR     = os.path.join(BASE, 'studyM')
os.makedirs(OUT_DIR, exist_ok=True)
assert os.path.exists(MODULE_PATH), f"Missing frozen module: {MODULE_PATH}"
print(f"[path] OUT_DIR = {OUT_DIR}")

import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

_spec = importlib.util.spec_from_file_location("hj", MODULE_PATH)
hj = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hj)
normalise          = hj.normalise
build_windows      = hj.build_windows
unit_window_ranges = hj.unit_window_ranges
to_device          = hj.to_device
HNAVARModel        = hj.HNAVARModel
train_hnavar_joint = hj.train_hnavar_joint
print("hnavar_joint_final.py loaded (frozen)")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Cell 2: Configuration ──────────────────────────────────
SEED       = 4242              # distinct stream from Studies 5/7
N_REPS     = 8
N_vars     = 5
K          = 1
SIGMA2     = 0.25
SPARSITY   = 0.40
PERT       = 1.0               # equal true heterogeneity, all blocks
H          = 32
C_BLOCK    = 10
T_BLOCKS   = [80, 40, 25]      # long / mid / short
C_TOTAL    = C_BLOCK * len(T_BLOCKS)
ARMS       = ['mixed', 'homog80', 'homog40', 'homog25']

EP_JOINT, EP_WARMUP = 500, 50
LR, LR_TAU          = 3e-4, 3e-3
VAL_FRAC            = 0.10

if os.environ.get('STUDYM_SMOKE') == '1':
    N_REPS, EP_JOINT, EP_WARMUP, C_BLOCK = 2, 40, 5, 4
    C_TOTAL = C_BLOCK * len(T_BLOCKS)
    print("[SMOKE MODE] Mechanics check only; results not "
          "scientifically meaningful.")

CSV_PATH = os.path.join(OUT_DIR, 'studyM_results.csv')

# ── Cell 3: DGP — shared units, per-arm series length ─────
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

def draw_units(rng):
    """One A_pool + C_TOTAL stabilised unit matrices (frozen Study 7
    perturbation and shrinking logic, pert_scale = PERT)."""
    A_pool = sample_pool_graph(N_vars, K, SPARSITY, rng)
    mask = (A_pool != 0).astype(float)
    A_units = []
    for c in range(C_TOTAL):
        A_c = A_pool + rng.normal(0, PERT * 0.25, A_pool.shape) * mask
        if np.max(np.abs(np.linalg.eigvals(A_c[:, :, 0]))) > 0.95:
            for shrink in [0.5, 0.25, 0.1, 0.0]:
                A_c_try = A_pool + shrink * (A_c - A_pool)
                if np.max(np.abs(np.linalg.eigvals(A_c_try[:, :, 0]))) <= 0.90:
                    A_c = A_c_try; break
        A_units.append(A_c)
    return A_pool, A_units

def simulate_unit(A_c, T_c, rng):
    buf = np.zeros((T_c + K, N_vars))
    buf[:K] = rng.normal(0, 0.1, (K, N_vars))
    eps = rng.normal(0, np.sqrt(SIGMA2), (T_c, N_vars))
    for t in range(K, T_c + K):
        raw = A_c[:, :, 0].T @ buf[t-1, :]
        buf[t] = np.clip(raw + 0.15 * np.tanh(raw) + eps[t - K], -8, 8)
    return buf[K:]

def build_panel(A_units, T_per_unit, rng):
    parts = [simulate_unit(A_c, T, rng)
             for A_c, T in zip(A_units, T_per_unit)]
    return (np.vstack(parts).astype(np.float32), list(T_per_unit))

# ── Cell 4: Canonical diagnostics (identical to scripts 2/3) ─
K_DIAG, DOF_MIN = 1, 5
SE2_MAX, MIN_UNITS_ENTRY, VAR_TOL = 4.0, 5, 1e-8

def unit_design(Yc, Kd):
    L = len(Yc)
    Xl = np.hstack([Yc[Kd-1-k:L-1-k] for k in range(Kd)])
    Yo = Yc[Kd:]
    return np.hstack([Xl, np.ones((len(Xl), 1))]), Yo

def canonical_bv(Y, split_spec, Kd=K_DIAG, dof_min=DOF_MIN):
    """Returns (bv_nc, noise_floor) for a set of units on the common
    normalised scale. Subset-safe: pass pre-normalised data."""
    N = Y[0].shape[1] if isinstance(Y, list) else Y.shape[1]
    if not isinstance(Y, list):
        parts, start = [], 0
        for L in split_spec:
            parts.append(Y[start:start+L]); start += L
        Y = parts
    p = N * Kd
    betas, se2s = [], []
    for Yc in Y:
        L = len(Yc)
        col_ok = np.tile(Yc.std(0) >= VAR_TOL, Kd)
        p_used = int(col_ok.sum())
        dof = (L - Kd) - p_used - 1
        if p_used == 0 or dof < dof_min: continue
        Xl_full, Yo = unit_design(Yc, Kd)
        Xl = np.hstack([Xl_full[:, :p][:, col_ok], Xl_full[:, -1:]])
        XtX_inv = np.linalg.pinv(Xl.T @ Xl, rcond=1e-8, hermitian=True)
        B = XtX_inv @ Xl.T @ Yo
        E = Yo - Xl @ B
        sig2 = (E**2).sum(0) / dof
        se2_red = np.outer(np.diag(XtX_inv)[:p_used], sig2)
        bf = np.full((p, N), np.nan); sf = np.full((p, N), np.nan)
        bf[col_ok] = B[:p_used]; sf[col_ok] = se2_red
        bad = ~(sf <= SE2_MAX)
        bf[bad] = np.nan; sf[bad] = np.nan
        betas.append(bf.flatten()); se2s.append(sf.flatten())
    if len(betas) < 2: return np.nan, np.nan
    Bm, Sm = np.array(betas), np.array(se2s)
    ok = (~np.isnan(Bm)).sum(0) >= MIN_UNITS_ENTRY
    if not ok.any(): return np.nan, np.nan
    var_obs  = np.nanvar(Bm[:, ok], axis=0, ddof=1)
    mean_se2 = np.nanmean(Sm[:, ok], axis=0)
    tau2 = np.maximum(0.0, var_obs - mean_se2)
    return (float(np.sqrt(tau2).mean()),
            float(np.sqrt(mean_se2).mean()))

# ── Cell 5: Fit-and-score one arm ─────────────────────────
def split_train_val(ranges, val_frac=0.10):
    train_r, val_r = [], []
    for (ws, we) in ranges:
        n = we - ws
        n_val = max(1, int(round(n * val_frac))) if n > 0 else 0
        train_r.append((ws, we - n_val))
        val_r.append((we - n_val, we))
    return train_r, val_r

def fit_arm(Y, split_spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Yn = normalise(Y)
    Xw, yw = build_windows(Yn, split_spec, K)
    Xd, yd = to_device(Xw, DEVICE), to_device(yw, DEVICE)
    rngs = unit_window_ranges(split_spec, K)
    train_r, val_r = split_train_val(rngs, VAL_FRAC)
    m = HNAVARModel(len(split_spec), N_vars, K, H).to(DEVICE)
    train_hnavar_joint(m, Xd, yd, train_r,
                       epochs=EP_JOINT, lr=LR, lr_tau=LR_TAU,
                       warmup_epochs=EP_WARMUP, verbose=False)
    m.eval()
    per_unit = []
    with torch.no_grad():
        for c, ((tws, twe), (vws, vwe)) in enumerate(zip(train_r, val_r)):
            if vwe <= vws or twe <= tws:
                per_unit.append((np.nan, np.nan, np.nan)); continue
            yh_hn = m.predict_unit(Xd[vws:vwe], c)
            yh_pb = m.predict_pool_only(Xd[vws:vwe])
            mse_hn = float(((yh_hn - yd[vws:vwe])**2).mean().item())
            mse_pb = float(((yh_pb - yd[vws:vwe])**2).mean().item())
            dmag = float((m.predict_unit(Xd[tws:twe], c)
                          - m.predict_pool_only(Xd[tws:twe]))
                         .abs().mean().item())
            per_unit.append((mse_hn, mse_pb, dmag))
    tau2_lrn = float(m.tau2.mean().item())
    del m
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return per_unit, tau2_lrn, Yn

# ── Cell 6: Resume cache ──────────────────────────────────
done_set = set()
if os.path.exists(CSV_PATH):
    df_done = pd.read_csv(CSV_PATH).drop_duplicates(
        subset=['arm', 'rep', 'block_T'], keep='last')
    df_done.to_csv(CSV_PATH, index=False)
    for _, r in df_done.iterrows():
        done_set.add((str(r['arm']), int(r['rep'])))
    print(f"Resuming: {len(done_set)} (arm, rep) done")

def append_rows(rows):
    pd.DataFrame(rows).to_csv(
        CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH),
        index=False)

# ── Cell 7: Main loop ─────────────────────────────────────
ARM_OFFSET = {'mixed': 1, 'homog80': 2, 'homog40': 3, 'homog25': 4}

block_of_unit = sum(([T] * C_BLOCK for T in T_BLOCKS), [])
t0 = time.time()
for rep in range(N_REPS):
    unit_seed = SEED + 1000 * rep
    rng_units = np.random.default_rng(unit_seed)
    A_pool, A_units = draw_units(rng_units)
    for arm in ARMS:
        if (arm, rep) in done_set: continue
        T_per_unit = (block_of_unit if arm == 'mixed'
                      else [int(arm[5:])] * C_TOTAL)
        rng_sim = np.random.default_rng(unit_seed + ARM_OFFSET[arm])
        Y, spec = build_panel(A_units, T_per_unit, rng_sim)
        per_unit, tau2_lrn, Yn = fit_arm(Y, spec, seed=unit_seed + 7)
        # per-block canonical diagnostics on the fitted panel's scale
        rows = []
        for T in sorted(set(T_per_unit)):
            idx = [i for i, t in enumerate(T_per_unit) if t == T]
            # block units on the ARM-level normalised scale
            parts, start, unit_parts = [], 0, []
            for i, L in enumerate(spec):
                if i in idx: unit_parts.append(Yn[start:start+L])
                start += L
            bv_nc, floor = canonical_bv(unit_parts, None)
            mh = [per_unit[i][0] for i in idx]
            mp = [per_unit[i][1] for i in idx]
            dm = [per_unit[i][2] for i in idx]
            gaps = [a - b for a, b in zip(mh, mp)
                    if not (np.isnan(a) or np.isnan(b))]
            rows.append(dict(
                arm=arm, rep=rep, block_T=T, n_units=len(idx),
                mse_hn=float(np.nanmean(mh)),
                mse_poolB=float(np.nanmean(mp)),
                gap=float(np.mean(gaps)),
                win=float(np.mean([g < 0 for g in gaps])),
                delta_mag=float(np.nanmean(dm)),
                tau2_learned=tau2_lrn,
                bv_nc=bv_nc, noise_floor=floor,
                snr=bv_nc / floor if floor and not np.isnan(floor)
                    else np.nan))
        append_rows(rows)
        done_set.add((arm, rep))
        print(f"[rep {rep} | {arm:8s}] tau2={tau2_lrn:.3f}  "
              f"gaps: " + "  ".join(
                  f"T{r['block_T']}={r['gap']:+.4f}" for r in rows)
              + f"   ({time.time()-t0:.0f}s)", flush=True)

# ── Cell 8: Contrasts and reports ─────────────────────────
df = pd.read_csv(CSV_PATH).drop_duplicates(
    subset=['arm', 'rep', 'block_T'], keep='last')

print("\n== Per-arm x block summary (mean over reps) ==")
summ = df.groupby(['arm', 'block_T']).agg(
    gap=('gap', 'mean'), gap_sd=('gap', 'std'), win=('win', 'mean'),
    delta_mag=('delta_mag', 'mean'), tau2=('tau2_learned', 'mean'),
    snr=('snr', 'mean')).round(4)
print(summ.to_string())

# Paired contrasts per rep
cons = []
for rep, g in df.groupby('rep'):
    def gp(arm, T):
        s = g[(g.arm == arm) & (g.block_T == T)]['gap']
        return float(s.iloc[0]) if len(s) else np.nan
    cons.append(dict(
        rep=rep,
        d25=gp('mixed', 25) - gp('homog25', 25),
        d40=gp('mixed', 40) - gp('homog40', 40),
        d80=gp('mixed', 80) - gp('homog80', 80)))
cdf = pd.DataFrame(cons)
cdf.to_csv(os.path.join(OUT_DIR, 'studyM_contrasts.csv'), index=False)

print("\n== PRIMARY CONTRAST: gap(mixed) - gap(homog), paired ==")
print("   positive = block does WORSE inside the mixed panel")
for col, lab in [('d25', 'T=25 (short)'), ('d40', 'T=40 (mid)'),
                 ('d80', 'T=80 (long)')]:
    v = cdf[col].dropna()
    if len(v) == 0: continue
    m_, s_ = v.mean(), v.std(ddof=1)
    t = m_ / (s_ / math.sqrt(len(v))) if s_ > 0 else np.nan
    pos = (v > 0).mean()
    print(f"  {lab:14s} mean={m_:+.4f}  sd={s_:.4f}  t={t:+.2f}  "
          f"frac>0={pos:.2f}  (n={len(v)})")

print("\n== tau2_learned by arm (mean over reps) ==")
print(df.groupby('arm')['tau2_learned'].mean().round(4).to_string())

with open(os.path.join(OUT_DIR, 'studyM_summary.json'), 'w') as f:
    json.dump({'config': dict(SEED=SEED, N_REPS=N_REPS, C_BLOCK=C_BLOCK,
                              T_BLOCKS=T_BLOCKS, PERT=PERT,
                              EP_JOINT=EP_JOINT),
               'per_arm_block': summ.reset_index()
                                    .to_dict(orient='records'),
               'contrasts': cdf.to_dict(orient='records')},
              f, indent=2, default=float)

fig, ax = plt.subplots(figsize=(6.0, 3.6))
labels = ['T=25\n(short)', 'T=40\n(mid)', 'T=80\n(long)']
data = [cdf['d25'].dropna(), cdf['d40'].dropna(), cdf['d80'].dropna()]
ax.axhline(0, c='k', lw=0.8)
bp = ax.boxplot(data, labels=labels, widths=0.5)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1)
               + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)),
               d, s=18, alpha=0.7, zorder=3)
ax.set_ylabel('gap(mixed) − gap(homogeneous)')
ax.set_title('Shared-prior penalty by block '
             '(positive = worse inside mixed panel)', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_studyM_contrast.png'), dpi=200)
print(f"\n[done] artifacts in {OUT_DIR}   total {time.time()-t0:.0f}s")
