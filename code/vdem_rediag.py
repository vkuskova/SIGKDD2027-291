# ============================================================
# V-DEM RE-DIAGNOSTICS — Canonical Noise-Corrected Recipe (KDD)
# ============================================================
# Recomputes the paper's diagnostic values (full panel + OECD /
# Non-OECD non-SSA / SSA) under the canonical recipe frozen in
# recal_thresholds.json, replacing the NeurIPS values (BV=0.56 etc.)
# that mixed raw/normalised data, OLS/ridge, and K=1/K=2.
#
# For each group, reports:
#   bv_raw      raw BV under the canonical recipe (cross-unit std
#               of per-unit OLS slopes, mean over entries)
#   noise_floor mean over entries of sqrt(mean_c se2) — the part of
#               bv_raw attributable to sampling noise alone
#   bv_nc       DL-corrected BV (estimate of TRUE slope dispersion)
#   crc_raw / crc_nc   residual-correlation index, raw / null-debiased
#   predicted cell under the recalibrated thresholds
#
# The bv_raw vs noise_floor vs bv_nc decomposition per region is the
# quantitative answer to "how much of SSA's heterogeneity reading
# was estimation noise from short series".
#
# INPUTS (Drive):
#   /KDD_HNAVAR/data/HDL_merged_notdev_selected.csv
#   /KDD_HNAVAR/recalibration/recal_thresholds.json
# OUTPUTS (Drive):
#   /KDD_HNAVAR/vdem_diagnostics/vdem_diag_groups.csv
#   /KDD_HNAVAR/vdem_diagnostics/vdem_diag_summary.json
#   /KDD_HNAVAR/vdem_diagnostics/vdem_diag_excluded_units.csv
#   /KDD_HNAVAR/vdem_diagnostics/fig_vdem_bv_decomposition.png
#
# PANEL PREPARATION — frozen verbatim from
# democracy_application_final.py: same NAVAR_VARS, dropna,
# MIN_SERIES=15, VAL_YEARS=5 train split. Diagnostics run on the
# TRAIN portion, as in the frozen workflow. The panel is
# z-normalised ONCE on the full stacked train set; group
# diagnostics subset AFTER normalisation so all groups share one
# scale.
#
# Runtime: ~1-2 min CPU (pairwise CRC over 134 units).
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
DATA_PATH = os.path.join(BASE, 'data', 'HDL_merged_notdev_selected.csv')
THR_PATH  = os.path.join(BASE, 'recalibration', 'recal_thresholds.json')
OUT_DIR   = os.path.join(BASE, 'vdem_diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)
for p in (DATA_PATH, THR_PATH):
    assert os.path.exists(p), f"Missing input: {p}"
print(f"[path] OUT_DIR = {OUT_DIR}")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

with open(THR_PATH) as f:
    THR_BLOB = json.load(f)
THR    = THR_BLOB['thresholds']
K_DIAG = THR_BLOB['recipe']['K_DIAG']
DOF_MIN = THR_BLOB['recipe']['DOF_MIN']
print(f"[thresholds] bv_nc={THR['bv_nc']:.4f}  crc_nc={THR['crc_nc']:.4f}"
      f"  (K_DIAG={K_DIAG}, DOF_MIN={DOF_MIN})")

# ── Cell 2: Frozen panel preparation ──────────────────────
# Verbatim constants from democracy_application_final.py.
COUNTRY_COL = 'country_id'
NAME_COL    = 'country_name'
YEAR_COL    = 'year'
MIN_SERIES  = 15
VAL_YEARS   = 5
NAVAR_VARS = [
    "v2x_polyarchy", "v2csprtcpt", "v2clrspct", "v2xps_party",
    "v2csantimv", "v2csanmvch_6", "v2csanmvch_7",
    "ecgrowth", "navco_viol", "hardpres",
]
N_VARS = len(NAVAR_VARS)

df_raw = pd.read_csv(DATA_PATH)
df_panel = (df_raw[[COUNTRY_COL, NAME_COL, YEAR_COL] + NAVAR_VARS]
            .dropna(subset=NAVAR_VARS)
            .sort_values([COUNTRY_COL, YEAR_COL])
            .reset_index(drop=True))
lengths_all  = df_panel.groupby(COUNTRY_COL).size()
eligible_ids = lengths_all[lengths_all >= MIN_SERIES].index
df_use    = df_panel[df_panel[COUNTRY_COL].isin(eligible_ids)]
countries = sorted(df_use[COUNTRY_COL].unique())
C = len(countries)

train_parts, split_spec, unit_names = [], [], []
for cid in countries:
    g = df_use[df_use[COUNTRY_COL] == cid].sort_values(YEAR_COL)
    n = len(g)
    if n >= MIN_SERIES + VAL_YEARS:
        train_parts.append(g.iloc[:-VAL_YEARS])
        split_spec.append(n - VAL_YEARS)
    else:
        train_parts.append(g)
        split_spec.append(n)
    unit_names.append(g[NAME_COL].iloc[0])
train_df = pd.concat(train_parts, ignore_index=True)
Y_train  = train_df[NAVAR_VARS].to_numpy(np.float32)
assert sum(split_spec) == len(Y_train), "split_spec mismatch"
print(f"Panel: {C} countries, train rows {len(Y_train):,}, "
      f"T_c(train) range [{min(split_spec)}, {max(split_spec)}]")

# ── Cell 3: Frozen region partition (vdem_partition.py) ───
OECD_2007 = frozenset({
    "Australia", "Austria", "Belgium", "Canada", "Chile", "Denmark",
    "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Israel", "Italy", "Japan", "South Korea", "Mexico", "Netherlands",
    "New Zealand", "Norway", "Poland", "Portugal", "Spain", "Sweden",
    "Switzerland", "Turkey", "United Kingdom",
    "United States of America",
})
assert len(OECD_2007) == 28
SSA = frozenset({
    "Angola", "Benin", "Botswana", "Burundi", "Cameroon",
    "Central African Republic", "Chad", "Ethiopia", "Gabon", "Ghana",
    "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho",
    "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal",
    "Sierra Leone", "South Africa", "Swaziland", "Tanzania",
    "The Gambia", "Togo", "Uganda", "Zambia", "Zimbabwe",
})
assert len(SSA) == 35
assert OECD_2007.isdisjoint(SSA)

def assign_group(name):
    if name in OECD_2007: return 'OECD'
    if name in SSA:       return 'SSA'
    return 'Non-OECD non-SSA'

unit_groups = [assign_group(n) for n in unit_names]
from collections import Counter
print("Partition:", dict(Counter(unit_groups)))

# ── Cell 4: Canonical diagnostics (identical to script 1) ─
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

SE2_MAX         = 4.0   # entries with se2 above this carry no info
MIN_UNITS_ENTRY = 5     # entries estimable in fewer units are dropped
VAR_TOL         = 1e-8  # within-unit std below this = constant column

def diagnostics_units(unit_series, K=K_DIAG, dof_min=DOF_MIN):
    """unit_series: list of (name, group, Yc) with Yc ALREADY on the
    common normalised scale.

    Real panels contain variables that are CONSTANT within a unit's
    window (binary/event variables), making the full OLS design
    singular. Handling (inactive on synthetic data where all columns
    vary, so the recalibrated thresholds still apply):
      - per unit, drop lag columns with within-unit std < VAR_TOL and
        estimate on the reduced design (dof floor on the reduced p);
      - per-unit entries with se2 > SE2_MAX are masked (no slope
        information; they destabilise the var_obs - mean_se2 moment
        difference);
      - DL correction runs PER ENTRY over units where the entry is
        estimable; entries with < MIN_UNITS_ENTRY units are dropped;
      - BV aggregates over retained entries; coverage is reported.
    Returns group-level stats plus the exclusion list."""
    N = unit_series[0][2].shape[1]
    p = N * K
    betas, se2s, resids, used, excluded = [], [], [], [], []
    for name, grp, Yc in unit_series:
        L = len(Yc)
        col_ok = np.tile(Yc.std(0) >= VAR_TOL, K)      # (p,) lag cols
        p_used = int(col_ok.sum())
        dof = (L - K) - p_used - 1
        if p_used == 0 or dof < dof_min:
            excluded.append((name, grp, L, dof))
            continue
        Xl_full, Yo = unit_design(Yc, K)
        Xl = np.hstack([Xl_full[:, :p][:, col_ok],
                        Xl_full[:, -1:]])              # + intercept
        XtX = Xl.T @ Xl
        # pinv(hermitian) guarantees a PSD inverse (non-negative se2)
        # on near-singular designs; uninformative entries then carry
        # huge se2 and are masked by SE2_MAX below.
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-8, hermitian=True)
        B = XtX_inv @ Xl.T @ Yo                        # (p_used+1, N)
        E = Yo - Xl @ B
        sig2 = (E**2).sum(0) / dof
        se2_red = np.outer(np.diag(XtX_inv)[:p_used], sig2)
        beta_full = np.full((p, N), np.nan)
        se2_full  = np.full((p, N), np.nan)
        beta_full[col_ok] = B[:p_used]
        se2_full[col_ok]  = se2_red
        info_ok = se2_full <= SE2_MAX                  # info floor
        beta_full[~info_ok] = np.nan
        se2_full[~info_ok]  = np.nan
        betas.append(beta_full.flatten())
        se2s.append(se2_full.flatten())
        resids.append(E)
        used.append((name, grp, L))
    n_used = len(betas)
    if n_used < 2:
        return None, excluded, used
    Bmat  = np.array(betas)                            # (n_used, p*N)
    S2mat = np.array(se2s)
    n_per_entry = (~np.isnan(Bmat)).sum(axis=0)
    entry_ok = n_per_entry >= MIN_UNITS_ENTRY
    if not entry_ok.any():
        return None, excluded, used
    var_obs  = np.nanvar(Bmat[:, entry_ok], axis=0, ddof=1)
    mean_se2 = np.nanmean(S2mat[:, entry_ok], axis=0)
    tau2  = np.maximum(0.0, var_obs - mean_se2)
    bv_raw      = float(np.sqrt(var_obs).mean())
    bv_nc       = float(np.sqrt(tau2).mean())
    noise_floor = float(np.sqrt(mean_se2).mean())
    entry_coverage  = float(entry_ok.mean())
    mean_units_per_entry = float(n_per_entry[entry_ok].mean())
    tot_r, tot_nc, cnt = 0.0, 0.0, 0
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
    stats = dict(
        bv_raw=bv_raw, bv_nc=bv_nc, noise_floor=noise_floor,
        entry_coverage=entry_coverage,
        mean_units_per_entry=mean_units_per_entry,
        crc_raw=tot_r/cnt if cnt else 0.0,
        crc_nc=tot_nc/cnt if cnt else 0.0,
        n_units_used=n_used, n_excluded=len(excluded),
        mean_Tc=float(np.mean([L for _, _, L in used])),
        median_Tc=float(np.median([L for _, _, L in used])))
    return stats, excluded, used

def classify(bv, crc):
    hi_bv, hi_crc = bv > THR['bv_nc'], crc > THR['crc_nc']
    return {(False, False): 'A (Pool)', (True, False): 'B (Hierarchy)',
            (False, True): 'C (Demean+Pool)',
            (True, True): 'D (Demean+Hierarchy)'}[(hi_bv, hi_crc)]

# ── Cell 5: Build normalised unit series and run ──────────
Yn = normalise(Y_train)          # ONE global scale for all groups
unit_series_all = []
start = 0
for L, name, grp in zip(split_spec, unit_names, unit_groups):
    unit_series_all.append((name, grp, Yn[start:start+L]))
    start += L

t0 = time.time()
rows, excl_rows = [], []
targets = [('Full panel', unit_series_all)] + [
    (g, [u for u in unit_series_all if u[1] == g])
    for g in ['OECD', 'Non-OECD non-SSA', 'SSA']]
for label, subset in targets:
    stats, excluded, used = diagnostics_units(subset)
    if stats is None:
        print(f"{label}: <2 usable units, skipped"); continue
    stats['group'] = label
    stats['pred_cell'] = classify(stats['bv_nc'], stats['crc_nc'])
    rows.append(stats)
    if label == 'Full panel':
        excl_rows = [dict(country=n, group=g, T_train=L, dof=d)
                     for n, g, L, d in excluded]
res = pd.DataFrame(rows)[
    ['group', 'n_units_used', 'n_excluded', 'mean_Tc', 'median_Tc',
     'entry_coverage', 'bv_raw', 'noise_floor', 'bv_nc',
     'crc_raw', 'crc_nc', 'pred_cell']]
print(f"\n[diagnostics done  {time.time()-t0:.0f}s]\n")
print(res.round(4).to_string(index=False))
print(f"\nExcluded units (dof < {DOF_MIN}):",
      [e['country'] for e in excl_rows] if excl_rows else "none")
print("\nNoise share of observed dispersion (1 - bv_nc/bv_raw):")
for _, r in res.iterrows():
    print(f"  {r['group']:18s}  {1 - r['bv_nc']/r['bv_raw']:.1%}")

# ── Cell 6: Artifacts ─────────────────────────────────────
res.to_csv(os.path.join(OUT_DIR, 'vdem_diag_groups.csv'), index=False)
pd.DataFrame(excl_rows).to_csv(
    os.path.join(OUT_DIR, 'vdem_diag_excluded_units.csv'), index=False)
with open(os.path.join(OUT_DIR, 'vdem_diag_summary.json'), 'w') as f:
    json.dump({'thresholds_used': THR,
               'recipe': THR_BLOB['recipe'],
               'groups': res.to_dict(orient='records')}, f, indent=2)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(res)); w = 0.27
ax.bar(x - w, res.bv_raw,      w, label='raw BV')
ax.bar(x,     res.noise_floor, w, label='noise floor')
ax.bar(x + w, res.bv_nc,       w, label='noise-corrected BV')
ax.axhline(THR['bv_nc'], ls='--', c='k', lw=1,
           label=f"BV$_{{nc}}$ threshold ({THR['bv_nc']:.3f})")
ax.set_xticks(x); ax.set_xticklabels(res.group, fontsize=8)
ax.set_ylabel('BV'); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_vdem_bv_decomposition.png'),
            dpi=200)
print(f"\n[done] artifacts in {OUT_DIR}")
