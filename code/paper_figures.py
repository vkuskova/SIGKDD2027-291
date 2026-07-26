# ============================================================
# PAPER FIGURES 3 & 4 — from canonical Drive artifacts (KDD)
# ============================================================
# Figure 3 (fig_tcbv_plane.png):  sweep replications in the
#   (T_c, BV_nc) plane, colored by hierarchy win/loss, V-Dem
#   regions overlaid.
#   Inputs: tcbv_calibration/tcbv_merged.csv
#           vdem_diagnostics/vdem_diag_groups.csv
# Figure 4 (fig_studyM_contrast.png): paired shared-prior
#   contrasts per block.
#   Input:  studyM/studyM_contrasts.csv
#
# Both sized for a KDD column (3.4 in wide, 300 dpi).
# OUTPUT: /KDD_HNAVAR/paper_figures/
# Runtime: seconds, CPU.
# ============================================================

# ── Cell 1: Drive mount and paths ─────────────────────────
import os, sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/KDD_HNAVAR'
else:
    BASE = os.environ.get('RECAL_BASE', './KDD_HNAVAR_local')
MERGED_PATH = os.path.join(BASE, 'tcbv_calibration', 'tcbv_merged.csv')
DIAG_PATH   = os.path.join(BASE, 'vdem_diagnostics',
                           'vdem_diag_groups.csv')
CONTR_PATH  = os.path.join(BASE, 'studyM', 'studyM_contrasts.csv')
OUT_DIR     = os.path.join(BASE, 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)
for p in (MERGED_PATH, DIAG_PATH, CONTR_PATH):
    assert os.path.exists(p), f"Missing input: {p}"
print(f"[path] OUT_DIR = {OUT_DIR}")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Cell 2: Figure 3 — (T_c, BV_nc) plane ─────────────────
m = pd.read_csv(MERGED_PATH)
m['win'] = (m['mse_hn_vs_poolB'] < 0).astype(int)
dg = pd.read_csv(DIAG_PATH)
dg = dg[dg.group != 'Full panel'].copy()

fig, ax = plt.subplots(figsize=(3.4, 2.9))
jit = 1 + 0.03 * np.random.default_rng(0).standard_normal(len(m))
ax.scatter(m['T_c'] * jit, m['bv_nc'], c=m['win'], cmap='RdYlGn',
           vmin=0, vmax=1, s=14, alpha=0.75, edgecolors='none')
short = {'OECD': 'OECD', 'Non-OECD non-SSA': 'non-OECD',
         'SSA': 'SSA'}
for _, r in dg.iterrows():
    ax.scatter(r['mean_Tc'], r['bv_nc'], marker='*', s=180,
               edgecolors='k', linewidths=0.7, c='tab:blue',
               zorder=5)
    ax.annotate(short[r['group']], (r['mean_Tc'], r['bv_nc']),
                textcoords='offset points', xytext=(5, 5),
                fontsize=7)
ax.set_xlabel('$T_c$', fontsize=8)
ax.set_ylabel('BV$_{nc}$', fontsize=8)
ax.tick_params(labelsize=7)
plt.tight_layout()
fig3 = os.path.join(OUT_DIR, 'fig_tcbv_plane.png')
plt.savefig(fig3, dpi=300)
plt.close()
print(f"[fig 3] {fig3}   ({len(m)} replications, "
      f"{len(dg)} regions)")

# ── Cell 3: Figure 4 — shared-prior contrasts ─────────────
cdf = pd.read_csv(CONTR_PATH)
data = [cdf['d25'].dropna().to_numpy(),
        cdf['d40'].dropna().to_numpy(),
        cdf['d80'].dropna().to_numpy()]
print("[check] contrast means:",
      [round(float(d.mean()), 4) for d in data],
      "(canonical run: [-0.0657, +0.0303, +0.0261])")

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.axhline(0, c='k', lw=0.8)
ax.boxplot(data,
           tick_labels=['T=25\n(short)', 'T=40\n(mid)',
                        'T=80\n(long)'],
           widths=0.5)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1)
               + np.random.default_rng(0).uniform(-0.08, 0.08,
                                                  len(d)),
               d, s=14, alpha=0.75, zorder=3)
ax.set_ylabel('gap(mixed) $-$ gap(homog.)', fontsize=8)
ax.tick_params(labelsize=7)
plt.tight_layout()
fig4 = os.path.join(OUT_DIR, 'fig_studyM_contrast.png')
plt.savefig(fig4, dpi=300)
plt.close()
print(f"[fig 4] {fig4}   ({len(cdf)} replications)")
print("\n[done] copy both PNGs next to main.tex before compiling")
