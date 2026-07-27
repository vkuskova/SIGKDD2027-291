# ============================================================
# TASK 2 SELECTION AUDIT — split-half evaluation (canonical)
# ============================================================
# Verifies the split-half numbers reported in Sec 7.2 against the
# canonical tcbv_merged.csv on Drive. Deterministic; expected:
#   select 0-3, eval 4-7: selected=snr  eval AUC=0.933  bal_acc=0.848
#   select 4-7, eval 0-3: selected=snr  eval AUC=0.882  bal_acc=0.846
# INPUT:  /KDD_HNAVAR/tcbv_calibration/tcbv_merged.csv
# OUTPUT: /KDD_HNAVAR/tcbv_calibration/task2_split_audit.json
# ============================================================
import os, sys, json
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/KDD_HNAVAR'
else:
    BASE = os.environ.get('RECAL_BASE', './KDD_HNAVAR_local')
import pandas as pd, numpy as np
m = pd.read_csv(os.path.join(BASE, 'tcbv_calibration', 'tcbv_merged.csv'))
m['win'] = (m['mse_hn_vs_poolB'] < 0).astype(int)
m['tc_x_bvnc'] = m['T_c']*m['bv_nc']
m['snr'] = m['bv_nc']/m['noise_floor']
m['tc_x_snr'] = m['T_c']*m['snr']
STATS = ['bv_nc','tc_x_bvnc','snr','tc_x_snr']
def auc(x, y):
    r = pd.Series(x).rank().to_numpy(); n1 = y.sum(); n0 = len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def best_thr_bacc(x, y):
    xs = np.sort(np.unique(x)); cuts = (xs[1:]+xs[:-1])/2
    best = (float('nan'), 0.0); P, Nn = y.sum(), (1-y).sum()
    for c in cuts:
        p = (x > c).astype(int)
        b = ((p & y).sum()/P + ((1-p) & (1-y)).sum()/Nn)/2
        if b > best[1]: best = (float(c), float(b))
    return best
out = []
for sel, ev in [([0,1,2,3],[4,5,6,7]), ([4,5,6,7],[0,1,2,3])]:
    tr, te = m[m.rep.isin(sel)], m[m.rep.isin(ev)]
    picks = {st: best_thr_bacc(tr[st].to_numpy(), tr.win.to_numpy())
             for st in STATS}
    chosen = max(picks, key=lambda k: picks[k][1])
    thr = picks[chosen][0]
    x, y = te[chosen].to_numpy(), te.win.to_numpy()
    p = (x > thr).astype(int); P, Nn = y.sum(), (1-y).sum()
    bacc = ((p & y).sum()/P + ((1-p) & (1-y)).sum()/Nn)/2
    r = dict(select_reps=sel, eval_reps=ev, selected=chosen,
             threshold=thr, eval_auc=round(auc(x, y), 3),
             eval_bal_acc=round(float(bacc), 3))
    out.append(r); print(r)
with open(os.path.join(BASE, 'tcbv_calibration',
                       'task2_split_audit.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('[done] audit written')
