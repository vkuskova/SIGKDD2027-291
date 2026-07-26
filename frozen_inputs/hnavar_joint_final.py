# ============================================================
# hnavar_joint_final.py — Shared H-NAVAR module
# ============================================================
# Reference implementation of H-NAVAR (Hierarchical Neural Additive
# Vector AutoRegression) used by the synthetic studies and the V-Dem
# application.
#
# Public API
#   AdditiveVAR              Pooled NAVAR estimator (one per dataset).
#   HNAVARModel              Hierarchical NAVAR: shared pool + per-unit
#                            deltas with a learned scalar τ².
#   train_pooled             Train AdditiveVAR. Panel-aware when
#                            `ranges=` is supplied (per-unit held-out
#                            validation on each unit's last val_frac
#                            of windows).
#   train_hnavar_joint       Joint training of pool + deltas with
#                            held-out per-unit validation, weak τ²
#                            prior, and L1 sparsity on deltas.
#   run_cold_start_unit      Per-unit fresh AdditiveVAR (no pool
#                            borrowing).
#   normalise, build_windows,
#   unit_window_ranges,
#   to_device, _split_panel_ranges
#                            Data utilities.
#
# Identification choices
#   - Weak log-normal hyperprior on log τ² (centered log(0.5), σ=2,
#     weight 0.01). Provides scale identification between delta
#     magnitude and τ² without imposing a value.
#   - L1 sparsity on delta contributions (`lambda_l1_delta=0.02`)
#     encourages deltas to be zero where data do not support
#     unit-specific deviation.
#
# Setting `lambda_l1_delta=0` and `log_tau_prior_weight=0` recovers
# pure quadratic shrinkage (no L1, no scale prior).
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared utilities ───────────────────────────────────────

def normalise(Y):
    mu = Y.mean(0)
    sd = np.where(Y.std(0) < 1e-10, 1.0, Y.std(0))
    return (Y - mu) / sd


def build_windows(Y, split_spec, K):
    M = sum(max(0, L - K) for L in split_spec)
    X = np.zeros((M, Y.shape[1], K), dtype=np.float32)
    y = np.zeros((M, Y.shape[1]),    dtype=np.float32)
    w = start = 0
    for L in split_spec:
        for t in range(start + K, start + L):
            X[w] = Y[t - K:t, :].T
            y[w] = Y[t, :]
            w += 1
        start += L
    return X, y


def unit_window_ranges(split_spec, K):
    ranges, w = [], 0
    for L in split_spec:
        n_w = max(0, L - K)
        ranges.append((w, w + n_w))
        w += n_w
    return ranges


def to_device(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _split_panel_ranges(ranges, val_frac):
    """
    Given panel ranges, return (train_ranges, val_ranges) where for each
    unit (ws, we), val is the last val_frac of windows (held-out future)
    and train is the first (1 - val_frac).

    Units with fewer than 3 windows get all-train, empty-val (val_ranges
    entry is (we, we) — empty slice).
    """
    train_ranges, val_ranges = [], []
    for (ws, we) in ranges:
        n_unit = we - ws
        if n_unit < 3:
            train_ranges.append((ws, we))
            val_ranges.append((we, we))
            continue
        n_val_unit = max(1, int(round(n_unit * val_frac)))
        n_val_unit = min(n_val_unit, n_unit - 2)
        val_start  = we - n_val_unit
        train_ranges.append((ws, val_start))
        val_ranges.append((val_start, we))
    return train_ranges, val_ranges


# ── Neural additive VAR (single model) ────────────────────

class AdditiveVAR(nn.Module):
    """
    Neural Additive VAR (single unit or pool).
    Contribution c_{ij}(x) is a learned function of source i at all lags.
    Prediction: y_j = sum_i c_{ij}(x) + bias_j.
    Causal score: S_{ij} = std_t(c_{ij}(x_t)).
    """
    def __init__(self, N, K, H=32):
        super().__init__()
        self.N = N; self.K = K; self.H = H
        self.W1   = nn.Parameter(torch.randn(N, N, H, K) * 0.05)
        self.b1   = nn.Parameter(torch.zeros(N, N, H))
        self.W2   = nn.Parameter(torch.randn(N, N, H) * 0.05)
        self.b2   = nn.Parameter(torch.zeros(N, N))
        self.bias = nn.Parameter(torch.zeros(N))
        self.register_buffer('od', ~torch.eye(N, dtype=torch.bool))

    def _c(self, X):
        """Contribution tensor: (B, N_src, N_tgt)."""
        pre = torch.einsum('bik,ijhk->bijh', X, self.W1) + self.b1
        return (F.softplus(pre) * self.W2.unsqueeze(0)).sum(-1) + self.b2

    def forward(self, X):
        return self._c(X).sum(1) + self.bias

    @torch.no_grad()
    def scores(self, X, chunk=512):
        """Causal scores: S[i,j] = std_t(c_{ij}(x_t))."""
        self.eval()
        parts = [self._c(X[s:s + chunk]).cpu()
                 for s in range(0, len(X), chunk)]
        C = torch.cat(parts).numpy()
        u = C.std(axis=0)
        np.fill_diagonal(u, 0.0)
        return u


def train_pooled(model, Xd, yd, epochs=400, lr=3e-4,
                  lambda_l1=0.10, batch_size=128, val_frac=0.10,
                  ranges=None):
    """
    Train a single AdditiveVAR.

    If `ranges` is None: cold-start / single-unit mode. Validation is
      the LAST val_frac of Xd (held-out future). Training is the
      first (1 - val_frac).

    If `ranges` is provided: panel mode. For each unit (ws, we), the
      LAST val_frac of windows is held out for validation; the first
      portion is used for training. Validation MSE is averaged across
      units for checkpoint selection.

    Best validation state is restored at end.
    """
    device = Xd.device

    if ranges is None:
        # Single-unit / cold-start: last val_frac is val
        n   = len(Xd)
        n_v = max(1, int(n * val_frac))
        n_v = min(n_v, n - 2) if n > 2 else 0
        if n_v > 0:
            Xt, yt = Xd[:-n_v], yd[:-n_v]
            Xv, yv = Xd[-n_v:], yd[-n_v:]
        else:
            # Too short to split — train on everything, no val
            Xt, yt = Xd, yd
            Xv, yv = None, None
        train_idx_for_l1 = None  # not used in single-unit mode
    else:
        # Panel mode: per-unit last-val_frac split
        train_ranges, val_ranges = _split_panel_ranges(ranges, val_frac)
        # Build flat training tensor for batched SGD
        train_idx_list = []
        for (ws_t, we_t) in train_ranges:
            if we_t > ws_t:
                train_idx_list.append(torch.arange(ws_t, we_t, device=device))
        if not train_idx_list:
            # Degenerate: nothing to train on
            return model
        train_idx = torch.cat(train_idx_list)
        Xt = Xd[train_idx]
        yt = yd[train_idx]
        # Validation: gather all val windows; we'll evaluate per-unit MSE
        Xv_list, yv_list = [], []
        val_unit_lens = []
        for (ws_v, we_v) in val_ranges:
            if we_v > ws_v:
                Xv_list.append(Xd[ws_v:we_v])
                yv_list.append(yd[ws_v:we_v])
                val_unit_lens.append(we_v - ws_v)
        if Xv_list:
            Xv_panel = torch.cat(Xv_list, dim=0)
            yv_panel = torch.cat(yv_list, dim=0)
        else:
            Xv_panel = yv_panel = None

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    best_val, best_state = float('inf'), None
    chk = max(1, epochs // 5)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), batch_size):
            xb = Xt[perm[s:s + batch_size]]
            yb = yt[perm[s:s + batch_size]]
            Cb = model._c(xb)
            loss = (F.mse_loss(Cb.sum(1) + model.bias, yb) +
                    lambda_l1 * Cb[:, model.od].abs().mean())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep % chk == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                if ranges is None:
                    if Xv is not None:
                        v = float(F.mse_loss(model(Xv), yv))
                    else:
                        v = float('inf')   # no val possible
                else:
                    if Xv_panel is not None:
                        # Per-unit MSE averaged across units
                        # (compute per-unit then average to weight units equally,
                        #  matching the H-NAVAR validation convention)
                        offset = 0; total_mse = 0.0; n_units_v = 0
                        preds = model(Xv_panel)
                        for L in val_unit_lens:
                            yhat = preds[offset:offset + L]
                            yt_u = yv_panel[offset:offset + L]
                            total_mse += float(F.mse_loss(yhat, yt_u))
                            offset += L
                            n_units_v += 1
                        v = total_mse / max(n_units_v, 1)
                    else:
                        v = float('inf')

            if v < best_val:
                best_val = v
                best_state = {k: p.cpu().clone()
                              for k, p in model.state_dict().items()}

    if best_state:
        model.load_state_dict(
            {k: p.to(device) for k, p in best_state.items()})
    return model


# ── H-NAVAR model ─────────────────────────────────────────

class HNAVARModel(nn.Module):
    """
    Hierarchical NAVAR: pool + C unit-specific delta networks + log_tau.
    Unit prediction: f_c(X) = pool(X) + delta_c(X).
    Hierarchical prior penalises delta magnitude relative to tau2.
    """
    def __init__(self, C, N, K, H=32):
        super().__init__()
        self.C = C; self.N = N; self.K = K
        self.pool   = AdditiveVAR(N, K, H)
        self.deltas = nn.ModuleList([AdditiveVAR(N, K, H) for _ in range(C)])
        self.log_tau = nn.Parameter(
            torch.full((N, N), float(np.log(0.5))))
        g = torch.linspace(-2, 2, 20)
        # .clone() so the buffer is contiguous; .expand() returns a memory-
        # sharing view that breaks load_state_dict (multiple tensor entries
        # point at the same memory).
        self.register_buffer(
            '_grid', g.unsqueeze(1).unsqueeze(1).expand(20, N, K).contiguous().float())

    @property
    def tau2(self):
        return torch.exp(self.log_tau) + 1e-6

    def predict_unit(self, X, c):
        return self.pool(X) + self.deltas[c](X)

    def predict_pool_only(self, X):
        """Pool-only prediction (delta=0). Used for Pool B baseline."""
        return self.pool(X)

    def joint_loss(self, ranges, Xd, yd,
                   lambda_l1=0.10, with_penalty=True,
                   train_idx_for_l1=None,
                   lambda_l1_delta=0.02,
                   log_tau_prior_weight=0.01,
                   log_tau_prior_mean=None,
                   log_tau_prior_sigma=2.0):
        """
        Joint training loss:
          L = recon + with_penalty * (penalty + log_tau_prior)
              + lambda_l1 * L1_pool + lambda_l1_delta * L1_delta

        `ranges` should be the TRAINING ranges (per-unit train portions).
        `train_idx_for_l1` is a flat index tensor for L1 mini-batch sampling;
          if None, samples from the entire concatenated Xd (assumes Xd already
          covers training portion only).

        v3 additions:
        - lambda_l1_delta: L1 penalty on per-unit delta contributions.
          Encourages deltas to be sparse (zero where data don't support a
          unit-specific deviation). Default 0.02 (5x weaker than pool L1,
          since the hierarchical penalty already provides quadratic shrinkage).
        - log_tau_prior_weight: weight on weak Gaussian prior on log_tau.
          Provides scale identification for τ² without shaping its value.
          Default 0.01.
        """
        N, K, C    = self.N, self.K, self.C
        tau2       = self.tau2
        device     = Xd.device

        # Reconstruction over training ranges
        recon = torch.zeros(1, device=device)
        nu = sum(1 for ws, we in ranges if we > ws)
        for c, (ws, we) in enumerate(ranges):
            if we <= ws: continue
            recon = recon + F.mse_loss(
                self.predict_unit(Xd[ws:we], c), yd[ws:we])
        recon = recon / max(nu, 1)

        # L1 on pool contributions — sample from train indices only
        if train_idx_for_l1 is not None and len(train_idx_for_l1) > 0:
            n_sample = min(256, len(train_idx_for_l1))
            perm = torch.randperm(len(train_idx_for_l1), device=device)[:n_sample]
            sample_idx = train_idx_for_l1[perm]
        else:
            sample_idx = torch.randperm(len(Xd), device=device)[:min(256, len(Xd))]
        Cb_p = self.pool._c(Xd[sample_idx])
        l1_pool = Cb_p[:, self.pool.od].abs().mean()

        if not with_penalty:
            return recon + lambda_l1 * l1_pool, recon.item(), 0.0

        # Hierarchical penalty: E[delta_c(x)^2] / tau2 per edge
        H_dim = self.deltas[0].H
        NN    = N * N
        W1 = torch.stack([d.W1 for d in self.deltas]).view(C, NN, H_dim, K)
        b1 = torch.stack([d.b1 for d in self.deltas]).view(C, NN, H_dim)
        W2 = torch.stack([d.W2 for d in self.deltas]).view(C, NN, H_dim)
        b2 = torch.stack([d.b2 for d in self.deltas]).view(C, NN)
        si = torch.arange(NN, device=device) // N
        g  = self._grid[:, si, :]
        pre = (torch.einsum('gnk,cnhk->cgnh', g, W1) + b1.unsqueeze(1))
        h   = F.softplus(pre)
        c_r = (h * W2.unsqueeze(1)).sum(-1) + b2.unsqueeze(1)
        pen = (c_r.pow(2).mean(dim=1) / tau2.view(NN).unsqueeze(0)).mean()

        # ── v3: weak log-normal hyperprior on log_tau ──
        # Centers prior at initialization log(0.5) by default; sigma=2 in
        # log-space is very wide so any plausible tau² is essentially unpenalized.
        # Purpose: break scale non-identifiability between delta magnitude
        # and tau², not to constrain the learned tau² value.
        if log_tau_prior_mean is None:
            log_tau_prior_mean = float(np.log(0.5))
        log_tau_prior = log_tau_prior_weight * (
            (self.log_tau - log_tau_prior_mean).pow(2) /
            (log_tau_prior_sigma ** 2)
        ).mean()

        # ── v3: L1 sparsity on delta contributions ──
        # Sample delta contributions on the same mini-batch used for pool L1.
        # We sum over units — average per unit, then mean across units —
        # keeping the term comparable in scale to pool L1 regardless of C.
        l1_delta_sum = torch.zeros(1, device=device)
        n_units_l1 = 0
        for c, (ws, we) in enumerate(ranges):
            if we <= ws: continue
            # Use unit-specific windows from the ranges for this term
            # (sampling globally would mix unit identities, which doesn't make
            # sense for delta_c, which is unit-specific by construction).
            Cb_d = self.deltas[c]._c(Xd[ws:we])
            l1_delta_sum = l1_delta_sum + Cb_d[:, self.deltas[c].od].abs().mean()
            n_units_l1 += 1
        l1_delta = l1_delta_sum / max(n_units_l1, 1)

        total = (recon
                 + pen
                 + lambda_l1 * l1_pool
                 + lambda_l1_delta * l1_delta
                 + log_tau_prior)
        return total, recon.item(), pen.item()

    @torch.no_grad()
    def unit_scores(self, Xd, c, chunk=512):
        """Causal scores for unit c: pool + delta contributions."""
        self.eval()
        parts = []
        for s in range(0, len(Xd), chunk):
            Cp = self.pool._c(Xd[s:s + chunk])
            Cd = self.deltas[c]._c(Xd[s:s + chunk])
            parts.append((Cp + Cd).cpu())
        C_arr = torch.cat(parts).numpy()
        u = C_arr.std(axis=0)
        np.fill_diagonal(u, 0.0)
        return u

    @torch.no_grad()
    def pool_scores(self, Xd, chunk=512):
        """Causal scores from pool only (Pool B for scoring purposes)."""
        self.eval()
        parts = [self.pool._c(Xd[s:s + chunk]).cpu()
                 for s in range(0, len(Xd), chunk)]
        C_arr = torch.cat(parts).numpy()
        u = C_arr.std(axis=0)
        np.fill_diagonal(u, 0.0)
        return u


# ── Joint training (with held-out per-unit validation) ───

def train_hnavar_joint(model, Xd, yd, ranges,
                        epochs=600,
                        lr=3e-4,
                        lr_tau=3e-3,
                        lambda_l1=0.10,
                        lambda_l1_delta=0.02,
                        log_tau_prior_weight=0.01,
                        warmup_epochs=50,
                        val_frac=0.10,
                        verbose=False):
    """
    Joint training of HNAVARModel with held-out per-unit validation,
    weak τ² prior, and L1 sparsity on deltas.

    For each unit (ws, we): the LAST val_frac of windows is held out
    for validation; training uses only the first (1 - val_frac).

    Optional args:
      lambda_l1_delta: L1 weight on delta contributions (default 0.02).
      log_tau_prior_weight: weight on weak log-normal prior on log_tau
                            (default 0.01).
    """
    device = Xd.device
    train_ranges, val_ranges = _split_panel_ranges(ranges, val_frac)

    # Flat training-only index tensor for L1 sampling
    train_idx_list = []
    for (ws_t, we_t) in train_ranges:
        if we_t > ws_t:
            train_idx_list.append(torch.arange(ws_t, we_t, device=device))
    train_idx = (torch.cat(train_idx_list) if train_idx_list
                 else torch.arange(0, len(Xd), device=device))

    pool_delta_params = (list(model.pool.parameters()) +
                         list(model.deltas.parameters()))
    opt = torch.optim.Adam([
        {'params': pool_delta_params, 'lr': lr,     'weight_decay': 1e-3},
        {'params': [model.log_tau],   'lr': lr_tau, 'weight_decay': 0.0},
    ])

    best_val   = float('inf')
    best_state = None
    chk        = max(1, epochs // 10)

    for ep in range(1, epochs + 1):
        model.train()
        with_penalty = ep > warmup_epochs
        loss, recon, pen = model.joint_loss(
            train_ranges, Xd, yd, lambda_l1, with_penalty,
            train_idx_for_l1=train_idx,
            lambda_l1_delta=lambda_l1_delta,
            log_tau_prior_weight=log_tau_prior_weight)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(pool_delta_params, 1.0)
        nn.utils.clip_grad_norm_([model.log_tau],   5.0)
        opt.step()

        if ep % chk == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                # True held-out validation: per-unit MSE on val_ranges
                val_loss_j, nu_v = 0.0, 0
                for c, (ws_v, we_v) in enumerate(val_ranges):
                    if we_v <= ws_v: continue
                    Xv_c = Xd[ws_v:we_v]
                    yv_c = yd[ws_v:we_v]
                    val_loss_j += float(F.mse_loss(
                        model.predict_unit(Xv_c, c), yv_c))
                    nu_v += 1
                val_loss = val_loss_j / max(nu_v, 1)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: p.cpu().clone()
                              for k, p in model.state_dict().items()}
            if verbose and (ep % (chk * 5) == 0 or ep == epochs):
                tau_mean = float(model.tau2.mean().item())
                print(f"    ep={ep:>4}  val={val_loss:.5f}  "
                      f"recon={recon:.5f}  pen={pen:.5f}  "
                      f"tau2_mean={tau_mean:.4f}  "
                      f"penalty={'ON ' if with_penalty else 'off'}")

    if best_state:
        model.load_state_dict(
            {k: p.to(device) for k, p in best_state.items()})
    return model


def run_cold_start_unit(Xd_unit, yd_unit, N, K, H=32,
                         epochs=400, device=None):
    """
    Train a single AdditiveVAR from random init on one unit's data.
    Uses the single-unit branch of train_pooled (last val_frac is val).
    """
    if device is None:
        device = Xd_unit.device
    if len(Xd_unit) == 0:
        return np.zeros((N, N))
    m = AdditiveVAR(N, K, H).to(device)
    train_pooled(m, Xd_unit, yd_unit, epochs=epochs)  # ranges=None default
    return m.scores(Xd_unit)
