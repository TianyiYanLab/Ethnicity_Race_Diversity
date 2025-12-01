import os, random,glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients
import torch.optim.lr_scheduler as lr_scheduler
import math


SEED = 1
N_SPLITS = 5
BATCH_SIZE = 32
LR = 1e-3 
WEIGHT_DECAY = 1.5*1e-3
EPOCHS = 130
EARLY_STOP_PATIENCE = 15
DROPOUT = 0.4
HIDDEN = (128, 32)

PRINT_PER_EPOCH = True


# ======= device and seed  =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if device.type == "cuda": torch.cuda.manual_seed_all(SEED)
AMP_ENABLED = (device.type == "cuda")

####################
# ======= utilities =======

def lr_consine(epoch):
    warmup_epochs = 1
    total_epochs = EPOCHS
    min_lr=0.5*1e-4
    lr_base=LR

    if epoch < warmup_epochs:
       return (epoch+1) / warmup_epochs 
    else:
       return min_lr/lr_base+0.5 * (1 + math.cos((epoch+1 - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi))*((lr_base-min_lr)/lr_base)



def vectorize_fc_upper_triangle(fc_mats, include_diag=False):
    """fc_mats: (R,R,N) -> X: (N, E)"""
    R, _, N = fc_mats.shape
    iu = np.triu_indices(R, k=0 if include_diag else 1)
    return fc_mats[iu[0], iu[1], :].T

class Residualizer:
    """
    Residualize features X_tr with respect to covariates C_tr using a multivariate linear model:
    X_resid = X - [1 C] * B
    """
    def fit(self, C, X):
        C1 = np.column_stack([np.ones((C.shape[0], 1)), C])
        self.coef_ = np.linalg.pinv(C1) @ X   # (p+1, E)
        return self
    def transform(self, C, X):
        C1 = np.column_stack([np.ones((C.shape[0], 1)), C])
        return X - C1 @ self.coef_

def sample_balanced_group_indices(y, groups, n_aa_subj=101, n_wa_subj=101, seed=42, allow_replace=False):
    """
    Perform subject-level balanced sampling: first sample n_aa_subj / n_wa_subj subjects from each group,
    then collect all row indices corresponding to these subjects and shuffle them.

    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)

    # 1) Labels for each subject
    uniq_subj = np.unique(groups)
    subj_label = {}
    for s in uniq_subj:
        ys = y[groups == s]
        if not np.all(ys == ys[0]):
            raise ValueError(f"Subject {s} has mixed labels in y.")
        subj_label[s] = int(ys[0])

    # 2) Split the subject list by group
    aa_subj = np.array([s for s, lab in subj_label.items() if lab == 1])
    wa_subj = np.array([s for s, lab in subj_label.items() if lab == 0])

    if (len(aa_subj) < n_aa_subj or len(wa_subj) < n_wa_subj) and not allow_replace:
        raise ValueError(f"Not enough subjects: AA_subj={len(aa_subj)}, WA_subj={len(wa_subj)}, "
                        f"require n_aa_subj={n_aa_subj}, n_wa_subj={n_wa_subj}")

    pick_aa = rng.choice(aa_subj, size=n_aa_subj, replace=allow_replace)
    pick_wa = rng.choice(wa_subj, size=n_wa_subj, replace=allow_replace)
    picked_subj = np.concatenate([pick_aa, pick_wa])

    # 3) Expand to sample-level row indices and shuffle
    # sel_rows = np.flatnonzero(np.isin(groups, picked_subj))
    sel_rows = np.where(np.in1d(groups, picked_subj))[0]
    rng.shuffle(sel_rows)
    return sel_rows

####################
# ======= FNN model =======

class MLP(nn.Module):
    def __init__(self, d_in, hidden=(256, 64), p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden[0])
        #self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn1 = nn.LayerNorm(hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        #self.bn2 = nn.BatchNorm1d(hidden[1])
        self.bn2 = nn.LayerNorm(hidden[1])
        self.out = nn.Linear(hidden[1], 1)
        self.p = p  
        self.ac=nn.PReLU()

               
    def forward(self, x):
        x = self.fc1(x); x = self.bn1(x); x = self.ac(x); x = F.dropout(x, p=self.p, training=self.training)
        x = self.fc2(x); x = self.bn2(x); x = self.ac(x); x = F.dropout(x, p=self.p, training=self.training)
        return self.out(x).squeeze(-1)


def _train_one_epoch(model,loader, opt, scaler):
    model.train()
    losses = []
    all_y_true = [] 
    all_y_prob = []  
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb.float())
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        #ema.update(model)
        losses.append(loss.item())

        probs = torch.sigmoid(logits)
        all_y_prob.append(probs.detach().cpu().numpy()) 
        all_y_true.append(yb.cpu().numpy()) 
    
    y_true_np = np.concatenate(all_y_true) 
    y_prob_np = np.concatenate(all_y_prob).ravel() 
    return float(np.mean(losses)), y_true_np, y_prob_np

####################
# ======= training and validation in each fold =======

def train_eval_fold(X, y, C, tr_idx, va_idx,groups,
                    batch_size=BATCH_SIZE, lr=LR, wd=WEIGHT_DECAY,
                    epochs=EPOCHS, early_stop_pat=EARLY_STOP_PATIENCE,
                    dropout=DROPOUT, hidden=HIDDEN,
                    use_weighted_sampler=True):
    history = [] 
    # ========= 0) Outer-level preprocessing: fit on the outer training fold (tr_idx) =========
    imp_C = SimpleImputer(strategy='median').fit(C[tr_idx])
    C_tr = imp_C.transform(C[tr_idx]); C_va = imp_C.transform(C[va_idx])
    
    sc_C = StandardScaler().fit(C_tr)
    C_tr_s = sc_C.transform(C_tr); C_va_s = sc_C.transform(C_va)

    rez = Residualizer().fit(C_tr_s, X[tr_idx])
    X_tr_rez = rez.transform(C_tr_s, X[tr_idx])
    X_va_rez = rez.transform(C_va_s, X[va_idx])

    sc_X = StandardScaler().fit(X_tr_rez)
    X_tr_fin = sc_X.transform(X_tr_rez).astype(np.float32)
    X_va_fin = sc_X.transform(X_va_rez).astype(np.float32)

    y_tr_full = y[tr_idx]
    y_va_outer = y[va_idx]

    # ========= 1) Inner hold-out split (8:2 split for Early Stopping only) =========

    # y_tr_full: sample-level labels in the outer training fold
    # groups_tr_full: sample-level subject IDs in the outer training fold
    groups_tr_full = groups[tr_idx] 

    sgkf_in = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    # first pair: (tr_in_idx, val_in_idx)
    tr_in_idx, val_in_idx = next(sgkf_in.split(X_tr_fin, y_tr_full, groups_tr_full))
    
    X_tr_in  = X_tr_fin[tr_in_idx];  y_tr_in  = y_tr_full[tr_in_idx]
    X_val_in = X_tr_fin[val_in_idx]; y_val_in = y_tr_full[val_in_idx]

    # ========= 2) DataLoader =========
    tr_in_ds  = TensorDataset(torch.from_numpy(X_tr_in),  torch.from_numpy(y_tr_in).long())
    val_in_ds = TensorDataset(torch.from_numpy(X_val_in), torch.from_numpy(y_val_in).long())
    tr_full_ds= TensorDataset(torch.from_numpy(X_tr_fin), torch.from_numpy(y_tr_full).long())
    va_outer_ds=TensorDataset(torch.from_numpy(X_va_fin), torch.from_numpy(y_va_outer).long())

    if use_weighted_sampler:
        counts = np.bincount(y_tr_in)
        class_weights = 1.0 / np.maximum(counts, 1)
        sample_weights = class_weights[y_tr_in]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(y_tr_in),
            replacement=True
        )
        tr_in_loader = DataLoader(tr_in_ds, batch_size=batch_size, sampler=sampler,
                            pin_memory=True, num_workers=2,drop_last=True)
        
        counts_full = np.bincount(y_tr_full)
        class_weights_full = 1.0 / np.maximum(counts_full, 1)
        sample_weights_full = class_weights_full[y_tr_full]
        sampler_full = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights_full).double(),
            num_samples=len(y_tr_full), 
            replacement=True
        )
        
        tr_full_loader = DataLoader(tr_full_ds, batch_size=batch_size, sampler=sampler_full,
                                pin_memory=True, num_workers=2,drop_last=True)
        
    else:
        tr_in_loader = DataLoader(tr_in_ds, batch_size=batch_size, shuffle=True,
                                    pin_memory=True, num_workers=2,drop_last=True)
        tr_full_loader  = DataLoader(tr_full_ds,  batch_size=batch_size, shuffle=True,
                                    pin_memory=True, num_workers=2,drop_last=True)

    val_in_loader   = DataLoader(val_in_ds,   batch_size=batch_size, shuffle=False,
                                    pin_memory=True, num_workers=2)

    va_outer_loader = DataLoader(va_outer_ds, batch_size=batch_size, shuffle=False,
                                    pin_memory=True, num_workers=2)

    # ========= 3) Early Stopping on the inner validation set =========

    model = MLP(d_in=X_tr_fin.shape[1], hidden=hidden, p=dropout).to(device)
    #model_ema = MLP(d_in=X_tr_fin.shape[1], hidden=hidden, p=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = lr_scheduler.LambdaLR(opt, lr_consine) 
    scaler = GradScaler() 

    best_auc_in, best_state_in, best_epoch, patience = -1.0, None, 0, early_stop_pat
    #ema = EMA(model, decay=0.9)
    for ep in range(epochs):
        train_loss, tr_y_true, tr_y_prob = _train_one_epoch(model,tr_in_loader, opt, scaler)
        
        scheduler.step()
        tr_y_pred = (tr_y_prob > 0.5).astype(int) 
        tr_acc = accuracy_score(tr_y_true, tr_y_pred) 

        # --- Evaluate ACC on the inner validation set ---
        model.eval()
        #ema.load(model_ema)
        y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in val_in_loader:
                xb = xb.to(device, non_blocking=True)
                with autocast():
                    p = torch.sigmoid(model(xb))
                y_prob.append(p.detach().cpu().numpy()); y_true.append(yb.numpy())

        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob).ravel()
        y_pred = (y_prob > 0.5).astype(int)
        acc_in = accuracy_score(y_true, y_pred)
        auc_in = roc_auc_score(y_true, y_prob)

        if PRINT_PER_EPOCH:
            print(f"[Inner] Epoch {ep+1:03d} | train_loss={train_loss:.4f} | train_ACC={tr_acc:.4f} | valACC={acc_in:.4f} | valAUC={auc_in:.4f}")

        history.append({
            "epoch": ep+1,
            "train_ACC": tr_acc,
            "val_ACC": acc_in,
            "val_AUC": auc_in,
        })
        
        # Early Stopping：based on the maximum AUC
        if auc_in > best_auc_in:
            best_auc_in = auc_in
            best_state_in = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            patience = early_stop_pat
        else:
            patience -= 1
            if patience == 0:
                break

    # ========= 4) Load the best model weights =========
    if best_state_in is not None:
        model.load_state_dict(best_state_in)

    # ========= 5) Evaluate on the outer validation fold =========
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in va_outer_loader:
            xb = xb.to(device, non_blocking=True)
            with autocast():
                p = torch.sigmoid(model(xb))
            y_prob.append(p.detach().cpu().numpy()); y_true.append(yb.numpy())
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob).ravel()

    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = dict(
        AUC  = float(auc),
        ACC  = float(accuracy_score(y_true, y_pred)),
        F1   = float(f1_score(y_true, y_pred)),
        BACC = float(balanced_accuracy_score(y_true, y_pred)),
        BEST_EPOCH = int(best_epoch),       
        INNER_AUC  = float(best_auc_in)      
    )
    transforms = dict(imp_C=imp_C, sc_C=sc_C, rez=rez, sc_X=sc_X)
    return model.eval().to(device), metrics, transforms,history



# ======= IG weights =======
class BinaryScorer(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base = base_model
    def forward(self, x):
        logit = self.base(x)                 # (B,)
        return torch.stack([-logit, logit], dim=1)  # (B,2)

def compute_ig_for_mlp(
    model,                   # MLP model (outputting a single logit)
    X_resid_std,             # (N, E) features after imp_C -> sc_C -> residualization -> sc_X
    y_true=None,             # (N,)，rray of 0/1 labels；if None, attribution defaults toward class 1
    baseline="mean",         # "zero" | "mean" | ndarray(E,)
    n_steps=50,
    n_samples=None,
    seed=42,
    batch_size=64,
):
    device = next(model.parameters()).device
    model.eval()

    N, E = X_resid_std.shape
    if (n_samples is None) or (n_samples > N): n_samples = N

    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:n_samples].cpu().numpy()

    X = torch.from_numpy(X_resid_std[idx].astype(np.float32)).to(device)  # (M,E)
    if y_true is None:
        targets = torch.ones(len(idx), dtype=torch.long, device=device)   
    else:
        targets = torch.from_numpy(y_true[idx].astype(np.int64)).to(device)

    # baseline
    if isinstance(baseline, str):
        if baseline == "zero":
            base = torch.zeros_like(X)
        elif baseline == "mean":
            mu = torch.from_numpy(X_resid_std.mean(axis=0, keepdims=True).astype(np.float32)).to(device)
            base = mu.expand_as(X)
        else:
            raise ValueError("baseline must be 'zero' | 'mean' | ndarray")
    elif isinstance(baseline, np.ndarray):
        base_vec = torch.from_numpy(baseline.astype(np.float32)).to(device)
        base = base_vec.unsqueeze(0).expand_as(X)
    elif torch.is_tensor(baseline):
        base = baseline.to(device).unsqueeze(0).expand_as(X)
    else:
        raise ValueError("Unsupported baseline type.")

    scorer = BinaryScorer(model)             # (B,E) -> (B,2)
    ig = IntegratedGradients(scorer)

    attrs = []
    with torch.enable_grad():               
        for s in range(0, len(idx), batch_size):
            e  = min(s + batch_size, len(idx))
            xb = X[s:e].clone().requires_grad_(True)   # (b,E)
            bb = base[s:e]                              # (b,E)
            tb = targets[s:e]                           # (b,)

            attr = ig.attribute(xb, target=tb, baselines=bb, n_steps=n_steps)  # (b,E)
            attrs.append(attr.detach().cpu().numpy())

    features = np.concatenate(attrs, axis=0)  # (M,E)
    return features, idx


def prepare_features_for_ig(X, C, idx, transforms):
    
    """
        Use the transforms returned by train_eval_fold (imp_C, sc_C, rez, sc_X)
        to process the selected samples (idx) in the same way as training,
        and output the residualized and standardized features with shape (N_sel, E).
    """
    expected_keys = {"imp_C", "sc_C", "rez", "sc_X"}
    missing_keys = expected_keys - set(transforms.keys())
    if missing_keys:
        raise KeyError(f"transforms miss keys: {missing_keys}")
    imp_C, sc_C, rez, sc_X = transforms["imp_C"], transforms["sc_C"], transforms["rez"], transforms["sc_X"]
    C_s   = sc_C.transform(imp_C.transform(C[idx]))
    X_rez = rez.transform(C_s, X[idx])
    return sc_X.transform(X_rez).astype(np.float32)



def vec_to_mat_(edge_vec, R, include_diag=False):
    mat = np.zeros((R, R), dtype=np.float32)
    iu = np.triu_indices(R, k=0 if include_diag else 1)
    mat[iu] = edge_vec
    mat[(iu[1], iu[0])] = edge_vec
    if not include_diag:
        np.fill_diagonal(mat, 0.0)
    return mat


def run_ig_for_all_folds(X, y, C, fold_objs,seed,
                            baseline="mean", n_steps=64, n_samples=None,
                            save_dir=None, prefix="ig"):
    """
    For each fold, compute IG on the outer training fold (tr_idx) and outer validation fold (va_idx).
    Optionally save the results to an NPZ file.
    Return ig_results: a list of dicts, one per fold.
    """
    ig_results = []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for i, (model, transforms, tr_idx, va_idx) in enumerate(fold_objs, 1):
        print(f"[IG] Fold {i}")

        # 1) input features
        X_tr_fin = prepare_features_for_ig(X, C, tr_idx, transforms)  # (N_tr,E)
        X_va_fin = prepare_features_for_ig(X, C, va_idx, transforms)  # (N_va,E)

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # 2) compute IG（training folds）
        feats_tr, pick_tr = compute_ig_for_mlp(
            model, X_resid_std=X_tr_fin, y_true=y_tr,
            baseline=baseline, n_steps=n_steps, n_samples=n_samples, seed=42+i
        )
        # 3) compute IG（validation folds/ test folds）
        feats_va, pick_va = compute_ig_for_mlp(
            model, X_resid_std=X_va_fin, y_true=y_va,
            baseline=baseline, n_steps=n_steps, n_samples=n_samples, seed=777+i
        )

        fold_pack = dict(
            fold=i,
            train=dict(features=feats_tr, picked_idx=tr_idx[pick_tr], y=y_tr[pick_tr]),
            valid=dict(features=feats_va, picked_idx=va_idx[pick_va], y=y_va[pick_va]),
            meta=dict(baseline=baseline, n_steps=n_steps, n_samples=n_samples)
        )
        ig_results.append(fold_pack)

        if save_dir:
            np.savez_compressed(
                os.path.join(save_dir, f"{prefix}_{seed}_fold{i}.npz"),
                features_tr=feats_tr, sel_tr=tr_idx[pick_tr], y_tr=y_tr[pick_tr],
                features_va=feats_va, sel_va=va_idx[pick_va], y_va=y_va[pick_va],
                baseline=baseline, n_steps=n_steps, n_samples=(n_samples if n_samples else -1)
            )

    return ig_results


def consensus_from_npz_dir(dir_path, file_glob="*.npz", use="valid", top_p=0.2, agg="mean_abs"):
    """
    
    dir_path: directory where IG results are stored
    use: 'valid' or 'train'
    For each .npz file, load features_va or features_tr and aggregate within each fold → top-p → count.

    When agg == "mean_abs": select the top_p features based on |score| (one-sided).
    When agg == "mean": select the top_p/2 highest and top_p/2 lowest features (two-sided, both directions matter).

    """
    paths = sorted(glob.glob(os.path.join(dir_path, file_glob)))
    if not paths:
        raise FileNotFoundError(f"No files matched in {dir_path}/{file_glob}")

    counts = None
    n_folds = 0
    selected_per_fold = []

    for p in paths:
        data = np.load(p)
        feats_key = "features_va" if use == "valid" else "features_tr"
        if feats_key not in data:
            feats_key = "features_va" if "features_va" in data else "features_tr"

        feats = data[feats_key]     # (M,E)
        # ---- compute the score for each feature ----
        if agg == "mean_abs":
            score = np.mean(np.abs(feats), axis=0)
            # one-sided
            k = int(np.ceil(top_p * len(score)))
            top_idx = np.argsort(-score)[:k]
        elif agg == "mean":
            score = np.mean(feats, axis=0)
            # two-sided, both directions matter
            thr_hi = np.quantile(score, 1 - top_p / 2)
            thr_lo = np.quantile(score, top_p / 2)
            top_idx = np.flatnonzero((score >= thr_hi) | (score <= thr_lo))
        else:
            raise ValueError("agg must be 'mean_abs' or 'mean'")

        # ---- stats ----
        if counts is None:
            counts = np.zeros_like(score, dtype=np.int32)
        counts[top_idx] += 1
        selected_per_fold.append(top_idx)
        n_folds += 1

    consensus = counts.astype(np.float64) / n_folds
    return consensus, counts, selected_per_fold, paths

from scipy.stats import binom

def binomial_threshold_counts(
    counts,
    n_trials,
    p0,
    alpha=0.05,
    bonferroni=True
):
    """

    # counts: (E,) number of times each feature is selected in the top 20%
    # n_trials: total number of trials (e.g., 200 or 500)
    # p0: probability of being selected under the null (e.g., top_p = 0.2)
    # alpha: significance level
    # bonferroni: whether to apply Bonferroni correction

    # Returns:
    #   sig_mask: (E,) bool array indicating significant features (True = passes threshold)
    #   pvals: (E,) right-tailed p-values for each feature
    #   alpha_corr: Bonferroni-corrected significance threshold
    """
    counts = np.asarray(counts).astype(int)
    E = counts.shape[0]

    if bonferroni:
        alpha_corr = alpha / E
    else:
        alpha_corr = alpha

    # one-sided p value：P(X >= counts_j)
    # binom.sf(k-1, n, p) = P(X >= k)
    pvals = binom.sf(counts - 1, n_trials, p0)

    sig_mask = pvals < alpha_corr
    return sig_mask, pvals, alpha_corr



####################
#======= 5fold cross-validation =======

def run_cv(X, y, C,groups, n_splits=N_SPLITS, seed=SEED):
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_histories=[]
    
    # Print sample-level and subject-level counts
    uniq_subj, subj_counts = np.unique(groups, return_counts=True)
    subj_label = {}
    for s in uniq_subj:
        ys = y[groups == s]
        assert np.all(ys == ys[0]), f"Subject {s} has mixed labels."
        subj_label[s] = int(ys[0])
    n_subj_aa = sum(1 for v in subj_label.values() if v == 1)
    n_subj_wa = sum(1 for v in subj_label.values() if v == 0)

    print(f"Total rows: {len(y)} | AA samples: {(y==1).sum()} | WA samples: {(y==0).sum()}")
    print(f"Total subjects: {len(uniq_subj)} | AA subj: {n_subj_aa} | WA subj: {n_subj_wa}")

    all_metrics, fold_objs = [], []
    
    for k, (tr, va) in enumerate(sgkf.split(X, y,groups), 1):
        print(f"\n=== Fold {k}/{n_splits} (group-stratified) ===")
        model, met, trans, hist = train_eval_fold(
            X, y, C, tr, va,groups,
            batch_size=BATCH_SIZE, lr=LR, wd=WEIGHT_DECAY,
            epochs=EPOCHS, early_stop_pat=EARLY_STOP_PATIENCE,
            dropout=DROPOUT, hidden=HIDDEN,
            use_weighted_sampler=True
        )
        all_histories.append(hist)
        print(f"Fold{k} metrics: {met}")
        all_metrics.append(met); fold_objs.append((model, trans, tr, va))

    keys = all_metrics[0].keys()
    mean_ = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std_  = {k: float(np.std ([m[k] for m in all_metrics], ddof=1)) for k in keys}
    print("\n=== CV summary (mean ± std) ===")
    for k in keys: print(f"{k}: {mean_[k]:.4f} ± {std_[k]:.4f}")
    return fold_objs, all_metrics, (mean_, std_),all_histories

def run_cv_on_random_balanced_subset(X, y, C,groups,
                                        n_aa=101, n_wa=101,
                                        n_splits=5, seed=42,ig_prefix='mlp_ig',
                                     **kwargs):
    """
    Each call randomly selects n_aa AA subjects and n_wa WA subjects to form a 202-sample subset,
    then feeds this subset into the existing run_cv pipeline
    (outer 5-fold CV with 1/5 inner hold-out for early stopping).

    """
    subset_idx = sample_balanced_group_indices(y,groups, n_aa_subj=n_aa, n_wa_subj=n_wa, seed=seed)
    X_sub, y_sub, C_sub = X[subset_idx], y[subset_idx], C[subset_idx]
    groups_sub = groups[subset_idx]
    fold_objs, all_metrics, summary,all_histories = run_cv(X_sub, y_sub, C_sub,groups_sub, n_splits=n_splits, seed=seed)
    
    ig_results = run_ig_for_all_folds(
            X_sub, y_sub, C_sub, fold_objs,seed,
            baseline="mean", n_steps=64, n_samples=None, 
            save_dir="./ig_outputs", prefix=ig_prefix
        )
    
    return fold_objs, all_metrics, summary,all_histories