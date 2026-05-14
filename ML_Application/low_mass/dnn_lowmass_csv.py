#!/usr/bin/env python3
"""
dnn_lowmass_csv.py  —  DNN classifier for YH→bbγγ low-mass resonance search
===========================================================================

CSV-compatible version of dnn_lowmass.py.  Trains a PyTorch DNN to separate
NMSSM  X → YH → bbγγ  signal from non-resonant backgrounds (γγ+jets, γ+jets).

Data sources: CSV files instead of parquet.

How to run (on lxplus, with LCG_106):
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
    python3 dnn_lowmass_csv.py

Key ML ideas demonstrated
--------------------------
 1. Reading physics data from CSV files
 2. Event weights  (weight_central encodes  σ × L / N_gen × SFs)
 3. Class-imbalance handling via weight normalisation
 4. Data-leakage-free feature scaling  (fit scaler on train only)
 5. Early stopping to prevent over-training
 6. Evaluation: ROC curve, feature importance, DNN-score distributions
"""

# ============================================================================
#  Imports
# ============================================================================
import os
import json
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no X11 on lxplus)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
#  CONFIGURATION  —  edit this section to change signal / backgrounds / features
# ============================================================================

# --- Signal mass points -------------------------------------------------------

SIGNAL_POINTS_BY_X = {
    240: [50, 60, 70, 80, 90, 95, 100],
    280: [50, 60, 70, 80, 90, 95, 100],
    300: [50, 60, 70, 80, 90, 95, 100],
    320: [50, 60, 70, 80, 90, 95, 100],
    350: [50, 60, 70, 80, 90, 95, 100],
    400: [50, 60, 70, 80, 90, 95, 100],
    450: [50, 60, 70, 80, 90, 95, 100],
    500: [50, 60, 70, 80, 90, 95, 100],
    550: [50, 60, 70, 80, 90, 95, 100],
    600: [50, 60, 70, 80, 90, 95, 100],
    650: [50, 60, 70, 80, 90, 95, 100],
    700: [50, 60, 70, 80, 90, 95, 100],
    750: [50, 60, 70, 80, 90, 95, 100],
    800: [50, 60, 70, 80, 90, 95, 100],
    850: [50, 60, 70, 80, 90, 95, 100],
    900: [50, 60, 70, 80, 90, 95, 100],
    950: [50, 60, 70, 80, 90, 95, 100],
    1000: [50, 60, 70, 80, 90, 95, 100],
}

LOW_X_MASSES = [240, 280, 300, 320, 350]
MID_X_MASSES = [400, 450, 500, 550, 600]
HIGH_X_MASSES = [650, 700, 750, 800, 850, 900, 950, 1000]

MASS_CATEGORIES = {
    "low": LOW_X_MASSES,
    "mid": MID_X_MASSES,
    "high": HIGH_X_MASSES,
    "all": sorted(SIGNAL_POINTS_BY_X.keys()),
}

# Choose signal mass category: low / mid / high / all
MASS_CATEGORY = "low"

# Choose which eras to load.
ACTIVE_YEARS = ["2022preEE", "2022postEE"]

# Per-era paths and process naming (adjusted for CSV files).
YEAR_DATASETS = {
    "2022preEE": {
        "signal_dir": "/eos/user/b/bsinghal/analysis/output/2022preEE/csv",
        "signal_name_templates": [
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_csv/v3/Run3_2022/sim/preEE/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
    "2022postEE": {
        "signal_dir": "/eos/user/b/bartek/hhbbgg/systematics_v3/2022_postEE/csv",
        "signal_name_templates": [
            "NMSSM_X{mx}_Y{my}",
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_csv/v3/Run3_2022/sim/postEE/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
    "2023": {
        "signal_dir": "/eos/user/b/bsinghal/analysis/output/2023/csv",
        "signal_name_templates": [
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
            "NMSSM_X{mx}_Y{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_csv/v3/Run3_2023/sim/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
}

# --- Cross-sections (pb) and luminosity ----------------------------------------
# Luminosities copied from normalisation.py (do not import at runtime).
YEAR_LUMINOSITY_FB = {
    "2022preEE":  5.0104 + 2.9700,               # C + D
    "2022postEE": 5.8070 + 17.7819 + 3.0828,    # E + F + G
    "2023":       17.794 + 9.451,               # preBPix + postBPix
}


def get_year_luminosity_fb(year):
    if year not in YEAR_LUMINOSITY_FB:
        raise KeyError(
            f"Missing luminosity for year='{year}'. "
            f"Add it to YEAR_LUMINOSITY_FB."
        )
    return YEAR_LUMINOSITY_FB[year]

SIGNAL_XSEC_PB = 0.001

XSEC_PB = {
    "GGJets_MGG-80":                            88.75,
    "GJet_PT-20to40_DoubleEMEnriched_MGG-80": 242.5,
    "GJet_PT-40_DoubleEMEnriched_MGG-80":     919.1,
}

# --- Output -------------------------------------------------------------------
def _build_signal_points(points_by_x, category):
    if category not in MASS_CATEGORIES:
        raise ValueError(
            f"Unknown MASS_CATEGORY='{category}'. "
            f"Choose from {sorted(MASS_CATEGORIES)}"
        )
    selected_mx = MASS_CATEGORIES[category]
    return [
        (mx, my)
        for mx in selected_mx
        for my in points_by_x.get(mx, [])
    ]


SIGNAL_POINTS = _build_signal_points(SIGNAL_POINTS_BY_X, MASS_CATEGORY)

_mx_str = "_".join(str(mx) for mx, _ in SIGNAL_POINTS)
_my_set = sorted({my for _, my in SIGNAL_POINTS})
_my_str = "_".join(str(my) for my in _my_set)
_years_str = "_".join(ACTIVE_YEARS)

# --- Feature set selection ----------------------------------------------------
FEATURE_SETS = {
    "discriminating_vars": [
        "Res_CosThetaStar_jj",
        "Res_CosThetaStar_gg",
        "Res_CosThetaStar_CS",
        "Res_DeltaR_jg_min",
        "Res_DeltaR_j1g2",
        "Res_DeltaR_j2g1",
        "Res_DeltaR_j2g2",
        "Res_FirstJet_PtOverM",
        "Res_lead_bjet_btagPNetB",
        "Res_lead_bjet_PNetRegPtRawRes",
        "Res_pholead_PtOverM",
        "Res_phosublead_PtOverM",
        "Res_SecondJet_PtOverM",
        "Res_sublead_bjet_btagPNetB",
        "Res_sublead_bjet_PNetRegPtRawRes",
        "sigma_m_over_m",
        "pt_jj_over_M_jjgg",
        "pt_gg_over_M_jjgg",
    ],
}

# --- Features -----------------------------------------------------------------
# FEATURE_SET_NAME, DIRECT_FEATURES, ALL_FEATURES, SIG_LABEL, OUTPUT_DIR are
# set at the start of each iteration in the main() feature-set loop.

ENGINEERING_COLS = [
    "Res_dijet_pt",
    "pt",
    "Res_HHbbggCandidate_mass",
    "mass",
]

WEIGHT_COL = "weight"
FILTER_COL = "is_Res"

# --- DNN hyper-parameters -----------------------------------------------------
DNN_PARAMS = dict(
    hidden_layers       = [256, 128, 64, 32],  # one extra hidden layer
    dropout             = 0.4,             # dropout rate between layers
    batch_norm          = True,            # use batch normalisation
    activation          = "ReLU",          # activation function
)

TRAINING_PARAMS = dict(
    batch_size          = 1024,
    learning_rate       = 1e-3,
    weight_decay        = 1e-4,            # L2 regularisation
    max_epochs          = 200,
    early_stopping      = 20,              # patience (epochs)
    lr_scheduler_factor = 0.5,             # reduce LR by this factor
    lr_scheduler_patience = 10,            # after this many epochs w/o improvement
)

TEST_FRACTION         = 0.3
WEIGHT_CLIP_MAX       = 10.0


# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

ENGINEERED_FEATURE_NAMES = [
    "pt_jj_over_M_jjgg",
    "pt_gg_over_M_jjgg",
    "reduced_invariant_mX",
]

ENGINEERED_FEATURE_DEPENDENCIES = {
    "pt_jj_over_M_jjgg": ["Res_dijet_pt", "Res_HHbbggCandidate_mass"],
    "pt_gg_over_M_jjgg": ["pt", "Res_HHbbggCandidate_mass"],
    "reduced_invariant_mX": ["Res_HHbbggCandidate_mass", "mass"],
}

# Values at or below this threshold are treated as non-physical sentinels.
# In these samples, missing kinematics are often encoded as -999, not NaN.
SENTINEL_THRESHOLD = -900.0

# Mutable globals — overwritten at the start of each feature-set iteration
# in main().  Other functions reference these names so they must exist at
# module level; their actual values are set inside the loop.
FEATURE_SET_NAME = ""
DIRECT_FEATURES  = []
ALL_FEATURES     = []
SIG_LABEL        = ""
OUTPUT_DIR       = ""


def add_engineered_features(df):
    """Build derived features from raw columns."""
    m_jjgg = df["Res_HHbbggCandidate_mass"].replace(0, np.nan)
    diphoton_mass = df["mass"].replace(0, np.nan)
    df["pt_jj_over_M_jjgg"] = df["Res_dijet_pt"] / m_jjgg
    df["pt_gg_over_M_jjgg"] = df["pt"]            / m_jjgg
    df["reduced_invariant_mX"] = m_jjgg - (diphoton_mass - 125.0)
    return df


def clean_sentinel_values(df, features, threshold=SENTINEL_THRESHOLD):
    """
    Replace non-physical sentinel values (e.g. -999) with NaN.

    Why this matters for the DNN:
    - PyTorch computes gradients from every sample in every batch.
    - A -999 sentinel far outside the physical range will dominate the
      gradient update and push weights in the wrong direction.
    - StandardScaler will also absorb the sentinel into the mean/std,
      making all scaled values slightly off for every real event.
    - Converting to NaN and then dropping those rows ensures the network
      is trained and evaluated on physically valid events only.
    """
    total_replaced = 0
    per_feature = []

    for col in features:
        if col not in df.columns:
            continue
        mask = df[col] <= threshold
        n_bad = int(mask.sum())
        if n_bad > 0:
            df.loc[mask, col] = np.nan
            total_replaced += n_bad
            per_feature.append((col, n_bad))

    if per_feature:
        print(f"\nSentinel clean-up (<= {threshold}):")
        for col, n_bad in per_feature:
            print(f"  {col:40s} replaced {n_bad:,} entries")
        print(f"  Total sentinel entries replaced: {total_replaced:,}")
    else:
        print(f"\nSentinel clean-up: no values <= {threshold} found.")

    return df


# ============================================================================
#  DATA  LOADING  (CSV-based)
# ============================================================================

def _columns_to_read():
    """Return deduplicated list of all columns needed from CSV."""
    cols = DIRECT_FEATURES + ENGINEERING_COLS + [WEIGHT_COL]
    if FILTER_COL:
        cols.append(FILTER_COL)
    return list(dict.fromkeys(cols))


def load_csv(filepath, columns, label, filter_col=None):
    """Read a CSV file, keep only *columns*, optionally filter, add a label."""
    try:
        df = pd.read_csv(filepath, usecols=columns)
    except (FileNotFoundError, ValueError) as e:
        raise KeyError(
            f"Error reading {filepath}: {e}\n"
            f"Check that the file exists and contains the required columns."
        )

    tag = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    print(f"  Loaded {len(df):>8,} events  from  {tag}")

    if filter_col and filter_col in df.columns:
        n_before = len(df)
        df = df[df[filter_col] == 1].copy()
        print(f"    → {len(df):>8,} pass  {filter_col}  "
              f"(dropped {n_before - len(df):,})")

    df["label"] = label
    return df


def _find_existing_csv(base_dir, proc_name):
    """Try a few common CSV layouts and return the first existing path."""
    candidates = [
        os.path.join(base_dir, proc_name, "nominal", "merged.csv"),
        os.path.join(base_dir, proc_name, "merged.csv"),
        os.path.join(base_dir, f"{proc_name}.csv"),
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    return None


def load_signal(signal_points):
    """Load multiple signal mass-points across ACTIVE_YEARS and concatenate."""
    cols = _columns_to_read()
    dfs  = []

    print(f"\n{'='*60}")
    print(f"[Signal]  {len(signal_points)} mass points  | years={ACTIVE_YEARS}")
    print(f"{'='*60}")

    for year in ACTIVE_YEARS:
        conf = YEAR_DATASETS.get(year)
        if conf is None:
            print(f"  WARNING: year '{year}' not found in YEAR_DATASETS — skipping")
            continue

        lumi_fb = get_year_luminosity_fb(year)

        signal_dir = conf["signal_dir"]
        proc_templates = conf.get("signal_name_templates", ["NMSSM-XtoYH-MX-{mx}-MY-{my}"])

        for mx, my in signal_points:
            fp = None
            proc_name = None
            for proc_tpl in proc_templates:
                candidate = proc_tpl.format(mx=mx, my=my)
                fp = _find_existing_csv(signal_dir, candidate)
                if fp is not None:
                    proc_name = candidate
                    break

            if fp is None:
                print(f"  WARNING: signal MX={mx}, MY={my} not found in {year} — skipping")
                continue

            df = load_csv(fp, cols, label=1, filter_col=FILTER_COL)
            df[WEIGHT_COL] = df[WEIGHT_COL] * SIGNAL_XSEC_PB * lumi_fb * 1000
            print(f"  Applied  σ = {SIGNAL_XSEC_PB} pb  ×  L = {lumi_fb:.4f} fb⁻¹")

            df["process"] = proc_name
            df["year"] = year
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No signal files were loaded!")

    sig = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total signal:  {len(sig):,} events  "
          f"({len(dfs)} mass points)")
    return sig


def load_backgrounds():
    """Load and concatenate all background processes across ACTIVE_YEARS."""
    cols = _columns_to_read()
    dfs  = []

    print(f"\n{'='*60}")
    print(f"[Background]  years={ACTIVE_YEARS}")
    print(f"{'='*60}")

    for year in ACTIVE_YEARS:
        conf = YEAR_DATASETS.get(year)
        if conf is None:
            print(f"  WARNING: year '{year}' not found in YEAR_DATASETS — skipping")
            continue

        lumi_fb = get_year_luminosity_fb(year)

        for proc in conf["bkg_processes"]:
            fp = _find_existing_csv(conf["bkg_dir"], proc)
            if fp is None:
                print(f"  WARNING: {proc} not found in {year} — skipping")
                continue
            df = load_csv(fp, cols, label=0, filter_col=FILTER_COL)
            xs = XSEC_PB.get(proc, 1.0)
            df[WEIGHT_COL] = df[WEIGHT_COL] * xs * lumi_fb * 1000
            print(f"  Applied  σ = {xs} pb  ×  L = {lumi_fb:.4f} fb⁻¹")

            df["process"] = proc
            df["year"] = year
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No background files were loaded!")

    bkg = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total background:  {len(bkg):,} events")
    return bkg


# ============================================================================
#  DNN  MODEL
# ============================================================================

class DNNClassifier(nn.Module):
    """
    Fully connected feed-forward network for binary classification.

    Architecture per hidden layer:
        Linear → (BatchNorm) → Activation → Dropout
    Output layer:
        Linear(last_hidden, 1) → Sigmoid
    """

    def __init__(self, n_features, hidden_layers, dropout=0.3,
                 batch_norm=True, activation="ReLU"):
        super().__init__()

        act_fn = getattr(nn, activation)

        layers = []
        prev_size = n_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_size = h

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================================
#  TRAINING  LOOP
# ============================================================================

def train_dnn(model, train_loader, val_loader, device, params):
    """
    Train the DNN with weighted binary cross-entropy, early stopping,
    and learning-rate scheduling.

    Returns
    -------
    model      : trained model (best checkpoint)
    history    : dict with 'train_loss' and 'val_loss' per epoch
    """
    criterion = nn.BCELoss(reduction='none')  # per-sample, we apply weights manually
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params["lr_scheduler_factor"],
        patience=params["lr_scheduler_patience"],
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, params["max_epochs"] + 1):
        # ---- Training ----
        model.train()
        running_loss   = 0.0
        running_weight = 0.0

        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss_per_sample = criterion(pred, y_batch)
            loss = (loss_per_sample * w_batch).sum() / w_batch.sum()
            loss.backward()
            optimizer.step()

            running_loss   += (loss_per_sample * w_batch).sum().item()
            running_weight += w_batch.sum().item()

        train_loss = running_loss / running_weight

        # ---- Validation ----
        model.eval()
        val_loss   = 0.0
        val_weight = 0.0

        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)

                pred = model(X_batch)
                loss_per_sample = criterion(pred, y_batch)
                val_loss   += (loss_per_sample * w_batch).sum().item()
                val_weight += w_batch.sum().item()

        val_loss /= val_weight

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1 or patience_ctr == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={lr_now:.2e}  patience={patience_ctr}/{params['early_stopping']}")

        if patience_ctr >= params["early_stopping"]:
            print(f"\n  Early stopping at epoch {epoch}  "
                  f"(best val_loss={best_val_loss:.6f})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return model, history


# ============================================================================
#  PREDICTION  HELPER
# ============================================================================

def predict(model, X, device, batch_size=4096):
    """Run inference in batches, return numpy array of scores."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            preds.append(model(batch).cpu().numpy())

    return np.concatenate(preds)


# ============================================================================
#  PLOTTING  HELPERS
# ============================================================================

def plot_loss_history(history, outdir):
    """Training and validation loss vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-",  lw=2, label="Train loss")
    ax.plot(epochs, history["val_loss"],   "r--", lw=2, label="Validation loss")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Weighted BCE Loss", fontsize=13)
    ax.set_title(f"Training History  —  {SIG_LABEL}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "loss_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_dnn_scores(y_train, y_test, pred_train, pred_test, outdir):
    """DNN output-score histograms for signal vs background."""
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)

    ax.hist(pred_train[y_train == 1], bins=bins, density=True,
            alpha=0.35, color="blue",  label="Signal (train)")
    ax.hist(pred_train[y_train == 0], bins=bins, density=True,
            alpha=0.35, color="red",   label="Background (train)")

    for lbl, col, name in [(1, "blue", "Signal"), (0, "red", "Background")]:
        mask = y_test == lbl
        counts, edges = np.histogram(pred_test[mask], bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(centres, counts, fmt="o", color=col, ms=4,
                     label=f"{name} (test)")

    ax.set_xlabel("DNN score  (signal probability)", fontsize=13)
    ax.set_ylabel("Normalised events", fontsize=13)
    ax.set_title(f"DNN Output  —  {SIG_LABEL}", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(outdir, "dnn_score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc(y_test, pred_test, w_test, outdir):
    """Weighted and unweighted ROC curves + AUC values."""
    fpr_w, tpr_w, _ = roc_curve(y_test, pred_test, sample_weight=w_test)
    auc_w = auc(fpr_w, tpr_w)

    fpr_u, tpr_u, _ = roc_curve(y_test, pred_test)
    auc_u = auc(fpr_u, tpr_u)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_w, tpr_w, "b-",  lw=2, label=f"Weighted   AUC = {auc_w:.4f}")
    ax.plot(fpr_u, tpr_u, "g--", lw=2, label=f"Unweighted AUC = {auc_u:.4f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate  (bkg efficiency)", fontsize=13)
    ax.set_ylabel("True Positive Rate   (sig efficiency)", fontsize=13)
    ax.set_title(f"ROC Curve  —  {SIG_LABEL}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return auc_w, auc_u


def plot_feature_importance(model, X_test, y_test, w_test, device, outdir):
    """
    Permutation-based feature importance.

    For each feature, shuffle its values across the test set and measure
    how much the weighted AUC drops.  A large drop → the feature is important.
    """
    print("\nComputing permutation feature importance …")

    # Baseline AUC
    pred_base = predict(model, X_test, device)
    _, _, _ = roc_curve(y_test, pred_base, sample_weight=w_test)
    fpr_b, tpr_b, _ = roc_curve(y_test, pred_base, sample_weight=w_test)
    auc_base = auc(fpr_b, tpr_b)

    importances = np.zeros(len(ALL_FEATURES))
    rng = np.random.RandomState(42)

    for i in range(len(ALL_FEATURES)):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, i])
        pred_perm = predict(model, X_perm, device)
        fpr_p, tpr_p, _ = roc_curve(y_test, pred_perm, sample_weight=w_test)
        auc_perm = auc(fpr_p, tpr_p)
        importances[i] = auc_base - auc_perm  # positive = feature matters

    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(ALL_FEATURES)), importances[order], color="steelblue")
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in order], fontsize=10)
    ax.set_xlabel("ΔAUC  (permutation importance)", fontsize=13)
    ax.set_title(f"Feature Importance  —  {SIG_LABEL}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nFeature ranking  (most → least important):")
    for rank, idx in enumerate(reversed(order), 1):
        print(f"  {rank:2d}. {ALL_FEATURES[idx]:40s}  ΔAUC = {importances[idx]:.4f}")


def plot_correlation_matrix(data, outdir):
    """Plot feature-feature correlation matrix (Pearson)."""
    feature_df = data[ALL_FEATURES]

    short_names = []
    for f in ALL_FEATURES:
        name = f.replace("Res_", "").replace("_bjet_", "_")
        short_names.append(name)

    for subset, title_tag in [
        (data["label"] == 1, "Signal"),
        (data["label"] == 0, "Background"),
        (slice(None),        "Combined"),
    ]:
        corr = feature_df.loc[subset].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        n = len(ALL_FEATURES)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=90, fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)

        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        ax.set_title(f"Feature Correlation  —  {title_tag}  ({SIG_LABEL})",
                     fontsize=14)
        fig.tight_layout()

        fname = f"correlation_{title_tag.lower()}.png"
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


# ============================================================================
#  MAIN  TRAINING  PIPELINE
# ============================================================================

def main():
    global FEATURE_SET_NAME, DIRECT_FEATURES, ALL_FEATURES, SIG_LABEL, OUTPUT_DIR

    # Select device  (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    #  LOAD DATA ONCE  (union of all features across every feature set)   #
    # ------------------------------------------------------------------ #
    # Build a deduplicated superset so CSV files are read only once.
    # Exclude engineered feature names — they are derived at runtime, not
    # present as columns in the CSV files.
    all_possible_direct = list(dict.fromkeys(
        f for fs in FEATURE_SETS.values() for f in fs
        if f not in ENGINEERED_FEATURE_NAMES
    ))
    DIRECT_FEATURES = all_possible_direct   # used by _columns_to_read()

    sig = load_signal(SIGNAL_POINTS)
    bkg = load_backgrounds()

    if "process" in sig.columns:
        sig = sig.drop(columns=["process"])
    if "process" in bkg.columns:
        bkg = bkg.drop(columns=["process"])

    data_raw = pd.concat([sig, bkg], ignore_index=True)
    print(f"\nCombined:  {len(data_raw):,} events  "
          f"({(data_raw['label']==1).sum():,} sig  +  "
          f"{(data_raw['label']==0).sum():,} bkg)")

    print(f"\n{'='*60}")
    print(f"Will train {len(FEATURE_SETS)} models  "
          f"(one per feature set):  {list(FEATURE_SETS)}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    #  LOOP  OVER  FEATURE  SETS                                          #
    # ------------------------------------------------------------------ #
    for feat_name, feat_list in FEATURE_SETS.items():
        loop_start = time.perf_counter()
        print(f"\n{'#'*60}")
        print(f"#  Feature set: {feat_name}  ({len(feat_list)} features)")
        print(f"{'#'*60}")

        # Update mutable globals for this iteration.
        FEATURE_SET_NAME = feat_name
        ALL_FEATURES     = list(feat_list)
        engineered_needed = [f for f in ALL_FEATURES if f in ENGINEERED_FEATURE_NAMES]
        direct_needed = [f for f in ALL_FEATURES if f not in ENGINEERED_FEATURE_NAMES]
        raw_clean_cols = list(dict.fromkeys(
            direct_needed + [
                dep
                for engineered_name in engineered_needed
                for dep in ENGINEERED_FEATURE_DEPENDENCIES[engineered_name]
            ]
        ))
        SIG_LABEL = (
            f"{MASS_CATEGORY}_{_years_str}_"
            f"feat-{FEATURE_SET_NAME}"
        )
        OUTPUT_DIR = (
            "/eos/user/b/bsinghal/analysis/bbgg_low_res/"
            f"hhbbgg_AwkwardAnalyzer/ML_Application/low_mass/dnn_output_csv/{SIG_LABEL}/"
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ------------------------------------------------------------------ #
        #  1.  SUBSET  DATA                                                   #
        # ------------------------------------------------------------------ #
        data = data_raw.copy()

        # Clean only the raw columns needed by this feature set before
        # constructing derived variables.
        data = clean_sentinel_values(data, raw_clean_cols)

        # Build the engineered variables, then keep only the columns used by
        # this particular feature set.
        data = add_engineered_features(data)
        data = data[ALL_FEATURES + ["label", WEIGHT_COL]].copy()

        n_before = len(data)
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=ALL_FEATURES + [WEIGHT_COL])
        n_dropped = n_before - len(data)
        if n_dropped:
            print(f"Dropped {n_dropped:,} rows with NaN/inf (including converted "
                  f"sentinels)  ({n_dropped / n_before * 100:.1f} %)")

        # ------------------------------------------------------------------ #
        #  2.  CORRELATION  MATRIX                                            #
        # ------------------------------------------------------------------ #
        plot_correlation_matrix(data, OUTPUT_DIR)

        # ------------------------------------------------------------------ #
        #  3.  PREPARE  WEIGHTS                                               #
        # ------------------------------------------------------------------ #
        raw_weights = np.clip(data[WEIGHT_COL].values, 0, WEIGHT_CLIP_MAX)

        sig_mask = data["label"].values == 1
        bkg_mask = ~sig_mask

        print(f"\nWeight stats  (BEFORE normalisation):")
        print(f"  Signal      mean={raw_weights[sig_mask].mean():.6f}  "
              f"sum={raw_weights[sig_mask].sum():.2f}  N={sig_mask.sum():,}")
        print(f"  Background  mean={raw_weights[bkg_mask].mean():.6f}  "
              f"sum={raw_weights[bkg_mask].sum():.2f}  N={bkg_mask.sum():,}")

        sum_sig = raw_weights[sig_mask].sum()
        sum_bkg = raw_weights[bkg_mask].sum()
        weights = raw_weights.copy()
        weights[bkg_mask] *= sum_sig / sum_bkg

        print(f"\nWeight stats  (AFTER normalisation):")
        print(f"  Signal      sum={weights[sig_mask].sum():.2f}")
        print(f"  Background  sum={weights[bkg_mask].sum():.2f}  "
              f"(rescaled by {sum_sig/sum_bkg:.4f})")

        # ------------------------------------------------------------------ #
        #  4.  TRAIN / TEST  SPLIT                                            #
        # ------------------------------------------------------------------ #
        X = data[ALL_FEATURES].values
        y = data["label"].values

        (X_train, X_test,
         y_train, y_test,
         w_train, w_test) = train_test_split(
            X, y, weights,
            test_size    = TEST_FRACTION,
            random_state = 42,
            stratify     = y,
        )

        print(f"\nTrain:  {len(X_train):,}  "
              f"({(y_train==1).sum():,} sig  +  {(y_train==0).sum():,} bkg)")
        print(f"Test:   {len(X_test):,}  "
              f"({(y_test==1).sum():,} sig  +  {(y_test==0).sum():,} bkg)")

        # ------------------------------------------------------------------ #
        #  5.  FEATURE  SCALING                                               #
        # ------------------------------------------------------------------ #
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ------------------------------------------------------------------ #
        #  6.  BUILD  PYTORCH  DATALOADERS                                    #
        # ------------------------------------------------------------------ #
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            torch.tensor(w_test, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_PARAMS["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_PARAMS["batch_size"],
            shuffle=False,
        )

        # ------------------------------------------------------------------ #
        #  7.  BUILD  &  TRAIN  DNN                                           #
        # ------------------------------------------------------------------ #
        print(f"\n{'='*60}")
        print("Training DNN …")
        print(f"  Feature set: {FEATURE_SET_NAME}")
        print(f"  Engineered features: {engineered_needed if engineered_needed else 'none'}")
        print(f"  Mass category: {MASS_CATEGORY}")
        print(f"  Active years: {ACTIVE_YEARS}")
        print(f"  Architecture: {len(ALL_FEATURES)} → "
              f"{' → '.join(str(h) for h in DNN_PARAMS['hidden_layers'])} → 1")
        print(f"  Dropout: {DNN_PARAMS['dropout']}, "
              f"BatchNorm: {DNN_PARAMS['batch_norm']}, "
              f"Activation: {DNN_PARAMS['activation']}")
        print(f"{'='*60}")

        model = DNNClassifier(
            n_features    = len(ALL_FEATURES),
            hidden_layers = DNN_PARAMS["hidden_layers"],
            dropout       = DNN_PARAMS["dropout"],
            batch_norm    = DNN_PARAMS["batch_norm"],
            activation    = DNN_PARAMS["activation"],
        ).to(device)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, history = train_dnn(model, train_loader, val_loader, device,
                                   TRAINING_PARAMS)

        # ------------------------------------------------------------------ #
        #  8.  EVALUATE                                                       #
        # ------------------------------------------------------------------ #
        pred_train = predict(model, X_train, device)
        pred_test  = predict(model, X_test,  device)

        plot_loss_history(history, OUTPUT_DIR)
        plot_dnn_scores(y_train, y_test, pred_train, pred_test, OUTPUT_DIR)
        auc_w, auc_u = plot_roc(y_test, pred_test, w_test, OUTPUT_DIR)
        print(f"\n  Weighted   AUC = {auc_w:.4f}")
        print(f"  Unweighted AUC = {auc_u:.4f}")
        plot_feature_importance(model, X_test, y_test, w_test, device, OUTPUT_DIR)

        # ------------------------------------------------------------------ #
        #  9.  SAVE  MODEL  +  METADATA                                       #
        # ------------------------------------------------------------------ #
        model_path = os.path.join(OUTPUT_DIR, "dnn_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": {
                "n_features":    len(ALL_FEATURES),
                "hidden_layers": DNN_PARAMS["hidden_layers"],
                "dropout":       DNN_PARAMS["dropout"],
                "batch_norm":    DNN_PARAMS["batch_norm"],
                "activation":    DNN_PARAMS["activation"],
            },
        }, model_path)
        print(f"\nModel → {model_path}")

        onnx_path = os.path.join(OUTPUT_DIR, "dnn_model.onnx")
        dummy_input = torch.randn(1, len(ALL_FEATURES), device=device)
        model.eval()
        try:
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=["features"],
                output_names=["score"],
                dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
            )
            print(f"ONNX  → {onnx_path}")
        except Exception as e:
            print(f"ONNX export skipped ({e.__class__.__name__}: {e})")

        meta = dict(
            features        = ALL_FEATURES,
            feature_set     = FEATURE_SET_NAME,
            mass_category   = MASS_CATEGORY,
            active_years    = ACTIVE_YEARS,
            signal_label    = SIG_LABEL,
            runtime_seconds = float(time.perf_counter() - loop_start),
            scaler_mean     = scaler.mean_.tolist(),
            scaler_scale    = scaler.scale_.tolist(),
            signal_points   = [{"MX": mx, "MY": my} for mx, my in SIGNAL_POINTS],
            weighted_auc    = float(auc_w),
            unweighted_auc  = float(auc_u),
            n_signal        = int(sig_mask.sum()),
            n_background    = int(bkg_mask.sum()),
            best_epoch      = int(len(history["val_loss"]) -
                                  TRAINING_PARAMS["early_stopping"]),
            architecture    = DNN_PARAMS,
            training_params = TRAINING_PARAMS,
        )
        meta_path = os.path.join(OUTPUT_DIR, "dnn_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Meta  → {meta_path}")

        print(f"\n{'='*60}")
        print(f"Feature set '{feat_name}' done!  Output:  {OUTPUT_DIR}")
        print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"All {len(FEATURE_SETS)} feature sets complete.")
    print(f"{'='*60}")


# ============================================================================
if __name__ == "__main__":
    main()
