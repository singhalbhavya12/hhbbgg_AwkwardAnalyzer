#!/usr/bin/env python3
"""
bdt_lowmass.py  —  BDT classifier for HH→bbγγ low-mass resonance search
=========================================================================

Trains an XGBoost BDT to separate  NMSSM  X → YH → bbγγ  signal
from non-resonant backgrounds (γγ+jets, γ+jets).

How to run (on lxplus, with LCG_106):
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
    python3 bdt_lowmass.py

Key ML ideas demonstrated
--------------------------
 1. Reading physics data from parquet files
 2. Event weights  (weight_central encodes  σ × L / N_gen × SFs)
 3. Class-imbalance handling with  scale_pos_weight
 4. Data-leakage-free feature scaling  (fit scaler on train only)
 5. Early stopping to prevent over-training
 6. Evaluation: ROC curve, feature importance, BDT-score distributions
"""

# ============================================================================
#  Imports
# ============================================================================
import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no X11 on lxplus)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import xgboost as xgb

# ============================================================================
#  CONFIGURATION  —  edit this section to change signal / backgrounds / features
# ============================================================================

# --- Signal mass points -------------------------------------------------------
# List of (MX, MY) tuples to combine as signal.  All are merged into one
# signal class (label=1) and trained together against the common background.
# Add / remove entries freely.
SIGNAL_POINTS = [
    (240,  90),
    (280,  90),
    (300,  90),
    (320,  90),
    (350,  90),
]

# --- File paths ---------------------------------------------------------------
# Signal: 2022 preEE MC  (your own EOS production)
SIGNAL_DIR = "/eos/user/b/bsinghal/analysis/output/2022preEE/merged"

# Backgrounds: v3 central production, 2022 preEE era  (matches signal era)
BKG_DIR = (
    "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/preEE/"
)
BKG_PROCESSES = [
    "GGJets_MGG-80",                          # γγ+jets  (Mγγ > 80 GeV)
    "GJet_PT-20to40_DoubleEMEnriched_MGG-80",  # γ+jet   (20 < pT < 40)
    "GJet_PT-40_DoubleEMEnriched_MGG-80",      # γ+jet   (pT > 40)
    # "DDQCDGJets",                              # data-driven QCD + γ-jet
]

# --- Cross-sections (pb) and luminosity ----------------------------------------
# The weight column contains  genWeight × SFs / sum(genWeights),
# so we multiply by  σ × L  to get expected event yield.
# IMPORTANT:  replace the placeholder values below with the real ones!
LUMINOSITY = 7.9804  # fb⁻¹  (2022 preEE)

# Cross-sections keyed by process name.  Signal entries are built from
# SIGNAL_POINTS so you can set a different σ for each mass point.
XSEC_PB = {
    # --- signal  (TODO: fill real σ for each mass point) ---
    **{f"NMSSM-XtoYH-MX-{mx}-MY-{my}": 0.001 for mx, my in SIGNAL_POINTS},
    # --- backgrounds ---
    "GGJets_MGG-80":                                88.75,  # TODO: verify from XSDB
    "GJet_PT-20to40_DoubleEMEnriched_MGG-80":     242.5,   # TODO: verify
    "GJet_PT-40_DoubleEMEnriched_MGG-80":         919.1,   # TODO: verify
    # "DDQCDGJets":                                   1.0,
}

# --- Output -------------------------------------------------------------------
# Build a compact label like "MX240_280_300_MY100" for the output folder.
_mx_str = "_".join(str(mx) for mx, _ in SIGNAL_POINTS)
_my_set = sorted(set(my for _, my in SIGNAL_POINTS))
_my_str = "_".join(str(my) for my in _my_set)
SIG_LABEL = f"MX{_mx_str}_MY{_my_str}"

OUTPUT_DIR = (
    "/eos/user/b/bsinghal/analysis/bbgg_low_res/"
    f"hhbbgg_AwkwardAnalyzer/ML_Application/low_mass/bdt_output/corr_dijet/{SIG_LABEL}/"
)

# --- Features -----------------------------------------------------------------
# DIRECT_FEATURES are columns that exist as-is in the parquet files.
# They are chosen to capture the kinematics of HH → bbγγ events.
DIRECT_FEATURES = [
    # ---- Angular correlations (Collins-Soper frame) ----
    "Res_CosThetaStar_jj",        # cos θ*(bb):  dijet  in CS frame
    "Res_CosThetaStar_gg",        # cos θ*(γγ):  diphoton in CS frame
    "Res_CosThetaStar_CS",        # cos θ*(HH candidate) in CS frame

    # ---- ΔR between photons and jets ----
    "Res_DeltaR_jg_min",          # smallest ΔR(jet, photon) of the 4 combos
    "Res_DeltaR_j1g1",            # ΔR(lead jet,    lead photon)
    "Res_DeltaR_j1g2",            # ΔR(lead jet,    sublead photon)
    "Res_DeltaR_j2g1",            # ΔR(sublead jet, lead photon)
    "Res_DeltaR_j2g2",            # ΔR(sublead jet, sublead photon)

    # ---- Photon identification ----
    "lead_mvaID",                 # leading  γ ID MVA  (higher = more photon-like)
    "sublead_mvaID",              # sublead  γ ID MVA

    # ---- B-tagging ----
    # PNet is the Run-3 standard.  DeepJet (DeepFlav) columns exist for
    # generic jets (jet1_btagDeepFlav_B) but NOT for the Res_ b-jet pair.
    "Res_lead_bjet_btagPNetB",    # lead b-jet  PNet B-tag score
    "Res_sublead_bjet_btagPNetB", # sublead b-jet PNet B-tag score

    # ---- Resolution variables ----
    "lead_energyErr",             # lead γ energy uncertainty  (GeV)
    "sublead_energyErr",          # sublead γ energy uncertainty
    "sigma_m_over_m",             # diphoton mass resolution  σ_m / m(γγ)
    "Res_lead_bjet_PNetRegPtRawRes",    # lead b-jet pT resolution
    "Res_sublead_bjet_PNetRegPtRawRes", # sublead b-jet pT resolution

    # ---- Dimensionless pT / mass ratios ----
    "Res_pholead_PtOverM",        # pT(lead γ)    / m(γγ)
    "Res_phosublead_PtOverM",     # pT(sublead γ) / m(γγ)
    "Res_FirstJet_PtOverM",       # pT(lead jet)    / m(jj)
    "Res_SecondJet_PtOverM",      # pT(sublead jet) / m(jj)

    # ---- Invariant masses ----
    "Res_dijet_mass",             # m(jj)  — should peak near m_H for signal
    # "Res_dijet_mass_DNNreg",      # m(jj)  with DNN-regressed jet energies

    # ---- Helicity / compatibility ----
    # "Res_chi_t0",                 # reduced χ² for  H → bb  mass compatibility
    # "Res_chi_t1",                 # reduced χ² for  H → γγ  mass compatibility
]

# Columns needed only for engineering extra features:
ENGINEERING_COLS = [
    "Res_dijet_pt",               # pT(jj)   — numerator  for  pT(jj)/M(jjγγ)
    "pt",                         # pT(γγ)   — diphoton transverse momentum
    "Res_HHbbggCandidate_mass",   # M(jjγγ)  — four-body mass  (denominator)
]

# Weight and filter
WEIGHT_COL = "weight"            # per-event weight (genWeight × SFs / Σ genWeights)
FILTER_COL = "is_Res"            # keep only events in the "resolved" category

# --- XGBoost hyper-parameters -------------------------------------------------
XGB_PARAMS = dict(
    n_estimators        = 500,    # max boosting rounds
    max_depth           = 4,      # tree depth  (higher = more complex)
    learning_rate       = 0.05,   # step-size shrinkage
    subsample           = 0.8,    # fraction of events per tree
    colsample_bytree    = 0.8,    # fraction of features per tree
    min_child_weight    = 50,     # min sum-of-weights in a leaf
    gamma               = 1.0,    # min loss reduction per split
    reg_alpha           = 0.1,    # L1 regularisation
    reg_lambda          = 1.0,    # L2 regularisation
    eval_metric         = "logloss",
    random_state        = 42,
    n_jobs              = 4,
)

EARLY_STOPPING_ROUNDS = 30       # stop if val loss stalls for N rounds
TEST_FRACTION         = 0.3      # 30 % held out for evaluation
WEIGHT_CLIP_MAX       = 10.0     # clip extreme per-event weights


# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

# Names of features we will compute (appended to DIRECT_FEATURES for training)
ENGINEERED_FEATURE_NAMES = [
    "pt_jj_over_M_jjgg",         # pT(jj) / M(jjγγ)  — "boostedness"
    "pt_gg_over_M_jjgg",         # pT(γγ) / M(jjγγ)
]

# Complete ordered list that goes into XGBoost
ALL_FEATURES = DIRECT_FEATURES + ENGINEERED_FEATURE_NAMES

# Values at or below this threshold are treated as non-physical sentinels.
# In these samples, missing kinematics are often encoded as -999, not NaN.
SENTINEL_THRESHOLD = -900.0


def add_engineered_features(df):
    """
    Build derived features from raw columns.

    Dimensionless ratios are preferred because they are less sensitive to
    absolute energy-scale uncertainties.
    """
    # Guard against division by zero  (M = 0 should not happen after is_Res,
    # but just in case → replace 0 with NaN so the row is dropped later).
    m_jjgg = df["Res_HHbbggCandidate_mass"].replace(0, np.nan)

    df["pt_jj_over_M_jjgg"] = df["Res_dijet_pt"] / m_jjgg
    df["pt_gg_over_M_jjgg"] = df["pt"]            / m_jjgg

    return df


def clean_sentinel_values(df, features, threshold=SENTINEL_THRESHOLD):
    """
    Replace non-physical sentinel values (e.g. -999) with NaN.

    Why: Pearson correlation and model training can be strongly distorted if
    many features share the same extreme placeholder value in the same events.
    Converting placeholders to NaN lets the standard cleaning step drop those
    rows in a transparent, controlled way.
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
#  DATA  LOADING
# ============================================================================

def _columns_to_read():
    """Return deduplicated list of all columns needed from parquet."""
    cols = DIRECT_FEATURES + ENGINEERING_COLS + [WEIGHT_COL]
    if FILTER_COL:
        cols.append(FILTER_COL)
    return list(dict.fromkeys(cols))          # deduplicate, keep order


def load_parquet(filepath, columns, label, filter_col=None):
    """
    Read a parquet file, keep only *columns*, optionally filter, add a label.

    Parameters
    ----------
    filepath   : str   – path to .parquet
    columns    : list  – column names to read
    label      : int   – 1 = signal, 0 = background
    filter_col : str   – if given, keep rows where this column == 1

    Returns
    -------
    pd.DataFrame  with the requested columns + a ``label`` column.
    """
    # Only read the columns we need  → faster I/O, less memory
    available = set(pq.read_schema(filepath).names)
    missing   = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"Columns missing in {filepath}:\n  {missing}\n"
            f"Available columns (first 30): {sorted(available)[:30]}"
        )

    df = pd.read_parquet(filepath, columns=columns)
    tag = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    print(f"  Loaded {len(df):>8,} events  from  {tag}")

    if filter_col and filter_col in df.columns:
        n_before = len(df)
        df = df[df[filter_col] == 1].copy()
        print(f"    → {len(df):>8,} pass  {filter_col}  "
              f"(dropped {n_before - len(df):,})")

    df["label"] = label
    return df


def load_signal(signal_points):
    """
    Load multiple signal mass-points and concatenate them.

    Each mass point gets its own σ × L normalisation so that
    higher-cross-section points naturally contribute more events.
    """
    cols = _columns_to_read()
    dfs  = []

    print(f"\n{'='*60}")
    print(f"[Signal]  {len(signal_points)} mass points")
    print(f"{'='*60}")

    for mx, my in signal_points:
        proc_name = f"NMSSM-XtoYH-MX-{mx}-MY-{my}"
        fp = os.path.join(
            SIGNAL_DIR,
            proc_name,
            "nominal",
            "NOTAG_merged.parquet",
        )
        if not os.path.exists(fp):
            print(f"  WARNING: {proc_name} not found – skipping!")
            continue

        df = load_parquet(fp, cols, label=1, filter_col=FILTER_COL)

        # Multiply per-event weight by  σ × L  →  expected yield
        xs = XSEC_PB.get(proc_name, 1.0)
        df[WEIGHT_COL] = df[WEIGHT_COL] * xs * LUMINOSITY * 1000
        print(f"  Applied  σ = {xs} pb  ×  L = {LUMINOSITY} fb⁻¹")

        df["process"] = proc_name
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No signal files were loaded!")

    sig = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total signal:  {len(sig):,} events  "
          f"({len(dfs)} mass points)")
    return sig


def load_backgrounds():
    """Load and concatenate all background processes."""
    cols = _columns_to_read()
    dfs  = []

    print(f"\n{'='*60}")
    print(f"[Background]  {len(BKG_PROCESSES)} processes")
    print(f"{'='*60}")

    for proc in BKG_PROCESSES:
        fp = os.path.join(BKG_DIR, proc, "nominal/NOTAG_merged.parquet")
        if not os.path.exists(fp):
            print(f"  WARNING: {proc} not found – skipping!")
            continue
        df = load_parquet(fp, cols, label=0, filter_col=FILTER_COL)

        # Multiply per-event weight by  σ × L
        xs = XSEC_PB.get(proc, 1.0)
        df[WEIGHT_COL] = df[WEIGHT_COL] * xs * LUMINOSITY * 1000
        print(f"  Applied  σ = {xs} pb  ×  L = {LUMINOSITY} fb⁻¹")

        df["process"] = proc
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No background files were loaded!")

    bkg = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total background:  {len(bkg):,} events")
    return bkg


# ============================================================================
#  PLOTTING  HELPERS
# ============================================================================

def plot_bdt_scores(y_train, y_test, pred_train, pred_test, outdir):
    """
    BDT output-score histograms for signal vs background.

    Filled histograms = training set;  markers = test set.
    If the markers sit on top of the filled histograms the model is NOT
    over-trained.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)

    # Training (filled)
    ax.hist(pred_train[y_train == 1], bins=bins, density=True,
            alpha=0.35, color="blue",  label="Signal (train)")
    ax.hist(pred_train[y_train == 0], bins=bins, density=True,
            alpha=0.35, color="red",   label="Background (train)")

    # Test (markers) — for over-training check
    for lbl, col, name in [(1, "blue", "Signal"), (0, "red", "Background")]:
        mask = y_test == lbl
        counts, edges = np.histogram(pred_test[mask], bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(centres, counts, fmt="o", color=col, ms=4,
                     label=f"{name} (test)")

    ax.set_xlabel("BDT score  (signal probability)", fontsize=13)
    ax.set_ylabel("Normalised events", fontsize=13)
    ax.set_title(f"BDT Output  —  {SIG_LABEL}", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(outdir, "bdt_score_distribution.png")
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


def plot_feature_importance(model, outdir):
    """Horizontal bar chart of gain-based feature importance."""
    imp = model.feature_importances_
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(ALL_FEATURES)), imp[order], color="steelblue")
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in order], fontsize=10)
    ax.set_xlabel("Feature Importance  (gain)", fontsize=13)
    ax.set_title(f"Feature Importance  —  {SIG_LABEL}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nFeature ranking  (most → least important):")
    for rank, idx in enumerate(reversed(order), 1):
        print(f"  {rank:2d}. {ALL_FEATURES[idx]:40s}  {imp[idx]:.4f}")


def plot_correlation_matrix(data, outdir):
    """
    Plot feature-feature correlation matrix (Pearson) for signal and
    background separately, plus a combined one.

    Why this matters:
    - Highly correlated features (|r| > 0.8) carry redundant info.
      Removing one can simplify the model without losing performance.
    - Different correlation patterns between sig & bkg can reveal
      which feature *pairs* are discriminating.
    """
    feature_df = data[ALL_FEATURES]

    # Use shorter display names (strip common prefixes) for readability
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

        # Annotate cells with correlation values
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  1.  LOAD DATA                                                      #
    # ------------------------------------------------------------------ #
    sig = load_signal(SIGNAL_POINTS)
    bkg = load_backgrounds()

    # Drop the 'process' bookkeeping column before merging
    if "process" in sig.columns:
        sig = sig.drop(columns=["process"])
    if "process" in bkg.columns:
        bkg = bkg.drop(columns=["process"])

    data = pd.concat([sig, bkg], ignore_index=True)
    print(f"\nCombined:  {len(data):,} events  "
          f"({(data['label']==1).sum():,} sig  +  "
          f"{(data['label']==0).sum():,} bkg)")

    # ------------------------------------------------------------------ #
    #  2.  ENGINEER FEATURES                                              #
    # ------------------------------------------------------------------ #
    data = add_engineered_features(data)

    # ------------------------------------------------------------------ #
    #  2b. CLEAN SENTINEL PLACEHOLDERS                                    #
    # ------------------------------------------------------------------ #
    # Important study note:
    # - The parquet files can have zero NaNs and still contain invalid values.
    # - Here, -999 is a placeholder for "feature unavailable" in some events.
    # - If we keep these placeholders, many unrelated features can appear
    #   almost perfectly correlated (an analysis artifact, not physics).
    data = clean_sentinel_values(data, DIRECT_FEATURES)

    # ------------------------------------------------------------------ #
    #  3.  CLEAN  (drop rows with NaN/inf in any feature or weight)        #
    # ------------------------------------------------------------------ #
    n_before = len(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=ALL_FEATURES + [WEIGHT_COL])
    n_dropped = n_before - len(data)
    if n_dropped:
        print(f"Dropped {n_dropped:,} rows with NaN/inf (including converted "
              f"sentinels)  "
              f"({n_dropped / n_before * 100:.1f} %)")

    # ------------------------------------------------------------------ #
    #  3b. CORRELATION MATRIX  (before scaling, on physical values)        #
    # ------------------------------------------------------------------ #
    plot_correlation_matrix(data, OUTPUT_DIR)

    # ------------------------------------------------------------------ #
    #  4.  PREPARE  WEIGHTS                                               #
    # ------------------------------------------------------------------ #
    # At this point  weight = (genWeight × SFs / Σ genWeights) × σ × L.
    # Clip extreme values to stop single events from dominating.
    raw_weights = np.clip(data[WEIGHT_COL].values, 0, WEIGHT_CLIP_MAX)

    sig_mask = data["label"].values == 1
    bkg_mask = ~sig_mask

    print(f"\nWeight stats  (BEFORE normalisation):")
    print(f"  Signal      mean={raw_weights[sig_mask].mean():.6f}  "
          f"sum={raw_weights[sig_mask].sum():.2f}  N={sig_mask.sum():,}")
    print(f"  Background  mean={raw_weights[bkg_mask].mean():.6f}  "
          f"sum={raw_weights[bkg_mask].sum():.2f}  N={bkg_mask.sum():,}")

    # ---- Normalise so  Σ w_sig == Σ w_bkg ----
    # This prevents the model from over-training on whichever class has
    # a larger total weight.  We keep the *relative* weight differences
    # within each class (so a high-σ background process still contributes
    # more than a low-σ one), but equalise signal vs background overall.
    sum_sig = raw_weights[sig_mask].sum()
    sum_bkg = raw_weights[bkg_mask].sum()
    weights = raw_weights.copy()
    weights[bkg_mask] *= sum_sig / sum_bkg    # rescale bkg → same total as sig

    print(f"\nWeight stats  (AFTER normalisation):")
    print(f"  Signal      sum={weights[sig_mask].sum():.2f}")
    print(f"  Background  sum={weights[bkg_mask].sum():.2f}  "
          f"(rescaled by {sum_sig/sum_bkg:.4f})")

    # ------------------------------------------------------------------ #
    #  5.  TRAIN / TEST  SPLIT                                            #
    # ------------------------------------------------------------------ #
    X = data[ALL_FEATURES].values
    y = data["label"].values

    (X_train, X_test,
     y_train, y_test,
     w_train, w_test) = train_test_split(
        X, y, weights,
        test_size    = TEST_FRACTION,
        random_state = 42,
        stratify     = y,      # keeps the sig / bkg ratio identical in both sets
    )

    print(f"\nTrain:  {len(X_train):,}  "
          f"({(y_train==1).sum():,} sig  +  {(y_train==0).sum():,} bkg)")
    print(f"Test:   {len(X_test):,}  "
          f"({(y_test==1).sum():,} sig  +  {(y_test==0).sum():,} bkg)")

    # ------------------------------------------------------------------ #
    #  6.  FEATURE  SCALING                                               #
    # ------------------------------------------------------------------ #
    # StandardScaler:  x' = (x - μ) / σ   →  mean 0, std 1.
    #
    # IMPORTANT:  fit ONLY on training data.   If you fit on the full
    # dataset the scaler "sees" test-set values → data leakage → your
    # evaluation metric is optimistically biased.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)    # fit + transform  (train)
    X_test  = scaler.transform(X_test)         # transform only   (test)

    # ------------------------------------------------------------------ #
    #  7.  CLASS  IMBALANCE                                               #
    # ------------------------------------------------------------------ #
    # We already normalised weights so Σ w_sig == Σ w_bkg.
    # Therefore we do NOT need scale_pos_weight — set it to 1 (default).
    # Using both scale_pos_weight AND normalised sample_weight would
    # double-compensate and bias the model towards signal.
    n_sig_tr = (y_train == 1).sum()
    n_bkg_tr = (y_train == 0).sum()
    print(f"\nClass counts  bkg / sig = {n_bkg_tr / n_sig_tr:.2f}  "
          f"(handled via weight normalisation, not scale_pos_weight)")

    # ------------------------------------------------------------------ #
    #  8.  TRAIN  XGBOOST                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Training XGBoost …")
    print(f"{'='*60}")

    model = xgb.XGBClassifier(
        # scale_pos_weight = 1 (default) — imbalance is handled by
        # the normalised sample_weight instead.
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS,
    )

    # eval_set  → XGBoost evaluates on the test set after each round.
    # When the metric stops improving for EARLY_STOPPING_ROUNDS rounds
    # it reverts to the best iteration  → prevents over-training.
    # sample_weight_eval_set ensures the early-stopping metric uses the
    # same weighting as training — otherwise the stopping criterion is
    # inconsistent with the loss being optimised.
    model.fit(
        X_train, y_train,
        sample_weight          = w_train,
        eval_set               = [(X_test, y_test)],
        sample_weight_eval_set = [w_test],
        verbose                = 50,
    )

    best_iter = getattr(model, "best_iteration", XGB_PARAMS["n_estimators"])
    print(f"\nBest boosting round:  {best_iter}")

    # ------------------------------------------------------------------ #
    #  9.  EVALUATE                                                       #
    # ------------------------------------------------------------------ #
    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test  = model.predict_proba(X_test)[:, 1]

    # -- BDT score distributions --
    plot_bdt_scores(y_train, y_test, pred_train, pred_test, OUTPUT_DIR)

    # -- ROC curve --
    auc_w, auc_u = plot_roc(y_test, pred_test, w_test, OUTPUT_DIR)
    print(f"\n  Weighted   AUC = {auc_w:.4f}")
    print(f"  Unweighted AUC = {auc_u:.4f}")

    # -- Feature importance --
    plot_feature_importance(model, OUTPUT_DIR)

    # ------------------------------------------------------------------ #
    #  10. SAVE  MODEL  +  METADATA                                       #
    # ------------------------------------------------------------------ #
    model_path = os.path.join(OUTPUT_DIR, "bdt_model.json")
    model.save_model(model_path)
    print(f"\nModel → {model_path}")

    meta = dict(
        features       = ALL_FEATURES,
        scaler_mean    = scaler.mean_.tolist(),
        scaler_scale   = scaler.scale_.tolist(),
        signal_points  = [{"MX": mx, "MY": my} for mx, my in SIGNAL_POINTS],
        weighted_auc   = float(auc_w),
        unweighted_auc = float(auc_u),
        n_signal       = int(sig_mask.sum()),
        n_background   = int(bkg_mask.sum()),
        best_iteration = int(best_iter),
    )
    meta_path = os.path.join(OUTPUT_DIR, "bdt_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta  → {meta_path}")

    print(f"\n{'='*60}")
    print(f"Done!  Output directory:  {OUTPUT_DIR}")
    print(f"{'='*60}")


# ============================================================================
if __name__ == "__main__":
    main()
