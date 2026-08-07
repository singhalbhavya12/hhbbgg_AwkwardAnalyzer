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
import time
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
# Keep this in sync with dnn_optimized.py
SIGNAL_POINTS_BY_X = {
    240: [50, 60, 70, 80, 90, 95],
    280: [50, 60, 70, 80, 90, 95],
    300: [50, 60, 70, 80, 90, 95],
    320: [50, 60, 70, 80, 90, 95],
    350: [50, 60, 70, 80, 90, 95],
    400: [50, 60, 70, 80, 90, 95],
    450: [50, 60, 70, 80, 90, 95],
    500: [50, 60, 70, 80, 90, 95],
    550: [50, 60, 70, 80, 90, 95],
    600: [50, 60, 70, 80, 90, 95],
    650: [50, 60, 70, 80, 90, 95],
    700: [50, 60, 70, 80, 90, 95],
    750: [50, 60, 70, 80, 90, 95],
    800: [50, 60, 70, 80, 90, 95],
    850: [50, 60, 70, 80, 90, 95],
    900: [50, 60, 70, 80, 90, 95],
    950: [50, 60, 70, 80, 90, 95],
    1000: [50, 60, 70, 80, 90, 95],
}

LOW_X_MASSES = [240, 280, 300, 320, 350]
MID_X_MASSES = [400, 450, 500, 550, 600]
HIGH_X_MASSES = [650, 700, 750, 800, 850, 900, 950, 1000]

MASS_CATEGORIES = {
    "280_70": [280],
    "low": LOW_X_MASSES,
    "mid": MID_X_MASSES,
    "high": HIGH_X_MASSES,
    "all": sorted(SIGNAL_POINTS_BY_X.keys()),
}

# Choose signal mass category: 280_70 / low / mid / high / all
MASS_CATEGORY = "low"

# Choose which eras to load.
# For 2024-only training, keep only "2024" here.
ACTIVE_YEARS = ["2022preEE", "2022postEE"]

# Per-era paths and process naming.
YEAR_DATASETS = {
    "2022preEE": {
        "signal_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/bsahu/higgsdna_v7/2022preEE/merged",
        "signal_name_templates": [
            "NMSSM_X{mx}_Y{my}",
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2022/sim/preEE/",
        "bkg_processes": [
            "GGJets_MGG-80",
            # "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            # "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
    "2022postEE": {
        "signal_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/bsahu/higgsdna_v7/2022postEE/merged",
        "signal_name_templates": [
            "NMSSM_X{mx}_Y{my}",
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2022/sim/postEE/",
        "bkg_processes": [
            "GGJets_MGG-80",
            # "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            # "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
    "2023": {
        "signal_dir": "/eos/user/b/bsinghal/analysis/output/2023/merged",
        "signal_name_templates": [
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
            "NMSSM_X{mx}_Y{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2023/sim/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
    "2024": {
        "signal_dir": "/eos/user/b/bartek/hhbbgg/higgsdna_v7/2024/merged",
        "signal_name_templates": [
            "NMSSM_X{mx}_Y{my}",
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        # Set this to your 2024 background MC location.
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v7/Run3_2024/sim/",
        # Fill with available 2024 background process folder names.
        # If left empty, background loading for 2024 is skipped with a warning.
        "bkg_processes": [
            "GGJets_MGG-80",
            # "GJet_MGG-80-PT-20to40",
            # "GJet_MGG-80-PT-40"
        ],

        #  "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/postEE/",
        # "bkg_processes": [
        #     "GGJets_MGG-80",
        #     "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
        #     "GJet_PT-40_DoubleEMEnriched_MGG-80",
        # ],
    },
}

# --- Cross-sections (pb) and luminosity -------------------------------------
# Luminosities copied from normalisation.py (do not import at runtime).
YEAR_LUMINOSITY_FB = {
    "2022preEE": 5.0104 + 2.9700,               # C + D
    "2022postEE": 5.8070 + 17.7819 + 3.0828,   # E + F + G
    "2023": 17.794 + 9.451,                     # preBPix + postBPix
    # TODO: set to the integrated lumi (fb^-1) used for your 2024 MC normalisation.
    "2024": 109.82,
}


def get_year_luminosity_fb(year):
    if year not in YEAR_LUMINOSITY_FB:
        raise KeyError(
            f"Missing luminosity for year='{year}'. "
            f"Add it to YEAR_LUMINOSITY_FB."
        )
    val = YEAR_LUMINOSITY_FB[year]
    if val is None:
        raise KeyError(
            f"Luminosity for year='{year}' is None. "
            f"Set YEAR_LUMINOSITY_FB['{year}'] before training."
        )
    return val


SIGNAL_XSEC_PB = 0.001

XSEC_PB = {
    "GGJets_MGG-80": 88.75,
    "GJet_PT-20to40_DoubleEMEnriched_MGG-80": 242.5,
    "GJet_PT-40_DoubleEMEnriched_MGG-80": 919.1,
}


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

_years_str = "_".join(ACTIVE_YEARS)

# --- Features -----------------------------------------------------------------
FEATURE_SETS = {
    "baseline": [
        "Res_CosThetaStar_jj",
        "Res_CosThetaStar_gg",
        "Res_CosThetaStar_CS",
        "Res_DeltaR_jg_min",
        "Res_DeltaR_j1g1",
        "Res_DeltaR_j1g2",
        "Res_DeltaR_j2g1",
        "Res_DeltaR_j2g2",
        "lead_mvaID",
        "sublead_mvaID",
        "Res_lead_bjet_btagPNetB",
        "Res_sublead_bjet_btagPNetB",
        "lead_energyErr",
        "sublead_energyErr",
        "sigma_m_over_m",
        "Res_lead_bjet_PNetRegPtRawRes",
        "Res_sublead_bjet_PNetRegPtRawRes",
        "Res_pholead_PtOverM",
        "Res_phosublead_PtOverM",
        "Res_FirstJet_PtOverM",
        "Res_SecondJet_PtOverM",
    ],
    "disc_low": [
        "Res_CosThetaStar_gg",
        "Res_CosThetaStar_CS",
        "Res_lead_bjet_btagPNetB",
        "Res_sublead_bjet_btagPNetB",
    ],
    "disc_mid": [
        "Res_CosThetaStar_CS",
        "Res_DeltaR_j1g2",
        "Res_lead_bjet_btagPNetB",
        "Res_sublead_bjet_btagPNetB",
        "Res_lead_bjet_PNetRegPtRawRes",
        "Res_sublead_bjet_PNetRegPtRawRes",
        "Res_pholead_PtOverM",
        "Res_phosublead_PtOverM",
        "Res_FirstJet_PtOverM",
    ]
}

FEATURE_SET_NAME = "disc_low"
DIRECT_FEATURES = FEATURE_SETS[FEATURE_SET_NAME]

# Choose engineered feature set: none / boost_var / pt_jj_only / pt_gg_only
ENGINEERED_FEATURE_SETS = {
    "none": [],
    "boost_var": [
        "pt_jj_over_M_jjgg",      # pT(jj) / M(jjgg)
        "pt_gg_over_M_jjgg",      # pT(gg) / M(jjgg)
    ],
    "pt_jj_only": [
        "pt_jj_over_M_jjgg",
    ],
    "pt_gg_only": [
        "pt_gg_over_M_jjgg",
    ],
}

ENGINEERED_FEATURE_SET_NAME = "boost_var"
if ENGINEERED_FEATURE_SET_NAME not in ENGINEERED_FEATURE_SETS:
    raise ValueError(
        f"Unknown ENGINEERED_FEATURE_SET_NAME='{ENGINEERED_FEATURE_SET_NAME}'. "
        f"Choose from {sorted(ENGINEERED_FEATURE_SETS)}"
    )

ENGINEERED_FEATURE_NAMES = ENGINEERED_FEATURE_SETS[ENGINEERED_FEATURE_SET_NAME]

# Optional photon MVA-ID working-point selection.
# Set to "none" to keep all events.
# Available boolean columns in the input parquet files:
#   lead_mvaID_WP80, lead_mvaID_WP90,
#   sublead_mvaID_WP80, sublead_mvaID_WP90
MVAID_WP_USE_FILTER = False
MVAID_WP_FILTER_MODES = [
    "lead_wp80",
    # "lead_wp90",
    "sublead_wp80",
    # "sublead_wp90",
]
MVAID_WP_COLUMNS = [
    "lead_mvaID_WP80",
    "lead_mvaID_WP90",
    "sublead_mvaID_WP80",
    "sublead_mvaID_WP90",
]
MVAID_WP_MODE_TO_COLUMN = {
    "lead_wp80": "lead_mvaID_WP80",
    "lead_wp90": "lead_mvaID_WP90",
    "sublead_wp80": "sublead_mvaID_WP80",
    "sublead_wp90": "sublead_mvaID_WP90",
}
MVAID_WP_FILTERS = {
    "lead_wp80": ["lead_mvaID_WP80"],
    "lead_wp90": ["lead_mvaID_WP90"],
    "sublead_wp80": ["sublead_mvaID_WP80"],
    "sublead_wp90": ["sublead_mvaID_WP90"],
    "all": MVAID_WP_COLUMNS,
}


def _normalize_mvaid_wp_filters(filter_modes):
    if filter_modes is None:
        return []
    if isinstance(filter_modes, str):
        if filter_modes == "none":
            return []
        if filter_modes == "all":
            return list(MVAID_WP_MODE_TO_COLUMN)
        return [filter_modes]
    if any(mode == "all" for mode in filter_modes):
        return list(MVAID_WP_MODE_TO_COLUMN)
    return list(filter_modes)


MVAID_WP_FILTER_MODES = _normalize_mvaid_wp_filters(MVAID_WP_FILTER_MODES)
MVAID_WP_LABEL = "none" if not MVAID_WP_USE_FILTER or not MVAID_WP_FILTER_MODES else "__".join(MVAID_WP_FILTER_MODES)

SIG_LABEL = (
    f"{MASS_CATEGORY}_{_years_str}-{FEATURE_SET_NAME}"
    f"-eng_{ENGINEERED_FEATURE_SET_NAME}"
    f"-mvaid_{MVAID_WP_LABEL}"
)

# Columns needed only for engineering extra features:
ENGINEERING_COLS = [
    "Res_dijet_pt",
    "pt",
    "Res_HHbbggCandidate_mass",
    "mass",
]

# Weight and filter
WEIGHT_COL = "weight"            # per-event weight (genWeight × SFs / Σ genWeights)
FILTER_COL = "is_Res"            # keep only events in the "resolved" category

# --- Runtime optimization controls -------------------------------------------
OPTIMIZATION_SETTINGS = dict(
    plot_correlation_matrix = False,
    plot_feature_importance = False,
)

# --- XGBoost hyper-parameters -------------------------------------------------
XGB_PARAMS = dict(
    n_estimators        = 1000,    # max boosting rounds
    max_depth           = 4,      # tree depth (higher = more complex)
    learning_rate       = 0.05,   # step-size shrinkage
    subsample           = 0.8,    # fraction of events per tree
    colsample_bytree    = 0.8,    # fraction of features per tree
    min_child_weight    = 50,     # min sum-of-weights in a leaf
    gamma               = 1.0,    # min loss reduction per split
    reg_alpha           = 0.1,    # L1 regularisation
    reg_lambda          = 1.0,    # L2 regularisation
    eval_metric         = "logloss",
    random_state        = 42,
    n_jobs              = max(1, min(8, (os.cpu_count() or 2) // 2)),
    tree_method         = "hist",
)

EARLY_STOPPING_ROUNDS = 20       # stop if val loss stalls for N rounds
TEST_FRACTION         = 0.3      # 30 % held out for evaluation
WEIGHT_CLIP_MAX       = 10.0     # clip extreme per-event weights

# Train one model for each selected (MX, MY) signal point in addition
# to the category-level model.
TRAIN_INDIVIDUAL_POINTS = True

BASE_OUTPUT_DIR = (
    "/eos/user/b/bsinghal/analysis/bbgg_low_res/"
    "hhbbgg_AwkwardAnalyzer/ML_Application/low_mass/bdt_output/disc_low/mvaid_none"
)


def build_output_dir(run_label):
    return os.path.join(
        BASE_OUTPUT_DIR,
        f"{run_label}_iterations{XGB_PARAMS['n_estimators']}",
    )


OUTPUT_DIR = build_output_dir(SIG_LABEL)


# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

ENGINEERED_FEATURE_DEPENDENCIES = {
    "pt_jj_over_M_jjgg": ["Res_dijet_pt", "Res_HHbbggCandidate_mass"],
    "pt_gg_over_M_jjgg": ["pt", "Res_HHbbggCandidate_mass"],
}

REQUIRED_ENGINEERING_COLS = [
    col for col in ENGINEERING_COLS
    if any(col in ENGINEERED_FEATURE_DEPENDENCIES[f] for f in ENGINEERED_FEATURE_NAMES)
]

# Complete ordered list that goes into XGBoost
ALL_FEATURES = DIRECT_FEATURES + ENGINEERED_FEATURE_NAMES

# Values at or below this threshold are treated as non-physical sentinels.
# In these samples, missing kinematics are often encoded as -999, not NaN.
SENTINEL_THRESHOLD = -900.0


def add_engineered_features(df):
    """Build derived features from the raw columns."""
    if not ENGINEERED_FEATURE_NAMES:
        return df

    # Guard against division by zero. The mass should be positive after the
    # resolved-category selection, but keep the protection explicit.
    m_jjgg = df["Res_HHbbggCandidate_mass"].replace(0, np.nan)

    if "pt_jj_over_M_jjgg" in ENGINEERED_FEATURE_NAMES:
        df["pt_jj_over_M_jjgg"] = df["Res_dijet_pt"] / m_jjgg
    if "pt_gg_over_M_jjgg" in ENGINEERED_FEATURE_NAMES:
        df["pt_gg_over_M_jjgg"] = df["pt"] / m_jjgg

    return df


def apply_mvaid_wp_filter(
    df,
    use_filter=MVAID_WP_USE_FILTER,
    filter_modes=MVAID_WP_FILTER_MODES,
):
    """Optionally filter events using the boolean photon MVA-ID WP columns.

    The parquet files provide 0/1 flags for the lead and sublead photons at
    WP80 and WP90. The mode controls which of those flags are required:
    - none: keep all events
    - one or more named flags: require all selected flags to be true
    """
    if not use_filter or not filter_modes:
        return df

    unknown = [mode for mode in filter_modes if mode not in MVAID_WP_MODE_TO_COLUMN]
    if unknown:
        raise ValueError(
            f"Unknown MVAID_WP_FILTER_MODES={unknown}. "
            f"Choose from {sorted(MVAID_WP_MODE_TO_COLUMN)}"
        )

    missing = [c for c in MVAID_WP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns needed for MVA-ID WP filtering: {missing}. "
            f"Add them to the parquet input or load them through _columns_to_read()."
        )

    for col in MVAID_WP_COLUMNS:
        df[col] = df[col].astype(bool)

    required_cols = [MVAID_WP_MODE_TO_COLUMN[mode] for mode in filter_modes]
    pass_mask = np.ones(len(df), dtype=bool)
    for col in required_cols:
        pass_mask &= df[col].to_numpy(dtype=bool)

    label = "__".join(filter_modes)
    df[f"pass_mvaid_{label}"] = pass_mask

    n_before = len(df)
    df = df[pass_mask].copy()
    print(
        f"\nMVA-ID selection ({label}): kept {len(df):,} / {n_before:,} events "
        f"(dropped {n_before - len(df):,})"
    )
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
    # Only include columns that exist in the parquet files:
    # - Direct features (excluding engineered features)
    # - Engineering base columns needed by selected engineered features
    # - Photon MVA-ID working-point booleans
    # - Weight and filter columns
    cols = [f for f in DIRECT_FEATURES if f not in ENGINEERED_FEATURE_NAMES]
    cols += REQUIRED_ENGINEERING_COLS
    if MVAID_WP_USE_FILTER:
        cols += MVAID_WP_COLUMNS
    cols.append(WEIGHT_COL)
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


def _find_existing_file(base_dir, proc_name):
    """Try a few common parquet layouts and return the first existing path."""
    candidates = [
        os.path.join(base_dir, proc_name, "nominal", "NOTAG_merged.parquet"),
        os.path.join(base_dir, proc_name, "NOTAG_merged.parquet"),
        os.path.join(base_dir, f"{proc_name}.parquet"),
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
                fp = _find_existing_file(signal_dir, candidate)
                if fp is not None:
                    proc_name = candidate
                    break

            if fp is None:
                print(f"  WARNING: signal MX={mx}, MY={my} not found in {year} — skipping")
                continue

            df = load_parquet(fp, cols, label=1, filter_col=FILTER_COL)
            df[WEIGHT_COL] = df[WEIGHT_COL] * SIGNAL_XSEC_PB * lumi_fb * 1000
            print(f"  Applied  σ = {SIGNAL_XSEC_PB} pb  ×  L = {lumi_fb:.4f} fb⁻¹")

            df["process"] = proc_name
            df["year"] = year
            df["MX"] = mx
            df["MY"] = my
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No signal files were loaded!")

    sig = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total signal:  {len(sig):,} events  "
          f"({len(dfs)} loaded chunks)")
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

        processes = conf.get("bkg_processes", [])
        if not processes:
            print(f"  WARNING: no bkg_processes configured for year '{year}' — skipping")
            continue

        lumi_fb = get_year_luminosity_fb(year)

        for proc in processes:
            fp = _find_existing_file(conf["bkg_dir"], proc)
            if fp is None:
                print(f"  WARNING: {proc} not found in {year} — skipping")
                continue

            df = load_parquet(fp, cols, label=0, filter_col=FILTER_COL)
            xs = XSEC_PB.get(proc, 1.0)
            df[WEIGHT_COL] = df[WEIGHT_COL] * xs * lumi_fb * 1000
            print(f"  Applied  σ = {xs} pb  ×  L = {lumi_fb:.4f} fb⁻¹")

            df["process"] = proc
            df["year"] = year
            dfs.append(df)

    if not dfs:
        raise RuntimeError(
            "No background files were loaded! "
            "Check YEAR_DATASETS['<year>']['bkg_dir'] and ['bkg_processes']."
        )

    bkg = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total background:  {len(bkg):,} events")
    return bkg


# ============================================================================
#  PLOTTING  HELPERS
# ============================================================================

def plot_bdt_scores(y_train, y_test, pred_train, pred_test, outdir, plot_label):
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
    ax.set_title(f"BDT Output  —  {plot_label}", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(outdir, "bdt_score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc(y_test, pred_test, w_test, outdir, plot_label):
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
    ax.set_title(f"ROC Curve  —  {plot_label}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return auc_w, auc_u


def plot_feature_importance(model, outdir, plot_label):
    """Horizontal bar chart of gain-based feature importance."""
    imp = model.feature_importances_
    order = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(ALL_FEATURES)), imp[order], color="steelblue")
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in order], fontsize=10)
    ax.set_xlabel("Feature Importance  (gain)", fontsize=13)
    ax.set_title(f"Feature Importance  —  {plot_label}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nFeature ranking  (most → least important):")
    for rank, idx in enumerate(reversed(order), 1):
        print(f"  {rank:2d}. {ALL_FEATURES[idx]:40s}  {imp[idx]:.4f}")


def plot_correlation_matrix(data, outdir, plot_label):
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
        ax.set_title(f"Feature Correlation  —  {title_tag}  ({plot_label})",
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

def train_and_save(sig, bkg, outdir, run_label, signal_points_meta, mass_category_tag):
    """Train one BDT and save all outputs for a provided signal/background set."""
    start_time = time.perf_counter()
    os.makedirs(outdir, exist_ok=True)

    if len(sig) == 0:
        print(f"\n[SKIP] {run_label}: no signal events loaded.")
        return None
    if len(bkg) == 0:
        print(f"\n[SKIP] {run_label}: no background events loaded.")
        return None

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
    #  2a. OPTIONAL  MVA-ID  WORKING-POINT  SELECTION                     #
    # ------------------------------------------------------------------ #
    data = apply_mvaid_wp_filter(data)

    # ------------------------------------------------------------------ #
    #  2b. CLEAN SENTINEL PLACEHOLDERS                                    #
    # ------------------------------------------------------------------ #
    # Important study note:
    # - The parquet files can have zero NaNs and still contain invalid values.
    # - Here, -999 is a placeholder for "feature unavailable" in some events.
    # - If we keep these placeholders, many unrelated features can appear
    #   almost perfectly correlated (an analysis artifact, not physics).
    sentinel_cols = DIRECT_FEATURES + REQUIRED_ENGINEERING_COLS
    data = clean_sentinel_values(data, sentinel_cols)

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
    if OPTIMIZATION_SETTINGS["plot_correlation_matrix"]:
        plot_correlation_matrix(data, outdir, run_label)

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
    if n_sig_tr == 0:
        print(f"\n[SKIP] {run_label}: no signal events in train split.")
        return None
    print(f"\nClass counts  bkg / sig = {n_bkg_tr / n_sig_tr:.2f}  "
          f"(handled via weight normalisation, not scale_pos_weight)")

    # ------------------------------------------------------------------ #
    #  8.  TRAIN  XGBOOST                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Training XGBoost …")
    print(f"  Feature set: {FEATURE_SET_NAME}")
    print(f"  Run label: {run_label}")
    print(f"  Mass category: {mass_category_tag}")
    print(f"  Active years: {ACTIVE_YEARS}")
    print(f"  Tree method: {XGB_PARAMS['tree_method']}, n_jobs: {XGB_PARAMS['n_jobs']}")
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
    plot_bdt_scores(y_train, y_test, pred_train, pred_test, outdir, run_label)

    # -- ROC curve --
    auc_w, auc_u = plot_roc(y_test, pred_test, w_test, outdir, run_label)
    print(f"\n  Weighted   AUC = {auc_w:.4f}")
    print(f"  Unweighted AUC = {auc_u:.4f}")

    # -- Feature importance --
    if OPTIMIZATION_SETTINGS["plot_feature_importance"]:
        plot_feature_importance(model, outdir, run_label)

    # ------------------------------------------------------------------ #
    #  10. SAVE  MODEL  +  METADATA                                       #
    # ------------------------------------------------------------------ #
    model_path = os.path.join(outdir, "bdt_model.json")
    model.save_model(model_path)
    print(f"\nModel → {model_path}")

    meta = dict(
        feature_set     = FEATURE_SET_NAME,
        engineered_feature_set = ENGINEERED_FEATURE_SET_NAME,
        engineered_features = ENGINEERED_FEATURE_NAMES,
        mvaid_wp_use_filter = MVAID_WP_USE_FILTER,
        mvaid_wp_filter_modes = MVAID_WP_FILTER_MODES,
        mvaid_wp_label = MVAID_WP_LABEL,
        mvaid_wp_columns = MVAID_WP_COLUMNS,
        mass_category   = mass_category_tag,
        active_years    = ACTIVE_YEARS,
        signal_label    = run_label,
        features       = ALL_FEATURES,
        scaler_mean    = scaler.mean_.tolist(),
        scaler_scale   = scaler.scale_.tolist(),
        signal_points  = signal_points_meta,
        weighted_auc   = float(auc_w),
        unweighted_auc = float(auc_u),
        n_signal       = int(sig_mask.sum()),
        n_background   = int(bkg_mask.sum()),
        best_iteration = int(best_iter),
        runtime_seconds = float(time.perf_counter() - start_time),
        optimization_settings = OPTIMIZATION_SETTINGS,
        xgb_params = XGB_PARAMS,
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    )
    meta_path = os.path.join(outdir, "bdt_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta  → {meta_path}")

    print(f"\n{'='*60}")
    print(f"Done!  Output directory:  {outdir}")
    print(f"{'='*60}")

    return dict(
        weighted_auc=float(auc_w),
        unweighted_auc=float(auc_u),
        output_dir=outdir,
    )


def train_individual_points(sig_all, bkg):
    """Train one model per (MX, MY) signal point in selected MASS_CATEGORY."""
    print(f"\n{'='*60}")
    print("Training individual MX_MY models")
    print(f"{'='*60}")

    summary = []

    for mx, my in SIGNAL_POINTS:
        point_sig = sig_all[(sig_all["MX"] == mx) & (sig_all["MY"] == my)].copy()
        if len(point_sig) == 0:
            print(f"\n[SKIP] MX={mx}, MY={my}: no loaded signal events.")
            continue

        run_label = (
            f"MX{mx}_MY{my}_{_years_str}-{FEATURE_SET_NAME}"
            f"-eng_{ENGINEERED_FEATURE_SET_NAME}"
            f"-mvaid_{MVAID_WP_LABEL}"
        )
        outdir = build_output_dir(run_label)
        result = train_and_save(
            sig=point_sig,
            bkg=bkg,
            outdir=outdir,
            run_label=run_label,
            signal_points_meta=[{"MX": mx, "MY": my}],
            mass_category_tag=f"MX{mx}_MY{my}",
        )
        if result is not None:
            summary.append((mx, my, result["weighted_auc"], result["output_dir"]))

    print(f"\n{'='*60}")
    print("Per-point training summary")
    print(f"{'='*60}")
    if not summary:
        print("No per-point models were produced.")
        return

    for mx, my, auc_w, outdir in summary:
        print(f"MX={mx:4d} MY={my:3d} | weighted AUC={auc_w:.4f} | {outdir}")


def main():
    # ------------------------------------------------------------------ #
    #  1.  LOAD DATA                                                      #
    # ------------------------------------------------------------------ #
    sig = load_signal(SIGNAL_POINTS)
    bkg = load_backgrounds()

    # ------------------------------------------------------------------ #
    #  2.  CATEGORY-LEVEL TRAINING                                        #
    # ------------------------------------------------------------------ #
    train_and_save(
        sig=sig.copy(),
        bkg=bkg.copy(),
        outdir=OUTPUT_DIR,
        run_label=SIG_LABEL,
        signal_points_meta=[{"MX": mx, "MY": my} for mx, my in SIGNAL_POINTS],
        mass_category_tag=MASS_CATEGORY,
    )

    # ------------------------------------------------------------------ #
    #  3.  OPTIONAL: POINT-BY-POINT TRAINING                              #
    # ------------------------------------------------------------------ #
    if TRAIN_INDIVIDUAL_POINTS:
        train_individual_points(sig_all=sig, bkg=bkg)


# ============================================================================
if __name__ == "__main__":
    main()
