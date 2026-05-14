import os
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np

# Signal/background configuration copied from dnn_lowmass.py
# Add new mass points by extending the Y-mass list for each X mass.
SIGNAL_POINTS_BY_X = {
     240: [50,60,70, 80, 90, 95, 100],
    280: [50,60,70, 80, 90, 95, 100],
    300: [50,60,70, 80, 90, 95, 100],
    320: [50,60,70, 80, 90, 95, 100],
    350: [50,60,70, 80, 90, 95, 100],
    400: [50,60,70, 80, 90, 95, 100],
    450: [50,60,70, 80, 90, 95, 100],
    500: [50,60,70, 80, 90, 95, 100],
    550: [50,60,70, 80, 90, 95, 100],#[60,70, 80, 90, 95, 100],
    600: [50,60,70, 80, 90, 95, 100],#[60,70, 80, 90, 95, 100],
    650: [50,60,70, 80, 90, 95, 100],#[70, 80, 90, 95, 100],
    700: [50,60,70, 80, 90, 95, 100],#[70, 80, 90, 95, 100],
    750: [50,60,70, 80, 90, 95, 100],#[80, 90, 95, 100],
    800: [50,60,70, 80, 90, 95, 100],#[80, 90, 95, 100],
    850: [50,60,70, 80, 90, 95, 100],#[90, 95, 100],
    900: [50,60,70, 80, 90, 95, 100],#[90, 95, 100],
    950: [50,60,70, 80, 90, 95, 100],#[95, 100],
    1000: [50,60,70, 80, 90, 95, 100],#[100],
}

LOW_X_MASSES = [240, 280, 300, 320, 350]
MID_X_MASSES = [400, 450, 500, 550, 600]
HIGH_X_MASSES = [650, 700, 750, 800, 850, 900, 950, 1000]

SIGNAL_POINTS = [
    (mx, my)
    for mx, my_values in SIGNAL_POINTS_BY_X.items()
    for my in my_values
]

# Select which eras to load. Add/remove entries here.
# Example: ACTIVE_YEARS = ["2022preEE", "2023"]
ACTIVE_YEARS = ["2022preEE", "2022postEE"]  # <-- EDIT THIS LIST AS NEEDED

# Per-era paths and background process lists.
YEAR_DATASETS = {
    "2022preEE": {
        "signal_dir": "/eos/user/b/bsinghal/analysis/output/2022preEE/merged",
        "signal_name_templates": [
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/preEE/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            "GJet_PT-40_DoubleEMEnriched_MGG-80",
        ],
    },
    "2022postEE": {
        "signal_dir": "/eos/user/b/bartek/hhbbgg/systematics_v3/2022_postEE/merged",
        "signal_name_templates": [
            "NMSSM_X{mx}_Y{my}",
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/postEE/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_PT-20to40_DoubleEMEnriched_MGG-80",
            "GJet_PT-40_DoubleEMEnriched_MGG-80",
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
}

DIRECT_FEATURES = [
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
    "Res_dijet_mass",
    "n_jets",
    "Res_HHbbggCandidate_mass",
]

ENGINEERING_COLS = [
    "Res_dijet_pt",
    "pt",
    "Res_HHbbggCandidate_mass",
    "mass",           # diphoton mass column in HiggsDNA parquet
]

FILTER_COL = "is_Res"

# Extra derived variables to plot on top of DIRECT_FEATURES
EXTRA_PLOT_VARS = [
    "pt_jj_over_M_jjgg",
    "pt_gg_over_M_jjgg",
    "reduced_invariant_mX",
    "diphoton_mass",  # alias of 'mass' column (renamed for clarity in plots)
]

OUTDIR = "/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/kinematics/bf_kin"


def columns_to_read():
    cols = DIRECT_FEATURES + ENGINEERING_COLS
    cols.append(FILTER_COL)
    return list(dict.fromkeys(cols))


def load_parquet(filepath, columns):
    available = set(pq.read_schema(filepath).names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"Columns missing in {filepath}: {missing}. "
            f"Available sample: {sorted(available)[:30]}"
        )

    df = pd.read_parquet(filepath, columns=columns)
    if FILTER_COL in df.columns:
        df = df[df[FILTER_COL] == 1].copy()
    return df


def load_signal_frames(columns, active_years):
    rows = []
    for year in active_years:
        conf = YEAR_DATASETS.get(year)
        if conf is None:
            print(f"WARNING: year '{year}' is not defined in YEAR_DATASETS")
            continue
        signal_dir = conf["signal_dir"]
        name_templates = conf.get("signal_name_templates", ["NMSSM-XtoYH-MX-{mx}-MY-{my}"])
        file_templates = [
            os.path.join("{signal_dir}", "{proc}", "nominal", "NOTAG_merged.parquet"),
            os.path.join("{signal_dir}", "{proc}", "NOTAG_merged.parquet"),
            os.path.join("{signal_dir}", "{proc}.parquet"),
        ]
        for mx, my in SIGNAL_POINTS:
            fp = None
            proc = None
            for proc_tpl in name_templates:
                proc_candidate = proc_tpl.format(mx=mx, my=my)
                for fp_tpl in file_templates:
                    fp_candidate = fp_tpl.format(signal_dir=signal_dir, proc=proc_candidate)
                    if os.path.exists(fp_candidate):
                        proc = proc_candidate
                        fp = fp_candidate
                        break
                if fp is not None:
                    break

            if fp is None:
                tried = [tpl.format(mx=mx, my=my) for tpl in name_templates]
                print(f"WARNING: signal file not found ({year}) for MX={mx}, MY={my}. Tried names: {tried}")
                continue
            try:
                df = load_parquet(fp, columns)
                df["MX"] = mx
                df["year"] = year
                rows.append(df)
                print(f"Loaded signal {proc} ({year}): {len(df):,} events")
            except Exception as e:
                print(f"Error loading signal {proc} ({year}): {e}")
    return rows


def load_background_frames(columns, active_years):
    gg_rows = []
    gjet_rows = []
    for year in active_years:
        conf = YEAR_DATASETS.get(year)
        if conf is None:
            print(f"WARNING: year '{year}' is not defined in YEAR_DATASETS")
            continue
        bkg_dir = conf["bkg_dir"]
        bkg_processes = conf["bkg_processes"]
        for proc in bkg_processes:
            fp = os.path.join(bkg_dir, proc, "nominal", "NOTAG_merged.parquet")
            if not os.path.exists(fp):
                print(f"WARNING: background file not found ({year}): {fp}")
                continue
            try:
                df = load_parquet(fp, columns)
                df["year"] = year
                if "GGJets" in proc:
                    gg_rows.append(df)
                else:
                    gjet_rows.append(df)
                print(f"Loaded background {proc} ({year}): {len(df):,} events")
            except Exception as e:
                print(f"Error loading background {proc} ({year}): {e}")
    return gg_rows, gjet_rows


SENTINEL_THRESHOLD = -900.0


def clean_sentinels(df, features=DIRECT_FEATURES):
    """Replace values <= SENTINEL_THRESHOLD (e.g. -999) with NaN, then drop
    any row that has a NaN or inf in any feature column."""
    n_before = len(df)
    for col in features:
        if col not in df.columns:
            continue
        mask = df[col] <= SENTINEL_THRESHOLD
        if mask.any():
            df.loc[mask, col] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with sentinel / NaN / inf values")
    return df


def add_engineered_features(df):
    """Build the same engineered variables used in dnn_lowmass.py."""
    m_jjgg = df["Res_HHbbggCandidate_mass"].replace(0, np.nan)
    diphoton_mass = df["mass"].replace(0, np.nan)

    df["pt_jj_over_M_jjgg"] = df["Res_dijet_pt"] / m_jjgg
    df["pt_gg_over_M_jjgg"] = df["pt"] / m_jjgg
    df["reduced_invariant_mX"] = m_jjgg - (diphoton_mass - 125.0)

    # Keep legacy plotting aliases.
    df["diphoton_mass"] = diphoton_mass

    return df


def concat_or_empty(frames, columns):
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def plot_variable(var, lowX_df, midX_df, highX_df, ggJets_df, gJetPt_df):
    """Produce a single normalised histogram for *var* across all datasets."""

    def finite_vals(df):
        if df.empty or var not in df.columns:
            return np.array([])
        # sentinels and NaNs already removed in main(); guard against any remaining
        mask = np.isfinite(df[var].values)
        return df.loc[mask, var].values

    lowX_vals  = finite_vals(lowX_df)
    midX_vals  = finite_vals(midX_df)
    highX_vals = finite_vals(highX_df)
    gg_vals    = finite_vals(ggJets_df)
    gjet_vals  = finite_vals(gJetPt_df)

    all_vals = np.concatenate([a for a in [lowX_vals, midX_vals, highX_vals,
                                            gg_vals, gjet_vals] if len(a) > 0])
    if len(all_vals) == 0:
        print(f"  [{var}] WARNING: no finite entries — skipping")
        return

    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)
    if np.isclose(vmin, vmax):
        vmin, vmax = all_vals.min(), all_vals.max()
    bin_edges = np.linspace(vmin, vmax, 50)

    fig, ax = plt.subplots(figsize=(8, 6))

    # density=True normalises each histogram so the area under it equals 1
    if len(lowX_vals) > 0:
        ax.hist(lowX_vals,  bins=bin_edges, density=True, histtype="step",
                label="lowX",  color="red",   linestyle="-",  linewidth=1.5)
    if len(midX_vals) > 0:
        ax.hist(midX_vals,  bins=bin_edges, density=True, histtype="step",
                label="midX",  color="blue",  linestyle="--", linewidth=1.5)
    if len(highX_vals) > 0:
        ax.hist(highX_vals, bins=bin_edges, density=True, histtype="step",
                label="highX", color="green", linestyle="-.", linewidth=1.5)
    if len(gg_vals) > 0:
        ax.hist(gg_vals,    bins=bin_edges, density=True, histtype="stepfilled",
                label=r"$\gamma\gamma$+Jets", color="blue", alpha=0.35)
    if len(gjet_vals) > 0:
        ax.hist(gjet_vals,  bins=bin_edges, density=True, histtype="stepfilled",
                label=r"$\gamma$+Jets",      color="gray", alpha=0.35)

    ax.set_xlabel(var)
    ax.set_ylabel("Normalised to unit area")
    ax.set_title(f"{' + '.join(ACTIVE_YEARS)}  —  {var}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    out_png = os.path.join(OUTDIR, f"{var}.png")
    out_pdf = os.path.join(OUTDIR, f"{var}.pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    if not ACTIVE_YEARS:
        raise ValueError("ACTIVE_YEARS is empty. Add at least one year.")

    cols = columns_to_read()
    signal_frames = load_signal_frames(cols, ACTIVE_YEARS)
    gg_rows, gjet_rows = load_background_frames(cols, ACTIVE_YEARS)

    signal_df = concat_or_empty(signal_frames, cols + ["MX"])
    if signal_df.empty:
        raise RuntimeError("No signal events loaded. Check SIGNAL_DIR and SIGNAL_POINTS.")

    raw_features_for_cleaning = list(dict.fromkeys(DIRECT_FEATURES + ENGINEERING_COLS))

    print("\nCleaning signal ...")
    signal_df = clean_sentinels(signal_df, features=raw_features_for_cleaning)

    lowX_df  = signal_df[signal_df["MX"].isin(LOW_X_MASSES)].copy()
    midX_df  = signal_df[signal_df["MX"].isin(MID_X_MASSES)].copy()
    highX_df = signal_df[signal_df["MX"].isin(HIGH_X_MASSES)].copy()

    ggJets_df = concat_or_empty(gg_rows, cols)
    gJetPt_df = concat_or_empty(gjet_rows, cols)

    print("Cleaning backgrounds ...")
    ggJets_df = clean_sentinels(ggJets_df, features=raw_features_for_cleaning)
    gJetPt_df = clean_sentinels(gJetPt_df, features=raw_features_for_cleaning)

    for _df in [lowX_df, midX_df, highX_df, ggJets_df, gJetPt_df]:
        add_engineered_features(_df)

    all_vars = DIRECT_FEATURES + EXTRA_PLOT_VARS
    print(f"\nPlotting {len(all_vars)} variables → {OUTDIR}\n")
    for var in all_vars:
        print(f"Plotting: {var}")
        plot_variable(var, lowX_df, midX_df, highX_df, ggJets_df, gJetPt_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
