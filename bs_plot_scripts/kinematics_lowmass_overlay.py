import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# Select eras to combine.
ACTIVE_YEARS = ["2024"]

# Select only the mass points to overlay.
# Format: (MX, MY)
SELECTED_MASS_POINTS = [
    (240, 60),
    (240, 70),
    (240, 80),
    (240, 90),
    (240, 95),
    (280, 50),
    (280, 60),
    (280, 70),
    (280, 80),
]

# Select only the features to plot.
SELECTED_FEATURES = [
    "Res_CosThetaStar_CS",
    "Res_DeltaR_j1g2",
    "Res_lead_bjet_btagPNetB",
    "Res_sublead_bjet_btagPNetB",
    "Res_lead_bjet_PNetRegPtRawRes",
    "Res_sublead_bjet_PNetRegPtRawRes",
    "Res_pholead_PtOverM",
    "Res_phosublead_PtOverM",
    "Res_FirstJet_PtOverM",
    "pt_jj_over_M_jjgg",
    "pt_gg_over_M_jjgg",
    "reduced_invariant_mX",
]

# If True, background templates are overlaid together with the selected signals.
INCLUDE_BACKGROUNDS = True

OUTDIR = (
    "/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/kinematics/"
    f"{'_'.join(ACTIVE_YEARS)}/overlay_selected/with_bkg"
)

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
    "2024": {
        "signal_dir": "/eos/user/b/bartek/hhbbgg/higgsdna_v7/2024/merged",
        "signal_name_templates": [
            "NMSSM_X{mx}_Y{my}",
            "NMSSM-XtoYH-MX-{mx}-MY-{my}",
        ],
        "bkg_dir": "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v4/Run3_2024/sim/",
        "bkg_processes": [
            "GGJets_MGG-80",
            "GJet_MGG-80-PT-20to40",
            "GJet_MGG-80-PT-40",
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
    "Res_dijet_mass_DNNreg",
    "n_jets",
    "Res_HHbbggCandidate_mass",
]

ENGINEERING_COLS = [
    "Res_dijet_pt",
    "pt",
    "Res_HHbbggCandidate_mass",
    "mass",
]

FILTER_COL = "is_Res"
SENTINEL_THRESHOLD = -900.0

EXTRA_PLOT_VARS = {
    "pt_jj_over_M_jjgg",
    "pt_gg_over_M_jjgg",
    "reduced_invariant_mX",
    "diphoton_mass",
}


def safe_tag(text):
    return str(text).replace(" ", "_").replace("/", "-")


def parse_mass_points(items):
    out = []
    for item in items:
        cleaned = item.strip().replace("MX", "").replace("MY", "")
        if ":" in cleaned:
            mx_str, my_str = cleaned.split(":", 1)
        elif "," in cleaned:
            mx_str, my_str = cleaned.split(",", 1)
        else:
            raise ValueError(
                f"Invalid mass point '{item}'. Use MX:MY, e.g. 280:70"
            )
        out.append((int(mx_str), int(my_str)))
    return out


def columns_to_read(selected_features):
    cols = [FILTER_COL]

    for var in selected_features:
        if var in DIRECT_FEATURES:
            cols.append(var)
        elif var not in EXTRA_PLOT_VARS:
            # Assume user passed a raw parquet column name.
            cols.append(var)

    # Engineered variables need these input columns.
    if any(var in EXTRA_PLOT_VARS for var in selected_features):
        cols.extend(ENGINEERING_COLS)

    return list(dict.fromkeys(cols))


def load_parquet(filepath, columns, apply_res_filter=True):
    available = set(pq.read_schema(filepath).names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"Columns missing in {filepath}: {missing}. "
            f"Available sample: {sorted(available)[:30]}"
        )

    df = pd.read_parquet(filepath, columns=columns)
    if apply_res_filter and FILTER_COL in df.columns:
        df = df[df[FILTER_COL] == 1].copy()
    return df


def load_signal_frames(columns, active_years, selected_mass_points):
    rows = []
    for year in active_years:
        conf = YEAR_DATASETS.get(year)
        if conf is None:
            print(f"WARNING: year '{year}' is not defined in YEAR_DATASETS")
            continue

        signal_dir = conf["signal_dir"]
        name_templates = conf.get(
            "signal_name_templates", ["NMSSM-XtoYH-MX-{mx}-MY-{my}"]
        )
        file_templates = [
            os.path.join("{signal_dir}", "{proc}", "nominal", "NOTAG_merged.parquet"),
            os.path.join("{signal_dir}", "{proc}", "NOTAG_merged.parquet"),
            os.path.join("{signal_dir}", "{proc}.parquet"),
        ]

        for mx, my in selected_mass_points:
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
                print(
                    f"WARNING: signal file not found ({year}) for MX={mx}, MY={my}. "
                    f"Tried names: {tried}"
                )
                continue

            try:
                df = load_parquet(fp, columns)
                df["MX"] = mx
                df["MY"] = my
                df["year"] = year
                rows.append(df)
                print(f"Loaded signal {proc} ({year}): {len(df):,} events")
            except Exception as err:
                print(f"Error loading signal {proc} ({year}): {err}")

    return rows


def load_background_frames(columns, active_years):
    gg_rows = []
    gjet_rows = []
    columns_wo_filter = [c for c in columns if c != FILTER_COL]

    for year in active_years:
        conf = YEAR_DATASETS.get(year)
        if conf is None:
            print(f"WARNING: year '{year}' is not defined in YEAR_DATASETS")
            continue

        bkg_dir = conf["bkg_dir"]
        bkg_processes = conf["bkg_processes"]
        for proc in bkg_processes:
            fp_candidates = [
                os.path.join(bkg_dir, proc, "nominal", "NOTAG_merged.parquet"),
                os.path.join(bkg_dir, proc, "NOTAG_merged.parquet"),
                os.path.join(bkg_dir, f"{proc}.parquet"),
            ]

            fp = None
            for fp_cand in fp_candidates:
                if os.path.exists(fp_cand):
                    fp = fp_cand
                    break

            if fp is None:
                print(
                    f"WARNING: background file not found ({year}) for {proc}. "
                    f"Tried: {fp_candidates}"
                )
                continue

            try:
                # Try with the same resonant-event filter as signals.
                df = load_parquet(fp, columns, apply_res_filter=True)

                # Fallback: some background samples have no usable is_Res tagging.
                if df.empty:
                    print(
                        f"INFO: {proc} ({year}) has 0 rows after is_Res filter; "
                        "retrying without is_Res filter."
                    )
                    df = load_parquet(fp, columns_wo_filter, apply_res_filter=False)

                df["year"] = year
                if "GGJets" in proc:
                    gg_rows.append(df)
                else:
                    gjet_rows.append(df)
                print(
                    f"Loaded background {proc} ({year}): {len(df):,} events "
                    f"from {fp}"
                )
            except Exception as err:
                print(f"Error loading background {proc} ({year}): {err}")

    return gg_rows, gjet_rows


def clean_sentinels(df, features):
    n_before = len(df)
    for col in features:
        if col not in df.columns:
            continue
        mask = df[col] <= SENTINEL_THRESHOLD
        if mask.any():
            df.loc[mask, col] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    existing_features = [col for col in features if col in df.columns]
    if existing_features:
        df = df.dropna(subset=existing_features)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with sentinel / NaN / inf values")
    return df


def add_engineered_features(df):
    if "Res_HHbbggCandidate_mass" in df.columns and "mass" in df.columns:
        m_jjgg = df["Res_HHbbggCandidate_mass"].replace(0, np.nan)
        diphoton_mass = df["mass"].replace(0, np.nan)

        if "Res_dijet_pt" in df.columns:
            df["pt_jj_over_M_jjgg"] = df["Res_dijet_pt"] / m_jjgg
        if "pt" in df.columns:
            df["pt_gg_over_M_jjgg"] = df["pt"] / m_jjgg

        df["reduced_invariant_mX"] = m_jjgg - (diphoton_mass - 125.0)
        df["diphoton_mass"] = diphoton_mass

    return df


def concat_or_empty(frames, columns):
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def finite_values(df, var):
    if df.empty or var not in df.columns:
        return np.array([])
    vals = df[var].values
    vals = vals[np.isfinite(vals)]
    return vals


CMS_SIGNAL_PALETTE = [
    "#3f90da",  # blue
    "#ffa90e",  # orange
    "#bd1f01",  # red
    "#94a4a2",  # gray-green
    "#832db6",  # purple
    "#a96b59",  # brown
    "#e76300",  # dark orange
    "#b9ac70",  # khaki
    "#717581",  # slate gray
    "#92dadd",  # cyan
]


def distinct_signal_colors(n):
    """Return n visually distinct colors using a CMS-like palette."""
    if n <= 0:
        return []
    if n <= len(CMS_SIGNAL_PALETTE):
        return CMS_SIGNAL_PALETTE[:n]

    # Keep first colors CMS-like, then extend with non-repeating categorical colors.
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        points = np.linspace(0.0, 1.0, n - len(CMS_SIGNAL_PALETTE), endpoint=True)
        extra = [cmap(p) for p in points]
        return CMS_SIGNAL_PALETTE + extra

    # For larger n, sample a continuous map to avoid repeats.
    cmap = plt.get_cmap("turbo")
    points = np.linspace(0.02, 0.98, n - len(CMS_SIGNAL_PALETTE), endpoint=True)
    extra = [cmap(p) for p in points]
    return CMS_SIGNAL_PALETTE + extra


def plot_overlay(var, signal_data_by_label, gg_df, gjet_df, outdir, title_suffix):
    arrays = []
    for vals in signal_data_by_label.values():
        if len(vals) > 0:
            arrays.append(vals)

    gg_vals = finite_values(gg_df, var)
    gjet_vals = finite_values(gjet_df, var)
    if len(gg_vals) > 0:
        arrays.append(gg_vals)
    if len(gjet_vals) > 0:
        arrays.append(gjet_vals)

    if not arrays:
        print(f"  [{var}] WARNING: no finite entries found. Skipping.")
        return

    all_vals = np.concatenate(arrays)
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)
    if np.isclose(vmin, vmax):
        vmin = all_vals.min()
        vmax = all_vals.max()

    bins = np.linspace(vmin, vmax, 50)
    fig, ax = plt.subplots(figsize=(8, 6))

    labels_with_data = [
        label for label, vals in signal_data_by_label.items() if len(vals) > 0
    ]
    color_map = dict(
        zip(labels_with_data, distinct_signal_colors(len(labels_with_data)))
    )

    for label, vals in signal_data_by_label.items():
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color_map[label],
            label=label,
        )

    if len(gg_vals) > 0:
        ax.hist(
            gg_vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.50,
            color="#3f90da",
            edgecolor="#1f4e79",
            linewidth=1.2,
            zorder=1,
            label=r"$\gamma\gamma$+Jets",
        )

    if len(gjet_vals) > 0:
        ax.hist(
            gjet_vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.50,
            color="#ffa90e",
            edgecolor="#a35d00",
            linewidth=1.2,
            zorder=1,
            label=r"$\gamma$+Jets",
        )

    ax.set_xlabel(var)
    ax.set_ylabel("Normalised to unit area")
    ax.set_title(f"Overlay: {var} ({title_suffix})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"{safe_tag(var)}_overlay.png")
    out_pdf = os.path.join(outdir, f"{safe_tag(var)}_overlay.pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"  Saved: {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay selected features for selected NMSSM mass points."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=ACTIVE_YEARS,
        help="Years to combine (default: from ACTIVE_YEARS in script)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=SELECTED_FEATURES,
        help="Feature names to plot",
    )
    parser.add_argument(
        "--mass-points",
        nargs="+",
        default=[f"{mx}:{my}" for mx, my in SELECTED_MASS_POINTS],
        help="Mass points as MX:MY tokens, e.g. 240:70 280:70",
    )
    parser.add_argument(
        "--outdir",
        default=OUTDIR,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        default=INCLUDE_BACKGROUNDS,
        help="Overlay GGJets and GJets templates as filled histograms",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_mass_points = parse_mass_points(args.mass_points)
    selected_features = list(dict.fromkeys(args.features))

    if not args.years:
        raise ValueError("No years selected. Provide at least one year.")
    if not selected_features:
        raise ValueError("No features selected. Provide at least one feature.")
    if not selected_mass_points:
        raise ValueError("No mass points selected. Provide at least one MX:MY pair.")

    cols = columns_to_read(selected_features)

    signal_frames = load_signal_frames(cols, args.years, selected_mass_points)
    signal_df = concat_or_empty(signal_frames, cols + ["MX", "MY", "year"])
    if signal_df.empty:
        raise RuntimeError("No signal events loaded for selected mass points.")

    clean_cols = list(dict.fromkeys(cols + ENGINEERING_COLS))
    signal_df = clean_sentinels(signal_df, clean_cols)
    add_engineered_features(signal_df)

    gg_df = pd.DataFrame()
    gjet_df = pd.DataFrame()
    if args.include_background:
        gg_rows, gjet_rows = load_background_frames(cols, args.years)
        gg_df = concat_or_empty(gg_rows, cols + ["year"])
        gjet_df = concat_or_empty(gjet_rows, cols + ["year"])
        gg_df = clean_sentinels(gg_df, clean_cols)
        gjet_df = clean_sentinels(gjet_df, clean_cols)
        add_engineered_features(gg_df)
        add_engineered_features(gjet_df)

    year_tag = "+".join(args.years)
    mass_tag = ", ".join([f"MX{mx}_MY{my}" for mx, my in selected_mass_points])
    title_suffix = f"{year_tag}; {mass_tag}"

    print("\nStarting overlays")
    print(f"Years: {args.years}")
    print(f"Mass points: {selected_mass_points}")
    print(f"Features: {selected_features}")
    print(f"Output: {args.outdir}\n")

    for var in selected_features:
        signal_data_by_label = {}
        for mx, my in selected_mass_points:
            label = f"MX{mx}_MY{my}"
            df_sel = signal_df[(signal_df["MX"] == mx) & (signal_df["MY"] == my)]
            signal_data_by_label[label] = finite_values(df_sel, var)

        print(f"Plotting overlay: {var}")
        plot_overlay(var, signal_data_by_label, gg_df, gjet_df, args.outdir, title_suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
