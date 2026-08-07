import os
import sys
import re
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from binning import binning

def make_bin_edges_fixed(binning_dict, var, fixed_nbins=50):
    """Create fixed number of bins for all variables using their xmin, xmax from binning dict.
    fixed_nbins: number of bins to use for all variables (default 50).
    """
    nb, xmin, xmax = binning_dict[var]
    return np.linspace(xmin, xmax, fixed_nbins + 1)

def clip_to_edges(x, edges):
    """Option B: clip values to [xmin, xmax] so under/overflow go into edge bins."""
    xmin, xmax = edges[0], edges[-1]
    return np.clip(x, xmin, xmax)

def make_adaptive_edges(x, num_bins):
    """
    Create bin edges that cover the full range of data.
    Takes full data range and divides into num_bins equal-width bins.
    """
    x = np.asarray(x)
    m = np.isfinite(x)
    if not np.any(m):
        return np.linspace(0, 1, num_bins + 1)
    
    xmin, xmax = np.nanmin(x[m]), np.nanmax(x[m])
    
    # Add small padding to ensure xmax is included
    if xmax == xmin:
        xmin -= 0.5
        xmax += 0.5
    else:
        padding = (xmax - xmin) * 0.01  # 1% padding
        xmin -= padding
        xmax += padding
    
    return np.linspace(xmin, xmax, num_bins + 1)

def weighted_density_hist(x, w, edges):
    """
    Returns y (density) evaluated per-bin, such that integral over x is 1.
    Equivalent to density=True but with explicit control + clipping.
    """
    x = np.asarray(x)
    if w is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    # finite mask
    m = np.isfinite(x) & np.isfinite(w)
    x = x[m]; w = w[m]
    if x.size == 0:
        return np.zeros(len(edges) - 1)

    x = clip_to_edges(x, edges)

    counts, _ = np.histogram(x, bins=edges, weights=w)
    binw = np.diff(edges)

    total_w = np.sum(w)
    if total_w <= 0:
        return np.zeros(len(edges) - 1)

    # density: counts / (total_w * bin_width)
    return counts / (total_w * binw)

def weighted_density_hist_adaptive(x, w, num_bins):
    """Density histogram using adaptive bins that expand to cover full data range."""
    x = np.asarray(x)
    if w is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w)
    x = x[m]; w = w[m]
    if x.size == 0:
        return np.linspace(0, 1, num_bins + 1), np.zeros(num_bins)

    edges = make_adaptive_edges(x, num_bins)
    counts, _ = np.histogram(x, bins=edges, weights=w)
    binw = np.diff(edges)
    total_w = np.sum(w)
    if total_w <= 0:
        return edges, np.zeros(len(edges) - 1)

    y = counts / (total_w * binw)
    return edges, y

def weighted_density_hist_inrange(x, w, edges):
    """
    Density histogram normalized only by in-range events.
    Events outside edges are excluded from both histogram and normalization.
    Returns density such that integral of in-range data equals 1.
    """
    x = np.asarray(x)
    if w is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    # finite mask
    m = np.isfinite(x) & np.isfinite(w)
    x = x[m]; w = w[m]
    if x.size == 0:
        return np.zeros(len(edges) - 1)

    # Only keep values within edges (no clipping)
    xmin, xmax = edges[0], edges[-1]
    in_range = (x >= xmin) & (x <= xmax)
    x_inrange = x[in_range]
    w_inrange = w[in_range]
    
    if x_inrange.size == 0:
        return np.zeros(len(edges) - 1)

    counts, _ = np.histogram(x_inrange, bins=edges, weights=w_inrange)
    binw = np.diff(edges)
    
    total_w = np.sum(w_inrange)
    if total_w <= 0:
        return np.zeros(len(edges) - 1)

    # density: counts / (total_w * bin_width)
    return counts / (total_w * binw)

def step_plot(ax, edges, y, **kwargs):
    """Convenience for step plotting using left edges."""
    ax.step(edges[:-1], y, where="post", **kwargs)

def filled_plot(ax, edges, y, **kwargs):
    ax.fill_between(edges[:-1], y, step="post", **kwargs)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Dataframes
#backgrounds(ggJets, and gJetPt)
parent_dir = "/eos/user/b/bsinghal/analysis/hhbbgg_AwkwardAnalyzer/outputfiles/"
background_files = [
    (f"{parent_dir}22EE/bkg_trees_merged.root", "/GGJets_MGG-80/preselection"),
    (f"{parent_dir}22EE/bkg_trees_merged.root", "/GJet_PT-20to40_DoubleEMEnriched_MGG-80/preselection"),
    (f"{parent_dir}22EE/bkg_trees_merged.root", "/GJet_PT-40_DoubleEMEnriched_MGG-80/preselection"),
]
# low X boost factor <~2 240-350
lowX = [
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X240_Y100/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X280_Y100/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X300_Y100/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X350_Y100/preselection"),
]

# mid X 1.5<~ boost factor <~3 400-600
midX = [
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X400_Y100/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X450_Y100/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y100/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X500_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X550_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X550_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X550_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X550_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X550_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X550_Y100/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y50/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X600_Y100/preselection"),
    
]


# high X 1.5<~ boost factor <~3 650-1000
highX = [
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X650_Y100/preselection"),
    # X = 700
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y60/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X700_Y100/preselection"),

    # X = 750
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y60/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X750_Y100/preselection"),

    # X = 800
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y60/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y70/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X800_Y100/preselection"),

    # X = 850
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y60/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y70/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X850_Y100/preselection"),

    # X = 900
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y60/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y70/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y80/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X900_Y100/preselection"),

    # X = 950
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y60/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y70/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y80/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y90/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X950_Y100/preselection"),

    # X = 1000
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y50/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y60/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y70/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y80/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y90/preselection"),
    # (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y95/preselection"),
    (f"{parent_dir}22EE/hhbbgg_trees.root", "/NMSSM_X1000_Y100/preselection"),
]

# Load DataFrames
keys = ["weight_preselection",
    #photon_diphoton
    "lead_pho_eta", "lead_pho_phi",
    "sublead_pho_eta", "sublead_pho_phi",
    #jets dijet and HY
    "diphoton_eta", "diphoton_phi",
    "lead_bjet_eta", "lead_bjet_phi",
    "sublead_bjet_eta", "sublead_bjet_phi",
    "dibjet_eta", "dibjet_phi",
    "bbgg_eta","bbgg_phi",
    # ΔR
    "DeltaR_j1g1","DeltaR_j2g1","DeltaR_j1g2","DeltaR_j2g2", #missing Res DeltaR jg min (minimum ∆R(j, γ))
    #Helicity and Collins-Soper angles
    "CosThetaStar_CS","CosThetaStar_jj","CosThetaStar_gg",
    #Object ID variables
    "lead_pho_mvaID", "sublead_pho_mvaID", "lead_bjet_PNetB", "sublead_bjet_PNetB",
    #Event-level counts and MET
    "n_jets", "puppiMET_pt","puppiMET_phi", #missing n_leptons
    #Azimuthal Separations with MET
    "DeltaPhi_j1MET", "DeltaPhi_j2MET",
    #chi^2 constraints
    "Res_chi_t0", "Res_chi_t1",
    #Raw kinematics and masses
    "dibjet_pt", "dibjet_mass", "Res_dijet_mass_DNNreg", "bbgg_mass", "bbgg_pt", "diphoton_mass", "diphoton_pt",
    #Scaled pt over mass ratios
    "lead_pt_over_diphoton_mass", "lead_pt_over_dibjet_mass",
]
dfs = {}

# Load lowX files
for file, key in lowX:
    try:
        with uproot.open(file) as f:
            dfs[key] = f[key].arrays(keys, library="pd")
    except Exception as e:
        print(f"Error loading {file} with key {key}: {e}")

# Load midX files
for file, key in midX:
    try:
        with uproot.open(file) as f:
            dfs[key] = f[key].arrays(keys, library="pd")
    except Exception as e:
        print(f"Error loading {file} with key {key}: {e}")
        
# Load highX files
for file, key in highX:
    try:
        with uproot.open(file) as f:
            dfs[key] = f[key].arrays(keys, library="pd")
    except Exception as e:
        print(f"Error loading {file} with key {key}: {e}")
        

# Load background files
for file, key in background_files:
    try:
        with uproot.open(file) as f:
            dfs[key] = f[key].arrays(keys, library="pd")
    except Exception as e:
        print(f"Error loading {file} with key {key}: {e}")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Combine “low/mid/high” and also keep individual signal points
def is_nmssm_key(k):
    return k.startswith("/NMSSM_") and k.endswith("/preselection")

def masspoint_label_from_key(k):
    # "/NMSSM_X650_Y70/preselection" -> "NMSSM_X650_Y70"
    return k.strip("/").split("/")[0]

signal_keys = [k for k in dfs.keys() if is_nmssm_key(k)]
bkg_gg_keys = [k for k in dfs.keys() if k.startswith("/GGJets") and k.endswith("/preselection")]
bkg_gj_keys = [k for k in dfs.keys() if k.startswith("/GJet_") and k.endswith("/preselection")]

# individual signals dict: label -> df
signal_dfs = {masspoint_label_from_key(k): dfs[k] for k in signal_keys}

# groupings (you can keep your OR logic, but list-based is cleaner)
lowX_vals  = [240, 280, 300, 350]
midX_vals  = [400, 450, 500, 550, 600]
highX_vals = [650, 700, 750, 800, 850, 900, 950, 1000]

def in_X_list(k, xs):
    return any(f"X{x}" in k for x in xs)

lowX_df  = pd.concat([dfs[k] for k in signal_keys if in_X_list(k, lowX_vals)],  ignore_index=True)
midX_df  = pd.concat([dfs[k] for k in signal_keys if in_X_list(k, midX_vals)],  ignore_index=True)
highX_df = pd.concat([dfs[k] for k in signal_keys if in_X_list(k, highX_vals)], ignore_index=True)

ggJets_df = pd.concat([dfs[k] for k in bkg_gg_keys], ignore_index=True)
gJetPt_df = pd.concat([dfs[k] for k in bkg_gj_keys], ignore_index=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Plotting functions



def plot_groups_inrange(variable, binning_dict, outdir, tag="22postEE", fixed_nbins=50):
    """Plot grouped signals with backgrounds using in-range normalization.
    Only events within bin edges are used for both histogram and weight normalization.
    Uses fixed number of bins across all variables for consistency (area normalized to 1).
    fixed_nbins: number of bins for all variables (default 50)
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "group_inrange"), exist_ok=True)

    edges = make_bin_edges_fixed(binning_dict, variable, fixed_nbins)
    y_low = weighted_density_hist_inrange(lowX_df[variable], lowX_df["weight_preselection"], edges)
    y_mid = weighted_density_hist_inrange(midX_df[variable], midX_df["weight_preselection"], edges)
    y_high = weighted_density_hist_inrange(highX_df[variable], highX_df["weight_preselection"], edges)
    y_gg = weighted_density_hist_inrange(ggJets_df[variable], ggJets_df["weight_preselection"], edges)
    y_gj = weighted_density_hist_inrange(gJetPt_df[variable], gJetPt_df["weight_preselection"], edges)

    fig, ax = plt.subplots(figsize=(8, 6))

    step_plot(ax, edges, y_low,  label="lowX",  linestyle="-",  linewidth=1.6)
    step_plot(ax, edges, y_mid,  label="midX",  linestyle="--", linewidth=1.6)
    step_plot(ax, edges, y_high, label="highX", linestyle="-.", linewidth=1.6)

    filled_plot(ax, edges, y_gg, label=r"$\gamma\gamma$+Jets", alpha=0.25)
    filled_plot(ax, edges, y_gj, label=r"$\gamma$+Jets",      alpha=0.25)

    ax.set_xlabel(variable)
    ax.set_ylabel("Density (area=1 in-range)")
    ax.set_title(f"{tag} {variable} (low/mid/high + bkg) [In-Range Norm]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(outdir, f"group_inrange/{variable}_groups.png"))
    fig.savefig(os.path.join(outdir, f"group_inrange/{variable}_groups.pdf"))
    plt.close(fig)


def plot_each_signal_with_bkg(variable, binning_dict, outdir, tag="22postEE", fixed_nbins=50):
    os.makedirs(outdir, exist_ok=True)
    edges = make_bin_edges_fixed(binning_dict, variable, fixed_nbins)

    # precompute backgrounds once (saves time)
    y_gg = weighted_density_hist_inrange(ggJets_df[variable], ggJets_df["weight_preselection"], edges)
    y_gj = weighted_density_hist_inrange(gJetPt_df[variable], gJetPt_df["weight_preselection"], edges)

    for sig_label, df in signal_dfs.items():
        if variable not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        y_sig = weighted_density_hist_inrange(df[variable], df["weight_preselection"], edges)
        step_plot(ax, edges, y_sig, label=sig_label, linewidth=1.8)

        filled_plot(ax, edges, y_gg, label=r"$\gamma\gamma$+Jets", alpha=0.25)
        filled_plot(ax, edges, y_gj, label=r"$\gamma$+Jets",      alpha=0.25)

        ax.set_xlabel(variable)
        ax.set_ylabel("Density (area=1)")
        ax.set_title(f"{tag} {variable} ({sig_label} + bkg)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()

        fig.savefig(os.path.join(outdir, f"individual/{variable}_{sig_label}.png"))
        fig.savefig(os.path.join(outdir, f"individual/{variable}_{sig_label}.pdf"))
        plt.close(fig)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---- choose only the variables you want ----
vars_to_plot = [
    #photon_diphoton
    "lead_pho_eta", "lead_pho_phi",
    "sublead_pho_eta", "sublead_pho_phi",
    #jets dijet and HY
    "diphoton_eta", "diphoton_phi",
    "lead_bjet_eta", "lead_bjet_phi",
    "sublead_bjet_eta", "sublead_bjet_phi",
    "dibjet_eta", "dibjet_phi",
    "bbgg_eta","bbgg_phi",
    # ΔR
    "DeltaR_j1g1","DeltaR_j2g1","DeltaR_j1g2","DeltaR_j2g2", #missing Res DeltaR jg min (minimum ∆R(j, γ))
    #Helicity and Collins-Soper angles
    "CosThetaStar_CS","CosThetaStar_jj","CosThetaStar_gg",
    #Object ID variables
    "lead_pho_mvaID", "sublead_pho_mvaID", "lead_bjet_PNetB", "sublead_bjet_PNetB",
    #Event-level counts and MET
    "n_jets", "puppiMET_pt","puppiMET_phi", #missing n_leptons
    #Azimuthal Separations with MET
    "DeltaPhi_j1MET", "DeltaPhi_j2MET",
    #chi^2 constraints
    "Res_chi_t0", "Res_chi_t1",
    #Raw kinematics and masses
    "dibjet_pt", "dibjet_mass", "Res_dijet_mass_DNNreg", "bbgg_mass", "bbgg_pt", "diphoton_mass", "diphoton_pt",
    #Scaled pt over mass ratios
    "lead_pt_over_diphoton_mass", "lead_pt_over_dibjet_mass",
]

# Optional: enforce they exist in your binning dict
vars_to_plot = [v for v in vars_to_plot if v in binning["preselection"]]

outdir = "/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/kinematics/"

# Fixed number of bins to use for all variables (ensures consistent binning across all plots)
FIXED_NBINS = 50

for variable in vars_to_plot:
    # (Optional) skip if not present in at least one DF (avoids KeyError)
    # You can comment this out if you're sure they're present everywhere.
    if variable not in lowX_df.columns:
        print(f"[skip] {variable} not found in loaded columns")
        continue

    # In-range normalization with fixed binning (area normalized to 1)
    plot_groups_inrange(variable, binning["preselection"], outdir, tag="2022 postEE", fixed_nbins=FIXED_NBINS)
    
    # plot_each_signal_with_bkg(variable, binning["preselection"], outdir, tag="2022 postEE")
    print(f"[done] {variable}")
