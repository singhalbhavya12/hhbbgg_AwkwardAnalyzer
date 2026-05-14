#!/usr/bin/env python3
"""
Diagnose why some features show r=1.00 (perfect correlation).

Common causes:
1. Features are constant-valued (all same number → std=0 → correlation undefined)
2. Features are filled with NaN → Pearson drops NaN automatically → small sample
3. Features are perfectly linearly related (one is 2x another)
4. Data structure issue: all events at boundary values
"""

import pandas as pd
import numpy as np
import sys

# Suspicious features that showed r=1.00
SUSPICIOUS = [
    "Res_CosThetaStar_jj",
    "Res_CosThetaStar_gg",
    "Res_CosThetaStar_CS",
    "Res_DeltaR_jg_min",
    "Res_DeltaR_j1g1",
    "Res_DeltaR_j1g2",
    "Res_DeltaR_j2g1",
    "Res_DeltaR_j2g2",
    "Res_lead_bjet_btagPNetB",
    "Res_sublead_bjet_btagPNetB",
]

def inspect_feature(df, name):
    """Print statistics for one feature."""
    if name not in df.columns:
        print(f"  ✗ Column '{name}' not in DataFrame")
        return
    
    vals = df[name].values
    n_total = len(vals)
    n_nan = np.isnan(vals).sum()
    n_valid = n_total - n_nan
    
    if n_valid == 0:
        print(f"  ✗ ALL NaN ({n_total} rows)")
        return
    
    valid_vals = vals[~np.isnan(vals)]
    
    print(f"  Total: {n_total:,} | NaN: {n_nan:,} ({100*n_nan/n_total:.1f}%) | Valid: {n_valid:,}")
    print(f"  Mean: {valid_vals.mean():.6f}, Std: {valid_vals.std():.6f}")
    print(f"  Min: {valid_vals.min():.6f}, Max: {valid_vals.max():.6f}")
    print(f"  Unique values: {len(set(valid_vals))}")
    
    # Check for extreme clustering
    val_counts = pd.Series(valid_vals).value_counts()
    top3 = val_counts.head(3)
    print(f"  Top 3 values:")
    for val, count in top3.items():
        pct = 100 * count / n_valid
        print(f"    {val:.6f}: {count:,} ({pct:.1f}%)")
    
    # If std is very small, check if constant
    if valid_vals.std() < 1e-6:
        print(f"  ⚠️  CONSTANT-VALUED (std ≈ 0)")
    
    # Check correlation with itself (should be 1.0)
    auto_corr = np.corrcoef(valid_vals, valid_vals)[0, 1]
    print(f"  Auto-correlation: {auto_corr:.4f}")


print("=" * 80)
print("SIGNAL: Load sample")
print("=" * 80)

sig_fp = "/eos/user/b/bsinghal/analysis/output/2022preEE/merged/NMSSM-XtoYH-MX-300-MY-90/nominal/NOTAG_merged.parquet"
cols_to_read = SUSPICIOUS + ["is_Res"]

try:
    sig = pd.read_parquet(sig_fp, columns=cols_to_read)
    print(f"Loaded {len(sig)} signal events")
except Exception as e:
    print(f"Failed to load signal: {e}")
    sys.exit(1)

print("\nBEFORE is_Res filter:")
for col in SUSPICIOUS:
    print(f"\n{col}:")
    inspect_feature(sig, col)

print("\n" + "=" * 80)
print("AFTER is_Res == 1 filter:")
print("=" * 80)

sig_filt = sig[sig["is_Res"] == 1].copy()
print(f"After filtering: {len(sig_filt)} events (dropped {len(sig) - len(sig_filt)})")

for col in SUSPICIOUS:
    print(f"\n{col}:")
    inspect_feature(sig_filt, col)

# Check correlations among the suspicious features
print("\n" + "=" * 80)
print("CORRELATION MATRIX (filtered signal)")
print("=" * 80)

corr_matrix = sig_filt[SUSPICIOUS].corr()
print("\nFull correlation matrix:")
print(corr_matrix)

# Find any r=1.00 pairs
print("\nPerfectly correlated pairs (|r| ≥ 0.9999):")
found_perfect = False
for i, col1 in enumerate(SUSPICIOUS):
    for j, col2 in enumerate(SUSPICIOUS):
        if i >= j:  # avoid duplicates and self-correlation
            continue
        corr_val = corr_matrix.loc[col1, col2]
        if abs(abs(corr_val) - 1.0) < 0.0001:  # essentially 1.00
            print(f"  {col1:40s} <--> {col2:40s}: r = {corr_val:.6f}")
            found_perfect = True

if not found_perfect:
    print("  (none found)")

print("\n" + "=" * 80)
print("BACKGROUND: GGJets sample")
print("=" * 80)

bkg_fp = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/preEE/GGJets_MGG-80/nominal/NOTAG_merged.parquet"

try:
    bkg = pd.read_parquet(bkg_fp, columns=cols_to_read)
    print(f"Loaded {len(bkg)} bkg events")
except Exception as e:
    print(f"Failed to load bkg: {e}")
    sys.exit(1)

bkg_filt = bkg[bkg["is_Res"] == 1].copy()
print(f"After is_Res==1: {len(bkg_filt)} events (dropped {len(bkg) - len(bkg_filt)})")

print("\nBackground feature statistics (filtered):")
for col in SUSPICIOUS[:5]:  # just show first 5
    print(f"\n{col}:")
    inspect_feature(bkg_filt, col)

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
