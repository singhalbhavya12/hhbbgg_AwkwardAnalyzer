import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kinematics_lowmass import OUTDIR, SIGNAL_POINTS, YEAR_DATASETS

MVA_VARS = [
	"lead_mvaID",
	"sublead_mvaID",
	"lead_mvaID_WP90",
	"lead_mvaID_WP80",
	"sublead_mvaID_WP90",
	"sublead_mvaID_WP80",
]

# Prefer explicit background files per year for quick checks.
# Fill these with the actual parquet files you want to use.
BKG_FILES_BY_YEAR = {
	"2022preEE": [
		"/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/preEE/GGJets_MGG-80/nominal/NOTAG_merged.parquet",
		"/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/preEE/GJet_PT-20to40_DoubleEMEnriched_MGG-80/nominal/NOTAG_merged.parquet",
		"/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/preEE/GJet_PT-40_DoubleEMEnriched_MGG-80/nominal/NOTAG_merged.parquet",
	],
	"2022postEE": [
		"/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/postEE/GGJets_MGG-80/nominal/NOTAG_merged.parquet",
		"/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/postEE/GJet_PT-20to40_DoubleEMEnriched_MGG-80/nominal/NOTAG_merged.parquet",
		"/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v3/Run3_2022/sim/postEE/GJet_PT-40_DoubleEMEnriched_MGG-80/nominal/NOTAG_merged.parquet",
	],
}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Quick MVA-ID plotting using 2022preEE/2022postEE paths from kinematics_lowmass.py"
	)
	parser.add_argument("--mx", type=int, default=None, help="Signal MX (default: first signal point in kinematics_lowmass.py)")
	parser.add_argument("--my", type=int, default=None, help="Signal MY (default: first signal point in kinematics_lowmass.py)")
	parser.add_argument(
		"--outdir",
		default="/eos/user/b/bsinghal/analysis/bbgg_low_res/hhbbgg_AwkwardAnalyzer/bs_plot_scripts/mvaid",
		help="Output directory for plots (default: OUTDIR from kinematics_lowmass.py)",
	)
	parser.add_argument("--signal-label", default="Signal", help="Legend label for signal")
	parser.add_argument("--background-label", default="2022 GJet+GGJets", help="Legend label for background")
	parser.add_argument(
		"--filter-col",
		default="is_Res",
		help="Optional event filter column. Keep rows where this equals 1 if present.",
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=50,
		help="Number of bins for continuous variables",
	)
	return parser.parse_args()


def load_df(path, cols, filter_col):
	if not os.path.exists(path):
		raise FileNotFoundError(f"Input file does not exist: {path}")

	all_cols = pd.read_parquet(path, engine="pyarrow").columns
	requested = list(cols)
	if filter_col not in requested:
		requested.append(filter_col)

	available = [c for c in requested if c in all_cols]
	missing = [c for c in cols if c not in all_cols]
	if not available:
		raise ValueError(f"None of the requested columns were found in {path}")

	df = pd.read_parquet(path, columns=available)
	if filter_col in df.columns:
		df = df[df[filter_col] == 1].copy()

	for col in missing:
		print(f"WARNING: missing column in {path}: {col}")

	return df


def resolve_signal_point(mx_arg, my_arg):
	if mx_arg is None and my_arg is None:
		if not SIGNAL_POINTS:
			raise ValueError("No SIGNAL_POINTS found in kinematics_lowmass.py")
		return SIGNAL_POINTS[0]

	if mx_arg is None or my_arg is None:
		raise ValueError("Provide both --mx and --my, or provide neither.")

	return mx_arg, my_arg


def find_signal_file(year, mx, my):
	conf = YEAR_DATASETS.get(year)
	if conf is None:
		return None

	signal_dir = conf["signal_dir"]
	name_templates = conf.get("signal_name_templates", ["NMSSM-XtoYH-MX-{mx}-MY-{my}"])
	file_templates = [
		os.path.join("{signal_dir}", "{proc}", "nominal", "NOTAG_merged.parquet"),
		os.path.join("{signal_dir}", "{proc}", "NOTAG_merged.parquet"),
		os.path.join("{signal_dir}", "{proc}.parquet"),
	]

	for proc_tpl in name_templates:
		proc = proc_tpl.format(mx=mx, my=my)
		for fp_tpl in file_templates:
			fp = fp_tpl.format(signal_dir=signal_dir, proc=proc)
			if os.path.exists(fp):
				return fp

	return None


def collect_background_files(year):
	# First use explicit paths if provided in BKG_FILES_BY_YEAR.
	explicit_paths = BKG_FILES_BY_YEAR.get(year, [])
	if explicit_paths:
		resolved = []
		for fp in explicit_paths:
			if os.path.exists(fp):
				resolved.append(fp)
			else:
				print(f"WARNING: background file not found ({year}): {fp}")
		if resolved:
			return resolved
		print(f"WARNING: no explicit background files were found for {year}; trying YEAR_DATASETS fallback")

	# Fallback: infer paths from YEAR_DATASETS + bkg_processes.
	conf = YEAR_DATASETS.get(year)
	if conf is None:
		return []

	bkg_dir = conf["bkg_dir"]
	procs = conf.get("bkg_processes", [])
	paths = []
	file_templates = [
		os.path.join("{bkg_dir}", "{proc}", "nominal", "NOTAG_merged.parquet"),
		os.path.join("{bkg_dir}", "{proc}", "NOTAG_merged.parquet"),
		os.path.join("{bkg_dir}", "{proc}.parquet"),
	]
	for proc in procs:
		found = False
		for fp_tpl in file_templates:
			fp = fp_tpl.format(bkg_dir=bkg_dir, proc=proc)
			if os.path.exists(fp):
				paths.append(fp)
				found = True
				break
		if not found:
			print(f"WARNING: background file not found ({year}) for process {proc}")
	return paths


def finite_vals(df, col):
	if col not in df.columns:
		return np.array([])
	vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
	return vals[np.isfinite(vals)]


def choose_bins(sig_vals, bkg_vals, n_bins):
	all_vals = np.concatenate([a for a in [sig_vals, bkg_vals] if len(a) > 0])
	if len(all_vals) == 0:
		return None

	unique_vals = np.unique(all_vals)
	if np.all(np.isin(unique_vals, [0.0, 1.0])):
		return np.array([-0.5, 0.5, 1.5])

	vmin = np.percentile(all_vals, 1)
	vmax = np.percentile(all_vals, 99)
	if np.isclose(vmin, vmax):
		vmin, vmax = np.min(all_vals), np.max(all_vals)
	if np.isclose(vmin, vmax):
		vmin -= 1e-6
		vmax += 1e-6
	return np.linspace(vmin, vmax, n_bins)


def plot_var(var, sig_df, bkg_df, outdir, signal_label, background_label, n_bins):
	sig_vals = finite_vals(sig_df, var)
	bkg_vals = finite_vals(bkg_df, var)

	if len(sig_vals) == 0 and len(bkg_vals) == 0:
		print(f"Skipping {var}: no finite entries")
		return

	bins = choose_bins(sig_vals, bkg_vals, n_bins)
	if bins is None:
		print(f"Skipping {var}: could not choose bins")
		return

	fig, ax = plt.subplots(figsize=(8, 6))

	if len(sig_vals) > 0:
		ax.hist(
			sig_vals,
			bins=bins,
			density=True,
			histtype="step",
			linewidth=1.8,
			color="crimson",
			label=signal_label,
		)

	if len(bkg_vals) > 0:
		ax.hist(
			bkg_vals,
			bins=bins,
			density=True,
			histtype="stepfilled",
			alpha=0.35,
			color="steelblue",
			label=background_label,
		)

	ax.set_xlabel(var)
	ax.set_ylabel("Normalized to unit area")
	ax.set_title(f"MVA-ID check: {var}")
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()

	out_png = os.path.join(outdir, f"{var}.png")
	out_pdf = os.path.join(outdir, f"{var}.pdf")
	fig.savefig(out_png)
	fig.savefig(out_pdf)
	plt.close(fig)
	print(f"Saved: {out_png}")


def main():
	args = parse_args()
	mx, my = resolve_signal_point(args.mx, args.my)
	outdir = args.outdir if args.outdir else OUTDIR
	os.makedirs(outdir, exist_ok=True)

	years = ["2022preEE", "2022postEE"]

	sig_frames = []
	bkg_frames = []

	for year in years:
		signal_fp = find_signal_file(year, mx, my)
		if signal_fp is None:
			print(f"WARNING: signal not found for {year}, MX={mx}, MY={my}")
		else:
			print(f"Loading signal: {signal_fp}")
			sig_frames.append(load_df(signal_fp, MVA_VARS, args.filter_col))

		for bkg_fp in collect_background_files(year):
			print(f"Loading background: {bkg_fp}")
			bkg_frames.append(load_df(bkg_fp, MVA_VARS, args.filter_col))

	if not sig_frames:
		raise RuntimeError("No signal files were loaded for 2022preEE/2022postEE.")

	sig_df = pd.concat(sig_frames, ignore_index=True)
	if bkg_frames:
		bkg_df = pd.concat(bkg_frames, ignore_index=True)
	else:
		print("WARNING: no background files loaded; plotting signal-only distributions")
		bkg_df = pd.DataFrame(columns=MVA_VARS)

	if args.signal_label == "Signal":
		signal_label = f"MX{mx}_MY{my}"
	else:
		signal_label = args.signal_label

	print(f"Signal events after filtering: {len(sig_df):,}")
	print(f"Background events after filtering: {len(bkg_df):,}")
	print(f"Output directory: {outdir}")

	for var in MVA_VARS:
		plot_var(
			var,
			sig_df,
			bkg_df,
			outdir,
			signal_label,
			args.background_label,
			args.bins,
		)

	print("Done")


if __name__ == "__main__":
	main()
