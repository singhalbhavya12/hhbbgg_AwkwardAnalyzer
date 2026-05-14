import matplotlib.pyplot as plt
import os

# Define the mass points
mass_points = {
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

# Custom dark color palette (manually selected)
dark_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#393b79', '#637939',
    '#8c6d31', '#843c39', '#7b4173', '#3182bd'
]

# Assign colors to each unique Y
unique_y_values = sorted({y for ys in mass_points.values() for y in ys})
y_color_map = {y: dark_colors[i % len(dark_colors)] for i, y in enumerate(unique_y_values)}


def make_table(headers, rows):
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)) + " |")
    lines.append(sep)
    return "\n".join(lines)

# Setup the plot
plt.figure(figsize=(8, 6))

# Plot each point
mp = 0
records = []
for mX, mYs in mass_points.items():
    for mY in mYs:
        boost_factor = mX / (mY + 125)
        plt.scatter(mX, boost_factor, color=y_color_map[mY], s=40)
        records.append((mX, mY, boost_factor))
        mp = mp+1

# Labels and formatting
plt.xlabel(r'$m_X$ (Spin-0) [GeV]', fontsize=14, loc='right')
plt.ylabel(r'Boost factor: $m_X / (m_Y + 125)$', fontsize=14, loc='top')

# Legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=y_color_map[y], markeredgecolor='black',
                      markersize=8, label=str(y)) for y in unique_y_values]
# plt.legend(handles=handles, title='Y', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.legend(
    handles=handles,
    title='Y',
    title_fontsize=12,
    fontsize=11,
    markerscale=1.5,              # Bigger legend markers
    handlelength=2.0,             # Longer lines for markers
    handletextpad=0.8,            # Space between marker and label
    borderpad=1.0,                # Padding inside legend border
    labelspacing=0.7,             # Space between labels
    bbox_to_anchor=(1.05, 1),     # Push legend out to the right
    loc='upper left',
    frameon=True                  # Optional: adds border box
)
plt.title(f" Low mass region, {mp} mass points")
plt.tight_layout()

output_dir = "/eos/user/b/bsinghal/analysis/bbgg_low_res/hhbbgg_AwkwardAnalyzer/bs_plot_scripts"#"/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/boost_factor1.png")
# plt.savefig(f"{output_dir}/boost_factor.pdf")

# Build detailed table
records_sorted = sorted(records, key=lambda x: (x[1], x[0]))
detailed_headers = ["mX [GeV]", "mY [GeV]", "Boost factor"]
detailed_rows = [(mX, mY, f"{bf:.4f}") for mX, mY, bf in records_sorted]
detailed_table = make_table(detailed_headers, detailed_rows)

# Category-wise summary by mY
my_summary = []
for mY in unique_y_values:
    bfs = [bf for mX, y, bf in records if y == mY]
    my_summary.append((
        mY,
        len(bfs),
        f"{min(bfs):.4f}",
        f"{max(bfs):.4f}",
        f"{sum(bfs)/len(bfs):.4f}",
    ))

my_summary_headers = ["mY category [GeV]", "N", "BF min", "BF max", "BF avg"]
my_summary_table = make_table(my_summary_headers, my_summary)

# Category-wise summary by boost-factor range
bf_ranges = [
    ("lowX  (BF <= 2.0)",        lambda bf: bf <= 2.0),
    ("midX  (2.0 < BF <= 3.0)",  lambda bf: 2.0 < bf <= 3.0),
    ("highX (3.0 < BF <= 5.0)",  lambda bf: 3.0 < bf <= 5.0),
    ("boosted (BF > 5.0)",        lambda bf: bf > 5.0),
]

range_summary = []
for label, selector in bf_ranges:
    selected = [bf for _, _, bf in records if selector(bf)]
    range_summary.append((
        label,
        len(selected),
        f"{min(selected):.4f}" if selected else "-",
        f"{max(selected):.4f}" if selected else "-",
        f"{(sum(selected)/len(selected)):.4f}" if selected else "-",
    ))

range_summary_headers = ["Boost-factor category", "N", "BF min", "BF max", "BF avg"]
range_summary_table = make_table(range_summary_headers, range_summary)

# X-mass category table per mY value
def x_cat(bf):
    if bf <= 2.0:
        return "lowX"
    elif bf <= 3.0:
        return "midX"
    elif bf <= 5.0:
        return "highX"
    else:
        return "boosted"

x_cat_headers = ["mY [GeV]", "lowX (BF<=2)", "midX (2<BF<=3)", "highX (3<BF<=5)", "boosted (BF>5)"]
x_cat_rows = []
for mY in unique_y_values:
    bins = {"lowX": [], "midX": [], "highX": [], "boosted": []}
    for mX, y, bf in records:
        if y == mY:
            bins[x_cat(bf)].append(mX)
    x_cat_rows.append((
        mY,
        ", ".join(str(v) for v in sorted(bins["lowX"]))    or "-",
        ", ".join(str(v) for v in sorted(bins["midX"]))    or "-",
        ", ".join(str(v) for v in sorted(bins["highX"]))   or "-",
        ", ".join(str(v) for v in sorted(bins["boosted"])) or "-",
    ))
x_cat_table = make_table(x_cat_headers, x_cat_rows)

# Write report to TXT
txt_path = f"{output_dir}/boost_factor_report1.txt"
with open(txt_path, "w") as f:
    f.write(f"Low mass region, {mp} mass points\n\n")
    f.write("Detailed boost factors (point-wise):\n")
    f.write(detailed_table)
    f.write("\n\nCategory-wise summary by mY:\n")
    f.write(my_summary_table)
    f.write("\n\nCategory-wise summary by boost-factor range:\n")
    f.write(range_summary_table)
    f.write("\n\nmX categories per mY (based on boost factor):\n")
    f.write("  lowX: BF <= 2.0 | midX: 2.0 < BF <= 3.0 | highX: 3.0 < BF <= 5.0 | boosted: BF > 5.0\n\n")
    f.write(x_cat_table)
    f.write("\n")

print(f"Saved plot: {output_dir}/boost_factor1.png")
# print(f"Saved plot: {output_dir}/boost_factor1.pdf")
print(f"Saved report: {txt_path}")
print("\nCategory-wise summary by mY:")
print(my_summary_table)
print("\nCategory-wise summary by boost-factor range:")
print(range_summary_table)
# plt.show()
# print(f"Number of signal mass points considered for Low mass analysis = {mp}")