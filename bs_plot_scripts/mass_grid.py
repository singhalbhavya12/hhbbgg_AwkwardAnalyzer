import matplotlib.pyplot as plt

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
    550: [60,70, 80, 90, 95, 100],
    600: [60,70, 80, 90, 95, 100],
    650: [70, 80, 90, 95, 100],
    700: [70, 80, 90, 95, 100],
    750: [80, 90, 95, 100],
    800: [80, 90, 95, 100],
    850: [90, 95, 100],
    900: [90, 95, 100],
    950: [95, 100],
    1000: [100],
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

# Setup the plot
plt.figure(figsize=(8, 6))

# Plot each point
mp = 0
for mX, mYs in mass_points.items():
    for mY in mYs:
        plt.scatter(mX, mY, color=y_color_map[mY], s=40)
        mp = mp+1

# Labels and formatting
plt.xlabel(r'$m_X$ (Spin-0) [GeV]', fontsize=14, loc='right')
plt.ylabel(r'$m_Y$ [GeV]', fontsize=14, loc='top')

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
plt.savefig("/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/grid_point_mx_1000_dark.png")
plt.savefig("/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/grid_point_mx_1000_dark.pdf")
# plt.show()
# print(f"Number of signal mass points considered for Low mass analysis = {mp}")