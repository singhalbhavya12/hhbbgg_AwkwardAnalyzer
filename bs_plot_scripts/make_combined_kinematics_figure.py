import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


DEFAULT_BASE_DIR = Path("/eos/user/b/bsinghal/analysis/www/CUA/XYH/signal/kinematics/2022preEE_2022postEE")
DEFAULT_GROUP_DIR = DEFAULT_BASE_DIR / "group_bf"
DEFAULT_INDIVIDUAL_DIR = DEFAULT_BASE_DIR / "individual"
DEFAULT_POINTS = [(280, 80), (500, 80), (750, 80)]


def build_image_path(feature, directory, mx=None, my=None):
    if mx is None or my is None:
        return directory / f"{feature}.png"
    return directory / f"{feature}_MX{mx}_MY{my}.png"


def load_image(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(path)


def add_panel(ax, image_path, title):
    ax.imshow(load_image(image_path))
    ax.axis("off")
    ax.set_title(title, fontsize=13)


def discover_group_features(group_dir):
    return sorted(path.stem for path in group_dir.glob("*.png"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine grouped and individual kinematics PNGs into one 2x2 figure per feature."
    )
    parser.add_argument("--feature", default=None, help="Optional single feature to combine")
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR, help="Directory containing grouped PNGs")
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=DEFAULT_INDIVIDUAL_DIR,
        help="Directory containing individual PNGs",
    )
    parser.add_argument(
        "--points",
        nargs=6,
        type=int,
        default=[280, 80, 500, 80, 750, 80],
        metavar=("MX1", "MY1", "MX2", "MY2", "MX3", "MY3"),
        help="Three (MX, MY) points for the non-grouped panels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "combined",
        help="Directory where the combined figures will be written",
    )
    return parser.parse_args()


def make_combined_figure(feature, group_dir, individual_dir, points, output_dir):
    grouped_path = build_image_path(feature, group_dir)
    panel_paths = [
        grouped_path,
        build_image_path(feature, individual_dir, *points[0]),
        build_image_path(feature, individual_dir, *points[1]),
        build_image_path(feature, individual_dir, *points[2]),
    ]

    labels = [
        "Grouped",
        f"MX{points[0][0]} MY{points[0][1]}",
        f"MX{points[1][0]} MY{points[1][1]}",
        f"MX{points[2][0]} MY{points[2][1]}",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for ax, path, label in zip(axes.flat, panel_paths, labels):
        add_panel(ax, path, label)

    fig.suptitle(feature, fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{feature}_combined.png"
    out_pdf = output_dir / f"{feature}_combined.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def main():
    args = parse_args()

    points = [(args.points[i], args.points[i + 1]) for i in range(0, len(args.points), 2)]

    features = [args.feature] if args.feature else discover_group_features(args.group_dir)
    if not features:
        raise RuntimeError(f"No grouped PNG files found in {args.group_dir}")

    for feature in features:
        grouped_path = build_image_path(feature, args.group_dir)
        if not grouped_path.exists():
            print(f"Skipping {feature}: missing grouped image {grouped_path}")
            continue

        if any(not build_image_path(feature, args.individual_dir, *point).exists() for point in points):
            print(f"Skipping {feature}: one or more individual images are missing")
            continue

        make_combined_figure(feature, args.group_dir, args.individual_dir, points, args.output_dir)


if __name__ == "__main__":
    main()