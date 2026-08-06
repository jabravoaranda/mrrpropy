from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mrrpropy.rain_process_classification.hexagram import generate_rgb_hex


def main() -> None:
    output_dir = Path("workbench/output/poster")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rgb_hexagram_k11_transparent_font24.png"

    k = 11
    valid_threshold = -0.5
    r_hex, g_hex, b_hex, _ = generate_rgb_hex(k=k)
    r = np.asarray(r_hex, dtype=float)
    g = np.asarray(g_hex, dtype=float)
    b = np.asarray(b_hex, dtype=float)
    valid = (r > valid_threshold) & (g > valid_threshold) & (b > valid_threshold)

    rgba = np.zeros((*r.shape, 4), dtype=float)
    rgba[..., 0] = np.clip(r, 0.0, 1.0)
    rgba[..., 1] = np.clip(g, 0.0, 1.0)
    rgba[..., 2] = np.clip(b, 0.0, 1.0)
    rgba[..., 3] = valid.astype(float)

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_title("RGB Hexagram | k=11", pad=10)
    ax.set_xlabel("hex_x")
    ax.set_ylabel("hex_y")
    ax.set_xlim(-0.5, rgba.shape[1] - 0.5)
    ax.set_ylim(-0.5, rgba.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
