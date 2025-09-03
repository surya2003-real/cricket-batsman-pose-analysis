import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Wedge, Circle, Rectangle
from matplotlib.colors import Normalize


def plot_shot_distribution_gradient(dist_df, output_path):
    """
    Generate separate colored-distribution plots for high- and low-intensity shots,
    saving two files with `_high` and `_low` suffixes.

    Args:
        dist_df (pd.DataFrame): DataFrame containing 'Side','High_Shots','Low_Shots'
            or with index named 'Side', matching region names case-insensitively.
        output_path (str): base filepath for saving PNGs; '_high' and '_low' appended.
    """
    # Prepare output filenames
    base, ext = os.path.splitext(output_path)
    output_high = f"{base}_high{ext}"
    output_low = f"{base}_low{ext}"

    # Normalize DataFrame for case-insensitive lookup
    df = dist_df.copy()
    if 'Side' in df.columns:
        df['__side_key'] = df['Side'].astype(str).str.lower()
        df2 = df.set_index('__side_key')
    else:
        df.index = df.index.astype(str).str.lower()
        df2 = df

    # Define angular regions for field
    regions = {
        'cover':      (180, 225),
        'mid off':    (225, 270),
        'mid on':     (270, 315),
        'mid wicket': (315, 360),
        'square leg': (0,   45),
        'fine leg':   (45,  90),
        'third man':  (90,  135),
        'point':      (135, 180),
    }

    region_sides = {
        'cover': 'off side',
        'mid off': 'off side',
        'mid on': 'leg side',
        'mid wicket': 'leg side',
        'square leg': 'leg side',
        'fine leg': 'leg side',
        'third man': 'off side',
        'point': 'off side'
    }

    # Plot for each intensity type
    for intensity, cmap_name, out_path in [
        ('High_Shots', 'Reds', output_high),
        ('Low_Shots',  'Blues', output_low)
    ]:
        # Gather values per region
        values = []
        for side in regions:
            if side in df2.index and intensity in df2.columns:
                try:
                    val = float(df2.at[side, intensity])
                except Exception:
                    val = 0.0
            else:
                val = 0.0
            values.append(val)

        # Normalize values for color mapping
        max_val = max(values)
        norm = Normalize(vmin=0, vmax=max_val if max_val > 0 else 1)
        cmap = plt.get_cmap(cmap_name)

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 7))
        # Field outline
        ax.add_patch(Circle((0, 0), 1, fill=False, linewidth=2))
        ax.add_patch(Rectangle((-0.05, -0.5), 0.1, 1.0, fill=False, linewidth=1.5))
        ax.add_patch(Circle((0, 0), 0.02, color='black'))

        # Draw wedges
        for (side, (start, end)), val in zip(regions.items(), values):
            color = cmap(norm(val))
            ax.add_patch(Wedge((0, 0), 1.0, start, end, facecolor=color, edgecolor='white'))
            mid = (start + end) / 2
            x = 0.7 * np.cos(np.deg2rad(mid))
            y = 0.7 * np.sin(np.deg2rad(mid))
            ax.text(x, y + 0.07, f"{val*100:.1f}%", ha='center', va='center', fontsize=8)
            ax.text(x, y - 0.05, f"{side}\n({region_sides[side]})", ha='center', va='center', fontsize=7)

        # Colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{intensity.replace('_', ' ')} Fraction")

        # Title and save
        title = "High-Intensity Shots" if intensity == 'High_Shots' else "Low-Intensity Shots"
        ax.set_title(f"{title} Distribution", pad=10)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
