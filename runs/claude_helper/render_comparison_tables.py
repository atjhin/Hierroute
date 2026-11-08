"""
Render the two Messy_background comparison tables (built by
build_comparison_tables.py) as styled PNG heatmap-tables instead of plain
markdown: cell color encodes accuracy (sequential blue ramp, light->dark =
low->high), and the best model in each row is bolded.

Run build_comparison_tables.py first -- this reads its CSV output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / 'runs' / 'logs'

# Sequential blue ramp, steps 100->700 (references/palette.md) -- light->dark.
BLUE_RAMP = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
    '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
    '#184f95', '#104281', '#0d366b',
]
CMAP = LinearSegmentedColormap.from_list('seq_blue', BLUE_RAMP)

TEXT_PRIMARY = '#0b0b0b'
TEXT_ON_DARK = '#ffffff'
HEADER_BG = '#f0efec'
GRID_LINE = '#000000'
GRID_LINEWIDTH = 1.2

TITLE_FONTSIZE = 15
HEADER_FONTSIZE = 12
CELL_FONTSIZE = 12
ROW_LABEL_FONTSIZE = 12


def text_width_in(text, fontsize, weight='normal'):
    """Rendered width of text at fontsize (pt), in inches -- backend-independent
    (no renderer/canvas needed), so this works before the real figure exists."""
    fp = FontProperties(size=fontsize, weight=weight)
    widths = [TextPath((0, 0), line, prop=fp).get_extents().width for line in text.split('\n')]
    return max(widths, default=0) / 72.0  # points -> inches


def relative_luminance(rgb):
    """WCAG relative luminance from a matplotlib (r,g,b[,a]) tuple in 0..1."""
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for(rgb):
    # White text has luminance 1.0; contrast ratio = (1+0.05)/(L_bg+0.05).
    # Switch to white once the black-on-bg contrast ratio drops below it.
    l_bg = relative_luminance(rgb)
    contrast_black = (l_bg + 0.05) / 0.05
    contrast_white = 1.05 / (l_bg + 0.05)
    return TEXT_ON_DARK if contrast_white > contrast_black else TEXT_PRIMARY


def wrap_header(label, max_chars=16):
    if len(label) <= max_chars:
        return label
    # split on the last space before max_chars, else the first space after
    idx = label.rfind(' ', 0, max_chars)
    if idx == -1:
        idx = label.find(' ')
    return label if idx == -1 else label[:idx] + '\n' + label[idx + 1:]


def render_table(df, title, out_path, summary_row=None, value_fmt='{:.3f}'):
    n_rows, n_cols = df.shape
    headers = [wrap_header(c) for c in df.columns]
    header_lines = max(h.count('\n') + 1 for h in headers)

    CELL_PAD = 0.25  # inches of horizontal breathing room on each side of a cell's text

    # Column width: just wide enough for its widest rendered line (header, bold since
    # header text is bold, or a bold cell value -- bold is always the wider weight) plus
    # padding, floored so no cell is exotically narrow.
    row_values = [value_fmt.format(v) for col in df.columns for v in df[col]]
    max_value_w = max((text_width_in(v, CELL_FONTSIZE, 'bold') for v in row_values), default=0)
    col_w = max(1.1, max_value_w + 2 * CELL_PAD)
    for header in headers:
        col_w = max(col_w, text_width_in(header, HEADER_FONTSIZE, 'bold') + 2 * CELL_PAD)

    row_h = 0.55
    header_h = row_h * max(1.4, 0.8 * header_lines + 0.6)
    label_texts = [str(n) for n in df.index]
    label_w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in label_texts) + 2 * CELL_PAD

    fig_w = label_w + col_w * n_cols
    fig_h = 0.9 + header_h + row_h * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY,
                 loc='left', x=0, pad=14)

    # --- header row (top-left corner cell + one cell per column, each bordered) ---
    header_y = 0
    ax.add_patch(Rectangle((0, header_y), label_w, header_h, facecolor=HEADER_BG,
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
    for j, header in enumerate(headers):
        x = label_w + col_w * j
        ax.add_patch(Rectangle((x, header_y), col_w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + col_w / 2, header_y + header_h / 2, header, ha='center', va='center',
                 fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)

    # --- body rows ---
    for i, (row_name, row) in enumerate(df.iterrows()):
        y = header_h + row_h * i
        is_summary = summary_row is not None and row_name == summary_row
        row_label_weight = 'bold' if is_summary else 'normal'

        ax.add_patch(Rectangle((0, y), label_w, row_h, facecolor='white',
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(CELL_PAD, y + row_h / 2, str(row_name), ha='left', va='center',
                 fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        best_col = row.idxmax()
        for j, col in enumerate(df.columns):
            val = row[col]
            x = label_w + col_w * j
            bg = CMAP(val)
            ax.add_patch(Rectangle((x, y), col_w, row_h, facecolor=bg,
                                    edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
            txt_color = text_color_for(bg)
            is_best = col == best_col
            ax.text(x + col_w / 2, y + row_h / 2, value_fmt.format(val),
                     ha='center', va='center', fontsize=CELL_FONTSIZE, color=txt_color,
                     fontweight='bold' if is_best else 'normal')

    # --- outer frame, drawn last so it sits crisp on top of every cell edge ---
    table_h = header_h + row_h * n_rows
    ax.add_patch(Rectangle((0, 0), fig_w, table_h, facecolor='none',
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH * 1.6, zorder=10))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


table1 = pd.read_csv(LOG_DIR / 'messy_background_leaf_accuracy.csv', index_col=0)
table2 = pd.read_csv(LOG_DIR / 'messy_background_level_accuracy.csv', index_col=0)

render_table(
    table1, 'Leaf-class accuracy on Messy_background (bold = best per row)',
    LOG_DIR / 'messy_background_leaf_accuracy.png', summary_row='Micro accuracy',
)
render_table(
    table2, 'Per-level accuracy on Messy_background (bold = best per row)',
    LOG_DIR / 'messy_background_level_accuracy.png',
)
