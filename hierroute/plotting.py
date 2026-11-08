"""
plotting.py — Styled PNG tables and stacked bar charts for evaluation results.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath

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
FOOTNOTE_FONTSIZE = 9


def text_width_in(text, fontsize, weight='normal'):
    fp = FontProperties(size=fontsize, weight=weight)
    widths = [TextPath((0, 0), line, prop=fp).get_extents().width for line in text.split('\n')]
    return max(widths, default=0) / 72.0


def relative_luminance(rgb):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for(rgb):
    l_bg = relative_luminance(rgb)
    contrast_black = (l_bg + 0.05) / 0.05
    contrast_white = 1.05 / (l_bg + 0.05)
    return TEXT_ON_DARK if contrast_white > contrast_black else TEXT_PRIMARY


def wrap_header(label, max_chars=16):
    if len(label) <= max_chars:
        return label
    idx = label.rfind(' ', 0, max_chars)
    if idx == -1:
        idx = label.find(' ')
    return label if idx == -1 else label[:idx] + '\n' + label[idx + 1:]


def render_table(df, title, out_path, summary_row=None, value_fmt='{:.3f}',
                  highlight_row_max=True, footnote=None, count_cols=frozenset(), label_cols=()):
    """
    label_cols: names of df columns to render as extra plain-text label columns (white cells,
    left-aligned, own width, never heat-mapped) between the index-label column and the
    heat-mapped value columns -- e.g. splitting a combined "Emily — 08-25" index into separate
    test_data / model_date columns. Every other column is treated as a numeric value column.
    """
    label_cols = list(label_cols)
    value_cols = [c for c in df.columns if c not in label_cols]

    summary_rows = {summary_row} if isinstance(summary_row, str) else set(summary_row or [])
    n_rows, n_value_cols = len(df), len(value_cols)
    headers = [wrap_header(c) for c in value_cols]
    label_headers = [wrap_header(c) for c in label_cols]
    header_lines = max([h.count('\n') + 1 for h in headers + label_headers], default=1)

    def format_val(col, val):
        if pd.isna(val):
            return 'N/A'
        return f'{int(val):,}' if col in count_cols else value_fmt.format(val)

    CELL_PAD = 0.25

    row_values = [format_val(col, v) for col in value_cols for v in df[col]]
    max_value_w = max((text_width_in(v, CELL_FONTSIZE, 'bold') for v in row_values), default=0)
    col_w = max(1.1, max_value_w + 2 * CELL_PAD)
    for header in headers:
        col_w = max(col_w, text_width_in(header, HEADER_FONTSIZE, 'bold') + 2 * CELL_PAD)

    row_h = 0.55
    header_h = row_h * max(1.4, 0.8 * header_lines + 0.6)

    # Left-side label block: the DataFrame index, plus one column per label_cols entry, each
    # sized to its own content/header (not a single shared width).
    index_texts = [str(n) for n in df.index]
    index_w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in index_texts) + 2 * CELL_PAD
    label_widths = [index_w]
    for col, header in zip(label_cols, label_headers):
        texts = [str(v) for v in df[col]]
        w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in texts)
        w = max(w, text_width_in(header, HEADER_FONTSIZE, 'bold')) + 2 * CELL_PAD
        label_widths.append(w)
    label_offsets = [sum(label_widths[:k]) for k in range(len(label_widths))]
    label_w = sum(label_widths)

    fig_w = label_w + col_w * n_value_cols

    # Wrap the footnote to fig_w instead of drawing it as one unbroken line -- an oversized
    # single-line footnote is wider than every other artist in the figure, which makes
    # fig.tight_layout() shrink the axes to fit it, squishing the (fixed-point-size) column
    # text into now-narrower column rectangles and causing header/cell text to overlap.
    if footnote:
        char_w = text_width_in('n' * 80, FOOTNOTE_FONTSIZE) / 80 or 0.05
        footnote = '\n'.join(textwrap.wrap(footnote, width=max(20, int(fig_w / char_w))))
    footnote_lines = footnote.count('\n') + 1 if footnote else 0
    footnote_h = 0.16 * footnote_lines + 0.22 if footnote else 0.0

    fig_h = 0.9 + header_h + row_h * n_rows + footnote_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, loc='left', x=0, pad=14)

    header_y = 0
    ax.add_patch(Rectangle((0, header_y), label_widths[0], header_h, facecolor=HEADER_BG,
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
    for k, (header, w) in enumerate(zip(label_headers, label_widths[1:]), start=1):
        x = label_offsets[k]
        ax.add_patch(Rectangle((x, header_y), w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + w / 2, header_y + header_h / 2, header, ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)
    for j, header in enumerate(headers):
        x = label_w + col_w * j
        ax.add_patch(Rectangle((x, header_y), col_w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + col_w / 2, header_y + header_h / 2, header, ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)

    for i, (row_name, row) in enumerate(df.iterrows()):
        y = header_h + row_h * i
        is_summary = row_name in summary_rows
        row_label_weight = 'bold' if is_summary else 'normal'

        ax.add_patch(Rectangle((0, y), label_widths[0], row_h, facecolor='white',
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(CELL_PAD, y + row_h / 2, str(row_name), ha='left', va='center',
                fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        for k, (col, w) in enumerate(zip(label_cols, label_widths[1:]), start=1):
            x = label_offsets[k]
            ax.add_patch(Rectangle((x, y), w, row_h, facecolor='white',
                                    edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
            ax.text(x + CELL_PAD, y + row_h / 2, str(row[col]), ha='left', va='center',
                    fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        scored = row[value_cols].drop(labels=[c for c in count_cols if c in value_cols])
        best_col = scored.idxmax() if len(scored) else None
        for j, col in enumerate(value_cols):
            val = row[col]
            x = label_w + col_w * j
            is_count = col in count_cols
            is_na = pd.isna(val)
            bg = 'white' if (is_count or is_na) else CMAP(val)
            ax.add_patch(Rectangle((x, y), col_w, row_h, facecolor=bg,
                                    edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
            txt_color = TEXT_PRIMARY if (is_count or is_na) else text_color_for(bg)
            is_best = highlight_row_max and not is_count and not is_na and (col == best_col) and n_value_cols > 1
            weight = row_label_weight if (is_count or is_na) else ('bold' if is_best else 'normal')
            ax.text(x + col_w / 2, y + row_h / 2, format_val(col, val),
                    ha='center', va='center', fontsize=CELL_FONTSIZE, color=txt_color,
                    fontweight=weight)

    table_h = header_h + row_h * n_rows
    ax.add_patch(Rectangle((0, 0), fig_w, table_h, facecolor='none',
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH * 1.6, zorder=10))

    if footnote:
        ax.text(0, table_h + footnote_h / 2, footnote, ha='left', va='center',
                fontsize=FOOTNOTE_FONTSIZE, style='italic', color='dimgray', linespacing=1.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out_path}')


# ---------------------------------------------------------------------------
# Dedicated renderer for hierarchy_level_metrics_table: color encodes hierarchy level (not
# value magnitude), since that table's rows are grouped by level rather than being one flat
# per-class list.
# ---------------------------------------------------------------------------
LEVEL_COLORS = {1: '#1f77b4', 2: '#d62728', 3: '#2ca02c', 4: '#9467bd', 5: '#ff7f0e'}


def _blend_with_white(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(alpha * c + (1 - alpha) for c in rgb)


def render_hierarchy_level_table(df, title, out_path, level_colors=None, value_fmt='{:.3f}',
                                  highlight_row_max=True, footnote=None, count_cols=frozenset()):
    """
    df: hierarchy_level_metrics_table's output (index = class name, columns 'Level'/N/Accuracy/
    Precision/Recall/F1, rows already sorted worst-Accuracy-first within each level's own block
    with a 'Total' row pinned last per block -- this renderer trusts that order and does not
    re-sort).

    Unlike render_table, color here encodes df['Level'] (via level_colors, default
    LEVEL_COLORS) rather than each cell's own value: every row's cells are bordered in its
    level's color, a 'Total' row's cells are additionally filled with that color at 20% opacity,
    and the value columns are otherwise plain white -- no CMAP heat-map, since color now means
    level, not magnitude. A small legend under the title spells out which color is which level.
    """
    level_colors = level_colors or LEVEL_COLORS
    value_cols = [c for c in df.columns if c != 'Level']
    n_rows, n_value_cols = len(df), len(value_cols)
    headers = [wrap_header(c) for c in value_cols]
    header_lines = max([h.count('\n') + 1 for h in headers], default=1)

    def format_val(col, val):
        if pd.isna(val):
            return 'N/A'
        return f'{int(val):,}' if col in count_cols else value_fmt.format(val)

    CELL_PAD = 0.25

    row_values = [format_val(col, v) for col in value_cols for v in df[col]]
    max_value_w = max((text_width_in(v, CELL_FONTSIZE, 'bold') for v in row_values), default=0)
    col_w = max(1.1, max_value_w + 2 * CELL_PAD)
    for header in headers:
        col_w = max(col_w, text_width_in(header, HEADER_FONTSIZE, 'bold') + 2 * CELL_PAD)

    row_h = 0.55
    header_h = row_h * max(1.4, 0.8 * header_lines + 0.6)

    index_texts = [str(n) for n in df.index]
    label_w = max(text_width_in(t, ROW_LABEL_FONTSIZE, 'bold') for t in index_texts) + 2 * CELL_PAD

    fig_w = label_w + col_w * n_value_cols

    if footnote:
        char_w = text_width_in('n' * 80, FOOTNOTE_FONTSIZE) / 80 or 0.05
        footnote = '\n'.join(textwrap.wrap(footnote, width=max(20, int(fig_w / char_w))))
    footnote_lines = footnote.count('\n') + 1 if footnote else 0
    footnote_h = 0.16 * footnote_lines + 0.22 if footnote else 0.0

    legend_h = row_h * 0.9
    fig_h = 0.9 + legend_h + header_h + row_h * n_rows + footnote_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, loc='left', x=0, pad=14)

    legend_x, swatch_w = 0.0, 0.28
    for level in sorted(int(l) for l in df['Level'].unique()):
        color = level_colors[level]
        ax.add_patch(Rectangle((legend_x, legend_h * 0.2), swatch_w, legend_h * 0.6,
                                facecolor=color, edgecolor='none'))
        label = f'Level {level}'
        ax.text(legend_x + swatch_w + 0.08, legend_h / 2, label, ha='left', va='center',
                fontsize=FOOTNOTE_FONTSIZE + 1, color=TEXT_PRIMARY)
        legend_x += swatch_w + 0.08 + text_width_in(label, FOOTNOTE_FONTSIZE + 1) + 0.35

    header_y = legend_h
    ax.add_patch(Rectangle((0, header_y), label_w, header_h, facecolor=HEADER_BG,
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
    for j, header in enumerate(headers):
        x = label_w + col_w * j
        ax.add_patch(Rectangle((x, header_y), col_w, header_h, facecolor=HEADER_BG,
                                edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH))
        ax.text(x + col_w / 2, header_y + header_h / 2, header, ha='center', va='center',
                fontsize=HEADER_FONTSIZE, fontweight='bold', color=TEXT_PRIMARY, linespacing=1.3)

    for i, (row_name, row) in enumerate(df.iterrows()):
        y = header_y + header_h + row_h * i
        color = level_colors.get(int(row['Level']), GRID_LINE)
        is_total = (row_name == 'Total')
        row_label_weight = 'bold' if is_total else 'normal'
        cell_bg = _blend_with_white(color, 0.2) if is_total else 'white'
        border_w = GRID_LINEWIDTH * (1.6 if is_total else 1.1)

        ax.add_patch(Rectangle((0, y), label_w, row_h, facecolor=cell_bg, edgecolor=color, linewidth=border_w))
        ax.text(CELL_PAD, y + row_h / 2, str(row_name), ha='left', va='center',
                fontsize=ROW_LABEL_FONTSIZE, fontweight=row_label_weight, color=TEXT_PRIMARY)

        scored = row[value_cols].drop(labels=[c for c in count_cols if c in value_cols])
        best_col = scored.idxmax() if len(scored) else None
        for j, col in enumerate(value_cols):
            val = row[col]
            x = label_w + col_w * j
            is_count = col in count_cols
            is_na = pd.isna(val)
            ax.add_patch(Rectangle((x, y), col_w, row_h, facecolor=cell_bg, edgecolor=color, linewidth=border_w))
            is_best = highlight_row_max and not is_count and not is_na and (col == best_col) and n_value_cols > 1
            weight = row_label_weight if (is_count or is_na) else ('bold' if is_best else 'normal')
            ax.text(x + col_w / 2, y + row_h / 2, format_val(col, val),
                    ha='center', va='center', fontsize=CELL_FONTSIZE, color=TEXT_PRIMARY, fontweight=weight)

    table_h = header_h + row_h * n_rows
    ax.add_patch(Rectangle((0, header_y), fig_w, table_h, facecolor='none',
                            edgecolor=GRID_LINE, linewidth=GRID_LINEWIDTH * 1.6, zorder=10))

    if footnote:
        ax.text(0, header_y + table_h + footnote_h / 2, footnote, ha='left', va='center',
                fontsize=FOOTNOTE_FONTSIZE, style='italic', color='dimgray', linespacing=1.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out_path}')


def plot_outcome_stacked_bar(proportions_by_group, title, keys, labels, colors, legend_side=False):
    """
    Stacked bar of outcome proportions, one bar per group (true class or run), stacked
    bottom -> top in ``keys`` order. ``legend_side=True`` puts the legend to the right, for
    long x labels such as run names.
    """
    groups = sorted(proportions_by_group.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 0.9), 5))
    bottom = np.zeros(len(groups))
    for key, label, color in zip(keys, labels, colors):
        vals = np.array([proportions_by_group[g].get(key, 0.0) for g in groups])
        ax.bar(groups, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    if legend_side:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(labels))
    ax.set_title(title)
    fig.tight_layout()
    return fig
