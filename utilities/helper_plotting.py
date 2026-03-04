#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba

# =============================================================================
# Reusable matplotlib formatting helpers for violin plots, box plots, and
# general axes styling.
#
# Convention for `column_labels`:
#   - list of str  → datapoints is (n_samples, n_conditions); string tick labels
#   - list of numeric → datapoints is (n_conditions, n_samples); numeric x-positions
# =============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _apply_spine_style(axs):
    """
    Apply the standard spine and tick style shared across all plot types:
    top and right spines hidden; bottom and left spines at linewidth 2.

    Parameters
    ----------
    axs : matplotlib Axes
    """
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_linewidth(2)
    axs.spines["left"].set_linewidth(2)
    axs.tick_params(width=2, length=10)


def _apply_grid(axs):
    """
    Add a subtle horizontal grid behind plot elements.

    Parameters
    ----------
    axs : matplotlib Axes
    """
    axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    axs.set_axisbelow(True)  # keep grid behind data


def _hide_x_ticks():
    """Remove tick marks on both edges of the x-axis (figure-level call)."""
    plt.tick_params(axis='x', which='both', bottom=False, top=False)


# ---------------------------------------------------------------------------
# Public Formatting Functions
# ---------------------------------------------------------------------------

def format_axs(axs):
    """
    Apply minimal spine / tick formatting to a general-purpose axes.

    Parameters
    ----------
    axs : matplotlib Axes

    Returns
    -------
    axs : matplotlib Axes  (modified in-place)
    """
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_linewidth(2)
    axs.spines["left"].set_linewidth(2)
    axs.tick_params(size=10, width=1, labelsize=8, length=10)
    return axs


def format_violinplot(axs, datapoints, column_labels, colors=None, y_label=None):
    """
    Draw and style a violin plot on the given axes.

    Two input orientations are supported, inferred from `column_labels` type:

    - **String labels** (categorical): `datapoints` is (n_samples, n_conditions).
      Medians are shown; tick labels are the strings, rotated 65°.
    - **Numeric labels** (continuous): `datapoints` is (n_conditions, n_samples).
      Means are shown; x-positions are the numeric label values.

    Parameters
    ----------
    axs : matplotlib Axes
    datapoints : np.ndarray
        Sample data — orientation depends on `column_labels` type (see above).
    column_labels : list of str or list of numeric
        Condition names (str) or x-axis positions (numeric).
    colors : list or None
        Per-violin fill colours. If None, matplotlib defaults are used.
    y_label : str or None
        Y-axis label. Defaults to 'probability'.

    Returns
    -------
    axs : matplotlib Axes  (modified in-place)
    """
    y_label = y_label or 'probability'

    # Determine orientation and display options from label type
    if isinstance(column_labels[0], str):
        nr_boxes = datapoints.shape[1]
        positions = list(np.arange(nr_boxes))
        show_medians, show_means = True, False
        use_string_labels = True
    else:
        nr_boxes = len(datapoints)
        positions = column_labels
        assert len(positions) == nr_boxes
        show_medians, show_means = False, True
        use_string_labels = False

    # Draw violin plot
    vp = axs.violinplot(
        datapoints,
        positions=positions,
        showmedians=show_medians,
        showmeans=show_means,
    )

    # Apply per-violin colours if provided
    if colors is not None:
        for body, color in zip(vp['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)

    # Style axes
    _apply_spine_style(axs)
    _apply_grid(axs)
    _hide_x_ticks()

    if use_string_labels:
        axs.set_xticks(np.arange(nr_boxes))
        axs.set_xticklabels(column_labels, rotation=65, fontsize=16)

    axs.set_ylabel(y_label)
    axs.tick_params(axis='both', labelsize=12)

    return axs


def format_boxplot(axs, datapoints, column_labels, colors=None, y_label=None,
                   paired=False, mrkr_size=10):
    """
    Draw and style a box plot on the given axes.

    Two input orientations are supported (same convention as `format_violinplot`).
    Optionally draws grey lines connecting paired samples across conditions.

    Parameters
    ----------
    axs : matplotlib Axes
    datapoints : np.ndarray
        Sample data — orientation depends on `column_labels` type.
    column_labels : list of str or list of numeric
        Condition names (str) or x-axis positions (numeric).
    colors : list or None
        Per-box fill colours. If None, boxes are drawn in black.
    y_label : str or None
        Y-axis label. Defaults to 'probability'. Pass '' to suppress the label.
    paired : bool
        If True, draw a light grey line for each sample across conditions.
    mrkr_size : int
        Marker size (currently reserved; not applied to the boxplot directly).

    Returns
    -------
    axs : matplotlib Axes  (modified in-place)
    """
    y_label = y_label if y_label is not None else 'probability'

    # Determine x-positions from label type
    if isinstance(column_labels[0], str):
        assert isinstance(datapoints, np.ndarray), \
            "datapoints must be np.ndarray when column_labels are strings."
        nr_boxes = datapoints.shape[1]
        positions = list(np.arange(nr_boxes))
        use_string_labels = True
    else:
        positions = column_labels
        nr_boxes = len(positions)
        use_string_labels = False

    # Common boxplot keyword arguments
    bp_kwargs = dict(
        notch=False, sym='.', vert=True,
        bootstrap=1, positions=positions,
        widths=0.5, showfliers=True, showmeans=True,
    )

    if colors is not None:
        # Coloured boxes require patch_artist=True
        bp = axs.boxplot(datapoints, patch_artist=True, **bp_kwargs)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    else:
        bp = axs.boxplot(datapoints, **bp_kwargs)
        plt.setp(bp['boxes'], color='black')

    plt.setp(bp['whiskers'], color='black')

    # Style axes
    _apply_spine_style(axs)
    _apply_grid(axs)
    _hide_x_ticks()

    if use_string_labels:
        axs.set_xticklabels(column_labels, rotation=65, fontsize=14)

    if y_label:
        axs.set_ylabel(y_label)

    axs.set_axisbelow(True)
    axs.tick_params(axis='both', labelsize=12)

    # Draw connecting lines between paired observations
    if paired:
        x_positions = list(np.arange(nr_boxes)) if use_string_labels else column_labels
        for i in range(datapoints.shape[0]):
            axs.plot(x_positions, datapoints[i], '-', color='grey', alpha=0.3)

    return axs
