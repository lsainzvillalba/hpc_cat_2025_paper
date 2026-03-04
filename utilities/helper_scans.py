#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:35:27 2024

@author: Laura Sainz Villalba
based on code by Adrian Roggenbach

# =============================================================================
# imaging_utils.py
# Utilities for two-photon imaging stack correction and ROI shape analysis.
#
# Public API:
#   Line-shift correction:
#     apply_shift_to_stack       – apply a known pixel shift and crop
#     correct_line_shift_stack   – find and apply the optimal shift in one call
#     find_shift_stack           – estimate optimal shift across sampled frames
#     find_shift_image           – estimate optimal shift for a single frame
#     shifted_corr               – lagged correlation between two 1-D arrays
#   ROI shape analysis:
#     get_shape_parameters       – diameter, perimeter, area, compactness
# =============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


# =============================================================================
# Line-shift correction
# =============================================================================

def apply_shift_to_stack(stack, shift, crop_left=50, crop_right=50):
    """
    Apply a sub-pixel line shift to correct misalignment between even and odd
    scan lines, then crop the edges to remove shift artefacts.

    A positive *shift* moves even lines (0, 2, 4, …) to the left by *shift*
    pixels. A negative *shift* moves odd lines (1, 3, 5, …) to the left by
    ``abs(shift)`` pixels.

    Parameters
    ----------
    stack : np.ndarray, shape (n_frames, n_rows, n_cols)
        Imaging stack to correct (modified in-place).
    shift : int
        Pixel shift to apply. Positive → shift even lines; negative → odd lines.
    crop_left : int, optional
        Pixels to remove from the right edge of each frame (default 50).
        Despite the parameter name, this crops the right side to match the
        convention used by the Scientifica blanking artefact removal.
    crop_right : int, optional
        Pixels to remove from the left edge of each frame (default 50).

    Returns
    -------
    np.ndarray
        Cropped and shift-corrected stack.
    """
    if shift > 0:
        # Shift even lines leftward by overwriting with content shifted right
        stack[:, ::2, :-shift] = stack[:, ::2, shift:]
    elif shift < 0:
        # Shift odd lines leftward by the absolute shift value
        shift = -shift
        stack[:, 1::2, :-shift] = stack[:, 1::2, shift:]

    if crop_left > 0:
        # Remove edge columns to eliminate shift artefacts and Scientifica
        # blanking onset artefacts on the left/right borders
        stack = stack[:, :, crop_right:-crop_left]

    return stack


def correct_line_shift_stack(stack, crop_left=0, crop_right=0,
                              nr_samples=100, nr_lags=10):
    """
    Find and apply the optimal even/odd line shift for an imaging stack.

    Convenience wrapper that calls `find_shift_stack` to estimate the shift
    and then `apply_shift_to_stack` to correct it.

    Parameters
    ----------
    stack : np.ndarray, shape (n_frames, n_rows, n_cols)
        Raw imaging stack.
    crop_left : int, optional
        Pixels to crop from the right after shifting (default 0).
    crop_right : int, optional
        Pixels to crop from the left after shifting (default 0).
    nr_samples : int, optional
        Number of randomly sampled frames used to estimate the shift (default 100).
    nr_lags : int, optional
        Maximum lag (in pixels) to test in each direction (default 10).

    Returns
    -------
    np.ndarray
        Shift-corrected (and optionally cropped) stack.
    """
    line_shift = find_shift_stack(stack, nr_lags=nr_lags, nr_samples=nr_samples)
    print(f'Correcting a shift of {line_shift} pixel(s).')
    return apply_shift_to_stack(stack, line_shift,
                                crop_left=crop_left, crop_right=crop_right)


def find_shift_stack(stack, nr_lags=10, nr_samples=100,
                     debug=False, return_all=False):
    """
    Estimate the optimal even/odd line shift by averaging over sampled frames.

    Randomly selects up to *nr_samples* frames, computes the lagged
    correlation for each, and returns the lag that maximises the mean
    correlation across frames.

    Parameters
    ----------
    stack : np.ndarray, shape (n_frames, n_rows, n_cols)
        Imaging stack to analyse.
    nr_lags : int, optional
        Maximum absolute lag (pixels) to test (default 10).
    nr_samples : int, optional
        Maximum number of frames to sample (default 100).
    debug : bool, optional
        If True, plot the mean correlation ± SEM across lags.
    return_all : bool, optional
        If True, return (lags, avg_corr); otherwise return only the optimal lag.

    Returns
    -------
    int or (np.ndarray, np.ndarray)
        Optimal lag as a single integer, or (lags, avg_corr) when
        *return_all* is True.
    """
    nr_frames = stack.shape[0]
    np.random.seed(123532)
    sampled_frames = np.random.choice(
        nr_frames, np.min([nr_samples, nr_frames]), replace=False
    )

    # Collect per-frame correlation vectors
    corrs = [
        find_shift_image(stack[frame], return_all=True)[1]
        for frame in sampled_frames
    ]
    corrs    = np.array(corrs)   # shape (n_sampled, n_lags)
    avg_corr = np.mean(corrs, axis=0)
    lags, _  = find_shift_image(stack[sampled_frames[0]], return_all=True)

    if debug:
        sem = np.std(corrs, axis=0) / np.sqrt(len(sampled_frames))
        plt.figure()
        plt.plot(lags, avg_corr, label='Mean')
        plt.fill_between(lags, avg_corr - sem, avg_corr + sem, alpha=0.3, label='SEM')
        plt.legend()
        plt.title(f'Optimal correlation at lag {lags[np.argmax(avg_corr)]}')

    if return_all:
        return lags, avg_corr
    return lags[np.argmax(avg_corr)]


def find_shift_image(image, nr_lags=10, debug=False, return_all=False):
    """
    Estimate the optimal even/odd line shift for a single 2-D image.

    Computes the lagged cross-correlation between flattened even and odd scan
    lines at each integer lag in ``[-nr_lags, nr_lags]`` and returns the lag
    with the highest correlation.

    Parameters
    ----------
    image : np.ndarray, shape (n_rows, n_cols)
        Single imaging frame.
    nr_lags : int, optional
        Maximum absolute lag to test (default 10).
    debug : bool, optional
        If True, plot the correlation vs lag curve.
    return_all : bool, optional
        If True, return (lags, corr); otherwise return only the optimal lag.

    Returns
    -------
    int or (np.ndarray, np.ndarray)
        Optimal lag, or (lags, corr) when *return_all* is True.
    """
    lags = np.arange(-nr_lags, nr_lags + 1, 1)
    corr = np.array([
        shifted_corr(image[::2].flatten(), image[1::2].flatten(), lag=lag)
        for lag in lags
    ])

    if debug:
        plt.figure()
        plt.plot(lags, corr)
        plt.title(f'Maximum at lag {lags[np.argmax(corr)]}')

    if return_all:
        return lags, corr
    return lags[np.argmax(corr)]


def shifted_corr(even_lines, odd_lines, lag=0):
    """
    Compute the Pearson correlation between two 1-D arrays at a given lag.

    A positive *lag* shifts *even_lines* forward (i.e. aligns
    ``even_lines[lag:]`` with ``odd_lines[:-lag]``).  A negative *lag* does
    the reverse.

    Parameters
    ----------
    even_lines : np.ndarray, shape (N,)
        Flattened even scan-line values.
    odd_lines : np.ndarray, shape (N,)
        Flattened odd scan-line values.
    lag : int, optional
        Integer pixel lag; positive shifts even_lines, negative shifts
        odd_lines (default 0).

    Returns
    -------
    float
        Pearson correlation coefficient at the specified lag.
    """
    if lag > 0:
        return np.corrcoef(even_lines[lag:],  odd_lines[:-lag])[0, 1]
    elif lag < 0:
        return np.corrcoef(even_lines[:lag],  odd_lines[-lag:])[0, 1]
    else:
        return np.corrcoef(even_lines, odd_lines)[0, 1]


# =============================================================================
# ROI shape analysis
# =============================================================================

def get_shape_parameters(maskneuronarray):
    """
    Compute morphological shape parameters for the largest connected region
    in a neuron mask array.

    Binarises *maskneuronarray*, labels connected components, and returns
    properties of the component with the largest major-axis length.

    Parameters
    ----------
    maskneuronarray : np.ndarray
        2-D mask array where non-zero pixels belong to the neuron ROI.

    Returns
    -------
    diameter : float
        Major-axis length of the largest region (pixels).
    perimeter : float
        Perimeter of the largest region (pixels).
    area : float
        Area of the largest region (pixels²).
    compactness : float
        area / perimeter² (dimensionless; higher = more circular).
    """
    # Binarise: any non-zero pixel → 1
    binary_mask = np.where(maskneuronarray > 0, 1, 0)
    regions     = regionprops(label(binary_mask))

    diameters  = [round(r.major_axis_length, 2) for r in regions]
    minors     = [round(r.minor_axis_length,  2) for r in regions]
    areas      = [round(r.area,               2) for r in regions]
    perimeters = [round(r.perimeter,          2) for r in regions]

    # Select the largest region by major-axis length
    best        = np.argmax(diameters)
    diameter    = diameters[best]
    perimeter   = perimeters[best]
    area        = areas[best]
    compactness = area / perimeter ** 2

    return diameter, perimeter, area, compactness