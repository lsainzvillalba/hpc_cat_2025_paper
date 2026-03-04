#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:31:45 2022

@author: Laura Sainz Villalba


# =============================================================================

# Utility functions for manipulating neural data tensors whose axes represent
# combinations of neurons, trials, and timepoints.
#
# Common tensor format string: ['neurons', 'trials', 'timepoints']
# =============================================================================
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

print("Calling helper_tensor.py from module script:", __name__)


# ---------------------------------------------------------------------------
# Shape & Format Utilities
# ---------------------------------------------------------------------------

def get_squared_tensor(tensor, min_nr_timepoints):
    """
    Convert a ragged tensor (lists of unequal length along the time axis) to a
    uniform 3-D NumPy array by truncating every trial to `min_nr_timepoints`.

    Parameters
    ----------
    tensor : array-like, shape (n_masks, n_trials)
        Each element is a 1-D array of timepoints (possibly different lengths).
    min_nr_timepoints : int
        Number of timepoints to keep per trial.

    Returns
    -------
    list : shape (n_masks, n_trials, min_nr_timepoints)
    """
    tensor = np.asarray(tensor)
    assert tensor.ndim == 2, "Input tensor must be 2-D (masks × trials)."

    nr_masks, nr_trials = tensor.shape
    squared = np.zeros((nr_masks, nr_trials, min_nr_timepoints))

    for m in range(nr_masks):
        for t in range(nr_trials):
            squared[m, t] = tensor[m][t][:min_nr_timepoints]

    return squared.tolist()


def change_tensor_format(tensor_data, current_tensor_format, desired_tensor_format):
    """
    Reorder tensor axes from one named format to another via transposition.

    Example
    -------
    ['neurons', 'trials', 'timepoints'] → ['timepoints', 'trials', 'neurons']

    Parameters
    ----------
    tensor_data : np.ndarray
    current_tensor_format : list of str
        Axis labels for the input tensor.
    desired_tensor_format : list of str
        Axis labels for the output tensor (a permutation of the input labels).

    Returns
    -------
    reformatted_tensor : np.ndarray
    desired_tensor_format : list of str  (echoed for convenience)
    """
    axis_order = tuple(current_tensor_format.index(ax) for ax in desired_tensor_format)
    return tensor_data.transpose(axis_order), desired_tensor_format


# ---------------------------------------------------------------------------
# Selection Utilities
# ---------------------------------------------------------------------------

def select_tensor_by_axisidx(tensor, ids, selected_ids, axis_idx):
    """
    Select a subset of elements along a given axis by matching IDs.

    Parameters
    ----------
    tensor : np.ndarray
    ids : array-like
        Full list of IDs currently on the target axis.
    selected_ids : array-like
        Subset of IDs to keep.
    axis_idx : int
        Axis along which to select.

    Returns
    -------
    selected_tensor : np.ndarray
    """
    selected_idx = pd.Index(ids).get_indexer(selected_ids)
    return np.take(tensor, selected_idx, axis=axis_idx)


def select_timepoints_byframe(timepoints, time_window_frames):
    """
    Extract a window of timepoints relative to the zero-crossing (event onset).

    Parameters
    ----------
    timepoints : array-like
        Full time axis; must contain at least one non-negative value.
    time_window_frames : (int, int)
        (start, end) offsets in frames relative to the onset index.
        start must be negative, end must be positive.

    Returns
    -------
    window_timepoints : array-like
        Slice of `timepoints` within the requested window.
    """
    start, end = time_window_frames
    assert start < 0, "start must be negative (pre-onset)."
    assert end > 0, "end must be positive (post-onset)."

    # Index of the first non-negative timepoint (event onset)
    ref_idx = np.where(np.array(timepoints) >= 0)[0][0]
    return timepoints[ref_idx + start: ref_idx + end]


def select_tensor_by_axis(tensor, tensor_format, ids, selected_ids, axis):
    """
    Select a subset of elements along a named tensor axis.

    Supports three selection modes determined by `axis`:

    - **Named discrete axis** (e.g. 'neurons', 'trials'):
      `selected_ids` is a list of IDs to look up in `ids`.

    - **'timepoints'**: `selected_ids` is a (start, end) time interval;
      all timepoints in [start, end) are selected.

    - **'frames'**: `selected_ids` is a (start_frame, end_frame) pair
      of offsets relative to the zero-crossing of `ids`.

    Parameters
    ----------
    tensor : np.ndarray
    tensor_format : list of str
    ids : array-like
        Labels / values currently on the target axis.
    selected_ids : array-like or (start, end) tuple
    axis : str
        Axis name as it appears in `tensor_format`, or 'frames'.

    Returns
    -------
    selected_tensor : np.ndarray
    """
    # Resolve axis index; 'frames' always maps to the last (timepoints) axis
    if axis == 'frames':
        assert len(tensor_format) == 3, "'frames' mode requires a 3-D tensor."
        axis_idx = 2
    else:
        axis_idx = tensor_format.index(axis)

    ids_arr = np.array(ids)

    if axis == 'timepoints':
        # Select a contiguous range of timepoints by value interval
        assert len(selected_ids) == 2, "Provide (start, end) for timepoints axis."
        start, end = selected_ids
        start_idx = np.where(ids_arr >= start)[0][0]
        end_idx = np.where(ids_arr >= end)[0][0]
        idx_set = list(range(start_idx, end_idx))

    elif axis == 'frames':
        # Select a contiguous range of frames relative to the zero-crossing
        assert len(selected_ids) == 2, "Provide (start_frame, end_frame) for frames axis."
        start_idx, end_idx = selected_ids
        ref_idx = np.where(ids_arr >= 0)[0][0]
        idx_set = list(range(ref_idx + start_idx, ref_idx + end_idx))

    else:
        # Select arbitrary elements by matching IDs
        idx_set = pd.Index(ids).get_indexer(selected_ids)

    selected_tensor = np.take(tensor, idx_set, axis=axis_idx)
    assert len(idx_set) == selected_tensor.shape[axis_idx]

    return selected_tensor


# ---------------------------------------------------------------------------
# Temporal Rebinning
# ---------------------------------------------------------------------------

def rebin_time_in_tensor(tensor_data, tensor_format,
                         timepoints, bin_window, overlap_window):
    """
    Rebin the time axis of a tensor using a sliding window.

    Each new bin aggregates `bin_window` original frames; the window slides
    forward by `bin_window - overlap_window` frames at each step (i.e.
    `overlap_window` frames are shared between adjacent bins).

    Aggregation rule:
    - ``'dff'``    → mean over the window (fluorescence).

    Parameters
    ----------
    tensor_data : np.ndarray
    tensor_format : list of str
    timepoints : array-like
        Timestamps corresponding to each frame on the time axis.
    bin_window : int
        Width of each new bin in original frames.
    overlap_window : int
        Number of frames shared between consecutive bins (0 = no overlap).

    Returns
    -------
    rebinned_tensor : np.ndarray
        Same format as the input tensor.
    rebinned_timepoints : list of float
        Mean timestamp of each new bin.
    """
    # Ensure the tensor is in canonical format before processing
    canonical = ['neurons', 'trials', 'timepoints']
    needs_reformat = tensor_format != canonical
    if needs_reformat:
        tensor_data, _ = change_tensor_format(tensor_data, tensor_format, canonical)

    dims = list(tensor_data.shape)
    nr_timepoints = dims[-1]
    step = bin_window - overlap_window  # stride between consecutive bin starts

    # Build bin start indices using a stride of (bin_window - overlap_window)
    bin_starts = [0]
    while bin_starts[-1] < nr_timepoints - bin_window:
        bin_starts.append(bin_starts[-1] + step)

    nr_bins = len(bin_starts)
    rebinned_dims = tuple(dims[:-1] + [nr_bins])
    rebinned_tensor = np.zeros(rebinned_dims)
    rebinned_timepoints = []

    for i, bin_idx in enumerate(bin_starts):
        window = tensor_data[:, :, bin_idx: bin_idx + bin_window]

        rebinned_tensor[:, :, i] = window.mean(axis=2)

        # Representative timestamp is the mean of the bin's original timestamps
        rebinned_timepoints.append(np.mean(timepoints[bin_idx: bin_idx + bin_window]))

    assert len(rebinned_timepoints) == rebinned_tensor.shape[2]

    # Restore original format if we reformatted above
    if needs_reformat:
        rebinned_tensor, _ = change_tensor_format(
            rebinned_tensor, canonical, tensor_format
        )

    return rebinned_tensor, rebinned_timepoints


# ---------------------------------------------------------------------------
# Trial Alignment
# ---------------------------------------------------------------------------

def get_align_croppings(onset_timepoints):
    """
    Compute the symmetric crop boundaries that keep all trials aligned to
    their event onset while staying within every trial's data range.

    For each trial, the reference frame is the first frame with a positive
    timestamp. The largest common pre-onset window and post-onset window are
    found by taking the minimum across all trials.

    Parameters
    ----------
    onset_timepoints : array-like, shape (n_trials, n_timepoints)
        Per-trial timestamp arrays; must be 2-D.

    Returns
    -------
    left_crop : int
        Number of frames to keep before the onset (pre-event window).
    right_crop : int
        Number of frames to keep from onset onwards (post-event window).
    left_paddings : list of int
        Per-trial index of the onset frame (used by `align_tensor`).
    """
    assert np.array(onset_timepoints).ndim == 2, \
        "onset_timepoints must be 2-D (trials × timepoints)."

    left_paddings = []
    right_paddings = []

    for timepoints in onset_timepoints:
        ref_frame = np.where(np.array(timepoints) > 0)[0][0]
        left_paddings.append(ref_frame)
        right_paddings.append(len(timepoints) - ref_frame)

    left_crop = min(left_paddings)
    right_crop = min(right_paddings)

    return left_crop, right_crop, left_paddings


def align_tensor(tensor_data, tensor_format, onset_timepoints):
    """
    Align all trials in a tensor to their respective event onsets.

    Each trial is cropped to a common window [−left_crop, +right_crop) frames
    relative to its onset, so that time zero corresponds to the event across
    all trials. The output tensor preserves the original axis format.

    Parameters
    ----------
    tensor_data : np.ndarray
    tensor_format : list of str
    onset_timepoints : array-like, shape (n_trials, n_timepoints)
        Per-trial timestamp arrays containing at least one positive value
        marking the event onset.

    Returns
    -------
    aligned_tensor : np.ndarray
        Shape matches `tensor_format` with the time axis truncated to
        left_crop + right_crop frames.
    aligned_timepoints : list of np.ndarray
        Per-trial timestamp arrays for the aligned window.
    """
    left_crop, right_crop, ref_frames = get_align_croppings(onset_timepoints)

    # Work in canonical format internally
    canonical = ['neurons', 'trials', 'timepoints']
    needs_reformat = tensor_format != canonical
    if needs_reformat:
        tensor_data, _ = change_tensor_format(tensor_data, tensor_format, canonical)

    nr_masks, nr_trials, nr_timepoints = tensor_data.shape

    assert nr_trials == len(ref_frames), \
        "Number of trials must match number of onset timepoint arrays."

    nr_selected_frames = left_crop + right_crop
    aligned_timepoints = [None] * nr_trials
    aligned_tensor = []

    for i in tqdm(range(nr_trials)):
        trial_timepoints = np.array(onset_timepoints[i])
        assert nr_timepoints == len(trial_timepoints)

        ref = ref_frames[i]

        # Crop the timestamp array for this trial
        aligned_timepoints[i] = trial_timepoints[ref - left_crop: ref + right_crop]

        # Crop each neuron's trace for this trial
        trial_neurons = []
        for j in range(nr_masks):
            trace = tensor_data[j, i]
            trial_neurons.append(trace[ref - left_crop: ref + right_crop])

        aligned_tensor.append(trial_neurons)

    # Build array in [trials, neurons, timepoints] order, then reformat
    aligned_tensor = np.array(aligned_tensor)  # (n_trials, n_masks, n_frames)
    assert aligned_tensor.shape == (nr_trials, nr_masks, nr_selected_frames)

    aligned_tensor, _ = change_tensor_format(
        aligned_tensor,
        ['trials', 'neurons', 'timepoints'],
        tensor_format,
    )

    return aligned_tensor, aligned_timepoints


