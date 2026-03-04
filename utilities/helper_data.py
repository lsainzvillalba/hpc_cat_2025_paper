#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:31:45 2022

@author: Laura Sainz Villalba

Helper functions for analysis

# =============================================================================
# 
# Utility functions for managing experimental phase metadata, trial-level
# tensor validation, timepoint reframing, and general dataset helpers.
# =============================================================================
"""

import numpy as np

print("Calling helper_data.py from module script:", __name__)


# ---------------------------------------------------------------------------
# Constants — Experimental Phase Metadata
# ---------------------------------------------------------------------------

# Short display labels for each training phase
TRAININGPOINT_SHORT = {
    'discrimination': 'DISC',
    'gentest_1':      'GT_1',
    'categorization_4': 'CAT',
    'gentest_2':      'GT_2',
}

# Full metadata per phase: [mode, stage, category_set_id]
TRAININGPOINT_DICT = {
    'discrimination':   ['periodshaping', 1, 1],
    'gentest_1':        ['experimental',  1, 0],
    'categorization_3': ['experimental',  2, 3],
    'categorization_4': ['experimental',  2, 4],
    'gentest_2':        ['experimental',  3, 0],
}


# ---------------------------------------------------------------------------
# Experimental Phase Lookups
# ---------------------------------------------------------------------------

def get_experimental_timepoint(phase):
    """
    Return the mode, stage, and category-set ID for a named training phase.

    Parameters
    ----------
    phase : str
        Key in TRAININGPOINT_DICT (e.g. 'discrimination', 'gentest_1').

    Returns
    -------
    mode : str           e.g. 'periodshaping' or 'experimental'
    stage : int          Training stage index.
    categoryset_id : int Category set used in this phase (0 = none).
    """
    mode, stage, categoryset_id = TRAININGPOINT_DICT[phase]
    return mode, stage, categoryset_id


def get_trainingpoint(session_value):
    """
    Reverse-lookup: return the phase name matching a [mode, stage, categoryset_id] value.

    Parameters
    ----------
    session_value : list
        A [mode, stage, categoryset_id] list to match against TRAININGPOINT_DICT.

    Returns
    -------
    str or None : Phase name if found, otherwise None.
    """
    for key, value in TRAININGPOINT_DICT.items():
        if session_value == value:
            return key
    return None


# ---------------------------------------------------------------------------
# Tensor Shape Validation
# ---------------------------------------------------------------------------

def get_bool_on_lengths(tensor, trial_timepoints):
    """
    Check whether all trials in a ragged tensor have the same number of timepoints.

    Parameters
    ----------
    tensor : array-like, shape (n_masks, n_trials)
        Each element is a 1-D array of timepoints (possibly variable length).
    trial_timepoints : array-like, length n_trials
        Timepoint arrays for each trial (used only for length validation here).

    Returns
    -------
    unequal_bool : bool
        True if trials have different lengths (tensor is ragged).
    min_nr_timepoints : int
        Shortest trial length across all masks and trials.
    dim_lastaxis : list of int
        Length of every (mask, trial) entry, in row-major order.
    """
    tensor_arr = np.asarray(tensor)
    assert tensor_arr.ndim == 2, "tensor must be 2-D (n_masks × n_trials)."

    nr_masks, nr_trials = tensor_arr.shape
    assert nr_trials == len(trial_timepoints), \
        "Number of trials must match len(trial_timepoints)."

    # Collect the length of every (mask, trial) time series
    dim_lastaxis = [
        len(tensor[m][t])
        for m in range(nr_masks)
        for t in range(nr_trials)
    ]

    unique_lengths = set(dim_lastaxis)
    unequal_bool = len(unique_lengths) > 1
    min_nr_timepoints = min(dim_lastaxis)

    print('min_nr_timepoints:', min_nr_timepoints)

    return unequal_bool, min_nr_timepoints, dim_lastaxis


# ---------------------------------------------------------------------------
# Timepoint Manipulation
# ---------------------------------------------------------------------------

def get_squared_timepoints(trial_timepoints, min_nr_timepoints):
    """
    Truncate each trial's timepoint array to a common length.

    Parameters
    ----------
    trial_timepoints : list of array-like
        Per-trial timestamp arrays (possibly different lengths).
    min_nr_timepoints : int
        Number of timepoints to keep (typically from `get_bool_on_lengths`).

    Returns
    -------
    list of array-like : Each entry truncated to `min_nr_timepoints` elements.
    """
    return [tp[:min_nr_timepoints] for tp in trial_timepoints]


def reframe_trial_timepoints(trials_timepoints, trial_starts, event_timestamps):
    """
    Express trial timepoints relative to each trial's event onset.

    For each trial:
        reframed = original_timepoints − trial_start − event_timestamp

    Parameters
    ----------
    trials_timepoints : list of array-like, length n_trials
        Absolute timepoints for each trial.
    trial_starts : array-like, length n_trials
        Absolute start time of each trial.
    event_timestamps : array-like, length n_trials
        Time of the event of interest within each trial.

    Returns
    -------
    np.ndarray, shape (n_trials, n_timepoints)
        Timepoints re-referenced to each trial's event onset.
    """
    assert len(trials_timepoints) == len(trial_starts) == len(event_timestamps), \
        "trials_timepoints, trial_starts, and event_timestamps must all have the same length."

    reframed = [
        np.array(tp) - trial_starts[i] - event_timestamps[i]
        for i, tp in enumerate(trials_timepoints)
    ]

    return np.array(reframed)


def reframe_all_events(event_array, event_timestamps):
    """
    Shift all event times in an array relative to per-trial event onsets.

    Parameters
    ----------
    event_array : np.ndarray, shape (n_event_types, n_trials)
        Absolute timestamps for each event type and trial.
    event_timestamps : array-like, length n_trials
        Reference event timestamp for each trial (subtracted from all events).

    Returns
    -------
    reframed_events : np.ndarray, shape (n_event_types, n_trials)
        Event times expressed relative to each trial's reference event.
    """
    assert len(event_timestamps) == event_array.shape[1], \
        "event_timestamps length must match the number of trials (axis 1)."

    reframed_events = np.zeros_like(event_array, dtype=float)

    for trial_idx in range(event_array.shape[1]):
        trial_events = event_array[:, trial_idx].copy().astype(float)
        # Replace any None values with NaN before arithmetic
        trial_events[trial_events == None] = np.nan  # noqa: E711
        reframed_events[:, trial_idx] = trial_events - event_timestamps[trial_idx]

    return reframed_events


# ---------------------------------------------------------------------------
# Trial Filtering
# ---------------------------------------------------------------------------

def get_trial_bool(event_timestamps):
    """
    Build a boolean mask of valid trials based on event timestamp availability.

    Trials with a None timestamp are excluded, and the first 5 trials are
    always excluded (passive warm-up period).

    Parameters
    ----------
    event_timestamps : list
        Per-trial event timestamps; None indicates a missing or invalid trial.

    Returns
    -------
    trial_bool : list of bool, length n_trials
    """
    trial_bool = [event is not None for event in event_timestamps]
    # Always exclude the first 5 passive trials regardless of timestamp
    trial_bool[:5] = [False] * 5
    return trial_bool


# ---------------------------------------------------------------------------
# General Dataset Helpers
# ---------------------------------------------------------------------------

def get_property(dataset, parameter):
    """
    Fetch a unique, ordered list of values for a parameter from a DataJoint table.

    Parameters
    ----------
    dataset : datajoint Table
        Any DataJoint table or query that supports `.fetch()`.
    parameter : str
        Column name to fetch.

    Returns
    -------
    list : Unique values in the order they first appear.
    """
    all_values = dataset.fetch(parameter)
    return list(dict.fromkeys(all_values))


def split_by(nr_elems, nr_sublists):
    """
    Split a range of indices into roughly equal contiguous sublists.

    The last sublist absorbs any remainder so that all indices are covered.

    Parameters
    ----------
    nr_elems : int
        Total number of elements to split.
    nr_sublists : int
        Desired number of sublists.

    Returns
    -------
    list of list of int : Contiguous index sublists.

    Example
    -------
    >>> split_by(10, 3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    """
    indices = list(range(nr_elems))
    chunk_size = nr_elems // nr_sublists

    subsets = []
    for i in range(nr_sublists):
        low = chunk_size * i
        high = low + chunk_size
        # Last sublist takes all remaining elements
        if i == nr_sublists - 1:
            subsets.append(indices[low:])
        else:
            subsets.append(indices[low:high])

    return subsets


