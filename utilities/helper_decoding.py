#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:41:49 2024

@author: Laura Sainz Villalba

# General-purpose utilities for neural decoding analysis: p-value computation,
# trial-condition handling, tensor manipulation, SVM training/testing, and
# video trace extraction.
#
# Public API (grouped by theme):
#   Statistical testing:
#     compute_pvalues_trace, pvalue_bootstrap_corrected,
#     compute_pvalues_trace_corrected, plot_pvalues_trace
#   Condition / trial helpers:
#     get_condition_response_variable, format_condition_variable,
#     get_dichotomies, get_trial_conditions, get_trials_by_conds,
#     get_available_set_trial_ids
#   Tensor manipulation:
#     stats_over_axis, zscore_neurons, get_X_tensors_at_t, get_y_for_X,
#     get_X_dataset, get_dataset, get_beh_dataset, get_cross_stim_tensors
#   Geometry:
#     angle_between_vectors, get_pointcloud, enclosed_volume
#   SVM decoding:
#     train_decoder, test_decoder
#   Video:
#     get_video_traces, get_video_tensor
# =============================================================================
"""

import numpy as np
import math
import pandas as pd
import copy

from statsmodels.stats.multitest import multipletests
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Statistical testing helpers
# =============================================================================

def _compute_pointwise_pvalue(accuracy, null_col, stat_type='one-sided larger'):
    """
    Compute a single timepoint p-value from an observed value and null column.

    Parameters
    ----------
    accuracy : float
        Observed statistic at this timepoint.
    null_col : np.ndarray, shape (n_iters,)
        Null distribution at this timepoint.
    stat_type : str
        'one-sided larger' or 'two-sided'.

    Returns
    -------
    float
        Raw p-value, or 1 if the null distribution has zero variance.
    """
    if math.isclose(np.var(null_col), 0):
        return 1   # near-zero variance → insufficient data, mark non-significant

    nr_null     = len(null_col)
    nr_decimals = str(1 / nr_null)[::-1].find('.')
    value       = round(accuracy, nr_decimals)
    n_dt        = np.round(null_col, decimals=nr_decimals)

    if stat_type == 'one-sided larger':
        return round(len(np.where(n_dt > value)[0]) / len(n_dt), nr_decimals)
    else:  # two-sided
        shifted = n_dt - np.mean(n_dt)
        return float(np.sum(np.abs(shifted) >= np.abs(value - np.mean(n_dt))) / len(n_dt))


def compute_pvalues_trace(var, null, multiple_correction=False, method='bonferroni'):
    """
    Compute per-timepoint p-values for an observed trace against a null distribution.

    Parameters
    ----------
    var : array-like, shape (n_timepoints,)
        Observed accuracy trace.
    null : np.ndarray, shape (n_iters, n_timepoints)
        Null distribution per timepoint.
    multiple_correction : bool, optional
        Whether to apply multiple-comparison correction.
    method : str, optional
        Correction method: 'bonferroni', 'holm', 'max', or 'cluster'.

    Returns
    -------
    list of float
        (Corrected) p-value per timepoint.
    """
    p_values = [
        _compute_pointwise_pvalue(acc, null[:, t])
        for t, acc in enumerate(var)
    ]

    if not multiple_correction:
        return p_values

    if method == 'max':
        # Max-statistic correction: compare against 95th percentile of null maxima
        threshold = np.round(np.max(np.percentile(null, 95, axis=0)), 5)
        return [0 if round(acc, 5) > threshold else 1 for acc in var]

    elif method == 'cluster':
        pass   # placeholder – not yet implemented

    else:
        _, p_values, _, _ = multipletests(p_values, method=method)

    return p_values


def pvalue_bootstrap_corrected(var, null, stat_type='one-sided larger', method='bonferroni'):
    """
    Compute raw and corrected per-timepoint p-values via bootstrap.

    Returns both standard bootstrap p-values and a max-statistic corrected
    version. When method != 'max', applies a multiple-testing correction
    (e.g. Bonferroni, Holm) to the raw p-values instead.

    Parameters
    ----------
    var : array-like, shape (n_timepoints,)
        Observed statistic trace.
    null : np.ndarray, shape (n_iters, n_timepoints)
        Null distribution per timepoint.
    stat_type : str, optional
        'one-sided larger' or 'two-sided'.
    method : str, optional
        Multiple-testing correction: 'max' (uses max-statistic thresholding)
        or any method accepted by `multipletests`.

    Returns
    -------
    p_values : list of float
        Raw bootstrap p-values per timepoint.
    p_values_corr : list of float
        Corrected p-values per timepoint.
    """
    # Pre-compute null percentile thresholds for the max-statistic correction
    perc_95  = np.round(np.max(np.percentile(null, 95,  axis=0)), 5)
    perc_97_5 = np.round(np.max(np.percentile(null, 97.5, axis=0)), 5)
    perc_2_5  = np.round(np.max(np.percentile(null, 2.5,  axis=0)), 5)

    p_values     = []
    corr_pvalues = []

    for t, x in enumerate(var):
        p_values.append(_compute_pointwise_pvalue(x, null[:, t], stat_type))

        # Max-statistic threshold correction
        if stat_type == 'one-sided larger':
            corr_pvalues.append(0 if x > perc_95 else 1)
        elif stat_type == 'two-sided':
            corr_pvalues.append(0 if (x > perc_97_5 or x < perc_2_5) else 1)

    if method != 'max':
        _, p_values_corr, _, _ = multipletests(p_values, method=method)
    else:
        p_values_corr = corr_pvalues

    return p_values, p_values_corr


def compute_pvalues_trace_corrected(var, null, stat_type='one-sided larger'):
    """
    Compute raw and max-statistic-corrected p-values for an observed trace.

    Simplified version of `pvalue_bootstrap_corrected` that always applies
    max-statistic correction without further multiple-testing methods.

    Parameters
    ----------
    var : array-like, shape (n_timepoints,)
        Observed statistic trace.
    null : np.ndarray, shape (n_iters, n_timepoints)
        Null distribution per timepoint.
    stat_type : str, optional
        'one-sided larger' or 'two-sided'.

    Returns
    -------
    p_values : list of float
        Raw bootstrap p-values per timepoint.
    corr_pvalues : list of float
        Max-statistic corrected p-values per timepoint.
    """
    perc_95   = np.round(np.max(np.percentile(null, 95,  axis=0)), 5)
    perc_97_5 = np.round(np.max(np.percentile(null, 97.5, axis=0)), 5)
    perc_2_5  = np.round(np.min(np.percentile(null, 2.5,  axis=0)), 5)

    p_values     = []
    corr_pvalues = []

    for t, x in enumerate(var):
        p_values.append(_compute_pointwise_pvalue(x, null[:, t], stat_type))

        if stat_type == 'one-sided larger':
            corr_pvalues.append(0 if x > perc_95 else 1)
        elif stat_type == 'two-sided':
            corr_pvalues.append(0 if (x > perc_97_5 or x < perc_2_5) else 1)

    return p_values, corr_pvalues


def plot_pvalues_trace(axs, pvalues, x, max_val, color, alpha=None,
                       p_thres=0.05, pbool=None):
    """
    Mark significant timepoints on an axes with upward-pointing triangles.

    Parameters
    ----------
    axs : matplotlib.axes.Axes
        Axes to draw on.
    pvalues : list of float
        Per-timepoint p-values.
    x : array-like
        Timepoint values aligned with *pvalues*.
    max_val : float
        y-position at which to place the markers.
    color : str or tuple
        Marker colour.
    alpha : float, optional
        Marker transparency (default 1).
    p_thres : float, optional
        Significance threshold (default 0.05).
    pbool : list of bool, optional
        Additional per-timepoint boolean gate (both *pvalues* and *pbool* must
        be True for a marker to appear).

    Returns
    -------
    matplotlib.axes.Axes
        The axes with markers added.
    """
    alpha = alpha if alpha is not None else 1

    for i, p in enumerate(pvalues):
        significant = (p < p_thres) and (pbool[i] if pbool is not None else True)
        if significant:
            axs.scatter(x[i], max_val, color=color, s=20, marker='^', alpha=alpha)

    return axs


# =============================================================================
# Condition / trial helpers
# =============================================================================

def get_condition_response_variable(condition):
    """
    Return the DataJoint Trial field name that encodes *condition*.

    Parameters
    ----------
    condition : str
        Condition label (e.g. 'choice', 'outcome', 'category', 'stimulus').

    Returns
    -------
    str or None
        Field name in the Trial table, or None if not recognised.
    """
    mapping = {
        'choice':        'response',
        'outcome':       'responsetype',
        'stimulus':      'stimulus_id',
        'stimulus_id':   'stimulus_id',
        'shift':         'response',
        'exception':     'stimulus_id',
        'stimulus_type': 'stimulus_id',
        'init_A':        'stimulus_id',
        'init_B':        'stimulus_id',
        'init':          'stimulus_id',
        'category':      'baited_port',
    }
    if condition in mapping:
        return mapping[condition]

    # Numeric string → treat as a specific stimulus ID
    try:
        int(condition)
        return 'stimulus_id'
    except ValueError:
        return None


def format_condition_variable(condition, variable, exception_stimulus, stim_of_int):
    """
    Convert raw Trial field values into binary or categorical class labels.

    Parameters
    ----------
    condition : str
        Condition type (e.g. 'choice', 'outcome', 'category').
    variable : list
        Raw values fetched from the Trial table.
    exception_stimulus : str or None
        Stimulus ID used for 'exception' trials.
    stim_of_int : str or None
        Target stimulus ID used for 'stimulus' decoding.

    Returns
    -------
    list
        Formatted labels aligned with *variable*.

    Raises
    ------
    Exception
        If *condition* is not implemented.
    """
    if condition == 'choice':
        return [0 if v == 0 else 1 for v in variable]

    elif condition == 'outcome':
        return [1 if v == 'correct' else 0 for v in variable]

    elif condition == 'category':
        return [0 if v == 0 else 1 for v in variable]

    else:
        raise Exception(f'Condition "{condition}" not implemented.')


def get_dichotomies(combinations, variable_indices):
    """
    Split combinations into binary dichotomies along each variable index.

    Parameters
    ----------
    combinations : list of tuple
        All trial-type combination tuples.
    variable_indices : list of int
        Indices of variables to split on.

    Returns
    -------
    list of list
        One [group_1, group_0] pair per variable index.
    """
    dichotomies = []
    for idx in variable_indices:
        group_1 = [c for c in combinations if c[idx]]
        group_0 = [c for c in combinations if not c[idx]]
        dichotomies.append([group_1, group_0])
    return dichotomies


def angle_between_vectors(u, v):
    """
    Compute the angle between two vectors in degrees.

    Parameters
    ----------
    u : np.ndarray
        First vector (flattened if multi-dimensional).
    v : np.ndarray
        Second vector (flattened if multi-dimensional).

    Returns
    -------
    float
        Angle in degrees in [0, 180].
    """
    u = u.reshape(-1)
    v = v.reshape(-1)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def get_trial_conditions(trial_session_entries, conditions, stim_of_int=None):
    """
    Build a per-condition, per-trial label dictionary for a session.

    Applies port-layout correction for category and choice variables,
    handles the 'exception' protocol (incorrect trials with matched choice),
    and filters stimuli for the 'stimulus_type' condition.

    Parameters
    ----------
    trial_session_entries : DataJoint table expression
        Active, responding trial rows for the session.
    conditions : list of str
        Conditions to extract labels for.
    stim_of_int : str or None, optional
        Target stimulus ID for 'stimulus' decoding.

    Returns
    -------
    dict
        {condition: {trial_id: label}} mapping.
    """
    port_layout = trial_session_entries.fetch('port_layout')[0]

    # Keep only active, responding, non-catch trials
    trial_session_entries = (trial_session_entries
                             & 'trialtype!="control_nostimulus"'
                             & 'trialtype="active"'
                             & 'responsetype!="no response"')

    condition_vars    = [get_condition_response_variable(c) for c in conditions]
    exception_stimulus = None

    
    trial_ids = list(trial_session_entries.fetch('trial_id'))

    # ------------------------------------------------------------------
    # Build the label dictionary for each condition
    # ------------------------------------------------------------------
    trial_condition_dict = {}

    for i, var in enumerate(condition_vars):
        condition = conditions[i]
        trial_condition_dict[condition] = {}
        variable = []

        for trial_id in trial_ids:
            entry = trial_session_entries & f'trial_id="{trial_id}"'
            val   = entry.fetch(var)[0]

            if condition == 'shift':
                prev_val = (trial_session_entries
                            & f'trial_id="{trial_id - 1}"').fetch(var)
                variable.append('stay' if val == prev_val else 'shift')

            elif condition in ('category', 'choice'):
                # Flip binary value when port layout is inverted
                variable.append(1 - val if port_layout else val)

            else:
                variable.append(val)

        # Numeric condition strings → treat as specific stimulus ID targets
        try:
            int(condition)
            stimulus_condition = True
        except ValueError:
            stimulus_condition = False

        if stimulus_condition:
            trial_condition = format_condition_variable(
                'stimulus', variable, None, condition
            )
        else:
            trial_condition = format_condition_variable(
                condition, variable, exception_stimulus, stim_of_int
            )

        for j, trial_id in enumerate(trial_ids):
            trial_condition_dict[condition][trial_id] = trial_condition[j]

    return trial_condition_dict


def get_trials_by_conds(trial_cond_dict, key, conditions, min_nr_trials,
                        nr_combinations, split=None):
    """
    Group trial IDs by their full condition-combination tuple.

    Validates that all expected combinations are present and that each has
    sufficient trials. When *split* is specified, further partitions combinations
    by the split variable and validates each partition independently.

    Parameters
    ----------
    trial_cond_dict : dict
        {condition: {trial_id: label}} from `get_trial_conditions`.
    key : dict
        Session key used for diagnostic prints.
    conditions : list of str
        Ordered condition labels matching trial_cond_dict keys.
    min_nr_trials : int or None
        Minimum trials required per combination (None skips this check).
    nr_combinations : int or None
        Expected number of combinations (None skips this check).
    split : str or None, optional
        Condition to use as a split variable for cross-condition analysis.

    Returns
    -------
    dict or np.nan
        {combination_tuple: [trial_id, ...]} or np.nan if validation fails.
    """
    # Normalise conditions for 'exception' protocol
    if 'exception' in conditions:
        conditions = list(trial_cond_dict.keys())

    condition_key = list(trial_cond_dict.keys())[0]
    trial_ids     = list(trial_cond_dict[condition_key].keys())

    # Build combination → trial-list mapping
    trials_by_conds = {}
    for trial in trial_ids:
        combo = tuple(trial_cond_dict[c][trial] for c in conditions)
        trials_by_conds.setdefault(combo, []).append(trial)

    if nr_combinations is None:
        # Just check minimum trial counts if requested
        if min_nr_trials is not None:
            for combo, tids in trials_by_conds.items():
                if len(tids) < min_nr_trials:
                    print(key, '| insufficient trials in combination:', len(tids))
                    return np.nan
        return trials_by_conds

    # ------------------------------------------------------------------
    # Validate without split
    # ------------------------------------------------------------------
    if split is None:
        n_found = len(trials_by_conds)
        if n_found < nr_combinations:
            print(key, f'| {n_found}/{nr_combinations} combinations found:', list(trials_by_conds.keys()))
            return np.nan
        assert n_found == nr_combinations, f"More combinations than expected: {n_found}"

        if min_nr_trials is not None:
            for combo, tids in trials_by_conds.items():
                if len(tids) < min_nr_trials:
                    print(key, '| insufficient trials:', len(tids))
                    return np.nan

    # ------------------------------------------------------------------
    # Validate with split variable
    # ------------------------------------------------------------------
    else:
        if len(trials_by_conds) < nr_combinations:
            print(key, '| missing combinations:', list(trials_by_conds.keys()))
            return np.nan

        split_idx    = conditions.index(split)
        split_values = list(dict.fromkeys(c[split_idx] for c in trials_by_conds))
        valid_splits = []

        for val in split_values:
            split_combinations = {
                combo: tids for combo, tids in trials_by_conds.items()
                if combo[split_idx] == val
            }
            complete = len(split_combinations) == nr_combinations
            enough   = True

            if min_nr_trials is not None:
                for combo, tids in split_combinations.items():
                    if len(tids) < min_nr_trials:
                        print(key, '| insufficient trials in split:', len(tids))
                        enough = False
                        break

            if complete and enough:
                valid_splits.append(split_combinations)

        if not valid_splits:
            return np.nan
        if len(valid_splits) == 1:
            trials_by_conds = valid_splits[0]

    return trials_by_conds


def get_available_set_trial_ids(list_trials_by_conds):
    """
    Collect available trial IDs per combination across multiple sessions.

    Parameters
    ----------
    list_trials_by_conds : list of dict
        One {combination: [trial_ids]} dict per session.

    Returns
    -------
    dict
        {combination: [[trial_ids_sess_0], [trial_ids_sess_1], ...]}
        with one inner list per session.
    """
    nr_sessions = len(list_trials_by_conds)

    # Collect all unique combinations across sessions
    all_combos = list({
        combo
        for session in list_trials_by_conds
        for combo in session
    })

    available_set_trials = {
        combo: [list(session.get(combo, [])) for session in list_trials_by_conds]
        for combo in all_combos
    }

    # Verify alignment
    for combo in available_set_trials:
        assert len(available_set_trials[combo]) == nr_sessions

    return available_set_trials


# =============================================================================
# Tensor manipulation helpers
# =============================================================================

def stats_over_axis(tensordata, tensor_format, axis_stats):
    """
    Compute mean and standard deviation over one named axis of a tensor.

    Parameters
    ----------
    tensordata : np.ndarray
        Input tensor.
    tensor_format : list of str
        Axis labels, e.g. ['neurons', 'trials', 'timepoints'].
    axis_stats : str
        Label of the axis to reduce over.

    Returns
    -------
    mean : np.ndarray
        Mean with *axis_stats* removed.
    std : np.ndarray
        Std with *axis_stats* removed.

    Raises
    ------
    Exception
        If the result shape does not match expected dimensions.
    """
    stat_axis  = tensor_format.index(axis_stats)
    mean       = np.nanmean(tensordata, axis=stat_axis)
    std        = np.nanstd(tensordata,  axis=stat_axis)

    expected_dims = [d for i, d in enumerate(tensordata.shape) if i != stat_axis]
    if list(mean.shape) != expected_dims:
        raise Exception(
            f'stats_over_axis: result shape {list(mean.shape)} does not match '
            f'expected {expected_dims}.'
        )
    return mean, std


def zscore_neurons(tensor_data, tensor_format, mean, std):
    """
    Z-score each feature (neuron/column) of a [trials × features] tensor.

    Parameters
    ----------
    tensor_data : np.ndarray, shape (n_trials, n_features)
        Input tensor.
    tensor_format : list of str
        Must be ['trials', 'neurons'] or ['trials', 'features'].
    mean : array-like, shape (n_features,)
        Per-feature mean.
    std : array-like, shape (n_features,)
        Per-feature standard deviation.

    Returns
    -------
    np.ndarray
        Z-scored tensor with the same shape as *tensor_data*.
    """
    assert tensor_format in (['trials', 'neurons'], ['trials', 'features'])
    n_features  = tensor_data.shape[1]
    assert len(mean) == n_features

    norm_tensor = np.zeros_like(tensor_data)
    for f in range(n_features):
        norm_tensor[:, f] = (tensor_data[:, f] - mean[f]) / std[f]
    return norm_tensor


def get_cross_stim_tensors(combinations, stim_dichotomy, n_data_comb, tensors):
    """
    Balance trial counts between the two sides of a stimulus dichotomy.

    Randomly samples *n_data_comb* trials for each side, ensuring both
    the target-stimulus group and the rest-of-stimuli group have exactly
    *n_data_comb* trials each.

    Parameters
    ----------
    combinations : list of tuple
        All combination tuples.
    stim_dichotomy : list of two lists
        [target_combinations, rest_combinations] from `get_dichotomies`.
    n_data_comb : int
        Target trial count per dichotomy side.
    tensors : np.ndarray, shape (n_combinations, n_trials, n_neurons)
        Trial tensors indexed by combination.

    Returns
    -------
    np.ndarray, dtype=object
        Resampled tensors aligned with *combinations*.
    """
    assert len(tensors.shape) == 3
    assert len(combinations) == tensors.shape[0]
    assert len(stim_dichotomy[0]) == 1   # exactly one target combination

    tensors  = list(tensors)
    n_rest   = len(stim_dichotomy[1])
    n_per    = int(n_data_comb / n_rest)
    n_extra  = n_data_comb - n_per * n_rest
    extra_groups = np.random.randint(0, n_rest, n_extra)

    # Initialise per-rest-combination row indices
    selected_rows = [
        list(np.random.randint(0, len(tensors[combinations.index(c)]), n_per))
        for c in stim_dichotomy[1]
    ]

    # Distribute any extra samples
    for g, extra_id in zip(extra_groups,
                           [np.random.randint(0, len(tensors[combinations.index(stim_dichotomy[1][g])]))
                            for g in extra_groups]):
        selected_rows[g].append(extra_id)

    for i, combo in enumerate(stim_dichotomy[1]):
        idx = combinations.index(combo)
        tensors[idx] = tensors[idx][selected_rows[i]]

    # Sample target combination
    target_idx      = combinations.index(stim_dichotomy[0][0])
    target_rows     = list(np.random.randint(0, len(tensors[target_idx]), n_data_comb))
    tensors[target_idx] = tensors[target_idx][target_rows]

    return np.array(tensors, dtype=object)


def get_X_tensors_at_t(available_set_trials, list_tensors, pseudopop_bool,
                        list_trial_ids, t, training_fraction,
                        n_data_comb, training_bool, cross):
    """
    Extract and balance training or testing tensors at timepoint(s) *t*.

    For each combination, randomly samples *n_data_comb* trials from the
    appropriate split (training or testing) in each session, then either
    horizontally stacks sessions into a pseudo-population tensor or returns
    a per-session tensor.

    Parameters
    ----------
    available_set_trials : dict
        {combination: [[trial_ids_sess_0], ...]} from `get_available_set_trial_ids`.
    list_tensors : list of np.ndarray
        Per-session tensors of shape (n_neurons, n_trials, n_timepoints).
    pseudopop_bool : bool
        If True, concatenate neurons across sessions.
    list_trial_ids : list
        Per-session ordered trial ID lists.
    t : int or list of int
        Timepoint index or [start, end] window.
    training_fraction : float
        Fraction of trials reserved for training.
    n_data_comb : int
        Trials to sample per combination per session.
    training_bool : bool
        True = sample from training split; False = testing split.
    cross : bool
        If True, always use the training split (for cross-decoding).

    Returns
    -------
    np.ndarray
        Pseudo-pop: (n_combinations, n_data_comb, total_neurons)
        Per-session: (n_combinations, n_sessions, n_data_comb, n_neurons)
    """
    combinations = list(available_set_trials.keys())
    assert len(list_trial_ids) == len(available_set_trials[combinations[0]])

    tensors = []
    for combination in combinations:
        tensors_combination = []
        for j in range(len(available_set_trials[combination])):
            nr_trials  = len(available_set_trials[combination][j])
            split_idx  = int(training_fraction * nr_trials)

            if cross or training_bool:
                trial_set = available_set_trials[combination][j][:split_idx]
            else:
                trial_set = available_set_trials[combination][j][split_idx:]

            selected = np.random.choice(trial_set, n_data_comb)
            idx      = pd.Index(list_trial_ids[j]).get_indexer(selected)

            if isinstance(t, list):
                # Average over a time window
                tensor = np.mean(
                    np.take(list_tensors[j][:, :, t[0]:t[1]], idx, axis=1),
                    axis=2
                ).T
            else:
                tensor = np.take(list_tensors[j][:, :, t], idx, axis=1).T

            tensors_combination.append(tensor)

        if pseudopop_bool:
            tensors.append(np.hstack(tensors_combination))
        else:
            tensors.append(tensors_combination)

    return np.array(tensors)


def get_y_for_X(combinations, conditions, decoding_var, tensors, pseudopop_bool):
    """
    Build target label arrays aligned with the trial tensors.

    Parameters
    ----------
    combinations : list of tuple
        All combination tuples.
    conditions : list of str
        Ordered condition labels.
    decoding_var : str
        Variable to decode; 'tuple' assigns a unique integer per combination.
    tensors : np.ndarray
        Trial tensors (used to infer per-combination trial counts).
    pseudopop_bool : bool
        If True, returns a single flat array; otherwise per-session arrays.

    Returns
    -------
    np.ndarray
        Flat label array (pseudopop) or (n_sessions, n_trials) array.
    """
    if pseudopop_bool:
        y_parts = []
        for k, combo in enumerate(combinations):
            label = k if decoding_var == 'tuple' else int(combo[conditions.index(decoding_var)])
            y_parts.append(np.repeat(label, len(tensors[k])))
        return np.hstack(y_parts)

    else:
        nr_sessions = len(tensors[0])
        y = []
        for i in range(nr_sessions):
            y_s = []
            for k, combo in enumerate(combinations):
                label = int(combo[conditions.index(decoding_var)])
                y_s.append(np.repeat(label, len(tensors[k, i])))
            y.append(np.hstack(y_s))
        return np.array(y)


def get_X_dataset(sessions_info, params, t, available_set_trials,
                  training_bool, subsampling_idx=None):
    """
    Build a pseudo-population feature tensor (no labels) for one timepoint.

    Parameters
    ----------
    sessions_info : dict
        Must contain 'list_tensors' and 'list_trial_ids'.
    params : dict
        Must contain 'training_fr', 'pseudopopulation' (True), 'n_datapoints'.
    t : int or list
        Timepoint index or [start, end] window.
    available_set_trials : dict
        From `get_available_set_trial_ids`.
    training_bool : bool
        Select training (True) or testing (False) split.
    subsampling_idx : list of array-like, optional
        Per-session neuron indices to subsample.

    Returns
    -------
    np.ndarray, shape (n_combinations, n_data_comb, total_neurons)
        Balanced pseudo-population tensors.
    """
    list_tensors = sessions_info['list_tensors']
    assert params['pseudopopulation'], "get_X_dataset requires pseudopopulation=True."

    if subsampling_idx is not None:
        list_tensors = [t[subsampling_idx[i]] for i, t in enumerate(list_tensors)]

    n_combos    = len(available_set_trials)
    n_data_comb = int(params['n_datapoints'] / n_combos)

    tensors = get_X_tensors_at_t(
        available_set_trials, list_tensors, True,
        sessions_info['list_trial_ids'], t,
        params['training_fr'], n_data_comb, training_bool, False
    )

    tot_neurons = sum(len(subsampling_idx[i]) for i in range(len(subsampling_idx)))
    assert tensors.shape == (n_combos, n_data_comb, tot_neurons)
    return tensors


def get_dataset(sessions_info, params, t, available_set_trials, training_bool,
                cross_stim=False, cross=False, subsampling_idx=None):
    """
    Build feature tensor *X* and label array *y* for SVM decoding.

    Handles pseudo-population and per-session modes, optional neuron
    subsampling, stimulus-vs-rest cross decoding, and training/testing splits.

    Parameters
    ----------
    sessions_info : dict
        Must contain 'list_tensors' and 'list_trial_ids'.
    params : dict
        Keys: 'training_fr', 'pseudopopulation', 'n_datapoints',
              'nr_neurons_session', 'variables', 'train_var', 'test_var'.
    t : int or list
        Timepoint(s).
    available_set_trials : dict
        From `get_available_set_trial_ids`.
    training_bool : bool
        Selects training or testing decoding variable and split.
    cross_stim : bool, optional
        If True, decode one stimulus vs rest (binary).
    cross : bool, optional
        If True, always use training-split trials (cross-decoding).
    subsampling_idx : list of array-like, optional
        Per-session neuron indices.

    Returns
    -------
    X_tensor : np.ndarray
        (n_trials, n_neurons) for pseudo-pop; (n_sessions, n_trials, n_neurons) otherwise.
    y : np.ndarray
        Trial labels.
    """
    list_tensors = sessions_info['list_tensors']

    if subsampling_idx is not None:
        list_tensors = [tensor[subsampling_idx[i]] for i, tensor in enumerate(list_tensors)]

    training_fraction = params['training_fr']
    pseudopop_bool    = params['pseudopopulation']
    n_datapoints      = params['n_datapoints']
    decoding_var      = params['train_var'] if training_bool else params['test_var']
    conditions        = params['variables']
    combinations      = list(available_set_trials.keys())

    nr_combinations   = 2 if cross_stim else len(available_set_trials)

    if pseudopop_bool:
        n_data_comb = int(n_datapoints / nr_combinations)
    else:
        n_data_comb = int((2 * params['nr_neurons_session']) / nr_combinations)

    tensors = get_X_tensors_at_t(
        available_set_trials, list_tensors, pseudopop_bool,
        sessions_info['list_trial_ids'], t, training_fraction,
        n_data_comb, training_bool, cross
    )

    # Rebalance for stimulus-vs-rest decoding
    if cross_stim:
        var_idx       = conditions.index(decoding_var)
        stim_dichotomy = get_dichotomies(combinations, [var_idx])[0]
        tensors        = get_cross_stim_tensors(combinations, stim_dichotomy, n_data_comb, tensors)
        assert len(tensors) == nr_combinations

    elif pseudopop_bool:
        tot_neurons = sum(len(subsampling_idx[i]) for i in range(len(subsampling_idx)))
        assert tensors.shape == (nr_combinations, n_data_comb, tot_neurons)

    y = get_y_for_X(combinations, conditions, decoding_var, tensors, pseudopop_bool)

    if pseudopop_bool:
        X_tensor = np.vstack(tensors)
        assert X_tensor.shape[0] == len(y)
    else:
        nr_sessions = len(list_tensors)
        X_tensor    = np.array([np.vstack(tensors[:, i]) for i in range(nr_sessions)])

    return X_tensor, y


def get_beh_dataset(sessions_info, params, t_idx, available_set_trials, training_bool):
    """
    Build video-feature tensor X and label array y for behavioural decoding.

    Parameters
    ----------
    sessions_info : dict
        Must contain 'list_video_tensors' and 'list_videotrial_ids'.
    params : dict
        Keys: 'training_fr', 'pseudopopulation', 'train_var', 'variables',
              'n_datapoints' (unused directly; inferred from trial count).
    t_idx : int
        Timepoint index into the video tensor.
    available_set_trials : dict
        From `get_available_set_trial_ids`.
    training_bool : bool
        Selects training or testing split.

    Returns
    -------
    X_tensor : np.ndarray
        Feature tensor.
    y : np.ndarray
        Trial labels.
    """
    list_video_tensors = sessions_info['list_video_tensors']
    list_videotrial_ids = sessions_info['list_videotrial_ids']
    training_fraction  = params['training_fr']
    pseudopop_bool     = params['pseudopopulation']
    decoding_var       = params['train_var']
    conditions         = params['variables']
    combinations       = list(available_set_trials.keys())
    nr_combinations    = len(combinations)
    nr_sessions        = len(list_video_tensors)

    if pseudopop_bool:
        # 4 video features per session; two-class balance → 2 * 4 * nr_sessions trials
        nr_trials_comb = int((2 * 4 * nr_sessions) / nr_combinations)
    else:
        nr_trials_comb = 6

    tensors = get_X_tensors_at_t(
        available_set_trials, list_video_tensors, pseudopop_bool,
        list_videotrial_ids, t_idx, training_fraction,
        nr_trials_comb, training_bool, cross=False
    )

    y = get_y_for_X(combinations, conditions, decoding_var, tensors, pseudopop_bool)

    if pseudopop_bool:
        X_tensor = np.vstack(tensors)
        assert X_tensor.shape[0] == len(y)
    else:
        X_tensor = np.array([np.vstack(tensors[:, i]) for i in range(nr_sessions)])

    return X_tensor, y


# =============================================================================
# Geometry helpers
# =============================================================================

def get_pointcloud(datapoints, plot=True):
    """
    Build a PyntCloud point-cloud object from a (N, 3) coordinate array.

    Parameters
    ----------
    datapoints : array-like, shape (N, 3)
        XYZ coordinates.
    plot : bool, optional
        If True, display the cloud interactively.

    Returns
    -------
    PyntCloud
        Point cloud with columns ['x', 'y', 'z'].
    """
    from pyntcloud import PyntCloud
    cloud = PyntCloud(pd.DataFrame(datapoints, columns=['x', 'y', 'z']))
    if plot:
        cloud.plot()
    return cloud


def enclosed_volume(pointcloud):
    """
    Compute the convex-hull volume of a PyntCloud point cloud.

    Parameters
    ----------
    pointcloud : PyntCloud
        Input point cloud.

    Returns
    -------
    float
        Convex-hull volume.
    """
    hull_id = pointcloud.add_structure("convex_hull")
    return pointcloud.structures[hull_id].volume


# =============================================================================
# SVM training and testing
# =============================================================================

def train_decoder(X_tensor, y, shuffle=False, behaviour=False,
                  auc_param=False, dDR=False):
    """
    Train one LinearSVC per session on (optionally shuffled) labels.

    Handles NaN imputation, z-scoring (neural) or standard scaling
    (behavioural), and optionally fits a calibrated classifier for AUC.

    Parameters
    ----------
    X_tensor : np.ndarray
        (n_trials, n_features) for single session; (n_sessions, n_trials,
        n_features) for multi-session.
    y : np.ndarray
        Trial labels.
    shuffle : bool, optional
        Shuffle labels before fitting (null model).
    behaviour : bool, optional
        Use StandardScaler instead of z-scoring.
    auc_param : bool, optional
        Also fit a calibrated classifier for AUC computation.
    dDR : bool, optional
        Return the z-scored X matrix (for dimensionality-reduction analyses).

    Returns
    -------
    svm_classifiers : list of LinearSVC
    mean : list of np.ndarray (or np.nan for behavioural)
    std  : list of np.ndarray (or np.nan for behavioural)
    [calibrated_classifiers] : only when auc_param=True
    [X] : only when dDR=True (last session's normalised X)
    """
    # Normalise to list-of-sessions format
    if len(X_tensor.shape) == 2:
        X_tensor, y = [X_tensor], [y]
    nr_sessions = X_tensor.shape[0] if len(X_tensor.shape) > 2 else len(X_tensor)

    svm_classifiers      = []
    calibrated_classifiers = []
    means, stds          = [], []

    for i in range(nr_sessions):
        Xi = X_tensor[i]

        # Impute NaN values
        if np.isnan(Xi).any():
            Xi = SimpleImputer(missing_values=np.nan, strategy='constant',
                               fill_value=0).fit_transform(Xi)

        if behaviour:
            X  = StandardScaler().fit_transform(Xi)
            m  = np.nan
            st = np.nan
        else:
            m, st = stats_over_axis(Xi, ['trials', 'neurons'], 'trials')
            X     = zscore_neurons(Xi, ['trials', 'neurons'], m, st)

        yi = copy.deepcopy(y[i])
        if shuffle:
            np.random.shuffle(yi)

        clf = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000)
        clf.fit(X, yi)
        svm_classifiers.append(clf)
        means.append(m)
        stds.append(st)

        if auc_param:
            cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
            cal_clf.fit(X, yi)
            calibrated_classifiers.append(cal_clf)

    if auc_param:
        return svm_classifiers, means, stds, calibrated_classifiers
    if dDR:
        return svm_classifiers, means, stds, X
    return svm_classifiers, means, stds


def test_decoder(svm_classifier, X_test, y_test, mean, std,
                 behaviour=False, calibrated_classifier=None, dDR=False):
    """
    Evaluate trained SVM classifiers on held-out test data.

    Parameters
    ----------
    svm_classifier : list of LinearSVC
        Trained classifiers from `train_decoder`.
    X_test : np.ndarray
        Test feature matrix; shape matches training.
    y_test : np.ndarray
        Test labels.
    mean, std : list of np.ndarray or None
        Per-session normalisation statistics; None skips z-scoring.
    behaviour : bool, optional
        Use StandardScaler instead of stored mean/std.
    calibrated_classifier : list or None, optional
        Calibrated classifiers for AUC; None skips AUC computation.
    dDR : bool, optional
        Return the normalised test X.

    Returns
    -------
    svm_accuracy : float or list of float
        Accuracy per session (list) or scalar for single-session.
    [di_auc] : only when calibrated_classifier is not None.
    [X] : only when dDR=True.
    """
    nr_sessions = len(svm_classifier)

    def _normalise(Xi, i):
        if behaviour:
            return StandardScaler().fit_transform(Xi)
        if mean is not None:
            return zscore_neurons(Xi, ['trials', 'neurons'], mean[i], std[i])
        return Xi

    if nr_sessions > 1:
        svm_accuracy = []
        di_auc       = []
        for i in range(nr_sessions):
            X = _normalise(X_test[i], i)
            svm_accuracy.append(svm_classifier[i].score(X, y_test[i]))

            if calibrated_classifier is not None:
                probs    = calibrated_classifier[i].predict_proba(X)[:, 1]
                fp, tp, _ = roc_curve(y_test[i], probs)
                di_auc.append((auc(fp, tp) - 0.5) * 2)

    else:
        X            = _normalise(X_test, 0)
        svm_accuracy = svm_classifier[0].score(X, y_test)

        di_auc = None
        if calibrated_classifier is not None:
            probs    = calibrated_classifier[0].predict_proba(X)[:, 1]
            fp, tp, _ = roc_curve(y_test, probs)
            di_auc   = (auc(fp, tp) - 0.5) * 2

    if calibrated_classifier is None:
        return (svm_accuracy, X) if dDR else svm_accuracy
    else:
        return svm_accuracy, di_auc


# =============================================================================
# Video trace helpers
# =============================================================================

def get_video_traces(video_info, params, trial_session_entries,
                     timepoints_axis, event_align):
    """
    Extract aligned licking and movement boolean traces for each trial.

    Aligns video frames to the specified event (stimulus onset, choice, or
    ports-off) and extracts a fixed-length window of binary behaviour flags.

    Parameters
    ----------
    video_info : DataJoint table expression
        Row containing video data for the session.
    params : dict
        Unused directly; reserved for future parameterisation.
    trial_session_entries : DataJoint table expression
        Trial rows to process.
    timepoints_axis : array-like
        [window_start, ..., window_end] in seconds relative to the event.
    event_align : str
        'stimulus_on', 'choice', or 'ports_off'.

    Returns
    -------
    dlc_videodata : dict
        {'licking', 'movement', 'lick_left', 'lick_right'}
        each mapping trial_id → binary array of length nr_videoframes_trial.
    video_times : np.ndarray
        Mean frame times relative to event, shape (nr_videoframes_trial,).
    """
    # Fetch all video and trial data up front
    lick_bool       = video_info.fetch('lick_bool')[0]
    movement_bool   = list(video_info.fetch('movement_bool')[0])
    lick_side       = video_info.fetch('lick_side')[0]
    missed_trials   = video_info.fetch('missed_trials')[0]
    videoframe_times = video_info.fetch('videoframe_times')[0]

    trial_ids      = trial_session_entries.fetch('trial_id')
    trial_starts   = trial_session_entries.fetch('trial_start')
    ports_on       = trial_session_entries.fetch('ports_on')
    ports_off      = trial_session_entries.fetch('ports_off')
    stimulus_on    = trial_session_entries.fetch('stimulus_on')
    reaction_time  = trial_session_entries.fetch('reaction_time')
    trialtype      = trial_session_entries.fetch('trialtype')
    responsetype   = trial_session_entries.fetch('responsetype')

    trial_window_onset  = timepoints_axis[0]
    trial_window_offset = timepoints_axis[-1]
    nr_frames           = int((trial_window_offset - trial_window_onset) * 45) + 1  # 45 Hz

    # Align videoframe_times length to lick_bool if off-by-one
    if len(videoframe_times) != len(lick_bool):
        videoframe_times = videoframe_times[:-1]
    assert len(videoframe_times) == len(lick_bool)
    assert len(videoframe_times) == len(movement_bool)

    dlc_videodata = {k: {} for k in ['licking', 'movement', 'lick_left', 'lick_right']}
    video_times   = []

    for i, trial_id in enumerate(trial_ids):
        if trial_id in missed_trials:
            continue  # no video for missed trials

        # Determine event time for this trial
        if event_align == 'choice':
            event_time = (trial_starts[i] + ports_on[i][0] + reaction_time[i]
                          if responsetype[i] != 'no response' else None)
        elif event_align == 'stimulus_on':
            event_time = (trial_starts[i] + stimulus_on[i][0]
                          if trialtype[i] != 'control_nostimulus' else None)
        elif event_align == 'ports_off':
            event_time = trial_starts[i] + ports_off[i][0]
        else:
            event_time = None

        if event_time is None:
            continue

        # Find first video frame inside the analysis window
        start_idx = np.where(videoframe_times > (event_time + trial_window_onset))[0][0]
        frame_slice = slice(start_idx, start_idx + nr_frames)

        video_times.append(videoframe_times[frame_slice] - event_time)

        lick_bools   = lick_bool[frame_slice]
        mov_bools    = movement_bool[frame_slice]
        lick_sides   = lick_side[frame_slice]

        dlc_videodata['licking'][trial_id]    = lick_bools
        dlc_videodata['movement'][trial_id]   = mov_bools
        dlc_videodata['lick_left'][trial_id]  = [1 if s == 0 else 0 for s in lick_sides]
        dlc_videodata['lick_right'][trial_id] = [1 if s == 1 else 0 for s in lick_sides]

    return dlc_videodata, np.mean(video_times, axis=0)


def get_video_tensor(video_info, params, trial_session_entries,
                     timepoints_axis, event_align):
    """
    Stack video traces into a (n_features, n_trials, n_frames) tensor.

    Parameters
    ----------
    video_info : DataJoint table expression
        Video data row.
    params : dict
        Passed through to `get_video_traces`.
    trial_session_entries : DataJoint table expression
        Trial rows.
    timepoints_axis : array-like
        Analysis window timepoints.
    event_align : str
        Event to align to.

    Returns
    -------
    video_tensor : np.ndarray, shape (n_features, n_trials, n_frames)
    video_trial_ids : list
        Trial IDs with valid video data.
    video_times : np.ndarray
        Mean frame times relative to event.
    """
    dlc_videodata, video_times = get_video_traces(
        video_info, params, trial_session_entries, timepoints_axis, event_align
    )

    video_trial_ids = list(dlc_videodata['licking'].keys())
    features        = list(dlc_videodata.keys())
    nr_frames       = len(dlc_videodata['licking'][video_trial_ids[0]])

    video_tensor = np.zeros((len(features), len(video_trial_ids), nr_frames))
    for i, feat in enumerate(features):
        for j, trial_id in enumerate(video_trial_ids):
            video_tensor[i, j, :] = dlc_videodata[feat][trial_id]

    return video_tensor, video_trial_ids, video_times