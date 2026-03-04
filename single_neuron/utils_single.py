#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:58:34 2024

@author: Laura Sainz Villalba

# =============================================================================
# utils_single.py
# Utility functions and the Zeta_analysis class supporting single-neuron
# analysis: similarity metrics, selectivity indices, ZETA responsiveness
# tests, trial-combination bookkeeping, and neuron registration lookups.
#
# Public API:
#   get_similarity_responses         – pairwise similarity matrix across PETHs
#   compute_selective_diff_significance – ZETA-style permutation test
#   compute_selectivity_index        – normalised within/across group index
#   compute_null_index               – permutation null distribution
#   get_event_timestamps             – behavioural event times per trial
#   get_trials_by_combination        – trial IDs grouped by condition tuple
#   get_neuron_ids_phase             – cross-session neuron IDs via registration
#   Zeta_analysis                    – stateful ZETA test runner per session
# =============================================================================
"""

import os, sys, inspect
import numpy as np
import datajoint as dj
import pandas as pd
from zetapy import zetatstest
from zetapy.ts_dependencies import getTimeseriesOffsetTwo, uniquetol
from zetapy.dependencies import getZetaP

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utilities import select_tensor_by_axis, get_condition_response_variable
from data_import import Trial, Neuron_registration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Variables whose binary values define the trial-type combination tuples
CONDITION_VARS = ['category', 'choice', 'outcome']


# =============================================================================
# Similarity and selectivity helpers
# =============================================================================

def get_similarity_responses(responses, window_idx, metric, combinations):
    """
    Compute a pairwise similarity matrix between mean PETHs for each
    trial-type combination.

    Parameters
    ----------
    responses : list
        Nested list of trial arrays. Shape depends on *window_idx*:
          - If window_idx is not None: responses[combo][window][trial, time]
          - If window_idx is None:     responses[combo][trial, time]
    window_idx : int or None
        Index of the event-alignment window to use, or None when responses
        are already single-window arrays.
    metric : str
        Similarity measure: 'cosine' or 'euclidean'.
    combinations : list
        Ordered list of trial-type combination labels.

    Returns
    -------
    np.ndarray, shape (n_combinations, n_combinations)
        Pairwise similarity values between combination mean responses.
    """
    n_combos = len(combinations)

    # Validate structure and infer timepoint count
    if window_idx is not None:
        assert len(responses[0]) == 2, "Expected two alignment windows."
        nr_timepoints = len(responses[0][0][0])
    else:
        assert len(responses[0]) != 2, "window_idx=None requires single-window responses."
        nr_timepoints = len(responses[0][0])

    # Compute mean response across trials for each combination
    means = np.zeros((n_combos, nr_timepoints))
    for i in range(n_combos):
        src = responses[i][window_idx] if window_idx is not None else responses[i]
        means[i] = np.mean(src, axis=0)

    # Define the chosen metric as a lambda for clean pairwise application
    if metric == 'euclidean':
        sim_fn = lambda x, y: np.linalg.norm(np.array(x) - np.array(y))
    elif metric == 'cosine':
        sim_fn = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'cosine' or 'euclidean'.")

    similarities = np.zeros((n_combos, n_combos))
    for i in range(n_combos):
        for j in range(n_combos):
            similarities[i, j] = sim_fn(means[i], means[j])

    return similarities


def compute_selective_diff_significance(vecRefT1, vecRefT2,
                                        matTracePerTrial1, matTracePerTrial2):
    """
    Test whether two sets of neural traces differ significantly using a
    ZETA-style permutation approach (Montijn et al.).

    Computes the maximum absolute timeseries offset (dblMaxD) between the
    two trial groups, then estimates its null distribution by randomly
    reassigning trials across groups.

    Reference: https://github.com/JorritMontijn/zetapy

    Parameters
    ----------
    vecRefT1 : array-like
        Reference timepoints for the first trial group.
    vecRefT2 : array-like
        Reference timepoints for the second trial group.
    matTracePerTrial1 : np.ndarray, shape (n_trials_1, n_timepoints)
        Neural traces for the first group.
    matTracePerTrial2 : np.ndarray, shape (n_trials_2, n_timepoints)
        Neural traces for the second group.

    Returns
    -------
    float
        p-value; low values indicate a significant difference between groups.
    """
    # Parameters for the permutation test
    dblSuperResFactor  = 100
    intResampNum       = 250
    boolDirectQuantile = False

    # Build a common high-resolution time reference
    dblSampInterval = (np.median(np.diff(vecRefT1)) + np.median(np.diff(vecRefT2))) / 2.0
    dblTol          = dblSampInterval / dblSuperResFactor
    vecRefTime      = uniquetol(np.concatenate((vecRefT1, vecRefT2)), dblTol)

    # Observed maximum absolute difference
    vecRealDiff, _, _ = getTimeseriesOffsetTwo(matTracePerTrial1, matTracePerTrial2)
    dblMaxD           = np.abs(vecRealDiff[np.argmax(np.abs(vecRealDiff))])

    # Permutation null distribution
    matAggregateTrials = np.concatenate((matTracePerTrial1, matTracePerTrial2), axis=0)
    intTrials1         = matTracePerTrial1.shape[0]
    intTrials2         = matTracePerTrial2.shape[0]
    intTotTrials       = intTrials1 + intTrials2

    vecMaxRandD = np.full((intResampNum, 1), np.nan)

    for r in range(intResampNum):
        idx1     = np.random.randint(intTotTrials, size=intTrials1)
        idx2     = np.random.randint(intTotTrials, size=intTrials2)
        randDiff, _, _ = getTimeseriesOffsetTwo(
            matAggregateTrials[idx1], matAggregateTrials[idx2]
        )
        add_val              = np.max(np.abs(randDiff))
        vecMaxRandD[r]       = add_val if (add_val is not None and add_val != 0) else dblMaxD

    dblZetaP, _ = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)
    return dblZetaP


def compute_selectivity_index(similarities, metric, combinations, var):
    """
    Compute a normalised selectivity index for *var* from a similarity matrix.

    Groups combination pairs into 'within-group' (same variable value) and
    'across-group' (different values), then computes:
        index = (mean_within – mean_across) / (mean_within + mean_across)

    For Euclidean distance, the index is negated so that higher always means
    more selective (large distance = dissimilar = across-group is larger).

    Parameters
    ----------
    similarities : np.ndarray, shape (n_combos, n_combos)
        Pairwise similarity matrix from `get_similarity_responses`.
    metric : str
        'cosine' or 'euclidean' (sign convention differs).
    combinations : list
        Ordered combination labels.
    var : str
        Variable to compute selectivity for ('category', 'choice', 'outcome').

    Returns
    -------
    float
        Selectivity index in [–1, 1]; higher = more selective.
    """
    var_idx  = CONDITION_VARS.index(var)
    n_combos = len(combinations)

    # Group combination indices by their value for *var*
    grouped = [[], []]
    for i, combination in enumerate(combinations):
        grouped[combination[var_idx]].append(i)

    # All unique ordered pairs
    all_pairs    = [[i, j] for i in range(n_combos) for j in range(i + 1, n_combos)]
    grouped_set  = set(map(tuple, grouped))

    # Within-group: both indices share the same variable value
    within_pairs  = [p for p in all_pairs if tuple(p) in grouped_set]
    # Across-group: indices belong to different variable values
    across_pairs  = [p for p in all_pairs if tuple(p) not in grouped_set]

    mean_within = np.nanmean([similarities[a, b] for a, b in within_pairs])
    mean_across = np.nanmean([similarities[a, b] for a, b in across_pairs])

    index = (mean_within - mean_across) / (mean_within + mean_across)

    # Euclidean distance: larger distance = less similar, so flip sign
    if metric == 'euclidean':
        index = -index

    return index


def compute_null_index(peths, window_idx, metric, combinations, var):
    """
    Build a permutation null distribution for the selectivity index.

    Randomly shuffles trial rows across combinations (preserving trial counts)
    and recomputes the selectivity index on each shuffle.

    Parameters
    ----------
    peths : list
        PETH arrays indexed as peths[combination][window][trial, time].
    window_idx : int
        Event-alignment window to use.
    metric : str
        'cosine' or 'euclidean'.
    combinations : list
        Ordered combination labels.
    var : str
        Variable to compute selectivity for.

    Returns
    -------
    list of float
        Null selectivity index values (length = nr_iter).
    """
    nr_iter   = 1000
    n_combos  = len(combinations)

    # Stack all trials into a single pool and record original counts
    peth_list = [peths[i][window_idx] for i in range(n_combos)]
    nr_trials = [len(p) for p in peth_list]
    all_peths = np.vstack(peth_list)
    assert len(all_peths) == np.sum(nr_trials)

    null = []
    for _ in range(nr_iter):
        np.random.shuffle(all_peths)      # shuffle in-place along trial axis

        # Reconstruct per-combination arrays with the same trial counts
        shuffled, counter = [], 0
        for nr in nr_trials:
            shuffled.append(all_peths[counter:counter + nr])
            counter += nr

        assert [len(s) for s in shuffled] == nr_trials

        sims  = get_similarity_responses(shuffled, None, metric, combinations)
        index = compute_selectivity_index(sims, metric, combinations, var)
        null.append(index)

    return null


# =============================================================================
# Trial and event helpers
# =============================================================================

def get_event_timestamps(session_key, rec_trial_ids):
    """
    Retrieve absolute timestamps for key behavioural events for a set of trials.

    Event times are stored relative to trial_start in the Trial table, so each
    is converted to an absolute time by adding trial_start.

    Parameters
    ----------
    session_key : dict
        DataJoint key identifying the session.
    rec_trial_ids : array-like
        Trial IDs that were recorded (subset to process).

    Returns
    -------
    dict with keys: 'trial_id', 'trial_start', 'stimulus_on', 'ports_on',
                    'reward', 'choice'
        Each value is a list aligned to rec_trial_ids; entries are None when
        the event did not occur (e.g. no response, no reward).
    """
    trials = Trial() & session_key

    event_dict = {k: [] for k in
                  ['trial_id', 'trial_start', 'stimulus_on',
                   'ports_on', 'reward', 'choice']}

    for trial_id in rec_trial_ids:
        entry       = trials & f'trial_id="{trial_id}"'
        trial_start = entry.fetch('trial_start')[0]
        event_dict['trial_id'].append(trial_id)
        event_dict['trial_start'].append(trial_start)

        # Stimulus onset
        stimulus_on = entry.fetch('stimulus_on')[0]
        event_dict['stimulus_on'].append(
            trial_start + stimulus_on[0] if stimulus_on is not None else None
        )

        # Ports-on (decision window opens)
        ports_on = entry.fetch('ports_on')[0]
        event_dict['ports_on'].append(
            trial_start + ports_on[0] if ports_on is not None else None
        )

        # Reward delivery
        water = entry.fetch('water')[0]
        event_dict['reward'].append(
            trial_start + water[0] if water is not None else None
        )

        # Choice time: ports_on + reaction_time (None for no-response trials)
        responsetype  = entry.fetch('responsetype')[0]
        reaction_time = entry.fetch('reaction_time')[0]
        event_dict['choice'].append(
            trial_start + ports_on + reaction_time
            if responsetype != 'no response' else None
        )

    return event_dict


def get_trials_by_combination(key, selected_trial_ids):
    """
    Group trial IDs by their (category, choice, outcome) binary combination.

    Only active trials with a committed response that overlap with
    *selected_trial_ids* are included. Combinations with no matching trials
    are removed from the output.

    Parameters
    ----------
    key : dict
        DataJoint key identifying the session.
    selected_trial_ids : array-like
        Trial IDs present in the neural recording.

    Returns
    -------
    dict
        {(cat, cho, out): [trial_id, ...]} where each key is a 3-tuple of
        binary values (0 or 1) for category, choice, and outcome respectively.
    """
    trial_entries = (Trial() & key
                     & 'trialtype="active"'
                     & 'responsetype!="no response"')

    port_layout = trial_entries.fetch('port_layout')[0]

    # Restrict to trials that were actually recorded
    trial_ids = list(
        np.intersect1d(list(trial_entries.fetch('trial_id')), selected_trial_ids)
    )

    # DataJoint field names for each condition variable
    condition_fields = [get_condition_response_variable(c) for c in CONDITION_VARS]

    # Initialise all 2^3 = 8 possible combinations
    trials_by_combination = {
        tuple(int(b) for b in format(i, f'0{len(CONDITION_VARS)}b')): []
        for i in range(1 << len(CONDITION_VARS))
    }

    for trial_id in trial_ids:
        entry      = trial_entries & f'trial_id="{trial_id}"'
        combination = []

        for i, (condition, field) in enumerate(zip(CONDITION_VARS, condition_fields)):
            val = entry.fetch(field)[0]

            if condition in ('category', 'choice'):
                # Flip value when port layout is inverted
                if port_layout:
                    val = 1 - val
            elif condition == 'outcome':
                val = 1 if val == 'correct' else 0

            combination.append(val)

        trials_by_combination[tuple(combination)].append(trial_id)

    # Remove empty combinations (not all 8 may occur in a session)
    return {k: v for k, v in trials_by_combination.items() if v}


def get_neuron_ids_phase(mask_ids, key, comparison):
    """
    Resolve cross-session neuron IDs for a list of mask IDs.

    Looks up the Neuron_registration table to map mask IDs in the current
    session to consistent neuron IDs across the two-session comparison.
    Falls back to returning mask_ids directly if no registration entry exists.

    Parameters
    ----------
    mask_ids : list
        Mask IDs from the current recording session.
    key : dict
        DataJoint key containing 'animal_id' and 'session_id'.
    comparison : list of int
        The two session IDs in the registered pair, e.g. [sess_id_1, sess_id_2].

    Returns
    -------
    list
        Neuron IDs aligned with *mask_ids*; entries are None for unregistered
        neurons.
    """
    session_id  = key['session_id']
    animal_id   = key['animal_id']
    session_idx = comparison.index(session_id)
    comp_str    = f'{comparison[0]}-{comparison[1]}'

    reg_entry = (Neuron_registration()
                 & f'animal_id="{animal_id}"'
                 & f'comparison="{comp_str}"')

    if len(reg_entry) == 0:
        return mask_ids   # no registration available; use mask IDs as-is

    all_neuron_ids, pair_mask_ids = reg_entry.fetch('neuron_id', 'mask_ids')

    # Extract the mask ID for the current session from each registered pair
    registered_mask_ids = [pair[session_idx] for pair in pair_mask_ids]

    neuron_ids = []
    for mask_id in mask_ids:
        if mask_id in registered_mask_ids:
            neuron_ids.append(all_neuron_ids[registered_mask_ids.index(mask_id)])
        else:
            neuron_ids.append(None)   # neuron not registered across sessions

    return neuron_ids


# =============================================================================
# Zeta_analysis – stateful ZETA test runner
# =============================================================================

class Zeta_analysis:
    """
    Stateful wrapper for running ZETA time-series tests on one recording session.

    Loads the neural tensor and behavioural events once at construction, then
    exposes `set_to_alignment_event` and `select_by_variable` to configure the
    analysis window and trial grouping before calling `compute_for_mask` in
    parallel for each neuron.

    Parameters
    ----------
    activity_entry : DataJoint table expression
        Single-session row from Raw_trialtensor_data.

    Attributes
    ----------
    mask_ids : list
        Accepted mask IDs (neurons) in the session.
    activity_tensor : np.ndarray
        Neural traces shape (n_neurons, n_trials, n_timepoints).
    timepoints : np.ndarray
        Per-trial timepoint arrays, shape (n_trials, n_timepoints).
    trials_by_combination : dict
        Trial IDs grouped by condition combination.
    event_dict : dict
        Absolute event timestamps per trial.
    """

    # ZETA test hyper-parameters
    _USE_MAX_DUR   = 1.5    # analysis window after event onset (seconds)
    _RESAMP_NUM    = 150    # number of jitter resamplings
    _JITTER_SIZE   = 6.0    # jitter magnitude

    def __init__(self, activity_entry):
        print('Loading tensor ...')
        tensortype      = 'tensor_data_dff'
        activity_tensor = np.array(activity_entry.fetch(tensortype)[0], dtype=object)
        print('Done.')

        session_key         = activity_entry.fetch(dj.key)[0]
        self.tensor_format  = activity_entry.fetch('tensor_format')[0]
        assert self.tensor_format == ['neurons', 'trials', 'timepoints']

        mask_ids, self.rec_trial_ids, timepoints = \
            activity_entry.fetch('tensor_dim_values')[0]

        # Keep only accepted (quality-filtered) neurons
        accepted      = activity_entry.fetch('accepted')[0]
        accepted_bool = [bool(ac) for ac in accepted]
        self.mask_ids         = list(np.array(mask_ids)[accepted_bool])
        self.activity_tensor  = activity_tensor[accepted_bool]
        assert len(self.activity_tensor) == len(self.mask_ids)

        self.timepoints = np.array(timepoints)

        # Pre-compute trial groupings and event timestamps (used across calls)
        self.trials_by_combination = get_trials_by_combination(
            session_key, self.rec_trial_ids
        )
        self.event_dict = get_event_timestamps(session_key, self.rec_trial_ids)

    # ------------------------------------------------------------------
    # Configuration methods (call before compute_for_mask)
    # ------------------------------------------------------------------

    def set_to_alignment_event(self, event):
        """
        Set the event to align neural data to for subsequent ZETA tests.

        Builds the list of active trial IDs (committed responses, present in
        the recording) and stores the corresponding event timestamps.

        Parameters
        ----------
        event : str
            Event key in `event_dict` (e.g. 'stimulus_on', 'choice').
        """
        self.event = event
        self.event_timestamps_starts = np.array(
            self.event_dict[event], dtype=object
        )
        assert len(self.event_timestamps_starts) == len(self.rec_trial_ids)

        # Union of all active trial IDs intersected with recorded trials
        active_trial_ids = [t for trials in self.trials_by_combination.values()
                            for t in trials]
        self.trial_ids_event = list(
            set(active_trial_ids) & set(self.rec_trial_ids)
        )

    def select_by_variable(self, var):
        """
        Select trials and prepare data for the ZETA test for *var*.

        For var='all', uses all active recorded trials as a single group.
        For other variables, splits trials into two groups based on the
        binary value of that variable.

        Parameters
        ----------
        var : str
            'all', 'category', 'choice', or 'outcome'.
        """
        self.var = var

        if var == 'all':
            # Single trial group: all active recorded trials
            idx_set = pd.Index(self.rec_trial_ids).get_indexer(self.trial_ids_event)

            self.selected_tensor = select_tensor_by_axis(
                self.activity_tensor, self.tensor_format,
                self.rec_trial_ids, self.trial_ids_event, 'trials'
            )
            self.trial_ids = self.trial_ids_event

            selected_timepoints  = np.take(self.timepoints, idx_set, axis=0)
            self.event_timestamps = np.take(
                self.event_timestamps_starts, idx_set, axis=0
            )[1:-2].astype(float)
            self.vecTimestamps = np.concatenate(selected_timepoints)

            assert len(selected_timepoints) == len(self.trial_ids_event)

        else:
            # Two trial groups split by the binary value of *var*
            var_idx      = CONDITION_VARS.index(var)
            combinations = list(self.trials_by_combination.keys())

            self.trial_ids       = [[], []]
            self.selected_tensor = [[], []]
            self.event_timestamps = [[], []]
            self.vecTimestamps   = [[], []]

            for combination in combinations:
                bool_idx = combination[var_idx]
                self.trial_ids[bool_idx] += self.trials_by_combination[combination]

            for i, trials_var in enumerate(self.trial_ids):
                # Intersect with recorded trials
                trials_var         = list(set(trials_var) & set(self.rec_trial_ids))
                self.trial_ids[i]  = trials_var

                self.selected_tensor[i] = select_tensor_by_axis(
                    self.activity_tensor, self.tensor_format,
                    self.rec_trial_ids, trials_var, 'trials'
                )
                idx_set = pd.Index(self.rec_trial_ids).get_indexer(trials_var)
                selected_timepoints         = np.take(self.timepoints, idx_set, axis=0)
                self.event_timestamps[i]    = np.take(
                    self.event_timestamps_starts, idx_set, axis=0
                )[1:-2].astype(float)
                self.vecTimestamps[i]       = np.concatenate(selected_timepoints)

                assert len(selected_timepoints) == len(trials_var)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_for_mask(self, mask_id):
        """
        Run the ZETA time-series test for one neuron.

        For var='all', runs a single test across all active trials.
        For other variables, runs one test per binary value of the variable
        and returns both p-values.

        Parameters
        ----------
        mask_id : int
            Mask ID of the neuron to test.

        Returns
        -------
        float or list of float
            Single p-value for var='all'; [p_val_0, p_val_1] for other vars.
        """
        m_idx    = self.mask_ids.index(mask_id)
        zeta_kw  = dict(
            dblUseMaxDur       = self._USE_MAX_DUR,
            intResampNum       = self._RESAMP_NUM,
            boolPlot           = False,
            dblJitterSize      = self._JITTER_SIZE,
            boolDirectQuantile = False,
        )

        if self.var == 'all':
            vec_data   = np.concatenate(self.selected_tensor[m_idx])
            pvalue, _  = zetatstest(
                self.vecTimestamps, vec_data, self.event_timestamps, **zeta_kw
            )
            return pvalue

        else:
            pvalues = []
            for i in range(2):
                vec_data  = np.concatenate(self.selected_tensor[i][m_idx])
                pvalue, _ = zetatstest(
                    self.vecTimestamps[i], vec_data,
                    self.event_timestamps[i], **zeta_kw
                )
                pvalues.append(pvalue)
            return pvalues
