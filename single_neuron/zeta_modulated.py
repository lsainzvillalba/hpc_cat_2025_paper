#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:04:21 2024

@author: Laura Sainz Villalba

# =============================================================================
# Single-neuron analysis pipeline: PETHs, ZETA modulation, selectivity, and
# functional labelling across training phases.
#
# Schemas / tables:
#   zeta_modulated_hpc_cat_2025
#     - Neuron_peths       : per-neuron peri-event time histograms
#     - Neuron_modulated   : ZETA responsiveness p-values per event/variable
#     - Selectivity_analysis : selectivity indices and null distributions
#   neuron_labels
#     - Labels             : functional label per neuron
#
# Module-level population functions:
#   populate_neuronmodulated  – fills Neuron_modulated
#   populate_selectivity      – fills Selectivity_analysis
#   populate_labels           – fills Labels
# =============================================================================
"""

import os, sys, inspect
import numpy as np
import datajoint as dj
from time import time
from multiprocessing import Pool
from tqdm import tqdm
from zetapy.ifr_dependencies import getPeak

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from .utils_single import (get_trials_by_combination, get_neuron_ids_phase,
                            Zeta_analysis, compute_selective_diff_significance,
                            get_similarity_responses, compute_selectivity_index,
                            compute_null_index)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_import import Raw_trialtensor_data, Aligned_trialtensor_data
from utilities import select_tensor_by_axis, select_timepoints_byframe, bootstrap_pval

# ---------------------------------------------------------------------------
# DataJoint configuration – two schemas used in this module
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema  = dj.schema('zeta_modulated_hpc_cat_2025', locals(), create_tables=True)
schema2 = dj.schema('neuron_labels',               locals(), create_tables=True)

# ---------------------------------------------------------------------------
# Experiment-wide constants
# ---------------------------------------------------------------------------
ANIMAL_IDS = ['BK4947_R', 'BK4936_L', 'BK4956_LR',
              'BK4933_LR', 'BK4937_R', 'BK4926_L']

PHASES = ['discrimination', 'gentest_1', 'categorization_4', 'gentest_2']

# Variables analysed in selectivity and labelling pipelines
STATE_VARS = ['category', 'choice', 'outcome']

# Minimum trials per combination required to process a session
MIN_TRIALS_PER_COMBINATION = 3

# Number of trial-type combinations expected per session
NR_TRIALTYPES = 4


# =============================================================================
# Shared session-key helpers
# =============================================================================

def _build_session_keys(table):
    """
    Build a list of session primary keys and a neuron-registration dictionary
    for all animals and phases that have entries in *table*.

    For each (animal_id, phase) pair that has exactly two sessions, those
    session IDs are stored in `nr_session_dict` to enable cross-session
    neuron registration.

    Parameters
    ----------
    table : DataJoint table class
        Table to query (e.g. Raw_trialtensor_data).

    Returns
    -------
    session_keys : list of dict
        Primary-key dicts for the most recent two sessions per animal/phase.
    nr_session_dict : dict
        {(animal_id, phase): [session_id_1, session_id_2]} for pairs with
        exactly two sessions available.
    """
    session_keys    = []
    nr_session_dict = {}

    for animal_id in ANIMAL_IDS:
        for phase in PHASES:
            entries = (table
                       & f'animal_id="{animal_id}"'
                       & f'experimental_timepoint="{phase}"')
            if len(entries) == 0:
                continue

            selected = entries.fetch(dj.key)[-2:]
            session_keys += list(selected)

            if len(selected) == 2:
                nr_session_dict[(animal_id, phase)] = [k['session_id'] for k in selected]

    return session_keys, nr_session_dict


def _check_session_completeness(session_key):
    """
    Verify that a session has the required number of trial-type combinations
    and sufficient trials per combination.

    Parameters
    ----------
    session_key : dict
        DataJoint primary-key dict identifying the session.

    Returns
    -------
    bool
        True if the session passes both checks; False otherwise.
    trials_by_combination : dict
        Mapping from combination label to list of trial IDs.
    """
    activity_entry      = Raw_trialtensor_data() & session_key
    rec_trial_ids       = activity_entry.fetch('tensor_dim_values')[0][1]
    trials_by_combination = get_trials_by_combination(session_key, rec_trial_ids)

    combinations_complete = len(trials_by_combination) == NR_TRIALTYPES
    enough_trials         = all(
        len(t) > MIN_TRIALS_PER_COMBINATION
        for t in trials_by_combination.values()
    )
    return combinations_complete and enough_trials, trials_by_combination


def _resolve_neuron_ids(mask_ids, session_key, nr_session_dict):
    """
    Return neuron IDs for *mask_ids*, using cross-session registration when
    available.

    Parameters
    ----------
    mask_ids : array-like
        Mask IDs from the recording session.
    session_key : dict
        Primary-key dict for the session.
    nr_session_dict : dict
        Registration dictionary from `_build_session_keys`.

    Returns
    -------
    array-like
        Neuron IDs aligned with *mask_ids*.
    """
    animal_id = session_key['animal_id']
    phase     = session_key['experimental_timepoint']
    if (animal_id, phase) in nr_session_dict:
        return get_neuron_ids_phase(mask_ids, session_key,
                                    nr_session_dict[(animal_id, phase)])
    return mask_ids


# =============================================================================
# Neuron_peths – Manual DataJoint table
# =============================================================================

@schema
class Neuron_peths(dj.Manual):
    definition = """ # Per-neuron peri-event time histograms across trial-type combinations

    animal_id              : varchar(128)   # Mouse unique identifier
    mask_id                : int            # Mask ID for this neuron in the session
    experimental_timepoint : varchar(256)   # Training phase label
    session_id             : int            # Session ID within the phase
    ---
    combinations   : blob      # Trial-type combination labels [category, choice, outcome]
    neuron_id=NULL : int        # Cross-session neuron ID (NULL if not registered)
    time           : longblob  # Timepoints within the analysis window (frames)
    peths          : longblob  # Activity tensor: (n_combinations x n_alignments x n_trials x n_timepoints)
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_aligned_tensors(key, window_events):
        """
        Fetch aligned activity tensors for each event window.

        Parameters
        ----------
        key : dict
            Session primary-key dict.
        window_events : list of str
            Event-alignment labels (e.g. ['stimulus_on', 'choice']).

        Returns
        -------
        tensor_activity : list of np.ndarray
            One tensor per event window.
        trial_ids_window : list of array-like
            Trial IDs per window.
        timepoints : list of array-like
            Timestamps per window.
        tensor_format : str
            Format string from the last fetched entry.
        mask_ids : array-like
            Mask IDs from the last fetched entry.
        """
        tensor_activity  = []
        trial_ids_window = []
        timepoints       = []

        for window in window_events:
            tensor_entry = (Aligned_trialtensor_data() & key
                            & f'event_to_align="{window}"')
            tensor       = tensor_entry.fetch('tensor_data')[0]
            mask_ids, trial_ids, timestamps = tensor_entry.fetch('tensor_dim_values')[0]

            assert tensor.shape == (len(mask_ids), len(trial_ids), len(timestamps))

            tensor_activity.append(tensor)
            trial_ids_window.append(trial_ids)
            timepoints.append(timestamps)

        tensor_format = tensor_entry.fetch('tensor_format')[0]
        return tensor_activity, trial_ids_window, timepoints, tensor_format, mask_ids

    @staticmethod
    def _build_peths(combinations, trials_by_combination, window_events,
                     tensor_activity, trial_ids_window, timepoints,
                     time_window_frames, nr_timepoints, tensor_format, mask_ids):
        """
        Slice tensors by trial-type combination and event window to build PETHs.

        Parameters
        ----------
        combinations : list
            Ordered list of trial-type combination labels.
        trials_by_combination : dict
            Trial IDs per combination.
        window_events : list of str
            Event-alignment labels.
        tensor_activity : list of np.ndarray
            Aligned tensors per window.
        trial_ids_window : list
            Trial IDs per window.
        timepoints : list
            Timestamps per window.
        time_window_frames : list of int
            [start, end] frame indices defining the analysis window.
        nr_timepoints : int
            Expected number of timepoints after slicing.
        tensor_format : str
            Axis-ordering format string for `select_tensor_by_axis`.
        mask_ids : array-like
            Mask IDs to iterate over.

        Returns
        -------
        peths : list
            Nested list [n_neurons][n_combinations][n_alignments] of trial arrays.
        window_timepoints : array-like
            Selected timepoint indices for the analysis window.
        """
        n_neurons    = len(mask_ids)
        n_combos     = len(combinations)
        n_alignments = len(window_events)

        # Initialise empty PETH structure: n_neurons x n_combos x n_alignments
        peths = [[[[] for _ in range(n_alignments)]
                  for _ in range(n_combos)]
                 for _ in range(n_neurons)]

        window_timepoints = None

        for i, combination in enumerate(combinations):
            type_trial_ids = trials_by_combination[combination]

            for j, _ in enumerate(window_events):
                timestamps        = timepoints[j]
                trial_ids         = trial_ids_window[j]
                window_timepoints = select_timepoints_byframe(timestamps, time_window_frames)

                # Select matching trials then matching timepoints
                sliced = select_tensor_by_axis(
                    tensor_activity[j], tensor_format,
                    trial_ids, type_trial_ids, 'trials'
                )
                sliced = select_tensor_by_axis(
                    sliced, tensor_format,
                    timestamps, time_window_frames, 'frames'
                )
                assert sliced.shape == (n_neurons, len(type_trial_ids), nr_timepoints)

                for k in range(n_neurons):
                    peths[k][i][j] = sliced[k]

        return peths, window_timepoints

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parallel_peths(self, key, window_events, nr_session_dict,
                       nr_eventsalignments, nr_trialtypes,
                       time_window_frames, nr_timepoints):
        """
        Compute and store PETHs for all neurons in one session.

        Fetches aligned tensors, slices by trial-type combination and time
        window, and inserts one row per neuron into the table.

        Parameters
        ----------
        key : dict
            Session primary-key dict (must include animal_id,
            experimental_timepoint, session_id).
        window_events : list of str
            Event-alignment labels to process.
        nr_session_dict : dict
            Cross-session registration dictionary.
        nr_eventsalignments : int
            Number of event alignment windows (len(window_events)).
        nr_trialtypes : int
            Expected number of trial-type combinations.
        time_window_frames : list of int
            [start, end] frame indices for the analysis window.
        nr_timepoints : int
            Expected number of timepoints after slicing.
        """
        activity_entry = Raw_trialtensor_data() & key
        rec_trial_ids  = activity_entry.fetch('tensor_dim_values')[0][1]
        trials_by_combination = get_trials_by_combination(key, rec_trial_ids)
        combinations   = list(trials_by_combination.keys())
        key['combinations'] = combinations

        # Fetch aligned tensors for each event window
        (tensor_activity, trial_ids_window,
         timepoints, tensor_format, mask_ids) = self._fetch_aligned_tensors(
            key, window_events
        )

        # Resolve neuron IDs (with cross-session registration if available)
        neuron_ids = _resolve_neuron_ids(mask_ids, key, nr_session_dict)

        # Build PETH structure
        peths, window_timepoints = self._build_peths(
            combinations, trials_by_combination, window_events,
            tensor_activity, trial_ids_window, timepoints,
            time_window_frames, nr_timepoints, tensor_format, mask_ids
        )

        # Insert one row per neuron
        for k in tqdm(range(len(mask_ids))):
            neuron_entry             = key.copy()
            neuron_entry['mask_id']  = mask_ids[k]
            neuron_entry['neuron_id'] = neuron_ids[k]
            neuron_entry['peths']    = peths[k]
            neuron_entry['time']     = window_timepoints
            self.insert1(neuron_entry, skip_duplicates=True)

    def populate_peths(self):
        """
        Populate the table for all sessions across animals and phases.

        For each session, checks that the full set of trial-type combinations
        is present with sufficient trials before computing PETHs.
        """
        window_events       = ['stimulus_on', 'choice']
        nr_eventsalignments = len(window_events)
        time_window_frames  = [-15, 45]   # analysis window in frames
        nr_timepoints       = time_window_frames[1] - time_window_frames[0]

        session_keys, nr_session_dict = _build_session_keys(Raw_trialtensor_data)

        for session_key in session_keys:
            if len(Neuron_peths() & session_key) != 0:
                continue  # already processed

            complete, _ = _check_session_completeness(session_key)
            if complete:
                print(f'Processing session: {session_key}')
                self.parallel_peths(
                    session_key, window_events, nr_session_dict,
                    nr_eventsalignments, NR_TRIALTYPES,
                    time_window_frames, nr_timepoints
                )


# =============================================================================
# Neuron_modulated – Manual DataJoint table
# =============================================================================

@schema
class Neuron_modulated(dj.Manual):
    definition = """ # ZETA responsiveness p-values per neuron, event window, and trial subset

    animal_id              : varchar(128)   # Mouse unique identifier
    mask_id                : int            # Mask ID for this neuron
    experimental_timepoint : varchar(256)   # Training phase label
    session_id             : int            # Session ID within the phase
    ---
    neuron_id=NULL  : int       # Cross-session neuron ID
    pval_stim       : float     # ZETA p-value for stimulus-aligned window (all trials)
    pval_choice     : float     # ZETA p-value for choice-aligned window (all trials)
    pval_cat_a      : tinyblob  # ZETA p-values for category A trials [stim, choice]
    pval_cat_b      : tinyblob  # ZETA p-values for category B trials [stim, choice]
    pval_cho_l      : tinyblob  # ZETA p-values for left-choice trials [stim, choice]
    pval_cho_r      : tinyblob  # ZETA p-values for right-choice trials [stim, choice]
    pval_out_nrw    : tinyblob  # ZETA p-values for no-reward trials [stim, choice]
    pval_out_rw     : tinyblob  # ZETA p-values for reward trials [stim, choice]
    """


# =============================================================================
# ZETA session analysis (module-level, not a table method)
# =============================================================================

def zeta_session_entries(session_key, nr_session_dict):
    """
    Run ZETA tests for all neurons in one session and insert into Neuron_modulated.

    For each event alignment window and trial-grouping variable, runs parallel
    ZETA tests across all mask IDs, then assembles per-neuron p-value vectors
    and inserts one row per neuron.

    Parameters
    ----------
    session_key : dict
        Session primary-key dict.
    nr_session_dict : dict
        Cross-session registration dictionary from `_build_session_keys`.
    """
    start          = time()
    activity_entry = Raw_trialtensor_data() & session_key
    events         = ['stimulus_on', 'choice']
    variables      = ['all', 'category', 'choice', 'outcome']

    # Maps each variable to the p-value fields it populates
    pval_label_dict = {
        'all':     ['pval_stim',    'pval_choice'],
        'category':['pval_cat_a',   'pval_cat_b'],
        'choice':  ['pval_cho_l',   'pval_cho_r'],
        'outcome': ['pval_out_nrw', 'pval_out_rw'],
    }

    # Initialise accumulators: scalar lists for 'all', nested lists for others
    neuron_entry_dict = {
        'pval_stim':    [],
        'pval_choice':  [],
        'pval_cat_a':   [[], []],
        'pval_cat_b':   [[], []],
        'pval_cho_l':   [[], []],
        'pval_cho_r':   [[], []],
        'pval_out_nrw': [[], []],
        'pval_out_rw':  [[], []],
    }

    zeta_session = Zeta_analysis(activity_entry)

    for i, event in enumerate(events):
        zeta_session.set_to_alignment_event(event)

        for var in variables:
            pval_labels = pval_label_dict[var]
            zeta_session.select_by_variable(var)

            # Parallel ZETA computation across all neurons
            with Pool(10) as pool:
                list_pvals = pool.starmap(
                    zeta_session.compute_for_mask,
                    [(mask_id,) for mask_id in zeta_session.mask_ids]
                )

            if var == 'all':
                # One scalar p-value per neuron for this window
                neuron_entry_dict[pval_labels[i]] += list_pvals
            else:
                # Two p-values per neuron (one per variable value)
                list_pvals = np.array(list_pvals)
                for j, pval_label in enumerate(pval_labels):
                    neuron_entry_dict[pval_label][i] += list(list_pvals[:, j])

    # Sanity check: each variable-value list should have one entry per neuron
    assert len(neuron_entry_dict['pval_out_rw'][0]) == len(zeta_session.mask_ids)

    neuron_ids = _resolve_neuron_ids(zeta_session.mask_ids, session_key, nr_session_dict)
    print(f'ZETA session time: {time() - start:.1f} s')

    for i in tqdm(range(len(zeta_session.mask_ids))):
        entry = session_key.copy()
        entry['mask_id']      = zeta_session.mask_ids[i]
        entry['neuron_id']    = neuron_ids[i]
        entry['pval_stim']    = neuron_entry_dict['pval_stim'][i]
        entry['pval_choice']  = neuron_entry_dict['pval_choice'][i]
        # Convert nested [event][neuron] lists to per-neuron [event] vectors
        for field in ['pval_cat_a', 'pval_cat_b', 'pval_cho_l',
                      'pval_cho_r', 'pval_out_nrw', 'pval_out_rw']:
            entry[field] = list(np.array(neuron_entry_dict[field])[:, i])

        Neuron_modulated().insert1(entry, skip_duplicates=True)


def populate_neuronmodulated():
    """
    Populate Neuron_modulated for all sessions not yet processed.

    """
    session_keys, nr_session_dict = _build_session_keys(Raw_trialtensor_data)

    for session_key in session_keys:
        if len(Neuron_modulated() & session_key) != 0:
            continue  # already processed

        complete, _ = _check_session_completeness(session_key)
        if complete:
            zeta_session_entries(session_key, nr_session_dict)


# =============================================================================
# Selectivity_analysis – Manual DataJoint table
# =============================================================================

@schema
class Selectivity_analysis(dj.Manual):
    definition = """ # Selectivity indices and null distributions per neuron and variable

    animal_id              : varchar(128)   # Mouse unique identifier
    mask_id                : int            # Mask ID for this neuron
    experimental_timepoint : varchar(256)   # Training phase label
    session_id             : int            # Session ID within the phase
    ---
    neuron_id=NULL  : int        # Cross-session neuron ID
    pval_cat_sel    : tinyblob   # Selectivity difference p-values for category [n_alignments]
    similarity_cat  : blob       # Similarity matrices [n_metrics x n_alignments x n_combos x n_combos]
    sel_index_cat   : tinyblob   # Selectivity indices [n_alignments x n_metrics]
    pval_cat_index  : tinyblob   # Index p-values [n_alignments x n_metrics]
    null_cat_index  : longblob   # Null index distributions [n_alignments x n_metrics x n_iters]
    pval_cho_sel    : tinyblob   # Selectivity difference p-values for choice
    similarity_cho  : blob       # Similarity matrices for choice
    sel_index_cho   : tinyblob   # Selectivity indices for choice
    pval_cho_index  : tinyblob   # Index p-values for choice
    null_cho_index  : longblob   # Null distributions for choice
    pval_out_sel    : tinyblob   # Selectivity difference p-values for outcome
    similarity_out  : blob       # Similarity matrices for outcome
    sel_index_out   : tinyblob   # Selectivity indices for outcome
    pval_out_index  : tinyblob   # Index p-values for outcome
    null_out_index  : longblob   # Null distributions for outcome
    """


# =============================================================================
# Selectivity computation (module-level)
# =============================================================================

def neuron_selectivity(neuron_key):
    """
    Compute selectivity indices and null distributions for one neuron.

    For each state variable (category, choice, outcome) and each event
    alignment window, this function:
      1. Tests significance of response difference between variable values.
      2. Computes cosine and Euclidean similarity matrices across combinations.
      3. Derives a selectivity index from the similarity structure.
      4. Builds a null distribution by permutation and computes a bootstrap p-value.

    Parameters
    ----------
    neuron_key : dict
        Primary-key dict identifying the neuron row in Neuron_peths.

    Returns
    -------
    dict
        Entry dict ready for insertion into Selectivity_analysis.
    """
    metrics   = ['cosine', 'euclidean']
    events    = ['stimulus_on', 'choice']
    nr_iters  = 1000    # permutation iterations for null distribution
    n_vars    = len(STATE_VARS)
    n_events  = len(events)
    n_metrics = len(metrics)

    neuron_entry  = Neuron_peths() & neuron_key
    combinations  = neuron_entry.fetch('combinations')[0]
    x_time        = neuron_entry.fetch('time')[0]
    peths         = neuron_entry.fetch('peths')[0]   # n_combos x n_alignments
    neuron_id     = neuron_entry.fetch('neuron_id')[0]
    n_combos      = len(combinations)

    # First post-event timepoint index
    ref0 = np.where(x_time > 0)[0][0]

    # Pre-allocate result arrays
    pval_sel   = np.zeros((n_vars, n_events))
    similarity = np.zeros((n_vars, n_events, n_metrics, n_combos, n_combos))
    sel_index  = np.zeros((n_vars, n_events, n_metrics))
    pval_index = np.zeros((n_vars, n_events, n_metrics))
    null_index = np.zeros((n_vars, n_events, n_metrics, nr_iters))

    # Restrict all PETHs to the post-event period
    for i, combination in enumerate(combinations):
        for j in range(n_events):
            peths[i][j] = peths[i][j][:, ref0:]

    for var_idx, var in enumerate(STATE_VARS):
        for j in range(n_events):
            # Group PETHs by the two values of this variable
            peths_by_var = [[], []]
            for i, combination in enumerate(combinations):
                bool_idx = combination[var_idx]
                peths_by_var[bool_idx].append(peths[i][j])
            peths_by_var = [np.vstack(g) for g in peths_by_var]

            # Significance of response difference between variable values
            pval_sel[var_idx, j] = compute_selective_diff_significance(
                x_time[ref0:], x_time[ref0:],
                peths_by_var[0], peths_by_var[1]
            )

            for k, metric in enumerate(metrics):
                # Similarity matrix across all combination pairs
                sims = get_similarity_responses(peths, j, metric, combinations)
                similarity[var_idx, j, k] = sims

                # Selectivity index derived from the similarity structure
                sel_index[var_idx, j, k] = compute_selectivity_index(
                    sims, metric, combinations, var
                )

                # Null distribution via permutation and bootstrap p-value
                null = compute_null_index(peths, j, metric, combinations, var)
                null_index[var_idx, j, k] = null
                pval_index[var_idx, j, k] = bootstrap_pval(sel_index[var_idx, j, k], null)

    # Assemble output entry (one slice per variable)
    entry = neuron_key.copy()
    entry['neuron_id'] = neuron_id

    for v_idx, var_str in enumerate([v[:3] for v in STATE_VARS]):
        entry[f'pval_{var_str}_sel']   = list(pval_sel[v_idx])
        entry[f'similarity_{var_str}'] = similarity[v_idx]
        entry[f'sel_index_{var_str}']  = list(sel_index[v_idx])
        entry[f'pval_{var_str}_index'] = pval_index[v_idx]
        entry[f'null_{var_str}_index'] = null_index[v_idx]

    return entry


def populate_selectivity():
    """
    Populate Selectivity_analysis in parallel for all neurons in Neuron_peths.

    """
    neuron_keys = (Neuron_peths()).fetch(dj.key)

    with Pool(2) as pool:
        entries = pool.starmap(neuron_selectivity, [(k,) for k in neuron_keys])

    for entry in entries:
        Selectivity_analysis().insert1(entry, skip_duplicates=True)


# =============================================================================
# Labels – Manual DataJoint table (schema2)
# =============================================================================

@schema2
class Labels(dj.Manual):
    definition = """ # Functional label assigned to each neuron based on selectivity profile

    animal_id              : varchar(128)   # Mouse unique identifier
    mask_id                : int            # Mask ID for this neuron
    experimental_timepoint : varchar(256)   # Training phase label
    session_id             : int            # Session ID within the phase
    ---
    neuron_id=NULL : int           # Cross-session neuron ID
    label          : varchar(128)  # Functional label (e.g. 'category*outcome', 'not modulated')
    """


# =============================================================================
# Neuron labelling (module-level)
# =============================================================================

# Maps each variable to its ZETA field and the event-window index to evaluate
_VAR_WINDOW = {
    'category': ('pval_stim',   0),
    'choice':   ('pval_choice', 1),
    'outcome':  ('pval_choice', 1),
}

# Maps each variable to its two value labels used in ZETA field names
_VAR_VALUES = {
    'category': ('a',   'b'),
    'choice':   ('l',   'r'),
    'outcome':  ('nrw', 'rw'),
}

# For singly selective neurons, specifies which variable to test for interaction
_INTERACTION_VAR = {
    'category': 'outcome',
    'choice':   'outcome',
    'outcome':  'choice',
}


def compute_label(neuron_key):
    """
    Assign a functional label to one neuron based on its selectivity profile.

    Label logic (applied in order):
      1. For each state variable, check whether the selectivity difference test
         and at least one similarity-index test are significant.
      2. If so, check the global ZETA test; if not significant, check the
         per-value ZETA tests as a fallback.
      3. Build a label string by joining significant variables with '*'.
      4. For singly selective neurons, test for an interaction with the
         corresponding interaction variable; require significance by both
         similarity metrics.
      5. If no selectivity is found, classify as 'stimulus', 'choice
         responsive', 'outcome responsive', or 'not modulated' based on the
         global ZETA tests and the window with the highest peak response.

    Parameters
    ----------
    neuron_key : dict
        Primary-key dict identifying the neuron row in Neuron_peths.

    Returns
    -------
    dict
        Entry dict ready for insertion into Labels.
    """
    selectivity_entry = Selectivity_analysis() & neuron_key
    zeta_entry        = Neuron_modulated()     & neuron_key
    peth_entry        = Neuron_peths()         & neuron_key

    x_time      = peth_entry.fetch('time')[0]
    ref0        = np.where(x_time > 0)[0][0]
    combinations = peth_entry.fetch('combinations')[0]
    peths        = peth_entry.fetch('peths')[0]
    neuron_id    = peth_entry.fetch('neuron_id')[0]

    # ------------------------------------------------------------------
    # Step 1–3: build selectivity label
    # ------------------------------------------------------------------
    label = ''

    for var in STATE_VARS:
        var_str    = var[:3]
        window_str, window_idx = _VAR_WINDOW[var]

        pval_sel   = selectivity_entry.fetch(f'pval_{var_str}_sel')[0]
        pval_index = selectivity_entry.fetch(f'pval_{var_str}_index')[0]
        index_sig  = [p < 0.05 for p in pval_index[window_idx]]

        print(f'var: {var} | diff_test: {pval_sel[window_idx]:.3f} | index_sig: {index_sig}')

        if not (pval_sel[window_idx] < 0.05 and True in index_sig):
            continue  # neither test significant – skip this variable

        # Check global ZETA significance for the relevant window
        pval_window = zeta_entry.fetch(window_str)[0]
        print(f'  global ZETA p-value: {pval_window:.3f}')

        if pval_window < 0.05:
            is_significant = True
        else:
            # Fallback: check per-value ZETA tests
            val_sigs = [
                zeta_entry.fetch(f'pval_{var_str}_{val}')[0][window_idx] < 0.05
                for val in _VAR_VALUES[var]
            ]
            print(f'  per-value sigs: {val_sigs}')
            is_significant = True in val_sigs

        if is_significant:
            label = label + ('*' if label else '') + var

    print(f'Preliminary label: {label}')

    # ------------------------------------------------------------------
    # Step 4: test interaction for singly selective neurons
    # ------------------------------------------------------------------
    if label and '*' not in label:
        var_str    = label[:3]
        window_str, window_idx = _VAR_WINDOW[label]
        pval_index = selectivity_entry.fetch(f'pval_{var_str}_index')[0]

        # Interaction test requires significance by both similarity metrics
        if all(p < 0.05 for p in pval_index[window_idx]):
            int_var  = _INTERACTION_VAR[label]
            var_idx  = STATE_VARS.index(label)
            _, int_window_idx = _VAR_WINDOW[int_var]

            # Group PETHs by the two values of the primary variable
            peths_by_var = [[], []]
            for i, combination in enumerate(combinations):
                peths_by_var[combination[var_idx]].append(peths[i][int_window_idx])

            int_bool  = []
            pvals_int = []
            for n in range(2):
                peth_0, peth_1 = peths_by_var[n]
                pval_int = compute_selective_diff_significance(
                    x_time[ref0:], x_time[ref0:],
                    peth_0[:, ref0:], peth_1[:, ref0:]
                )
                pvals_int.append(pval_int)
                int_bool.append(pval_int < 0.01)

            print(f'  interaction p-values: {pvals_int} | sig: {int_bool}')

            if True in int_bool:
                label += '*' + int_var
        else:
            # Both metrics must be significant; otherwise label is invalid
            label = ''

    # ------------------------------------------------------------------
    # Step 5: fallback classification for non-selective neurons
    # ------------------------------------------------------------------
    if not label:
        pval_stim   = zeta_entry.fetch('pval_stim')[0]
        pval_choice = zeta_entry.fetch('pval_choice')[0]
        sig_bools   = [pval_stim < 0.05, pval_choice < 0.05]
        print(f'  fallback ZETA sigs: {sig_bools}')

        if not any(sig_bools):
            label = 'not modulated'

        elif all(sig_bools):
            # Both windows significant: assign to whichever has the higher peak
            peths_by_window = [
                np.vstack([peths[i][w] for i in range(len(combinations))])
                for w in range(2)
            ]
            window_peaks = []
            for w in range(2):
                mean_trace = np.mean(peths_by_window[w], axis=0)
                assert len(mean_trace) == len(x_time)
                peak_info = getPeak(mean_trace, x_time)
                if np.isnan(peak_info['dblPeakValue']):
                    window_peaks.append(max(mean_trace[0], mean_trace[-1]))
                else:
                    window_peaks.append(peak_info['dblPeakValue'])

            print(f'  window peaks: {window_peaks}')
            label = 'choice responsive' if np.argmax(window_peaks) else 'stimulus'

        else:
            label = 'stimulus' if sig_bools[0] else 'outcome responsive'

    print(f'Final label: {label}')

    entry             = neuron_key.copy()
    entry['neuron_id'] = neuron_id
    entry['label']     = label
    return entry


def populate_labels():
    """
    Populate Labels in parallel for all neurons in Neuron_peths.

    Uses 8 worker processes to call `compute_label` for every neuron key.
    """
    neuron_keys = Neuron_peths().fetch(dj.key)

    with Pool(8) as pool:
        entries = pool.starmap(compute_label, [(k,) for k in neuron_keys])

    for entry in entries:
        Labels().insert1(entry, skip_duplicates=True)
   