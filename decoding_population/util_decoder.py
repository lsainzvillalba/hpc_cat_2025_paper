#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:06:02 2024

@author: Laura Sainz Villalba


Utility functions shared across all decoding modules.  Covers:

  1.  Time-rebinning helpers (``rebin_times``, ``rebin_data``)
  2.  Session-key gathering and filtering
     (``get_session_keys``, ``get_valid_sessions``, ``get_nr_subsample_neurons``,
      ``get_common_time_axis_params``)
  3.  Session-data assembly (``sessions_info_to_decode``)
  4.  Single-iteration decoder workers, designed to be called via
      ``multiprocessing.Pool.starmap``:
        - ``one_iter_crosstime_decode``   : cross-time SVM decoding
        - ``one_iter_cross_var``          : cross-variable CCGP + parallelism score
        - ``one_iter_geometry_intime``    : geometry (CCGP / PS) at one timepoint
        - ``one_iter_behavior_decode_at_t``: behavioural SVM decoding at one timepoint
        - ``one_iter_all_window_decode``  : window-averaged decoding + CCGP
        - ``window_decode``               : window decoding with dDR projection
        - ``window_decode_allvars``       : joint all-variable window decoding
  5.  Multi-CV wrappers that call the workers above inside a Pool:
        - ``cross_var_at_t``             : cross-variable geometry at one timepoint
        - ``acc_geom_estimation_at_t``   : accuracy + geometry at one timepoint
        - ``behaviour_decode``           : full behavioural decoding sweep
"""

# import libraries
import os, sys, inspect
import numpy as np
from tqdm import tqdm
import datajoint as dj
from multiprocessing import Pool
from sklearn.svm import LinearSVC
import copy
from sklearn.impute import SimpleImputer

# ── Add parent directory to path so data_import and utilities can be found ──
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_import import Trial, Aligned_trialtensor_data, Dlc_videoinfo
from utilities import rebin_time_in_tensor, get_video_tensor, get_trials_by_conds, \
            get_dataset, get_property, train_decoder, test_decoder, get_trial_conditions, \
                get_available_set_trial_ids, get_dichotomies, angle_between_vectors, \
                    get_beh_dataset, get_X_dataset, stats_over_axis, zscore_neurons, \
                        get_y_for_X

# ===========================================================================
# TIME-REBINNING HELPERS
# ===========================================================================

def rebin_times(timepoints, bin_window, overlap_window):
    """
    Compute the starting frame indices and mean timestamps for a sliding
    window binning of a 1-D timepoints array.

    Parameters
    ----------
    timepoints     : array-like  Original per-frame timestamps.
    bin_window     : int         Number of frames per bin.
    overlap_window : int         Step size between consecutive bin starts
                                 (frames; overlap = bin_window - overlap_window).

    Returns
    -------
    bin_starts          : list of int    First frame index of each bin.
    rebinned_timepoints : list of float  Mean timestamp within each bin.
    """
    nr_timepoints = len(timepoints)

    # Advance bin start by (bin_window - overlap_window) each step
    bin_starts = [0]
    while bin_starts[-1] < (nr_timepoints - bin_window):
        new_start = bin_starts[-1] + bin_window - overlap_window
        bin_starts.append(new_start)

    # Drop the last bin if it would be shorter than bin_window
    if (nr_timepoints - bin_starts[-1]) < bin_window:
        bin_starts = bin_starts[:-1]

    # print('bin',bin_starts[-1],nr_timepoints)

    # Compute mean timestamp for each bin
    rebinned_timepoints = []
    for i, bin_idx in enumerate(bin_starts):
        timepoint_window = np.array(timepoints[bin_idx: bin_idx + bin_window])
        rebinned_timestamp = timepoint_window.mean()
        rebinned_timepoints.append(rebinned_timestamp)

    return bin_starts, rebinned_timepoints


def rebin_data(imputer, tensor, timepoints, bin_window, overlap_window):
    """
    Sliding-window average a 3-D tensor (features × trials × timepoints),
    imputing NaN values within each bin before averaging.

    Parameters
    ----------
    imputer        : sklearn imputer  Fitted/fittable imputer (e.g. SimpleImputer).
    tensor         : ndarray          Shape (features, trials, timepoints).
    timepoints     : array-like       Per-frame timestamps (length = tensor.shape[-1]).
    bin_window     : int              Frames per bin.
    overlap_window : int              Step size between bin starts.

    Returns
    -------
    rebinned_tensor : ndarray  Shape (features, trials, nr_bins).
    """
    # features x trials x timepoints
    dims = list(tensor.shape)
    nr_timepoints = dims[-1]
    assert len(timepoints) == nr_timepoints

    # Compute bin layout
    bin_starts, rebinned_timepoints = rebin_times(timepoints, bin_window, overlap_window)
    nr_bins = len(bin_starts)

    # Allocate output tensor (same dims, last axis = nr_bins)
    # features x trials x timepoints
    rebinned_dims = dims
    rebinned_dims[-1] = nr_bins
    rebinned_tensor = np.zeros(tuple(rebinned_dims))
    # print('rebinned_tensor',rebinned_tensor.shape)

    for i, bin_idx in enumerate(bin_starts):
        window_tensor = tensor[:, :, bin_idx: bin_idx + bin_window]
        # print('window_tensor.shape',window_tensor.shape)
        # print('timestemps', nr_timesteps)

        # Impute NaNs for each frame within the bin, then average across frames
        window = []
        for t in range(bin_window):
            window.append(imputer.fit_transform(window_tensor[:, :, t]))
        window = np.array(window)
        # print('window.shape',window.shape)
        rebinned_tensor_frame = window.mean(axis=0)
        # print('rebinned_tensor_frame.shape',rebinned_tensor_frame.shape)
        rebinned_tensor[:, :, i] = rebinned_tensor_frame

    rebinned_tensor = np.array(rebinned_tensor)
    # print('rebinned_tensor.shape',rebinned_tensor.shape)
    assert len(rebinned_timepoints) == rebinned_tensor.shape[-1]

    return rebinned_tensor


# ===========================================================================
# SESSION MANAGEMENT HELPERS
# ===========================================================================

def get_subsampling_idx(sessions_info, n_neurons):
    """
    Draw random neuron subsampling indices for each session tensor.

    Parameters
    ----------
    sessions_info : dict
        Must contain 'list_tensors' (list of per-session tensors) and
        'session_keys' (list of key dicts) if ``n_neurons`` is a dict.
    n_neurons : int or dict
        If int, the same number of neurons is sampled from every session.
        If dict, keys are animal_ids and values are per-animal sample counts.

    Returns
    -------
    subsampling_idx : list of ndarray
        Per-session arrays of randomly drawn neuron indices.
    """
    list_tensors = sessions_info['list_tensors']

    if isinstance(n_neurons, dict):
        # Map per-animal sample counts to the session order in list_tensors
        animal_ids = [key['animal_id'] for key in sessions_info['session_keys']]
        n_neurons  = [n_neurons[animal_id] for animal_id in animal_ids]
    else:
        assert isinstance(n_neurons, int)
        n_neurons = [n_neurons] * len(list_tensors)

    subsampling_idx = [np.random.randint(0, len(list_tensors[i]), n_neurons[i])
                       for i in range(len(list_tensors))]
    return subsampling_idx


def shuffle_available_set(available_set_trials):
    """
    Randomly shuffle the trial ID lists within each trial-combination slot,
    then return a deep copy so the original is not mutated.

    Parameters
    ----------
    available_set_trials : dict
        {combination_label: [list_of_trial_ids_per_session]}.

    Returns
    -------
    frozen_shuffle : dict  Deep copy with shuffled trial orderings.
    """
    # print(available_set_trials.keys())
    for combination in available_set_trials:
        # print(len(available_set_trials[combination]))
        for i in range(len(available_set_trials[combination])):
            np.random.shuffle(available_set_trials[combination][i])
    # Freeze the shuffle state so parallel workers see a consistent copy
    frozen_shuflle = copy.deepcopy(available_set_trials)
    return frozen_shuflle


def get_session_keys(params, event_to_align):
    """
    Gather the DB session keys to include in the decoding analysis.

    For each phase and subject, at most ``nr_sessions_phase`` sessions are
    kept (the most recent ones)

    Parameters
    ----------
    params        : dict  Must contain 'phases', 'nr_sessions_phase'
    event_to_align : str  Alignment event label (e.g. 'choice', 'stimulus_on').

    Returns
    -------
    session_keys_dict : dict  {phase: {animal_id: [key_dicts]}}.
    """
    phases            = params['phases']
    nr_sessions_phase = params['nr_sessions_phase']

    # Fetch all aligned-tensor entries for this event and trace type
    recording_entries = Aligned_trialtensor_data() \
        & 'event_to_align="%s"' % event_to_align 

    session_keys_dict = {}
    for phase in phases:
        session_keys_dict[phase] = {}
        entries_phase = recording_entries & 'experimental_timepoint="%s"' % phase
        animal_ids    = get_property(entries_phase, 'animal_id')

        # ── Keep at most the last nr_sessions_phase sessions per subject ───────
        for animal_id in animal_ids:
            entries_subject = entries_phase & 'animal_id="%s"' % animal_id
            _, _, timepoints = entries_subject.fetch('tensor_dim_values')[0]
            # print('timepoints[-1]',timepoints[-1])
            session_keys_dict[phase][animal_id] = entries_subject.fetch(dj.key)[-nr_sessions_phase:]

    return session_keys_dict


def get_valid_sessions(session_keys, conditions, min_nr_trials, nr_combinations, split=None):
    """
    Filter session keys to those with enough trials per condition combination.

    Iterates over all sessions, checks whether each has at least
    ``min_nr_trials`` trials in every condition combination, and removes
    sessions (and empty subjects) that do not pass the check.  The dict is
    modified in-place; returns None.

    Parameters
    ----------
    session_keys    : dict   {phase: {animal_id: [key_dicts]}} — modified in-place.
    conditions      : list   Variable names defining the condition space.
    min_nr_trials   : int    Minimum trials required per combination.
    nr_combinations : int    Number of condition combinations to check.
    split           : optional  Passed through to get_trials_by_conds.
    """
    nr_valid_sessions = 0
    assert isinstance(session_keys, dict)
    total_sessions = np.sum([len(session_keys[phase][animal_id])
                             for phase in session_keys
                             for animal_id in session_keys[phase]])
    pbar0 = tqdm(total=total_sessions, position=0, leave=True)
    empty = []  # collect (phase, animal_id) pairs to remove after iteration

    for phase in session_keys:
        for animal_id in session_keys[phase]:
            valid_sessions = []
            sessions = session_keys[phase][animal_id]

            for key in sessions:
                # print('before trial cond dict')
                trialset       = Trial() & key
                trial_cond_dict = get_trial_conditions(trialset, conditions, stim_of_int=None)
                # print('after trial cond dict')
                trials_by_conds = get_trials_by_conds(trial_cond_dict, key,
                                                      conditions, min_nr_trials,
                                                      nr_combinations, split)

                # get_trials_by_conds returns float (nan) when the session is invalid
                if not isinstance(trials_by_conds, float):
                    valid_sessions.append(True)
                    nr_valid_sessions += 1
                else:
                    valid_sessions.append(False)
                    #print('invalid key ', key)

                pbar0.update(1)

            # Replace session list with only the valid subset
            valid = list(np.array(sessions)[valid_sessions])
            if len(valid) != 0:
                session_keys[phase][animal_id] = valid
            else:
                empty.append([phase, animal_id])

    # Remove subjects that ended up with no valid sessions
    for phase, animal_id in empty:
        print('EMPTY')
        print('animal_id', animal_id)
        print('phase', phase)
        del session_keys[phase][animal_id]

    print('nr valid sessions ', nr_valid_sessions)
    return


def get_nr_subsample_neurons(session_keys, subsampling_fr=0.9, nr_sessions_phase=2,
                             across_axis='phases'):
    """
    Determine how many neurons to subsample per session so that decoding
    comparisons are fair across subjects / phases.

    The subsampling count is set to ``subsampling_fr × min_total_neurons``
    where the minimum is taken either across phases (``across_axis='phases'``)
    or across subjects within each phase (``across_axis='subjects'``).
    The per-session count is then the phase/subject total divided equally
    across sessions.

    Parameters
    ----------
    session_keys      : dict   {phase: {animal_id: [key_dicts]}}.
    subsampling_fr    : float  Fraction of the minimum neuron count to use.
    nr_sessions_phase : int    How many sessions per phase to count neurons from.
    across_axis       : str    'phases' or 'subjects'.

    Returns
    -------
    nr_neurons_session : dict   {phase: {animal_id: int}}  Neurons per session.
    n_datapoints       : int    2 × total subsample (used for pseudopopulation size).
    """
    assert isinstance(session_keys, dict)
    nr_neurons_session = {}

    if across_axis == 'phases':
        # ── Count total neurons per phase across all subjects ─────────────────
        nr_neurons_phase = []
        for phase in session_keys:
            print('phases ', phase)
            nr_neurons = 0
            for animal_id in session_keys[phase]:
                for key in session_keys[phase][animal_id]:
                    entry_subject = Aligned_trialtensor_data() & key
                    nr_neurons += np.sum(entry_subject.fetch('nr_accepted')[-nr_sessions_phase:])
            nr_neurons_phase.append(nr_neurons)
            print('nr_neurons_phase ', nr_neurons_phase)

        # Subsample to a fraction of the phase with fewest neurons
        subsampling = int(subsampling_fr * np.min(nr_neurons_phase))

        # Divide the budget equally across subjects and sessions within each phase
        for i, phase in enumerate(session_keys):
            nr_neurons_session[phase] = {}
            nr_subject = len(session_keys[phase])
            for j, animal_id in enumerate(session_keys[phase]):
                nr_sessions_subject = len(session_keys[phase][animal_id])
                nr_neurons_session[phase][animal_id] = int(subsampling / (nr_subject * nr_sessions_subject))

    elif across_axis == 'subjects':
        # ── Count total neurons per subject across all phases ─────────────────
        nr_neurons_subject = []
        for phase in session_keys:
            for animal_id in session_keys[phase]:
                nr_neurons = 0
                for key in session_keys[phase][animal_id]:
                    entry_subject = Aligned_trialtensor_data() & key
                    nr_neurons += np.sum(entry_subject.fetch('nr_accepted')[-nr_sessions_phase:])
                if nr_neurons != 0:
                    nr_neurons_subject.append(nr_neurons)

        # Subsample to a fraction of the subject with fewest neurons
        subsampling = int(subsampling_fr * np.min(nr_neurons_subject))

        # Divide the budget equally across sessions for each subject
        for i, phase in enumerate(session_keys):
            # print('phase', phase)
            nr_neurons_session[phase] = {}
            for j, animal_id in enumerate(session_keys[phase]):
                nr_sessions_subject = len(session_keys[phase][animal_id])
                if nr_sessions_subject != 0:
                    nr_neurons_session[phase][animal_id] = int(subsampling / nr_sessions_subject)

    n_datapoints = 2 * subsampling   # pseudopopulation size = 2 × subsample budget

    return nr_neurons_session, n_datapoints


def get_common_time_axis_params(session_keys):
    """
    Compute the common cropping bounds so all sessions share the same
    number of timepoints relative to t=0 (the alignment event).

    For each session the index of t=0 is located; the final left/right ranges
    are the minimum across all sessions, guaranteeing every session has
    enough timepoints on both sides.

    Parameters
    ----------
    session_keys : dict  {phase: {animal_id: [key_dicts]}}.

    Returns
    -------
    zero_left  : dict   {phase: {animal_id: [zero_left_idx per session]}}.
    left_range : int    Minimum number of frames before t=0 across all sessions.
    right_range: int    Minimum number of frames after  t=0 across all sessions.
    """
    zero_left      = {}
    all_zero_lefts  = []
    all_zero_rights = []

    for i, phase in enumerate(session_keys):
        zero_left[phase] = {}
        for j, animal_id in enumerate(session_keys[phase]):
            zero_left[phase][animal_id] = []
            for key in session_keys[phase][animal_id]:
                entry_subject = Aligned_trialtensor_data() & key
                _, _, timepoints = entry_subject.fetch('tensor_dim_values')[0]
                timepoints     = np.array(timepoints)
                # Index of the first frame after t=0
                zero_left_idx  = np.where(timepoints > 0)[0][0]
                zero_left[phase][animal_id].append(zero_left_idx)
                all_zero_lefts.append(zero_left_idx)
                all_zero_rights.append(len(timepoints) - 1 - zero_left_idx)

    left_range  = np.min(all_zero_lefts)
    right_range = np.min(all_zero_rights)

    return zero_left, left_range, right_range


# ===========================================================================
# SESSION DATA ASSEMBLY
# ===========================================================================

def sessions_info_to_decode(session_keys, params):
    """
    Load and pre-process all neural and behavioural tensors for decoding.

    For each session in ``session_keys`` the function:
      1. Fetches the aligned neural tensor from DB
      2. Crops to the common time window and rebins
      3. Optionally loads and rebins the DLC video tensor
      4. Handles the special 'exception' condition

    The returned ``sessions_info`` dict centralises everything the decoding
    workers need, avoiding repeated DB fetches during the decode loop.

    Parameters
    ----------
    session_keys : list or dict
        list  → per-subject (within-subject) sessions
        dict  → {animal_id: [key_dicts]} (across-subject / pseudopopulation)
    params : dict
        Must contain: 'zero_left', 'left_range', 'right_range', 'bin_window',
        'overlap_window', 'nr_combinations', 'event_align',
        'variables', 'min_nr_trials_comb'.  Optionally 'split'.

    Returns
    -------
    sessions_info : dict
        Keys: 'session_keys', 'port_layouts', 'timepoint_axis', 'list_tensors',
              'list_trial_ids', 'list_mask_ids', 'available_set_trials',
              'list_trial_cond_dict', 'list_video_tensors', 'list_videotrial_ids',
              'video_times', 'exception_cond', 'list_stimuli_id'.
    """
    # ── Determine animal list and processing mode ─────────────────────────────
    if isinstance(session_keys, list):   # within-subject: single animal
        animal_ids = [session_keys[0]['animal_id']]
    elif isinstance(session_keys, dict): # across-subject: pseudopopulation
        animal_ids = (session_keys.keys())

    nr_subjects = len(animal_ids)
    assert nr_subjects != 0

    # ── Unpack params ─────────────────────────────────────────────────────────
    zero_left      = params['zero_left']
    left_range     = params['left_range']
    right_range    = params['right_range']
    bin_window     = params['bin_window']
    overlap_window = params['overlap_window']
    nr_combinations = params['nr_combinations']
    event_align    = params['event_align']
    conditions     = params['variables']
    # print('conditions in session info',conditions)
    min_nr_trials  = params['min_nr_trials_comb']

    split = params['split'] if 'split' in params else None

    # ── Per-session accumulators ──────────────────────────────────────────────
    list_tensors        = []
    list_trial_ids      = []
    list_mask_ids       = []
    list_trial_conds    = []
    list_trial_cond_dict = []
    list_video_tensors  = []
    list_videotrial_ids = []
    list_videotimepoints = []
    timepoint_axis      = []
    list_sessionkeys    = []
    list_portlayouts    = []
    list_exception_cond = []
    list_stimuli_id     = []

    #print('animal_ids',animal_ids)
    for animal_id in animal_ids:
        # Select the session list depending on single- vs multi-subject mode
        if nr_subjects != 1:
            subject_sessions = session_keys[animal_id]
        else:
            subject_sessions = session_keys

        for i, key in enumerate(subject_sessions):

            # ── Fetch neural tensor ───────────────────────────────────────────
            tensor_entry  = Aligned_trialtensor_data() & key
            tensor_data   = tensor_entry.fetch('tensor_data')[0]
            tensor_format = tensor_entry.fetch('tensor_format')[0]
            assert tensor_format == ['neurons', 'trials', 'timepoints']
            mask_ids, trial_ids, timepoints = tensor_entry.fetch('tensor_dim_values')[0]

            # Build trial-condition mapping for this session
            trial_cond_dict = get_trial_conditions(Trial() & key, conditions, stim_of_int=None)
            # print('after trial cond dict')
            trials_by_conds = get_trials_by_conds(trial_cond_dict, key,
                                                  conditions, min_nr_trials,
                                                  nr_combinations, split)
            assert not isinstance(trials_by_conds, float)

            # ── Crop to common time window ────────────────────────────────────
            # print(tensor_data.shape)
            if nr_subjects != 1:
                zero_left = params['zero_left'][animal_id]
            else:
                zero_left = params['zero_left']
            # Keep [zero_left - left_range : zero_left + right_range] frames
            tensor_data = tensor_data[:, :, zero_left[i]-left_range: zero_left[i]+right_range]
            timepoints  = timepoints[zero_left[i]-left_range: zero_left[i]+right_range]
            # print(timepoints[0],timepoints[-1])

            # ── Rebin neural tensor in time ───────────────────────────────────
            rebinned_tensor, rebinned_timepoints = rebin_time_in_tensor(
                tensor_data, tensor_format,
                timepoints, bin_window, overlap_window,
            )
            # print(rebinned_tensor.shape)
            # print(rebinned_timepoints)
            assert len(rebinned_timepoints) == rebinned_tensor.shape[-1]

            # ── Load and rebin DLC video tensor (if available) ─────────────────
            video_info = Dlc_videoinfo() & key
            if len(video_info) != 0:
                assert len(video_info) == 1
                video_tensor, videotrial_ids, video_times = get_video_tensor(
                    video_info, params, Trial() & key,
                    rebinned_timepoints, event_align,
                )

                # Convert bin sizes from neural Hz (30 Hz) to video Hz (45 Hz)
                beh_bin_window     = round(bin_window     * (45/30))
                beh_overlap_window = round(overlap_window * (45/30))

                # Rebin video tensor to match neural temporal resolution
                _, rebinned_videotimes = rebin_times(video_times, beh_bin_window,
                                                     beh_overlap_window)
                imputer = SimpleImputer(missing_values=np.nan, strategy='constant',
                                        fill_value=0)
                rebinned_video_tensor = rebin_data(imputer, video_tensor,
                                                   video_times, beh_bin_window,
                                                   beh_overlap_window)
                # print('nr rebinned video timepoints',len(rebinned_videotimes))
                list_video_tensors.append(rebinned_video_tensor)
                list_videotrial_ids.append(videotrial_ids)
                list_videotimepoints.append(rebinned_videotimes)
            else:
                # No video data available for this session — use sentinel NaN
                list_video_tensors.append([np.nan])
                list_videotrial_ids.append([np.nan])

            

            # ── Accumulate per-session data ────────────────────────────────────
            list_sessionkeys.append(key)
            list_portlayouts.append((Trial() & key).fetch('port_layout')[0])
            list_exception_cond.append((Trial() & key).fetch('condition')[0])
            list_tensors.append(rebinned_tensor)
            list_trial_ids.append(trial_ids)
            list_mask_ids.append(mask_ids)
            list_trial_conds.append(trials_by_conds)
            timepoint_axis.append(rebinned_timepoints)
            list_trial_cond_dict.append(trial_cond_dict)

    # ── Build the trial availability set (intersection across sessions) ───────
    available_set_trials = get_available_set_trial_ids(list_trial_conds)

    # ── Pack everything into a single dict for downstream workers ─────────────
    sessions_info = {}
    sessions_info['session_keys']         = list_sessionkeys
    sessions_info['port_layouts']         = list_portlayouts
    sessions_info['timepoint_axis']       = np.mean(timepoint_axis, axis=0)  # mean across sessions
    sessions_info['list_tensors']         = list_tensors
    sessions_info['list_trial_ids']       = list_trial_ids
    sessions_info['list_mask_ids']        = list_mask_ids
    # sessions_info['list_neuron_ids'] = get_neuron_ids(session_keys,list_mask_ids)
    sessions_info['available_set_trials'] = available_set_trials
    sessions_info['list_trial_cond_dict'] = list_trial_cond_dict
    sessions_info['list_video_tensors']   = list_video_tensors
    sessions_info['list_videotrial_ids']  = list_videotrial_ids
    sessions_info['video_times']          = np.mean(list_videotimepoints, axis=0)
    sessions_info['exception_cond']       = list_exception_cond
    sessions_info['list_stimuli_id']      = list_stimuli_id

    return sessions_info


# ===========================================================================
# SINGLE-ITERATION DECODER WORKERS
# (intended to be called via Pool.starmap)
# ===========================================================================

def one_iter_all_window_decode(sessions_info, params, window_idx, available_set_trials,
                               subsampling_idx, shuffle):
    """
    One cross-validation iteration of window-averaged decoding + CCGP for a
    single variable (per-variable neuron sets, standard SVM).

    Trains on ``window_idx`` timepoints, tests on a held-out set, then
    evaluates CCGP for all cross-variable dichotomies.
    When ``shuffle=False``, also returns the fraction CCGP / accuracy,
    decoder weights, and neuron IDs.

    Parameters
    ----------
    sessions_info        : dict   Output of ``sessions_info_to_decode``.
    params               : dict   Must contain 'train_var', 'variables', 'pseudopopulation'.
    window_idx           : list   [start_frame, end_frame] of the decoding window.
    available_set_trials : dict   Trial ID sets per combination.
    subsampling_idx      : list   Per-session neuron subsampling indices.
    shuffle              : bool   If True, shuffle trial labels (null distribution).

    Returns
    -------
    results : dict  Keys: 'acc', 'ccgp'; plus 'fraction', 'weight_distribution',
                    'neuron_ids_decode' when shuffle=False.
    """
    variable = params['train_var']
    # nr_sessions = len(sessions_info['session_keys'])
    variables        = params['variables']
    assert params['pseudopopulation']
    var_idx          = variables.index(variable)
    variable_indices = list(np.arange(len(variables)))
    variable_indices.pop(var_idx)  # indices of all OTHER variables (for CCGP)
    combinations     = list(available_set_trials.keys())

    # ── Train and test main decoder ───────────────────────────────────────────
    train_tensor, y_train = get_dataset(sessions_info, params,
                                        window_idx, available_set_trials,
                                        training_bool=True, cross=False,
                                        subsampling_idx=subsampling_idx)
    svm_classifier_acc, mean, std = train_decoder(train_tensor, y_train, shuffle=shuffle)

    test_tensor, y_test = get_dataset(sessions_info, params,
                                      window_idx, available_set_trials,
                                      training_bool=False, cross=False,
                                      subsampling_idx=subsampling_idx)
    accuracy = test_decoder(svm_classifier_acc, test_tensor, y_test, mean, std)

    # ── CCGP: evaluate for every cross-variable dichotomy ─────────────────────
    dichotomy_combinations = get_dichotomies(combinations, variable_indices)
    cross_accuracy = []

    for dichotomies in dichotomy_combinations:
        ccgp_dichotomy = []
        # Split combination space into two dichotomy sets
        dichotomy_sets = [{}, {}]
        for i in range(2):  # two sets per dichotomy
            for combination in dichotomies[i]:
                dichotomy_sets[i][combination] = available_set_trials[combination]

        for i in range(2):
            # Train on one half of the dichotomy, test on the other
            train_tensor, y_train = get_dataset(sessions_info, params,
                                                window_idx, dichotomy_sets[i],
                                                training_bool=True, cross=True,
                                                subsampling_idx=subsampling_idx)
            svm_classifier, mean, std = train_decoder(train_tensor, y_train, shuffle=shuffle)

            # train in one condition given in one dichotomoy set
            # test in the other dichotomy set
            test_tensor, y_test = get_dataset(sessions_info, params,
                                              window_idx, dichotomy_sets[1-i],
                                              training_bool=False, cross=True,
                                              subsampling_idx=subsampling_idx)
            ccgp_dichotomy.append(test_decoder(svm_classifier, test_tensor, y_test, mean, std))

        cross_accuracy.append(np.mean(ccgp_dichotomy))

    # print('ccgp shape', ccgp.shape)
    print('nr classifiers ', len(svm_classifier_acc))

    results = {
        'acc':  accuracy,
        'ccgp': cross_accuracy,
    }
    if not shuffle:
        # Extra diagnostics only needed for the real (non-shuffled) estimate
        results['fraction']           = np.array(cross_accuracy) / accuracy
        results['weight_distribution'] = svm_classifier_acc[0].coef_[0]
        results['neuron_ids_decode']  = subsampling_idx

    return results


def window_decode_allvars(sessions_info, params, window_idx, available_set_trials,
                          subsampling_idx, shuffle):
    """
    One cross-validation iteration of joint all-variable decoding on a
    shared neuron set with dDR-projected train/test matrices.

    All variables are decoded from the SAME X_train / X_test matrices
    (same neuron subsampling); individual LinearSVC classifiers are fitted
    per variable.  Returns accuracy, CCGP, parallelism score, decoder
    vectors, and the full train/test matrices for downstream geometry analysis.

    Parameters
    ----------
    sessions_info        : dict   Output of ``sessions_info_to_decode``.
    params               : dict   Must contain 'variables', 'pseudopopulation'.
    window_idx           : list   [start_frame, end_frame] of the decoding window.
    available_set_trials : dict   Trial ID sets per combination.
    subsampling_idx      : list   Per-session neuron subsampling indices.
    shuffle              : bool   (not used for labels here; geometry is always real)

    Returns
    -------
    results : dict  Keys: 'acc', 'ccgp', 'ps', 'decoder_vec', 'combinations',
                    'neuron_ids_decode', 'X_train', 'X_test'.
    """
    assert params['pseudopopulation']
    combinations = list(available_set_trials.keys())

    # ── Build shared normalised train / test matrices ─────────────────────────
    # decoding accuracy x tensors
    train_tensors = get_X_dataset(sessions_info, params,
                                  window_idx, available_set_trials,
                                  training_bool=True,
                                  subsampling_idx=subsampling_idx)
    #print('train_tensors ',train_tensors.shape)

    test_tensors = get_X_dataset(sessions_info, params,
                                 window_idx, available_set_trials,
                                 training_bool=False,
                                 subsampling_idx=subsampling_idx)
    #print('test_tensors ',test_tensors.shape)

    # Stack across combinations: (nr_combinations * nr_trials) × nr_neurons
    X_train = np.vstack(train_tensors)
    X_test  = np.vstack(test_tensors)

    # Z-score neurons using training-set statistics
    m, st    = stats_over_axis(X_train, ['trials', 'neurons'], 'trials')
    X_train  = zscore_neurons(X_train, ['trials', 'neurons'], m, st)
    X_test   = zscore_neurons(X_test,  ['trials', 'neurons'], m, st)

    # ── Decode each variable from the shared matrices ─────────────────────────
    accuracies        = []
    cross_accuracies  = []
    parallelism_scores = []
    dec_vecs          = []

    for v in params['variables']:
        variables = params['variables']
        params['train_var'] = v
        params['test_var']  = v
        var_idx          = variables.index(v)
        variable_indices = list(np.arange(len(variables)))
        variable_indices.pop(var_idx)

        # Build label vectors for this variable
        y_train = get_y_for_X(combinations, params['variables'],
                               v, train_tensors, pseudopop_bool=True)
        y_test  = get_y_for_X(combinations, params['variables'],
                               v, test_tensors,  pseudopop_bool=True)

        # Train a LinearSVC on the shared X_train
        svm_classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000)
        svm_classifier.fit(X_train, y_train)
        accuracies.append(svm_classifier.score(X_test, y_test))
        # Store decoder weight vector for this variable
        dec_vecs.append(svm_classifier.coef_[0])

        # ── CCGP and parallelism score for cross-variable dichotomies ──────────
        dichotomy_combinations = get_dichotomies(combinations, variable_indices)
        cross_accuracy = []
        ps             = []

        for dichotomies in dichotomy_combinations:
            ccgp_dichotomy = []
            coding_vectors = []
            dichotomy_sets = [{}, {}]
            for i in range(2):  # two sets per dichotomy
                for combination in dichotomies[i]:
                    dichotomy_sets[i][combination] = available_set_trials[combination]

            for i in range(2):
                # Train on one half, collect coding vector
                train_tensor, y_train = get_dataset(sessions_info, params,
                                                    window_idx, dichotomy_sets[i],
                                                    training_bool=True, cross=True,
                                                    subsampling_idx=subsampling_idx)
                svm_classifier, mean, std = train_decoder(train_tensor, y_train, shuffle=False)
                coding_vectors.append(svm_classifier[0].coef_[0])

                # train in one condition given in one dichotomoy set
                # test in the other dichotomy set
                test_tensor, y_test = get_dataset(sessions_info, params,
                                                  window_idx, dichotomy_sets[1-i],
                                                  training_bool=False, cross=True,
                                                  subsampling_idx=subsampling_idx)
                ccgp_dichotomy.append(test_decoder(svm_classifier, test_tensor, y_test, mean, std))

            # Parallelism score = angle between the two dichotomy coding vectors
            ps.append(angle_between_vectors(coding_vectors[0], coding_vectors[1]))
            cross_accuracy.append(np.mean(ccgp_dichotomy))

        cross_accuracies.append(cross_accuracy)
        parallelism_scores.append(ps)

    accuracies         = np.array(accuracies)
    cross_accuracies   = np.array(cross_accuracies)
    dec_vecs           = np.array(dec_vecs)
    parallelism_scores = np.array(parallelism_scores)

    results = {
        'acc':              accuracies,
        'ccgp':             cross_accuracies,
        'ps':               parallelism_scores,
        'decoder_vec':      dec_vecs,        # shape: (nr_vars, nr_neurons)
        'combinations':     combinations,
        'neuron_ids_decode': subsampling_idx,
        'X_train':          X_train,
        'X_test':           X_test,
    }
    return results


def window_decode(sessions_info, params, window_idx, available_set_trials,
                  subsampling_idx, shuffle):
    """
    One cross-validation iteration of window-averaged decoding + CCGP for a
    single variable with dDR projection (returns train/test matrices).

    Identical logic to ``one_iter_all_window_decode`` but uses
    ``train_decoder(..., dDR=True)`` and ``test_decoder(..., dDR=True)``
    so that the projected X matrices are also returned.

    Parameters
    ----------
    Same as ``one_iter_all_window_decode``.

    Returns
    -------
    results : dict  Keys: 'acc', 'ccgp'; plus 'decoder_vec', 'neuron_ids_decode',
                    'X_train', 'X_test', 'combinations' when shuffle=False.
    """
    variable         = params['train_var']
    variables        = params['variables']
    assert params['pseudopopulation']
    var_idx          = variables.index(variable)
    variable_indices = list(np.arange(len(variables)))
    variable_indices.pop(var_idx)
    combinations     = list(available_set_trials.keys())

    # ── Train with dDR projection; also returns the projected X_train ─────────
    train_tensor, y_train = get_dataset(sessions_info, params,
                                        window_idx, available_set_trials,
                                        training_bool=True, cross=False,
                                        subsampling_idx=subsampling_idx)
    svm_classifier_acc, mean, std, X_train = train_decoder(train_tensor, y_train,
                                                            shuffle=shuffle, dDR=True)

    # Test with dDR; also returns the projected X_test
    test_tensor, y_test = get_dataset(sessions_info, params,
                                      window_idx, available_set_trials,
                                      training_bool=False, cross=False,
                                      subsampling_idx=subsampling_idx)
    accuracy, X_test = test_decoder(svm_classifier_acc, test_tensor, y_test, mean, std, dDR=True)

    # ── CCGP: evaluate for every cross-variable dichotomy ─────────────────────
    dichotomy_combinations = get_dichotomies(combinations, variable_indices)
    cross_accuracy = []

    for dichotomies in dichotomy_combinations:
        ccgp_dichotomy = []
        dichotomy_sets = [{}, {}]
        for i in range(2):  # two sets per dichotomy
            for combination in dichotomies[i]:
                dichotomy_sets[i][combination] = available_set_trials[combination]

        for i in range(2):
            # Train on one half, test on the other
            train_tensor, y_train = get_dataset(sessions_info, params,
                                                window_idx, dichotomy_sets[i],
                                                training_bool=True, cross=True,
                                                subsampling_idx=subsampling_idx)
            svm_classifier, mean, std = train_decoder(train_tensor, y_train, shuffle=shuffle)

            # train in one condition given in one dichotomoy set
            # test in the other dichotomy set
            test_tensor, y_test = get_dataset(sessions_info, params,
                                              window_idx, dichotomy_sets[1-i],
                                              training_bool=False, cross=True,
                                              subsampling_idx=subsampling_idx)
            ccgp_dichotomy.append(test_decoder(svm_classifier, test_tensor, y_test, mean, std))

        cross_accuracy.append(np.mean(ccgp_dichotomy))

    results = {
        'acc':  accuracy,
        'ccgp': cross_accuracy,
    }
    if not shuffle:
        # Store decoder artefacts only for real (non-shuffled) runs
        results['decoder_vec']      = svm_classifier_acc[0].coef_[0]
        results['neuron_ids_decode'] = subsampling_idx
        results['X_train']          = X_train
        results['X_test']           = X_test
        results['combinations']     = combinations

    return results


def one_iter_cross_var(sessions_info, params, t_idx, available_set_trials,
                       subsampling_idx, shuffle):
    """
    One cross-validation iteration of cross-variable CCGP + parallelism score
    at a single training timepoint.

    For each cross-variable dichotomy, trains on one half of the combination
    space and tests on the other, collecting both accuracy (CCGP) and the
    angle between the two dichotomy coding vectors (parallelism score).

    Parameters
    ----------
    sessions_info        : dict
    params               : dict   Must contain 'train_var', 'variables', 'pseudopopulation'.
    t_idx                : int    Training timepoint index.
    available_set_trials : dict
    subsampling_idx      : list
    shuffle              : bool

    Returns
    -------
    cross_accuracy : list  Mean CCGP per dichotomy.
    cross_ps       : list  Parallelism score per dichotomy.
    """
    train_var        = params['train_var']
    nr_sessions      = len(sessions_info['session_keys'])
    variables        = params['variables']
    pseudopop_bool   = params['pseudopopulation']
    var_idx          = variables.index(train_var)
    variable_indices = list(np.arange(len(variables)))
    variable_indices.pop(var_idx)
    combinations     = list(available_set_trials.keys())

    cross_accuracy         = []
    cross_ps               = []
    dichotomy_combinations = get_dichotomies(combinations, variable_indices)
    nr_dichotomies         = len(dichotomy_combinations)
    assert nr_dichotomies == len(variable_indices)

    for dichotomies in dichotomy_combinations:
        coding_vectors = []
        ccgp_dichotomy = []
        dichotomy_sets = [{}, {}]
        for i in range(2):  # two sets per dichotomy
            for combination in dichotomies[i]:
                dichotomy_sets[i][combination] = available_set_trials[combination]

        for i in range(2):
            train_tensor, y_train = get_dataset(sessions_info, params,
                                                t_idx, dichotomy_sets[i],
                                                training_bool=True, cross=True,
                                                subsampling_idx=subsampling_idx)
            svm_classifier, mean, std = train_decoder(train_tensor, y_train, shuffle=shuffle)

            # Collect coding vector (pseudopop: 1 classifier; multi-session: one per session)
            if pseudopop_bool:
                assert len(mean) == 1
                assert len(svm_classifier) == 1
                coding_vectors.append(svm_classifier[0].coef_)
            else:
                assert len(mean) > 1
                coding_vectors.append([clf.coef_ for clf in svm_classifier])

            # train in one condition given in one dichotomoy set
            # test in the other dichotomy set
            test_tensor, y_test = get_dataset(sessions_info, params,
                                              t_idx, dichotomy_sets[1-i],
                                              training_bool=False, cross=True,
                                              subsampling_idx=subsampling_idx)
            ccgp_dichotomy.append(test_decoder(svm_classifier, test_tensor, y_test, mean, std))

        # Parallelism score = angle between the two dichotomy coding vectors
        if pseudopop_bool:
            angle = angle_between_vectors(coding_vectors[0], coding_vectors[1])
        else:
            coding_vectors = np.array(coding_vectors)
            print('coding_vectors', coding_vectors.shape)
            assert coding_vectors.shape == (2, nr_sessions)
            # Per-session angle
            angle = [angle_between_vectors(code_vec[0], code_vec[1])
                     for code_vec in coding_vectors.T]

        cross_ps.append(angle)
        cross_accuracy.append(np.mean(ccgp_dichotomy))

    return cross_accuracy, cross_ps


def one_iter_behavior_decode_at_t(sessions_info, params, t_idx, available_set_trials, shuffle):
    """
    One cross-validation iteration of behavioural (DLC-feature) decoding at
    a single timepoint.

    Parameters
    ----------
    sessions_info        : dict
    params               : dict   Must contain 'train_var', 'pseudopopulation'.
    t_idx                : int    Timepoint index in the behavioural tensor.
    available_set_trials : dict
    shuffle              : bool

    Returns
    -------
    beh_acc : float (pseudopop) or list of float (per-session)
    """
    pseudopop_bool = params['pseudopopulation']
    nr_sessions    = len(sessions_info['list_video_tensors'])

    # ── Train behavioural decoder ──────────────────────────────────────────────
    # y -> nr sessions x trials or (pseudopol) trials
    # get train tensor  nr_sessions x trials x behaviour features or (pseudopol) trials x features
    train_tensor, y_train = get_beh_dataset(sessions_info, params,
                                            t_idx, available_set_trials,
                                            training_bool=True)
    # print('train tensor done')
    svm_classifier, _, _ = train_decoder(train_tensor, y_train, shuffle=shuffle, behaviour=True)

    # ── Test behavioural decoder ───────────────────────────────────────────────
    # print('classifier fitted')
    test_tensor, y_test = get_beh_dataset(sessions_info, params,
                                          t_idx, available_set_trials,
                                          training_bool=False)
    beh_acc = test_decoder(svm_classifier, test_tensor, y_test, None, None, behaviour=True)
    # print('beh_acc',beh_acc)

    # ── Validate output shape ──────────────────────────────────────────────────
    # if not pseudopopulation then outputs of functions
    # correspond to list for each session in session keys
    if pseudopop_bool:
        assert isinstance(beh_acc, float)
    else:
        # transform to nr sessions x nr tested timepoints
        assert isinstance(beh_acc, list)
        assert len(beh_acc) == nr_sessions

    return beh_acc


def one_iter_crosstime_decode(sessions_info, params, t_idx, available_set_trials,
                              subsampling_idx, shuffle):
    """
    One cross-validation iteration of cross-time decoding.

    Trains at timepoint ``t_idx``; when ``shuffle=False`` tests at ALL
    timepoints (full cross-time matrix column); when ``shuffle=True`` tests
    only at ``t_idx`` (null distribution — no cross-time sweep needed).

    Parameters
    ----------
    sessions_info        : dict
    params               : dict   Must contain 'pseudopopulation'.
    t_idx                : int    Training timepoint index.
    available_set_trials : dict
    subsampling_idx      : list
    shuffle              : bool

    Returns
    -------
    accuracy_trace : list of float (real) or float (shuffled / pseudopop)
                     or ndarray shape (nr_sessions, nr_timepoints) (real / multi-session)
    """
    # subsampling_idx = get_subsampling_idx(sessions_info,n_neurons = params['nr_neurons_session'])
    # print('one iter crosstime')
    pseudopop_bool = params['pseudopopulation']
    nr_timepoints  = sessions_info['list_tensors'][0].shape[-1]
    nr_sessions    = len(sessions_info['list_tensors'])

    # ── Train at t_idx ────────────────────────────────────────────────────────
    # X tensor -> nr_sessions x trials x neurons or (pseudopol) trials x neurons
    # y -> nr sessions x trials or (pseudopol) trials
    train_tensor, y_train = get_dataset(sessions_info, params,
                                        t_idx, available_set_trials,
                                        training_bool=True,
                                        subsampling_idx=subsampling_idx)
    svm_classifier, mean, std = train_decoder(train_tensor, y_train, shuffle=shuffle)

    accuracy_trace = []

    if not shuffle:
        # ── Real model: test at every timepoint (cross-time generalisation) ────
        for t in range(nr_timepoints):
            test_tensor, y_test = get_dataset(sessions_info, params,
                                              t, available_set_trials,
                                              training_bool=False,
                                              subsampling_idx=subsampling_idx)
            accuracy_trace.append(test_decoder(svm_classifier, test_tensor, y_test, mean, std))
    else:
        # ── Null model: test only at t_idx (no cross-time needed for null) ─────
        test_tensor, y_test = get_dataset(sessions_info, params,
                                          t_idx, available_set_trials,
                                          training_bool=False,
                                          subsampling_idx=subsampling_idx)
        accuracy_trace = test_decoder(svm_classifier, test_tensor, y_test, mean, std)

    # ── Validate output shape ──────────────────────────────────────────────────
    if pseudopop_bool:
        if not shuffle:
            assert len(accuracy_trace) == nr_timepoints
        else:
            assert isinstance(accuracy_trace, float)
    else:
        if not shuffle:
            # Reshape to (nr_sessions × nr_timepoints)
            accuracy_trace = np.array(accuracy_trace).T
            assert accuracy_trace.shape == (nr_sessions, nr_timepoints)
        else:
            assert len(accuracy_trace) == nr_sessions

    return accuracy_trace


def one_iter_geometry_intime(sessions_info, params, t_idx, available_set_trials,
                             subsampling_idx, shuffle):
    """
    One cross-validation iteration of CCGP + parallelism score geometry at a
    single timepoint.

    For each cross-variable dichotomy trains on one half, tests on the other,
    and collects coding vectors to compute the parallelism score (angle between
    the two vectors).  Returns per-dichotomy means collapsed to scalars
    (pseudopop) or per-session vectors (multi-session).

    Parameters
    ----------
    sessions_info        : dict
    params               : dict   Must contain 'train_var', 'variables', 'pseudopopulation'.
    t_idx                : int    Timepoint index.
    available_set_trials : dict
    subsampling_idx      : list
    shuffle              : bool

    Returns
    -------
    ccgp : float or ndarray  Mean CCGP across dichotomies.
    ps   : float or ndarray  Mean parallelism score across dichotomies.
    """
    # print('one_iter_geometry_intime')
    variable         = params['train_var']
    pseudopop_bool   = params['pseudopopulation']
    nr_sessions      = len(sessions_info['session_keys'])
    variables        = params['variables']
    var_idx          = variables.index(variable)
    variable_indices = list(np.arange(len(variables)))
    variable_indices.pop(var_idx)
    combinations     = list(available_set_trials.keys())

    dichotomy_combinations = get_dichotomies(combinations, variable_indices)
    nr_dichotomies         = len(dichotomy_combinations)
    ccgp = []
    ps   = []

    for dichotomies in dichotomy_combinations:
        coding_vectors = []
        ccgp_dichotomy = []
        dichotomy_sets = [{}, {}]
        for i in range(2):  # two sets per dichotomy
            for combination in dichotomies[i]:
                dichotomy_sets[i][combination] = available_set_trials[combination]

        for i in range(2):
            train_tensor, y_train = get_dataset(sessions_info, params,
                                                t_idx, dichotomy_sets[i],
                                                training_bool=True, cross=True,
                                                subsampling_idx=subsampling_idx)
            svm_classifier, mean, std = train_decoder(train_tensor, y_train, shuffle=shuffle)

            # Collect coding vector (1 for pseudopop; one per session otherwise)
            if pseudopop_bool:
                assert len(mean) == 1
                assert len(svm_classifier) == 1
                coding_vectors.append(svm_classifier[0].coef_)
            else:
                assert len(mean) > 1
                coding_vectors.append([clf.coef_ for clf in svm_classifier])

            # train in one condition given in one dichotomoy set
            # test in the other dichotomy set
            test_tensor, y_test = get_dataset(sessions_info, params,
                                              t_idx, dichotomy_sets[1-i],
                                              training_bool=False, cross=True,
                                              subsampling_idx=subsampling_idx)
            ccgp_dichotomy.append(test_decoder(svm_classifier, test_tensor, y_test, mean, std))

        # Parallelism score = angle between the two dichotomy coding vectors
        if pseudopop_bool:
            angle = angle_between_vectors(coding_vectors[0], coding_vectors[1])
        else:
            coding_vectors = np.array(coding_vectors)
            print('coding_vectors', coding_vectors.shape)
            assert coding_vectors.shape == (2, nr_sessions)
            # Per-session angle
            angle = [angle_between_vectors(code_vec[0], code_vec[1])
                     for code_vec in coding_vectors.T]

        ps.append(angle)
        ccgp.append(np.mean(ccgp_dichotomy))

    ccgp = np.array(ccgp)
    ps   = np.array(ps)

    # ── Validate output shapes ─────────────────────────────────────────────────
    if pseudopop_bool:
        assert ccgp.shape == (nr_dichotomies,)   # (2*nr_dichotomies,)
        assert ps.shape   == (nr_dichotomies,)
    else:
        assert ccgp.shape == (nr_dichotomies, nr_sessions)  # (2*nr_dichotomies,nr_sessions)
        assert ps.shape   == (nr_dichotomies, nr_sessions)

    # Collapse across dichotomies to a single scalar (or per-session vector)
    ccgp = np.mean(ccgp, axis=0)
    ps   = np.mean(ps,   axis=0)

    return ccgp, ps


# ===========================================================================
# MULTI-CV POOL WRAPPERS
# ===========================================================================

def cross_var_at_t(sessions_info, params, available_set_trials, t_idx):
    """
    Run ``nr_crossvalidations`` parallel iterations of ``one_iter_cross_var``
    at timepoint ``t_idx`` and return stacked CCGP and PS arrays.

    Parameters
    ----------
    sessions_info        : dict
    params               : dict   Must contain 'nr_crossvalidations', 'nr_cores',
                                  'nr_neurons_session'.
    available_set_trials : dict
    t_idx                : int    Timepoint index.

    Returns
    -------
    cross_var_acc : ndarray  Shape (nr_crossvalidations, nr_dichotomies).
    cross_var_ps  : ndarray  Shape (nr_crossvalidations, nr_dichotomies).
    """
    #one_iter_cross_var(sessions_info,params,t_idx,available_set_trials,subsampling_idx,shuffle)
    nr_cores = params['nr_cores']

    with Pool(nr_cores) as pool:
        # Pre-generate trial shuffles and neuron subsampling for each CV fold
        available_sets      = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]
        subsampling_indices = [get_subsampling_idx(sessions_info, params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
        args    = [(sessions_info, params, t_idx, available_sets[n], subsampling_indices[n], False) for n in range(params['nr_crossvalidations'])]
        results = pool.starmap(one_iter_cross_var, args)

        results       = np.array(results)
        cross_var_acc = results[:, 0]  # CCGP across CV folds
        cross_var_ps  = results[:, 1]  # PS across CV folds

    return cross_var_acc, cross_var_ps


def acc_geom_estimation_at_t(sessions_info, params, available_set_trials, t_idx, condition):
    """
    Run ``nr_crossvalidations`` parallel iterations of both
    ``one_iter_crosstime_decode`` and ``one_iter_geometry_intime`` at
    timepoint ``t_idx``.

    Parameters
    ----------
    sessions_info        : dict
    params               : dict
    available_set_trials : dict
    t_idx                : int    Timepoint index.
    condition            : str    Variable name (unused here, passed by callers).

    Returns
    -------
    acc_fold  : list  Cross-time accuracy traces from each CV fold.
    geom_fold : list  (ccgp, ps) tuples from each CV fold.
    """
    # print('condition: ',condition)
    nr_cores = params['nr_cores']

    with Pool(nr_cores) as pool:
        # Shared set of shuffled trials and subsampling indices for both workers
        available_sets      = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]
        subsampling_indices = [get_subsampling_idx(sessions_info, params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
        args = [(sessions_info, params, t_idx, available_sets[n], subsampling_indices[n], False) for n in range(params['nr_crossvalidations'])]

        # Both accuracy and geometry use the same args (same shuffled sets / subsampling)
        acc_fold  = pool.starmap(one_iter_crosstime_decode, args)
        geom_fold = pool.starmap(one_iter_geometry_intime,  args)

    return acc_fold, geom_fold


def behaviour_decode(sessions_info, params):
    """
    Run the full time-resolved behavioural decoder for all variables,
    computing both cross-validated accuracy and a shuffled null distribution.

    For each variable and timepoint, two parallel pool runs are performed:
      - CV run (shuffle=False): estimates real behavioural accuracy
      - Null run (shuffle=True): generates the null distribution

    Sessions without video data (NaN tensors) are silently excluded.

    Parameters
    ----------
    sessions_info : dict   Output of ``sessions_info_to_decode``.
    params        : dict   Must contain 'nr_iter', 'nr_crossvalidations',
                            'nr_cores', 'pseudopopulation', 'variables'.

    Returns
    -------
    behavior : dict  {variable: ndarray (nr_crossvalidations × nr_timepoints)
                      or (nr_sessions × nr_timepoints) for multi-session mode}.
    null     : dict  {variable: ndarray (nr_iter × nr_timepoints)
                      or (nr_iter × nr_sessions × nr_timepoints)}.
    """
    list_video_tensors  = sessions_info['list_video_tensors']
    # list_trial_cond_dict = sessions_info['list_trial_cond_dict']
    list_videotrial_ids = sessions_info['list_videotrial_ids']
    available_set_trials = sessions_info['available_set_trials']
    video_times          = sessions_info['video_times']
    nr_timepoints        = len(video_times)
    nr_cores             = params['nr_cores']
    pseudopop_bool       = params['pseudopopulation']
    nr_iter              = params['nr_iter']
    nr_crossvalidations  = params['nr_crossvalidations']
    variables            = params['variables']
    nr_video_sessions    = len(list_video_tensors)
    combinations         = list(available_set_trials.keys())

    # ── Build valid trial sets: only sessions with actual video data ───────────
    assert len(list_video_tensors) == len(list_videotrial_ids)
    valid_set_trials = dict()

    for i, combination in enumerate(combinations):
        valid_set_trials[combination] = []
        assert len(available_set_trials[combination]) == nr_video_sessions
        for j in range(nr_video_sessions):
            if not np.isnan(list_video_tensors[j]).any():
                trial_ids    = available_set_trials[combination][j]
                # print('trialids in available set', trial_ids)
                # Keep only trials that appear in both the neural and video trial lists
                valid_trials = list(set(trial_ids).intersection(set(list_videotrial_ids[j])))
                valid_set_trials[combination].append(valid_trials)

    # ── Filter out sessions without video data ─────────────────────────────────
    valid_video_tensors  = []
    valid_video_trialids = []
    for j in range(nr_video_sessions):
        if not np.isnan(list_video_tensors[j]).any():
            valid_video_tensors.append(list_video_tensors[j])
            valid_video_trialids.append(list_videotrial_ids[j])

    print('valid_video_tensors[0].shape', valid_video_tensors[0].shape)
    assert valid_video_tensors[0].shape[-1] == nr_timepoints

    if len(valid_video_trialids) != len(valid_set_trials[combinations[0]]):
        print('len(valid_video_trialids)',         len(valid_video_trialids))
        print('len(valid_set_trials[combinations[0]])', len(valid_set_trials[combinations[0]]))

    # Replace with filtered lists in sessions_info so workers see only valid data
    sessions_info['list_video_tensors']  = valid_video_tensors
    sessions_info['list_videotrial_ids'] = valid_video_trialids

    # ── Decode each variable in time ──────────────────────────────────────────
    behavior = {}  # cross-validated accuracy
    null     = {}  # shuffled null accuracy

    for variable in variables:
        print('\n variable: %s...' % variable, flush=True)
        params['train_var'] = variable

        # Pre-allocate output arrays
        if pseudopop_bool:
            behavior[variable] = np.zeros((nr_crossvalidations, nr_timepoints))
            null[variable]     = np.zeros((nr_iter,             nr_timepoints))
        else:
            behavior[variable] = np.zeros((nr_video_sessions,  nr_timepoints))
            null[variable]     = np.zeros((nr_iter, nr_video_sessions, nr_timepoints))

        print('nr_timepoints', nr_timepoints)
        for t_idx in tqdm(range(nr_timepoints)):

            # ── CV run: real trial labels ──────────────────────────────────────
            available_sets = [shuffle_available_set(valid_set_trials) for n in range(nr_crossvalidations)]
            args           = [(sessions_info, params, t_idx, available_sets[n], False) for n in range(nr_crossvalidations)]
            # print('args,')
            with Pool(nr_cores) as pool:
                one_iter_result  = pool.starmap(one_iter_behavior_decode_at_t, args)
                one_iter_result  = np.array(one_iter_result)

            # print('one_iter_result.shape', np.array(one_iter_result).shape)

            # ── Null run: shuffled trial labels ────────────────────────────────
            available_sets = [shuffle_available_set(valid_set_trials) for n in range(nr_iter)]
            args           = [(sessions_info, params, t_idx, available_sets[n], True) for n in range(nr_iter)]
            # print('args,')
            with Pool(nr_cores) as pool:
                null_results = pool.starmap(one_iter_behavior_decode_at_t, args)
                null_results = np.array(null_results)
            # print('null_results.shape', np.array(null_results).shape)

            # ── Store results at this timepoint ────────────────────────────────
            if pseudopop_bool:
                behavior[variable][:, t_idx] = one_iter_result
                null[variable][:, t_idx]     = null_results
            else:
                behavior[variable][:, t_idx]    = one_iter_result
                null[variable][:, :, t_idx]     = null_results

    return behavior, null
