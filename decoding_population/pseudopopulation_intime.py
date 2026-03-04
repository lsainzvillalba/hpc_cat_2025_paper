#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:40:26 2024

@author: Laura Sainz Villalba

Time-resolved neural decoding of behavioural variables from a pseudopopulation
of neurons pooled across animals and sessions.

For each experimental phase and alignment event the pipeline:
  1.  Gathers valid session keys and subsamples neurons to a common count
  2.  Computes a common time axis across sessions
  3.  At every timepoint, fits a linear decoder and evaluates cross-validated
      accuracy (time-independent) and a shuffled null distribution (parallel)
  4.  Also runs a behavioural (lick-based) decoder for comparison
  5.  Inserts accuracy traces into ``Single_var`` and ``Behavior`` DB tables

DataJoint schema : pseudopopulation_intime_hpc_cat_2025
"""

# import libraries
import numpy as np
from tqdm import tqdm
import datajoint as dj
from time import time
from multiprocessing import Pool

dj.config["enable_python_native_blobs"] = True
schema = dj.schema('pseudopopulation_intime_hpc_cat_2025', locals(), create_tables = True)

print('calling in pseudopopulation_intime from: ',__name__)

# ── Conditional imports depending on execution context ──────────────────────
if __name__ == '__main__':
    from util_decoder import get_session_keys,get_valid_sessions,get_nr_subsample_neurons,\
        get_common_time_axis_params, sessions_info_to_decode, acc_geom_estimation_at_t, \
            shuffle_available_set,one_iter_crosstime_decode,behaviour_decode, \
            get_subsampling_idx
else:
    from .util_decoder import get_session_keys,get_valid_sessions,get_nr_subsample_neurons,\
        get_common_time_axis_params, sessions_info_to_decode, acc_geom_estimation_at_t, \
            shuffle_available_set,one_iter_crosstime_decode,behaviour_decode, \
            get_subsampling_idx


# ===========================================================================
# DataJoint table definitions
# ===========================================================================

@schema
class Single_var(dj.Manual):
    # train and test in each time point - time independent decoder
    # for each subject in phase
    definition = """ # 

    variable           : varchar(256)   # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    event_align        : varchar(128)   # event of alignment for time
    ---
    animal_ids          : blob          # list of animal ids included 
    session_ids         : blob          # list of session ids corresponding to animal ids used
    accuracies          : longblob      # nr cross validations x accuracy trace in time 
    null                : longblob      # iterations x timepoints for null distribution decoding
    timepoints    : longblob      # common timepoints for trace used in neural decoder
    """


@schema
class Behavior(dj.Manual):
    definition = """ # behavioural accuracy pseudopop in time w

    variable           : varchar(256)   # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    event_align        : varchar(128)   # event of alignment for time
    ---
    animal_ids          : blob          # list of animal ids included 
    session_ids         : blob          # list of session ids corresponding to animal ids used
    beh_accuracies          : longblob      # nr cross validations x behavioural accuracy trace in time 
    beh_null                : longblob      # iterations x timepoints for null distribution decoding
    beh_timepoints    : longblob      # common timepoints for trace used in behavioural decoder
    """


# ===========================================================================
# Core decoding functions
# ===========================================================================

def decoder_time_phase(_session_keys, params, phase, populate=True):
    """
    Run the time-resolved neural and behavioural decoder for one experimental
    phase, then insert results into ``Single_var`` and ``Behavior``.

    For each variable the function:
      - Builds a (nr_crossvalidations × timepoints × timepoints) cross-time
        accuracy matrix by training at each timepoint and testing at all others
      - Computes a shuffled null distribution in parallel using a process Pool
      - Extracts the diagonal of the cross-time matrix as the time-independent
        decoding accuracy
      - Runs a behavioural decoder on the same sessions for comparison

    Parameters
    ----------
    _session_keys : dict
        Nested dict {animal_id: [session_key_dicts]} for this phase.
    params : dict
        Decoding hyperparameters (see module-level ``params`` dict for keys).
    phase : str
        Label for the experimental phase (used as DB primary key).
    populate : bool
        If True (default), insert results into the DB.
    """
    # print('\n Gathering session keys...',flush=True)
    start__run = time()

    # ── Unpack decoding hyperparameters ───────────────────────────────────────
    nr_iter            = params['nr_iter']            # shuffle iterations for null model
    nr_crossvalidations = params['nr_crossvalidations'] # CV folds for accuracy estimate
    event_align        = params['event_align']         # alignment event label
    variables          = params['variables']           # list of variables to decode
    nr_cores           = params['nr_cores']            # parallel worker count

    # ── Flatten session keys into ordered animal / session ID lists ───────────
    session_ids = [key['session_id'] for animal_id in _session_keys for key in _session_keys[animal_id]]
    animal_ids  = [key['animal_id']  for animal_id in _session_keys for key in _session_keys[animal_id]]

    # ── Collect session tensors, trial sets, and common time axis ─────────────
    sessions_info        = sessions_info_to_decode(_session_keys, params)
    available_set_trials = sessions_info['available_set_trials']
    nr_timepoints        = sessions_info['list_tensors'][0].shape[-1]

    # ── Behavioural (lick-based) decoding ─────────────────────────────────────
    # print('\n Decoding behaviour...',flush=True)
    start = time()
    beh, beh_null = behaviour_decode(sessions_info, params)
    # print('behaviour', (time()-start)/60,'mins')
    print('\n Finished behavioral decoding', flush=True)

    # ── Neural decoding — one variable at a time ───────────────────────────────
    for variable in variables:
        print('\n Decoding for variable: ', variable, flush=True)
        # Set train and test variable to the same label (time-independent decoder)
        params['train_var'] = variable
        params['test_var']  = variable

        # Cross-time accuracy matrix: train at t_train, test at t_test
        cross_time_acc = np.zeros((nr_crossvalidations, nr_timepoints, nr_timepoints))
        # Null distribution: one shuffled accuracy value per iteration per timepoint
        acc_null = np.zeros((nr_iter, nr_timepoints))

        # ── Sweep over all training timepoints ────────────────────────────────
        for t_idx in tqdm(range(nr_timepoints)):
            start = time()

            # Cross-validated accuracy: train at t_idx, test at all timepoints
            acc, _ = acc_geom_estimation_at_t(sessions_info, params,
                                              available_set_trials,
                                              t_idx, variable)
            cross_time_acc[:, t_idx, :] = acc  # shape: (nr_crossvalidations, nr_timepoints)

            # ── Null distribution via parallel shuffles ────────────────────────
            with Pool(nr_cores) as pool:
                # Pre-generate shuffled trial sets and neuron subsampling indices
                available_sets      = [shuffle_available_set(available_set_trials) for n in range(nr_iter)]
                subsampling_indices = [get_subsampling_idx(sessions_info, params['nr_neurons_session']) for n in range(nr_iter)]
                args         = [(sessions_info, params, t_idx, available_sets[n], subsampling_indices[n], True) for n in range(nr_iter)]
                results_acc  = pool.starmap(one_iter_crosstime_decode, args)
                results_acc  = np.array(results_acc)
                acc_null[:, t_idx] = results_acc   # store shuffled accuracies

            print('decoding timepoint ', (time()-start)/60, 'mins')

        # Time-independent accuracy = diagonal of the cross-time matrix
        # (i.e. trained and tested at the same timepoint)
        acc = np.array([np.diagonal(cross_time_acc[i]) for i in range(len(cross_time_acc))])

        # ── Insert neural decoding results ─────────────────────────────────────
        print('\n populating in time decoding')
        in_time_entry = {
            'variable':               variable,
            'trace_type':             params['trace_type'],
            'experimental_timepoint': phase,
            'event_align':            event_align,
            'animal_ids':             animal_ids,
            'session_ids':            session_ids,
            'accuracies':             acc,
            'null':                   acc_null,
            'timepoints':             sessions_info['timepoint_axis'],
            }
        Single_var().insert1(in_time_entry, skip_duplicates=True)

        # ── Insert behavioural decoding results ────────────────────────────────
        print('\n populating behaviour decoding')
        behtime_entry = {
            'variable':               variable,
            'trace_type':             params['trace_type'],
            'experimental_timepoint': phase,
            'event_align':            event_align,
            'animal_ids':             animal_ids,
            'session_ids':            session_ids,
            'beh_accuracies':         beh[variable],
            'beh_null':               beh_null[variable],
            'beh_timepoints':         sessions_info['video_times'],
            }
        Behavior().insert1(behtime_entry, skip_duplicates=True)

    print('TIME run : ', (time()-start__run))


def decode_in_time_all_subjects(params):
    """
    Outer loop: run ``decoder_time_phase`` for every combination of alignment
    event and experimental phase defined in ``params``.

    For each event alignment the function:
      1. Retrieves valid session keys and filters by minimum trial count
      2. Determines the number of neurons to subsample per session
      3. Computes the common time axis parameters across phases
      4. Iterates over phases, skipping any already present in the DB

    Parameters
    ----------
    params : dict
        Experiment-wide hyperparameters (see module-level ``params`` dict).
    """
    print('Gathering sessions in experiment...', flush=True)

    # ── Unpack loop-level parameters ──────────────────────────────────────────
    event_alignments = ['choice', 'stimulus_on']  #,'choice','ports_off']
    variables        = params['variables']
    min_nr_trials    = params['min_nr_trials_comb']
    nr_combinations  = params['nr_combinations']
    trace_type       = params['trace_type']

    for event_align in event_alignments:
        params['event_align'] = event_align

        # ── Gather and validate session keys for this alignment event ──────────
        session_keys_dict = get_session_keys(params, event_align)
        get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)

        # Compute how many neurons to subsample per session (equalise across phases)
        nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
            session_keys_dict,
            params['subsampling_fr'],
            params['nr_sessions_phase'],
            across_axis='phases',
        )

        # Set pseudopopulation trial count (or None for single-session mode)
        if params['pseudopopulation']:
            params['n_datapoints'] = n_datapoints
        else:
            params['n_datapoints'] = None

        # ── Compute common time-axis bounds across all phases ─────────────────
        zero_left, left_range, right_range = get_common_time_axis_params(session_keys_dict)
        params['left_range']  = left_range
        params['right_range'] = right_range
        phases = params['phases']

        print('\n Processing all entries...', flush=True)
        pbar = tqdm(total=len(phases), position=0, leave=True)

        # ── Decode phase by phase  ───────────────────────
        for phase in phases:
            params['zero_left']          = zero_left[phase]
            params['nr_neurons_session'] = nr_neurons_session[phase]

            # Skip if results for this phase / event / trace type already exist
            entries = Single_var() \
                & 'experimental_timepoint="%s"' % phase \
                & 'event_align="%s"'            % event_align \
                & 'trace_type="%s"'             % trace_type
            if len(entries) == 0:
                print('\n phase: %s...' % phase, flush=True)
                print('event ', event_align, flush=True)
                decoder_time_phase(session_keys_dict[phase], params, phase, True)
            pbar.update(1)


# ===========================================================================
# Experiment-wide hyperparameters
# ===========================================================================

params = {}

# ── Experimental phases to decode (in processing order) ─────────────────────
params['phases']             = ['gentest_2', 'discrimination', 'gentest_1', 'categorization_4']

# ── Time-axis binning ─────────────────────────────────────────────────────────
params['bin_window']         = 5    # width of each time bin (frames)
params['overlap_window']     = 1    # step size between consecutive bins (frames)

# ── Session / neuron subsampling ──────────────────────────────────────────────
params['nr_sessions_phase']  = 2    # minimum sessions required per phase
params['subsampling_fr']     = 0.9  # fraction of neurons to subsample per session

# ── Decoder training / validation ─────────────────────────────────────────────
params['training_fr']        = 0.7  # fraction of trials used for training
params['nr_crossvalidations'] = 100 # number of cross-validation folds

# ── Variables to decode ───────────────────────────────────────────────────────
params['variables']          = ['category', 'choice', 'outcome']

# ── Null distribution ─────────────────────────────────────────────────────────
params['nr_iter']            = 500  # shuffle iterations for null model

# ── Trial filtering ───────────────────────────────────────────────────────────
params['min_nr_trials_comb'] = 2    # minimum trials per condition combination
params['nr_combinations']    = 4    # number of trial condition combinations

# ── Parallelisation ───────────────────────────────────────────────────────────
params['nr_cores']           = 9    # worker processes (run `nproc` in terminal to check)

# ── Pseudopopulation mode ─────────────────────────────────────────────────────
params['pseudopopulation']   = True # pool neurons across sessions into a pseudopopulation