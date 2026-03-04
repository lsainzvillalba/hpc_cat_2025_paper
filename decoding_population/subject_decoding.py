#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:32:57 2024

@author: Laura Sainz Villalba

Per-subject (single-animal) time-resolved and window-based neural decoding of
behavioural variables, with cross-variable geometry (CCGP / parallelism score)
analysis.  Results are stored in three DataJoint tables:

  - ``Single_var_subject``       : time-resolved decoding accuracy per variable
  - ``Cross_var_subject``        : cross-variable geometry (CCGP / PS) per variable
  - ``Window_subject``           : window-averaged decoding + geometry per variable
  - ``Window_subject_decoder``   : window-averaged decoding for all variables jointly,
                                   including decoder vectors and train/test matrices

DataJoint schema : subject_decoding_hpc_cat_2025
"""

import numpy as np
from tqdm import tqdm
import datajoint as dj
from time import time
from multiprocessing import Pool

dj.config["enable_python_native_blobs"] = True
schema = dj.schema('subject_decoding_hpc_cat_2025', locals(), create_tables = True)

# ── Conditional imports depending on execution context ──────────────────────
if __name__ == '__main__':
    from util_decoder import get_session_keys,get_valid_sessions,get_nr_subsample_neurons,\
        get_common_time_axis_params, sessions_info_to_decode, acc_geom_estimation_at_t, \
            shuffle_available_set,one_iter_crosstime_decode,cross_var_at_t, \
            one_iter_all_window_decode,get_subsampling_idx,one_iter_cross_var,\
                window_decode_allvars
else:
    from .util_decoder import get_session_keys,get_valid_sessions,get_nr_subsample_neurons,\
    get_common_time_axis_params, sessions_info_to_decode, acc_geom_estimation_at_t, \
        shuffle_available_set,one_iter_crosstime_decode,cross_var_at_t, \
        one_iter_all_window_decode,get_subsampling_idx,one_iter_cross_var,\
            window_decode_allvars


# ===========================================================================
# DataJoint table definitions
# ===========================================================================

@schema
class Single_var_subject(dj.Manual):
    # train and test in each time point - time independent decoder
    # for each subject in phase
    definition = """ # decoding accuracy pseudopop in time with permuted shuffled test bootstrap for null model

    animal_id          : varchar(128)   # unique name for animal subject
    variable           : varchar(256)   # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    event_align        : varchar(128)   # event of alignment for time
    ---
    session_ids         : blob          # list of session ids corresponding to animal ids used
    accuracies          : longblob      # nr cross validations x accuracy trace in time 
    null                : longblob      # iterations x timepoints for null distribution decoding
    timepoints    : longblob      # common timepoints for trace used in neural decoder
    """


@schema
class Cross_var_subject(dj.Manual):
    definition = """ # decoding accuracy, cross condition decoding - geometry of variables in time

    animal_id       : varchar(128)   # unique name for animal subject
    variable        : varchar(256)   # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    event_align      : varchar(128)   # event of alignment - cero time reference
    ---
    session_ids      : blob       # list of session ids corresponding to animal ids used
    cross_variables  : blob       # list of variables -cross condition decoding
    cross_var_acc      : longblob   # ccgp decoder accuracy cross variable decoding
    cross_var_null     : longblob   # null ccgp decoder accuracy cross variable decoding
    cross_var_ps       : longblob   # ccgp decoder accuracy cross variable decoding
    cross_var_ps_null  : longblob   # null ccgp decoder accuracy cross variable decoding
    timepoints    : longblob      # common timepoints for trace used in neural decoder
    """


# ===========================================================================
# Time-resolved per-subject decoder
# ===========================================================================

def decoder_subject(_session_keys, params, phase, populate=True):
    """
    Run the time-resolved neural decoder for one subject in one experimental
    phase, then insert results into ``Single_var_subject`` and
    ``Cross_var_subject``.

    For each variable the function:
      1. Sweeps over all timepoints, computing cross-validated decoding
         accuracy (train at t, test at all t) and a shuffled null distribution
      2. Extracts the diagonal of the cross-time matrix as the time-independent
         accuracy and inserts it into ``Single_var_subject``
      3. Computes cross-variable (CCGP) and parallelism-score geometry at
         every timepoint, with an independent shuffled null, and inserts into
         ``Cross_var_subject``

    Parameters
    ----------
    _session_keys : list of dict
        Session key dicts for a single animal in this phase.
    params : dict
        Decoding hyperparameters (see module-level ``params`` dict).
    phase : str
        Experimental phase label (used as DB primary key).
    populate : bool
        If True (default), insert results into the DB.
    """
    # ── Unpack decoding hyperparameters ───────────────────────────────────────
    nr_iter             = params['nr_iter']             # shuffle iterations for null model
    nr_crossvalidations = params['nr_crossvalidations'] # CV folds for accuracy estimate
    event_align         = params['event_align']          # alignment event label
    variables           = params['variables']            # list of variables to decode
    nr_cores            = params['nr_cores']             # parallel worker count
    nr_variables        = len(variables)

    # ── Build session tensors and trial availability sets ─────────────────────
    sessions_info        = sessions_info_to_decode(_session_keys, params)
    session_ids          = [key['session_id'] for key in _session_keys]
    available_set_trials = sessions_info['available_set_trials']
    nr_timepoints        = sessions_info['list_tensors'][0].shape[-1]

    # ── Decode each variable independently ────────────────────────────────────
    # print('behaviour', (time()-start)/60,'mins')
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
        # neural activity decoding
        for t_idx in tqdm(range(nr_timepoints)):
            start = time()
            # Cross-validated accuracy: train and test at t_idx
            acc, _ = acc_geom_estimation_at_t(sessions_info, params,
                                              available_set_trials,
                                              t_idx, variable)
            cross_time_acc[:, t_idx, :] = acc   # shape: (nr_crossvalidations, nr_timepoints)

            # ── Null distribution via shuffled labels ────────────────────────
            # decoding null distribution in pool parallelization
            with Pool(nr_cores) as pool:
                # Pre-generate shuffled trial sets and neuron subsampling indices
                # Prepare the data
                available_sets      = [shuffle_available_set(available_set_trials) for n in range(nr_iter)]
                subsampling_indices = [get_subsampling_idx(sessions_info, params['nr_neurons_session']) for n in range(nr_iter)]
                args        = [(sessions_info, params, t_idx, available_sets[n], subsampling_indices[n], True) for n in range(nr_iter)]
                results_acc = pool.starmap(one_iter_crosstime_decode, args)
                acc_null[:, t_idx] = np.array(results_acc)

            print('decoding timepoint ', (time()-start)/60, 'mins')

        # Time-independent accuracy = diagonal of the cross-time matrix
        # (trained and tested at the same timepoint)
        acc = np.array([np.diagonal(cross_time_acc[i]) for i in range(len(cross_time_acc))])

        # ── Insert time-resolved decoding results ──────────────────────────────
        print('\n populating in time decoding')
        in_time_entry = {
            'animal_id':              _session_keys[0]['animal_id'],
            'variable':               variable,
            'trace_type':             params['trace_type'],
            'experimental_timepoint': phase,
            'event_align':            event_align,
            'session_ids':            session_ids,
            'accuracies':             acc,
            'null':                   acc_null,
            'timepoints':             sessions_info['timepoint_axis'],
            }
        Single_var_subject().insert1(in_time_entry, skip_duplicates=True)

        # ── Cross-variable geometry (CCGP / parallelism score) ─────────────────
        # Remove the current variable from the list to build cross-variable pairs
        cross_variables   = variables[:]
        cross_variables.pop(variables.index(variable))
        # Accumulators: (crossvalidations × cross_vars × timepoints)
        cross_var_acc     = np.zeros((nr_crossvalidations, nr_variables-1, nr_timepoints))
        cross_var_ps      = np.zeros((nr_crossvalidations, nr_variables-1, nr_timepoints))
        cross_var_null    = np.zeros((nr_iter, nr_variables-1, nr_timepoints))
        cross_var_ps_null = np.zeros((nr_iter, nr_variables-1, nr_timepoints))

        # ── Sweep over all timepoints for cross-variable geometry ──────────────
        # neural activity decoding
        for t_idx in tqdm(range(nr_timepoints)):
            start = time()
            # CCGP and parallelism score at this timepoint
            acc, ps = cross_var_at_t(sessions_info, params,
                                     available_set_trials, t_idx)
            print('cross var acc.shape', acc.shape)
            print('ps.shape', ps.shape)

            cross_var_acc[:, :, t_idx] = acc
            cross_var_ps[:, :, t_idx]  = ps

            # ── Null distribution via shuffled labels ────────────────────────
            # decoding null distribution in pool parallelization
            with Pool(nr_cores) as pool:
                # Prepare the data
                available_sets      = [shuffle_available_set(available_set_trials) for n in range(nr_iter)]
                subsampling_indices = [get_subsampling_idx(sessions_info, params['nr_neurons_session']) for n in range(nr_iter)]
                args    = [(sessions_info, params, t_idx, available_sets[n], subsampling_indices[n], True) for n in range(nr_iter)]
                # print('running acc pool')
                results = pool.starmap(one_iter_cross_var, args)
                results = np.array(results)
                # results shape: (nr_iter, 2, nr_cross_vars) — dim 1: [ccgp_acc, ps]
                assert results.shape == (nr_iter, 2, len(cross_variables))
                print('results.shape', results.shape)
                print('null decoding Done')
                cross_var_null[:, :, t_idx]    = results[:, 0]  # CCGP null
                cross_var_ps_null[:, :, t_idx] = results[:, 1]  # PS null

        # ── Insert cross-variable geometry results ─────────────────────────────
        geometrytime_entry = {
            'animal_id':              _session_keys[0]['animal_id'],
            'variable':               variable,
            'trace_type':             params['trace_type'],
            'experimental_timepoint': phase,
            'event_align':            event_align,
            'session_ids':            session_ids,
            'cross_variables':        cross_variables,
            'cross_var_acc':          cross_var_acc,
            'cross_var_ps':           cross_var_ps,
            'cross_var_null':         cross_var_null,
            'cross_var_ps_null':      cross_var_ps_null,
            'timepoints':             sessions_info['timepoint_axis'],
            }
        Cross_var_subject().insert1(geometrytime_entry, skip_duplicates=True)


# ===========================================================================
# Window-based per-subject decoder
# ===========================================================================

@schema
class Window_subject(dj.Manual):
    definition = """ # geometry, decoding accuracy and ccgp of variables at different windows of interest

    animal_id          : varchar(128)    # Mouse id (unique id)
    variable           : varchar(256)    # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    window        : varchar(128)   # window_of_interest - post stimulus, pre choice, post choice, post outcome
    window_length    : float   # dff or spikes
    ---
    event_align         : varchar(128)   # event of alignment - cero time reference
    session_ids         : blob       # list of session ids corresponding to animal ids used
    accuracy            : longblob       # nr cross validations,  accuracy of decoder value for window
    null                : longblob   # nr iterations x, null distribution decoding
    ccgp                : longblob       # nr  cross validations, ccgp decoder accuracy cross variable decoding
    ccgp_null           : longblob   # nr iterations, null distribution for ccgp accuracies
    fraction_ccgp       : blob       # nr  cross validations, fraction ccgp respect of decoding accuracy
    weight_distribution : longblob   # nr  cross validations, weight values for corresponding neuron ids 
    neuron_ids_decode   :longblob    # nr  cross validations, neuron ids used in each cross validation
    """

    def window_subject(self, _session_keys, params, populate=True):
        """
        Decode all variables within a fixed time window for one subject,
        computing both cross-validated accuracy and a shuffled null distribution,
        then insert into ``Window_subject``.

        The window runs from t=0 (alignment event) to t=window_length seconds.
        Cross-validation and null-distribution runs are parallelised via a Pool.

        Parameters
        ----------
        _session_keys : list of dict
            Session key dicts for a single animal.
        params : dict
            Decoding hyperparameters (must include 'window' and 'window_length').
        populate : bool
            If True (default), insert results into the DB.
        """
        # print('\n Gathering session keys...',flush=True)
        start__run = time()

        # ── Extract window boundaries from the common time axis ───────────────
        window        = params['window']
        window_length = params['window_length']
        sessions_info = sessions_info_to_decode(_session_keys, params)
        session_ids   = [key['session_id'] for key in _session_keys]
        # nr_sessions = len(session_ids)
        available_set_trials = sessions_info['available_set_trials']

        nr_iter        = params['nr_iter']
        event_align    = params['event_align']
        variables      = params['variables']
        nr_cores       = params['nr_cores']
        timepoint_axis = sessions_info['timepoint_axis']
        # Frame indices corresponding to [t=0, t=window_length]
        window_idx = [np.where(timepoint_axis>0)[0][0],
                      np.where(timepoint_axis>window_length)[0][0]]

        for variable in variables:
            print('\n Decoding for variable: ', variable, flush=True)
            # pbar = tqdm(total = nr_decoder_runs)
            # train and test condition variables, allows cross condition testing
            #sessions_info['decoding_var'] = variable
            params['train_var'] = variable
            params['test_var']  = variable

            # ── Cross-validated accuracy and null distribution (parallel) ──────
            # null distribution in pool parallelization
            # print('available_set_trials',available_set_trials)
            with Pool(nr_cores) as pool:
                print('computing accuracy ...', flush=True)
                nr_cores = params['nr_cores']
                # Cross-validation: shuffle=False → real trial labels
                subsampling_idx = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
                available_sets  = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]
                args     = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], False) for n in range(params['nr_crossvalidations'])]
                results  = pool.starmap(one_iter_all_window_decode, args)

                print('computing null ...', flush=True)
                # Null distribution: shuffle=True → permuted trial labels
                # Prepare the data
                subsampling_idx = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(nr_iter)]
                available_sets  = [shuffle_available_set(available_set_trials) for n in range(nr_iter)]
                args        = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], True) for n in range(nr_iter)]
                results_null = pool.starmap(one_iter_all_window_decode, args)

            # ── Unpack per-field results from CV and null runs ─────────────────
            print('populating entry ...', flush=True)
            #print('results',results[:2])
            acc               = [result['acc']               for result in results]
            ccgp              = [result['ccgp']              for result in results]
            fraction_ccgp     = [result['fraction']          for result in results]
            weight_distribution = [result['weight_distribution'] for result in results]
            neuron_ids_decode = [result['neuron_ids_decode'] for result in results]
            null              = [result['acc']               for result in results_null]
            ccgp_null         = [result['ccgp']              for result in results_null]

            # ── Insert window decoding results ─────────────────────────────────
            entry = {
                'animal_id':              _session_keys[0]['animal_id'],
                'variable':               variable,
                'trace_type':             _session_keys[0]['trace_type'],
                'experimental_timepoint': _session_keys[0]['experimental_timepoint'],
                'window':                 window,
                'window_length':          window_length,
                'event_align':            event_align,
                'session_ids':            session_ids,
                'accuracy':               acc,
                'null':                   null,
                'ccgp':                   ccgp,
                'ccgp_null':              ccgp_null,
                'fraction_ccgp':          fraction_ccgp,
                'weight_distribution':    weight_distribution,
                'neuron_ids_decode':      neuron_ids_decode,
                }
            self.insert1(entry, skip_duplicates=True)

        print('TIME run : ', (time()-start__run))

    def window_decode_all_subjects(self, params):
        """
        Outer loop: run ``window_subject`` for every combination of window,
        phase, and animal present in the experiment.

        For each window definition the function:
          1. Retrieves valid session keys and filters by minimum trial count
          2. Determines per-subject neuron subsampling counts
          3. Iterates over phases and animals, skipping already-populated entries

        Parameters
        ----------
        params : dict
            Experiment-wide hyperparameters (see module-level ``params`` dict).
        """
        print('Gathering sessions in experiment...', flush=True)

        # ── Unpack loop-level parameters ──────────────────────────────────────
        windows_of_interest = params['windows']
        variables           = params['variables']
        min_nr_trials       = params['min_nr_trials_comb']
        nr_combinations     = params['nr_combinations']

        for i, window in enumerate(windows_of_interest):
            # Set window-specific params for this iteration
            params['window']       = window
            params['window_length'] = params['window_lengths'][i]
            params['event_align']  = params['event_alignments'][i]

            # ── Gather and validate session keys ──────────────────────────────
            session_keys_dict = get_session_keys(params, params['event_align'])
            # print(session_keys_dict['discrimination']['BK4933_LR'])
            get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)
            # print(session_keys_dict['discrimination']['BK4933_LR'])

            # Compute per-subject neuron subsampling counts
            nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
                session_keys_dict,
                params['subsampling_fr'],
                params['nr_sessions_phase'],
                across_axis='subjects',
            )
            # print(nr_neurons_session['discrimination']['BK4933_LR'])

            # Set pseudopopulation trial count (or None for single-session mode)
            if params['pseudopopulation']:
                params['n_datapoints'] = n_datapoints
            else:
                params['n_datapoints'] = None

            total_runs = np.sum([len(session_keys_dict[phase]) for phase in session_keys_dict])

            # ── Compute common time-axis bounds ────────────────────────────────
            zero_left, left_range, right_range = get_common_time_axis_params(session_keys_dict)
            params['left_range']  = left_range
            params['right_range'] = right_range
            phases = params['phases']

            print('\n Processing all entries...', flush=True)
            pbar = tqdm(total=total_runs, position=0, leave=True)

            # ── Iterate over phases and subjects; skip existing DB entries ─────
            for phase in phases[:4]:
                for animal_id in session_keys_dict[phase]:
                    print('\n phase: %s...' % phase, flush=True)
                    print('\n animal_id: %s...' % animal_id, flush=True)
                    print('\n window: %s...' % window, flush=True)
                    params['zero_left']          = zero_left[phase][animal_id]
                    params['nr_neurons_session'] = nr_neurons_session[phase][animal_id]
                    # Check if entry already exists
                    entries = self \
                        & 'experimental_timepoint="%s"' % phase \
                        & 'animal_id="%s"'              % animal_id \
                        & 'window="%s"'                 % params['window'] \
                        & 'window_length="%s"'          % params['window_length']

                    if len(entries) == 0:
                       self.window_subject(session_keys_dict[phase][animal_id],
                                           params, True)
                    pbar.update(1)


# ===========================================================================
# All-variable joint window decoder (decoder vectors + train/test matrices)
# ===========================================================================

@schema
class Window_subject_decoder(dj.Manual):
    definition = """ # decoding dimensions for each subject 

    animal_id          : varchar(128)    # Mouse id (unique id)
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    window        : varchar(128)   # window_of_interest - post stimulus, pre choice, post choice, post outcome
    window_length    : float   # dff or spikes
    ---
    variables           : tinyblob   # variables to decode
    event_align         : varchar(128)   # event of alignment - cero time reference
    session_ids    : blob       # list of session ids corresponding to animal ids used
    accuracy       : longblob       # mean accuracy of decoder value for window
    ccgp           : longblob       # mean ccgp decoder accuracy against the other variables
    ps             : longblob       # mean parallelism score against the other variables
    decoder_vec : longblob   # nr  cross validations, weight values for corresponding neuron ids 
    neuron_ids_decode   : longblob   # nr  cross validations, neuron ids used in each cross validation
    combinations   : longblob   # combination of cat, cho, out for trial types used in decoding
    x_train   : longblob   # decoder matrix fed for training trials x neurons, concatenated 4 trial types
    x_test   : longblob   # decoder matrix fed for testing, trials x neurons, concatenated 4 trial types
    """

    def per_subject(self, _session_keys, phase, params, populate=True):
        """
        Run the all-variable joint decoder for one subject in one window,
        storing mean accuracy, CCGP, parallelism score, and a random subset
        of cross-validation decoder vectors and train/test matrices.

        Cross-validation runs are parallelised; 200 runs are then sampled at
        random for storing decoder vectors and train/test data (to limit storage).

        Parameters
        ----------
        _session_keys : list of dict
            Session key dicts for a single animal.
        phase : str
            Experimental phase label.
        params : dict
            Decoding hyperparameters.
        populate : bool
            If True (default), insert results into the DB.
        """
        print('subject: ', _session_keys[0]['animal_id'])
        start_run = time()

        # ── Extract window boundaries from the common time axis ───────────────
        window        = params['window']
        window_length = params['window_length']
        sessions_info = sessions_info_to_decode(_session_keys, params)
        session_ids   = [key['session_id'] for key in _session_keys]
        # nr_sessions = len(session_ids)
        available_set_trials = sessions_info['available_set_trials']

        event_align    = params['event_align']
        variables      = params['variables']
        nr_cores       = params['nr_cores']
        timepoint_axis = sessions_info['timepoint_axis']
        # Frame indices corresponding to [t=0, t=window_length]
        window_idx = [np.where(timepoint_axis>0)[0][0],
                      np.where(timepoint_axis>window_length)[0][0]]

        # ── Pre-generate subsampling and trial-shuffle indices ─────────────────
        subsampling_idx = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
        available_sets  = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]

        # ── Run all-variable joint decoder in parallel ─────────────────────────
        with Pool(nr_cores) as pool:
            print('computing accuracy ...', flush=True)
            nr_cores = params['nr_cores']
            # Cross-validation: shuffle=False → real trial labels
            args    = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], False) for n in range(params['nr_crossvalidations'])]
            results = pool.starmap(window_decode_allvars, args)

        # ── Average accuracy metrics over all CV runs ──────────────────────────
        # pick 10 cross validation runs at random
        results = np.array(results, dtype=object)
        acc  = np.mean([result['acc']  for result in results], axis=0)
        ccgp = np.mean([result['ccgp'] for result in results], axis=0)
        ps   = np.mean([result['ps']   for result in results], axis=0)

        # ── Subsample 200 CV runs for storing decoder vectors + matrices ───────
        idx      = np.random.randint(0, len(results), 200)  # 40 runs selected randomly
        selected = results[idx]
        # from selected cross validations get results
        # nr selected x nr vars

        # Decoder weight vectors: nr selected x nr vars x nr neurons
        decoder_vec       = [result['decoder_vec']       for result in selected]
        neuron_ids_decode = results[0]['neuron_ids_decode']   # same across runs
        combinations      = results[0]['combinations']        # same across runs
        # Train / test matrices: nr selected x (nr_combinations*nr_trials) x nr_neurons
        X_train = [result['X_train'] for result in selected]
        print('x train ', len(X_train), X_train[0].shape, flush=True)
        X_test  = [result['X_test']  for result in selected]
        print('x test', len(X_test), X_test[0].shape, flush=True)

        # ── Insert all-variable decoding results ───────────────────────────────
        entry = {
            'animal_id':              _session_keys[0]['animal_id'],
            'experimental_timepoint': phase,
            'window':                 window,
            'window_length':          window_length,
            'variables':              variables,
            'event_align':            event_align,
            'session_ids':            session_ids,
            'accuracy':               acc,
            'ccgp':                   ccgp,
            'ps':                     ps,
            'decoder_vec':            decoder_vec,
            'neuron_ids_decode':      neuron_ids_decode,
            'combinations':           combinations,
            'x_train':                X_train,
            'x_test':                 X_test,
            }
        self.insert1(entry, skip_duplicates=True)

        print('TIME run : ', (time()-start_run))
        print('times 72')

    def decode_all_subjects(self, params):
        """
        Outer loop: run ``per_subject`` for every combination of window, phase,
        and animal present in the experiment.

        Parameters
        ----------
        params : dict
            Experiment-wide hyperparameters (see module-level ``params`` dict).
        """
        print('Gathering sessions in experiment...', flush=True)

        # ── Unpack loop-level parameters ──────────────────────────────────────
        windows_of_interest = params['windows']
        variables           = params['variables']
        min_nr_trials       = params['min_nr_trials_comb']
        nr_combinations     = params['nr_combinations']

        for i, window in enumerate(windows_of_interest):
            # Set window-specific params for this iteration
            params['window']        = window
            params['window_length'] = params['window_lengths'][i]
            params['event_align']   = params['event_alignments'][i]

            # ── Gather and validate session keys ──────────────────────────────
            session_keys_dict = get_session_keys(params, params['event_align'])
            get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)

            # Compute per-subject neuron subsampling counts
            nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
                session_keys_dict,
                params['subsampling_fr'],
                params['nr_sessions_phase'],
                across_axis='subjects',
            )

            # Set pseudopopulation trial count (or None for single-session mode)
            if params['pseudopopulation']:
                params['n_datapoints'] = n_datapoints
            else:
                params['n_datapoints'] = None

            total_runs = np.sum([len(session_keys_dict[phase]) for phase in session_keys_dict])

            # ── Compute common time-axis bounds ────────────────────────────────
            zero_left, left_range, right_range = get_common_time_axis_params(session_keys_dict)
            params['left_range']  = left_range
            params['right_range'] = right_range
            phases = params['phases']

            print('\n Processing all entries...', flush=True)
            pbar = tqdm(total=total_runs, position=0, leave=True)

            # ── Iterate over all phases and subjects; skip existing DB entries ──
            for phase in phases:
                for animal_id in session_keys_dict[phase]:
                    print('\n phase: %s...'   % phase,     flush=True)
                    print('\n animal_id: %s...' % animal_id, flush=True)
                    print('\n window: %s...'   % window,    flush=True)
                    params['zero_left']          = zero_left[phase][animal_id]
                    params['nr_neurons_session'] = nr_neurons_session[phase][animal_id]
                    # Check if entry already exists
                    entries = self \
                        & 'experimental_timepoint="%s"' % phase \
                        & 'animal_id="%s"'              % animal_id \
                        & 'window="%s"'                 % params['window'] \
                        & 'window_length="%s"'          % params['window_length']

                    if len(entries) == 0:
                        self.per_subject(session_keys_dict[phase][animal_id],
                                         phase, params, True)
                    pbar.update(1)


# ===========================================================================
# Experiment-wide hyperparameters
# ===========================================================================

params = {}

# ── Experimental phases to decode ─────────────────────────────────────────────
params['phases']             = ['discrimination', 'gentest_1', 'categorization_4', 'gentest_2']

# ── Time-axis binning ─────────────────────────────────────────────────────────
params['bin_window']         = 5    # width of each time bin (frames)
params['overlap_window']     = 1    # step size between consecutive bins (frames)

# ── Session / neuron subsampling ──────────────────────────────────────────────
params['nr_sessions_phase']  = 2    # minimum sessions required per phase
params['subsampling_fr']     = 0.9  # fraction of neurons to subsample per session

# ── Decoder training / validation ─────────────────────────────────────────────
params['training_fr']        = 0.7  # fraction of trials used for training
params['nr_crossvalidations'] = 250  # number of cross-validation folds

# ── Variables to decode ───────────────────────────────────────────────────────
params['variables']          = ['category', 'choice', 'outcome']

# ── Null distribution ─────────────────────────────────────────────────────────
params['nr_iter']            = 500  # shuffle iterations for null model

# ── Trial filtering ───────────────────────────────────────────────────────────
params['min_nr_trials_comb'] = 2    # minimum trials per condition combination
params['nr_combinations']    = 4    # number of trial condition combinations

# ── Parallelisation ───────────────────────────────────────────────────────────
params['nr_cores']           = 5    # worker processes (run `nproc` in terminal to check)

# ── Pseudopopulation mode ─────────────────────────────────────────────────────
params['pseudopopulation']   = True  # pool neurons across sessions into a pseudopopulation

# ── Window definitions (one entry per window; lists must be same length) ──────
params['event_alignments']   = ['choice', 'choice', 'choice']   # alignment event per window
params['window_lengths']     = [0.7, 1, 1.5]                    # window duration in seconds
params['windows']            = ['post_choice', 'post_choice', 'post_choice']  # window label


