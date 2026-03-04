#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:02:41 2024

@author: Laura Sainz Villalba

Pseudopopulation-level cross-variable geometry (CCGP / parallelism score) and
window-based decoding for a cohort of animals pooled across sessions.

Five DataJoint tables are populated:

  - ``Cross_var``                 : time-resolved CCGP + PS per variable
  - ``By_window``                 : window-averaged decoding, CCGP, weights (per variable)
  - ``By_window_decoder``         : window-averaged decoding + decoder vectors (per variable,
                                    10 randomly selected CV runs stored)
  - ``Window_cohort_neuronset``   : all-variable joint decoding on a shared neuron set
                                    (all CV runs for accuracy; 10 for decoder vectors)
  - ``Window_cohort_decspace``    : same as above but accuracies are averaged across CV runs
                                    (20 randomly selected runs for decoder vectors)

DataJoint schema : pseudopopulation_geometry_hpc_cat_2025
"""

import numpy as np
from tqdm import tqdm
import datajoint as dj
from time import time
from multiprocessing import Pool

dj.config["enable_python_native_blobs"] = True
schema = dj.schema('pseudopopulation_geometry_hpc_cat_2025', locals(), create_tables = True)

# print('calling in pseudopopulation_geometry from: ',__name__)

# ── Conditional imports depending on execution context ──────────────────────
if __name__ == '__main__':
    from util_decoder import get_session_keys,get_valid_sessions,get_nr_subsample_neurons,\
    get_common_time_axis_params, sessions_info_to_decode, cross_var_at_t, \
        shuffle_available_set,get_subsampling_idx, one_iter_cross_var, \
        one_iter_all_window_decode,window_decode,window_decode_allvars
else:
    from .util_decoder import get_session_keys,get_valid_sessions,get_nr_subsample_neurons,\
        get_common_time_axis_params, sessions_info_to_decode, cross_var_at_t, \
            shuffle_available_set,get_subsampling_idx, one_iter_cross_var, \
            one_iter_all_window_decode,window_decode,window_decode_allvars


# ===========================================================================
# DataJoint table definition — time-resolved cross-variable geometry
# ===========================================================================

@schema
class Cross_var(dj.Manual):
    definition = """ # cross variable decoding accuracy pseudopop geometry of variables in time

    variable        : varchar(256)   # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    event_align      : varchar(128)   # event of alignment - cero time reference
    ---
    session_ids      : blob       # list of session ids corresponding to animal ids used
    animal_ids       : blob       # list animal ids for pseudopopulation decoding
    cross_variables  : blob       # list of variables -cross condition decoding
    cross_var_acc      : longblob   # ccgp decoder accuracy cross variable decoding
    cross_var_null     : longblob   # null ccgp decoder accuracy cross variable decoding
    cross_var_ps       : longblob   # ccgp decoder accuracy cross variable decoding
    cross_var_ps_null  : longblob   # null ccgp decoder accuracy cross variable decoding
    timepoints    : longblob      # common timepoints for trace used in neural decoder
    """


# ===========================================================================
# Time-resolved cross-variable decoding function
# ===========================================================================

def cross_var_decode(_session_keys, params, phase, populate=True):
    """
    Run the time-resolved cross-variable (CCGP / PS) decoder for the full
    pseudopopulation in one phase, then insert into ``Cross_var``.

    For each variable the function:
      1. Sweeps over all timepoints, computing CCGP and parallelism score
         at each point
      2. In parallel, runs a shuffled null distribution for both metrics
      3. Inserts the full time-resolved geometry traces into ``Cross_var``

    Parameters
    ----------
    _session_keys : dict
        Nested dict {animal_id: [session_key_dicts]} for this phase.
    params : dict
        Decoding hyperparameters (see module-level ``params`` dict).
    phase : str
        Experimental phase label (used as DB primary key).
    populate : bool
        If True (default), insert results into the DB.
    """
    # print('\n Gathering session keys...',flush=True)
    #start_run = time()

    # ── Unpack decoding hyperparameters ───────────────────────────────────────
    nr_iter             = params['nr_iter']             # shuffle iterations for null model
    nr_crossvalidations = params['nr_crossvalidations'] # CV folds for accuracy estimate
    event_align         = params['event_align']          # alignment event label
    variables           = params['variables']            # list of variables to decode
    nr_cores            = params['nr_cores']             # parallel worker count
    nr_variables        = len(variables)

    # ── Build session tensors and trial availability sets ─────────────────────
    sessions_info        = sessions_info_to_decode(_session_keys, params)
    available_set_trials = sessions_info['available_set_trials']
    nr_timepoints        = sessions_info['list_tensors'][0].shape[-1]
    timepoints           = sessions_info['timepoint_axis']

    # ── Decode each variable independently ────────────────────────────────────
    for variable in variables:
        print('\n Decoding for variable: ', variable, flush=True)
        # Set train and test variable to the same label
        params['train_var'] = variable
        params['test_var']  = variable

        # Remove the current variable to build the cross-variable pairs
        cross_variables   = variables[:]
        cross_variables.pop(variables.index(variable))

        # Accumulators: (crossvalidations/iter × cross_vars × timepoints)
        cross_var_acc     = np.zeros((nr_crossvalidations, nr_variables-1, nr_timepoints))
        cross_var_ps      = np.zeros((nr_crossvalidations, nr_variables-1, nr_timepoints))
        cross_var_null    = np.zeros((nr_iter, nr_variables-1, nr_timepoints))
        cross_var_ps_null = np.zeros((nr_iter, nr_variables-1, nr_timepoints))

        # ── Sweep over all timepoints ──────────────────────────────────────────
        # neural activity decoding
        for t_idx in tqdm(range(nr_timepoints)):
            start = time()
            # CCGP and parallelism score at this timepoint
            acc, ps = cross_var_at_t(sessions_info, params,
                                     available_set_trials, t_idx)
            # print('acc.shape', acc.shape)
            # print('ps.shape', ps.shape)

            cross_var_acc[:, :, t_idx] = acc
            cross_var_ps[:, :, t_idx]  = ps

            # ── Null distribution via parallel shuffles ────────────────────────
            # decoding null distribution in pool parallelization
            with Pool(nr_cores) as pool:
                # Pre-generate shuffled trial sets and subsampling indices
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

            print('decoding timepoint ', (time()-start)/60, 'mins')
            print('multiply by timepoints, by 3 , by 27')

        # ── Flatten session keys into ordered lists ────────────────────────────
        session_ids = [key['session_id'] for animal_id in _session_keys for key in _session_keys[animal_id]]
        animal_ids  = [key['animal_id']  for animal_id in _session_keys for key in _session_keys[animal_id]]

        # ── Insert time-resolved geometry results ──────────────────────────────
        print('\n populating in geometry in time decoding')
        geometrytime_entry = {
            'variable':               variable,
            'experimental_timepoint': phase,
            'event_align':            event_align,
            'animal_ids':             animal_ids,
            'session_ids':            session_ids,
            'cross_variables':        cross_variables,
            'cross_var_acc':          cross_var_acc,
            'cross_var_ps':           cross_var_ps,
            'cross_var_null':         cross_var_null,
            'cross_var_ps_null':      cross_var_ps_null,
            'timepoints':             timepoints,
            }
        Cross_var().insert1(geometrytime_entry, skip_duplicates=True)


def decode_in_time_all_subjects(params):
    """
    Outer loop: run ``cross_var_decode`` for every combination of alignment
    event and experimental phase defined in ``params``.

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

    for event_align in event_alignments:
        params['event_align'] = event_align

        # ── Gather and validate session keys for this alignment event ──────────
        session_keys_dict = get_session_keys(params, event_align)
        get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)

        # Compute how many neurons to subsample per phase (equalise across phases)
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

        # ── Decode first 4 phases; skip if entry already exists ───────────────
        for phase in phases[:4]:
            params['zero_left']          = zero_left[phase]
            params['nr_neurons_session'] = nr_neurons_session[phase]
            # Check if entry already exists
            entries = Cross_var() \
                & 'experimental_timepoint="%s"' % phase \
                & 'event_align="%s"'            % event_align
            if len(entries) == 0:
                print('\n phase: %s...' % phase, flush=True)
                print('event ', event_align, flush=True)
                cross_var_decode(session_keys_dict[phase], params, phase, True)
            pbar.update(1)


# ===========================================================================
# Window-based cohort decoders
# ===========================================================================

@schema
class By_window(dj.Manual):
    definition = """ # 

    variable           : varchar(256)    # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    window        : varchar(128)   # window_of_interest - post stimulus, pre choice, post choice, post outcome
    ---
    event_align         : varchar(128)   # event of alignment - cero time reference
    session_ids      : blob       # list of session ids corresponding to animal ids used
    animal_ids       : blob       # list animal ids for pseudopopulation decoding
    accuracy            : longblob       # nr cross validations,  accuracy of decoder value for window
    null                : longblob   # nr iterations x, null distribution decoding
    ccgp                : longblob       # nr  cross validations, ccgp decoder accuracy cross variable decoding
    ccgp_null           : longblob   # nr iterations, null distribution for ccgp accuracies
    fraction_ccgp       : blob       # nr  cross validations, fraction ccgp respect of decoding accuracy
    weight_distribution : longblob   # nr  cross validations, weight values for corresponding neuron ids 
    neuron_ids_decode   : longblob   # nr  cross validations, neuron ids used in each cross validation
    
    """

    def decoder_in_phase(self, _session_keys, phase, params, populate=True):
        """
        Decode all variables within a fixed time window for one phase of the
        full pseudopopulation, storing both cross-validated accuracy and a
        shuffled null distribution.

        CV runs use shuffle=False (real trial labels); null runs use
        shuffle=True (permuted labels).  Both are parallelised via a Pool.

        Parameters
        ----------
        _session_keys : dict
            Nested {animal_id: [session_key_dicts]} for this phase.
        phase : str
            Experimental phase label.
        params : dict
            Decoding hyperparameters.
        populate : bool
            If True (default), insert results into the DB.
        """
        # print('\n Gathering session keys...',flush=True)
        start__run = time()

        # ── Extract window boundaries from the common time axis ───────────────
        window        = params['window']
        window_length = params['window_length']
        sessions_info = sessions_info_to_decode(_session_keys, params)
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
                subsampling_idx  = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(nr_iter)]
                available_sets   = [shuffle_available_set(available_set_trials) for n in range(nr_iter)]
                args         = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], True) for n in range(nr_iter)]
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

            # Flatten nested session keys into ordered lists
            session_ids = [key['session_id'] for animal_id in _session_keys for key in _session_keys[animal_id]]
            animal_ids  = [key['animal_id']  for animal_id in _session_keys for key in _session_keys[animal_id]]

            # ── Insert window decoding results ─────────────────────────────────
            entry = {
                'variable':               variable,
                'experimental_timepoint': phase,
                'window':                 window,
                'window_length':          window_length,
                'event_align':            event_align,
                'session_ids':            session_ids,
                'animal_ids':             animal_ids,
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

    def decode_cohort(self, params):
        """
        Outer loop: run ``decoder_in_phase`` for every combination of window
        and phase defined in ``params``.

        Parameters
        ----------
        params : dict
            Experiment-wide hyperparameters.
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
            # print(session_keys_dict['discrimination']['BK4933_LR'])
            get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)
            # print(session_keys_dict['discrimination']['BK4933_LR'])

            # Compute per-phase neuron subsampling counts
            nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
                session_keys_dict,
                params['subsampling_fr'],
                params['nr_sessions_phase'],
                across_axis='phases',
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

            # ── Iterate over all phases; skip existing DB entries ──────────────
            for phase in phases:
                print('\n phase: %s...' % phase, flush=True)
                print('\n window: %s...' % window, flush=True)
                params['zero_left']          = zero_left[phase]
                params['nr_neurons_session'] = nr_neurons_session[phase]
                # Check if entry already exists
                entries = self \
                    & 'experimental_timepoint="%s"' % phase \
                    & 'window="%s"'                 % params['window'] \
                    & 'window_length="%s"'          % params['window_length']

                if len(entries) == 0:
                    self.decoder_in_phase(session_keys_dict[phase], phase,
                                          params, True)
                pbar.update(1)


# ===========================================================================
# Per-variable window decoder with decoder vectors (By_window_decoder)
# ===========================================================================

@schema
class By_window_decoder(dj.Manual):
    definition = """ # g

    variable           : varchar(256)    # variable to decode
    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    window        : varchar(128)   # window_of_interest - post stimulus, pre choice, post choice, post outcome
    window_length    : float   # window length in seconds
    ---
    event_align         : varchar(128)   # event of alignment - cero time reference
    session_ids      : blob       # list of session ids corresponding to animal ids used
    animal_ids       : blob       # list animal ids for pseudopopulation decoding
    accuracy            : longblob       # nr cross validations,  accuracy of decoder value for window
    ccgp                : longblob       # nr  cross validations, ccgp decoder accuracy against the other variables
    decoder_vec : longblob   # nr  cross validations, weight values for corresponding neuron ids 
    neuron_ids_decode   : longblob   # nr  cross validations, neuron ids used in each cross validation
    combinations   : longblob   # combination of cat, cho, out for trial types used in decoding
    x_train   : longblob   # decoder matrix fed for training trials x neurons, concatenated 4 trial types
    x_test   : longblob   # decoder matrix fed for testing, trials x neurons, concatenated 4 trial types
    """
    # decoding of different variables is not done on the same set of neurons for each run as in Window_cohort_decspace
    # accuracy and ccgp have all the cross validation runs

    def decoder_in_phase(self, _session_keys, phase, params, populate=True):
        """
        Decode one variable per run within a fixed window for one phase,
        storing all CV accuracy values and 10 randomly selected decoder
        vectors / train–test matrices.

        Unlike ``Window_cohort_neuronset`` / ``Window_cohort_decspace``,
        different variables may be decoded on different neuron subsets across
        runs (no shared neuron set enforced).

        Parameters
        ----------
        _session_keys : dict
            Nested {animal_id: [session_key_dicts]} for this phase.
        phase : str
            Experimental phase label.
        params : dict
            Decoding hyperparameters.
        populate : bool
            If True (default), insert results into the DB.
        """
        # print('\n Gathering session keys...',flush=True)
        start__run = time()

        # ── Extract window boundaries from the common time axis ───────────────
        window        = params['window']
        window_length = params['window_length']
        sessions_info = sessions_info_to_decode(_session_keys, params)
        # nr_sessions = len(session_ids)
        available_set_trials = sessions_info['available_set_trials']

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

            # ── Cross-validated accuracy (parallel, real labels only) ──────────
            # null distribution in pool parallelization
            # print('available_set_trials',available_set_trials)
            with Pool(nr_cores) as pool:
                print('computing accuracy ...', flush=True)
                nr_cores = params['nr_cores']
                # Cross-validation: shuffle=False → real trial labels
                subsampling_idx = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
                available_sets  = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]
                args    = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], False) for n in range(params['nr_crossvalidations'])]
                results = pool.starmap(window_decode, args)

            # ── Unpack accuracy and CCGP from all CV runs ──────────────────────
            print('populating entry ...', flush=True)
            acc  = [result['acc']  for result in results]
            ccgp = [result['ccgp'] for result in results]

            # ── Subsample 10 CV runs for storing decoder vectors + matrices ────
            # pick 10 cross validation runs at random
            results   = np.array(results, dtype=object)
            idx       = np.random.randint(0, len(results), 10)
            selected  = results[idx]
            decoder_vec       = [result['decoder_vec']       for result in selected]
            neuron_ids_decode = [result['neuron_ids_decode'] for result in selected]
            combinations      = results[0]['combinations']   # same across all runs
            # Train / test matrices: 10 selected × (nr_combinations*nr_trials) × nr_neurons
            X_train = [result['X_train'] for result in selected]
            print('x train ', len(X_train), X_train[0].shape, flush=True)
            X_test  = [result['X_test']  for result in selected]
            print('x test', len(X_test), X_test[0].shape, flush=True)

            # Flatten nested session keys into ordered lists
            session_ids = [key['session_id'] for animal_id in _session_keys for key in _session_keys[animal_id]]
            animal_ids  = [key['animal_id']  for animal_id in _session_keys for key in _session_keys[animal_id]]

            # ── Insert results ─────────────────────────────────────────────────
            entry = {
                'variable':               variable,
                'experimental_timepoint': phase,
                'window':                 window,
                'window_length':          window_length,
                'event_align':            event_align,
                'session_ids':            session_ids,
                'animal_ids':             animal_ids,
                'accuracy':               acc,
                'ccgp':                   ccgp,
                'decoder_vec':            decoder_vec,
                'neuron_ids_decode':      neuron_ids_decode,
                'combinations':           combinations,
                'x_train':                X_train,
                'x_test':                 X_test,
                }
            self.insert1(entry, skip_duplicates=True)

        print('TIME run : ', (time()-start__run))

    def decode_cohort(self, params):
        """
        Outer loop: run ``decoder_in_phase`` for the hard-coded
        post_choice window (1 s) across all phases.

        Unlike ``By_window.decode_cohort``, this method uses a single
        fixed window rather than iterating over params['windows'].

        Parameters
        ----------
        params : dict
            Experiment-wide hyperparameters.
        """
        print('Gathering sessions in experiment...', flush=True)

        # ── Unpack loop-level parameters ──────────────────────────────────────
        variables       = params['variables']
        min_nr_trials   = params['min_nr_trials_comb']
        nr_combinations = params['nr_combinations']

        # Hard-coded window definition for this table
        params['window']        = 'post_choice'
        params['window_length'] = 1
        params['event_align']   = 'choice'

        # ── Gather and validate session keys ──────────────────────────────────
        session_keys_dict = get_session_keys(params, params['event_align'])
        # print(session_keys_dict['discrimination']['BK4933_LR'])
        get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)
        # print(session_keys_dict['discrimination']['BK4933_LR'])

        # Compute per-phase neuron subsampling counts
        nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
            session_keys_dict,
            params['subsampling_fr'],
            params['nr_sessions_phase'],
            across_axis='phases',
        )
        # print(nr_neurons_session['discrimination']['BK4933_LR'])

        # Set pseudopopulation trial count (or None for single-session mode)
        if params['pseudopopulation']:
            params['n_datapoints'] = n_datapoints
        else:
            params['n_datapoints'] = None

        total_runs = np.sum([len(session_keys_dict[phase]) for phase in session_keys_dict])

        # ── Compute common time-axis bounds ────────────────────────────────────
        zero_left, left_range, right_range = get_common_time_axis_params(session_keys_dict)
        params['left_range']  = left_range
        params['right_range'] = right_range
        phases = params['phases']

        print('\n Processing all entries...', flush=True)
        pbar = tqdm(total=total_runs, position=0, leave=True)

        # ── Iterate over first 4 phases; skip existing DB entries ─────────────
        for phase in phases[:4]:
            print('\n phase: %s...' % phase, flush=True)
            params['zero_left']          = zero_left[phase]
            params['nr_neurons_session'] = nr_neurons_session[phase]
            # Check if entry already exists
            entries = self \
                & 'experimental_timepoint="%s"' % phase \
                & 'window="%s"'                 % params['window'] \
                & 'window_length="%s"'          % params['window_length']

            if len(entries) == 0:
                self.decoder_in_phase(session_keys_dict[phase], phase,
                                      params, True)
            pbar.update(1)


# ===========================================================================
# Joint all-variable decoder on a shared neuron set (Window_cohort_neuronset)
# ===========================================================================

@schema
class Window_cohort_neuronset(dj.Manual):
    definition = """ # geometry with same set of neuron ids for all variables of interest

    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    window        : varchar(128)   # window_of_interest - post stimulus, pre choice, post choice, post outcome
    window_length    : float   # window length in seconds
    ---
    variables           : blob  # variables (category, choice, outcome)
    event_align         : varchar(128)   # event of alignment - cero time reference
    session_ids      : blob       # list of session ids corresponding to animal ids used
    animal_ids       : blob       # list animal ids for pseudopopulation decoding
    accuracies          : longblob    # nr cross validations x nr variables
    ccgps               : longblob    # nr  cross validations x nr variables x nr cross variables
    decoder_vec   : longblob   # nr  cross validations, weight values for corresponding neuron ids x nr variables
    neuron_ids_decode   : longblob   # nr  cross validations, neuron ids used in each cross validation
    combinations   : longblob   # combination of cat, cho, out for trial types used in decoding
    x_train   : longblob   # decoder matrix fed for training trials x neurons, concatenated 4 trial types for each cross validation
    x_test    : longblob   # decoder matrix fed for testing, trials x neurons, concatenated 4 trial types for each cross validation
    """

    def decoder_in_phase(self, _session_keys, phase, params, populate=True):
        """
        Decode all variables jointly within a fixed window using the same
        neuron set for each cross-validation run, then insert into
        ``Window_cohort_neuronset``.

        All CV accuracy values are kept; only 10 randomly selected runs are
        stored for decoder vectors and train/test matrices.

        Note: ``neuron_ids_decode`` and ``combinations`` are identical across
        all CV runs and taken from results[0].

        Parameters
        ----------
        _session_keys : dict
            Nested {animal_id: [session_key_dicts]} for this phase.
        phase : str
            Experimental phase label.
        params : dict
            Decoding hyperparameters.
        populate : bool
            If True (default), insert results into the DB.
        """
        # print('\n Gathering session keys...',flush=True)
        start__run = time()

        # ── Extract window boundaries from the common time axis ───────────────
        window        = params['window']
        window_length = params['window_length']
        sessions_info = sessions_info_to_decode(_session_keys, params)
        # nr_sessions = len(session_ids)
        available_set_trials = sessions_info['available_set_trials']

        event_align    = params['event_align']
        variables      = params['variables']
        nr_cores       = params['nr_cores']
        timepoint_axis = sessions_info['timepoint_axis']
        # Frame indices corresponding to [t=0, t=window_length]
        window_idx = [np.where(timepoint_axis>0)[0][0],
                      np.where(timepoint_axis>window_length)[0][0]]

        # Pre-generate shared subsampling and trial-shuffle indices
        subsampling_idx = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
        available_sets  = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]

        # ── Run all-variable joint decoder in parallel (real labels only) ──────
        # null distribution in pool parallelization
        with Pool(nr_cores) as pool:
            # print('computing accuracy ...',flush=True)
            nr_cores = params['nr_cores']
            # Cross-validation: shuffle=False → real trial labels
            args    = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], False) for n in range(params['nr_crossvalidations'])]
            results = pool.starmap(window_decode_allvars, args)

        # ── Unpack results — keep all CV runs for accuracy ─────────────────────
        # pick 10 cross validation runs at random
        results    = np.array(results, dtype=object)
        # nr crossvalidations x nr vars
        accuracies = [result['acc']  for result in results]
        ccgps      = [result['ccgp'] for result in results]

        # ── Subsample 10 CV runs for storing decoder vectors + matrices ────────
        idx      = np.random.randint(0, len(results), 10)
        selected = results[idx]

        # nr selected x nr vars x nr neurons
        decoder_vec       = [result['decoder_vec'] for result in selected]
        neuron_ids_decode = results[0]['neuron_ids_decode']   # same across runs
        combinations      = results[0]['combinations']        # same across runs
        # X_train nr selected x nr combinations*nr trials x nr neurons
        X_train = [result['X_train'] for result in selected]
        # print('x train ',len(X_train),X_train[0].shape,flush=True)
        X_test  = [result['X_test']  for result in selected]
        # print('x test',len(X_test),X_test[0].shape,flush=True)

        # Flatten nested session keys into ordered lists
        session_ids = [key['session_id'] for animal_id in _session_keys for key in _session_keys[animal_id]]
        animal_ids  = [key['animal_id']  for animal_id in _session_keys for key in _session_keys[animal_id]]

        # ── Insert results ─────────────────────────────────────────────────────
        print('inserting into table')
        entry = {
            'experimental_timepoint': phase,
            'window':                 window,
            'window_length':          window_length,
            'variables':              variables,
            'event_align':            event_align,
            'session_ids':            session_ids,
            'animal_ids':             animal_ids,
            'accuracies':             accuracies,
            'ccgps':                  ccgps,
            'decoder_vec':            decoder_vec,
            'neuron_ids_decode':      neuron_ids_decode,
            'combinations':           combinations,
            'x_train':                X_train,
            'x_test':                 X_test,
            }
        self.insert1(entry, skip_duplicates=True)

        print('TIME run : ', (time()-start__run))
        print('times 12')

    def decode_cohort(self, params):
        """
        Outer loop: run ``decoder_in_phase`` for every combination of window
        and phase defined in ``params``.

        Parameters
        ----------
        params : dict
            Experiment-wide hyperparameters.
        """
        print('Gathering sessions in experiment...', flush=True)

        # ── Unpack loop-level parameters ──────────────────────────────────────
        variables       = params['variables']
        min_nr_trials   = params['min_nr_trials_comb']
        nr_combinations = params['nr_combinations']

        for i, window in enumerate(params['windows']):
            # Set window-specific params for this iteration
            params['window']        = window
            params['window_length'] = params['window_lengths'][i]
            params['event_align']   = params['event_alignments'][i]

            # ── Gather and validate session keys ──────────────────────────────
            session_keys_dict = get_session_keys(params, params['event_align'])
            # print(session_keys_dict['discrimination']['BK4933_LR'])
            get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)
            # print(session_keys_dict['discrimination']['BK4933_LR'])

            # Compute per-phase neuron subsampling counts
            nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
                session_keys_dict,
                params['subsampling_fr'],
                params['nr_sessions_phase'],
                across_axis='phases',
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

            # ── Iterate over all phases; skip existing DB entries ──────────────
            for phase in phases:
                print('\n phase: %s...' % phase, flush=True)
                params['zero_left']          = zero_left[phase]
                params['nr_neurons_session'] = nr_neurons_session[phase]
                # Check if entry already exists
                entries = self \
                    & 'experimental_timepoint="%s"' % phase \
                    & 'window="%s"'                 % params['window'] \
                    & 'window_length="%s"'          % params['window_length']

                if len(entries) == 0:
                    self.decoder_in_phase(session_keys_dict[phase], phase,
                                          params, True)
                pbar.update(1)


# ===========================================================================
# Joint all-variable decoder with averaged accuracies (Window_cohort_decspace) + geometry parameters (ccgp, ps)
# ===========================================================================

@schema
class Window_cohort_decspace(dj.Manual):
    definition = """ # geometry with same set of neuron ids for all variables of interest

    experimental_timepoint : varchar(256)    # experimental phase... discrimination,generalization, etc
    window        : varchar(128)   # window_of_interest - post stimulus, pre choice, post choice, post outcome
    window_length    : float   # window length in seconds
    ---
    variables           : blob  # variables (category, choice, outcome)
    event_align         : varchar(128)   # event of alignment - cero time reference
    session_ids      : blob       # list of session ids corresponding to animal ids used
    animal_ids       : blob       # list animal ids for pseudopopulation decoding
    accuracy         : blob    # decoding accuracy x nr variables
    ccgp             : blob    # cross accuracy nr variables x nr cross variables
    ps            : blob    # parallelism score nr variables x nr cross variables
    decoder_vec   : longblob   # nr  cross validations, weight values for corresponding neuron ids x nr variables
    combinations   : longblob   # combination of cat, cho, out for trial types used in decoding
    x_train   : longblob   # decoder matrix fed for training trials x neurons, concatenated 4 trial types for each cross validation
    x_test    : longblob   # decoder matrix fed for testing, trials x neurons, concatenated 4 trial types for each cross validation
    """
    # diferencia con cohort neuronset es que accuracy es la media de las validaciones en vez de conservar todos los valores en cada iteracion

    def decoder_in_phase(self, _session_keys, phase, params, populate=True):
        """
        Decode all variables jointly within a fixed window, averaging accuracy
        and geometry metrics across CV runs before inserting.

        Differs from ``Window_cohort_neuronset`` in that ``accuracy``, ``ccgp``,
        and ``ps`` are stored as mean values rather than all individual CV runs.
        20 randomly selected runs are stored for decoder vectors and matrices.
        DataJointError is caught and printed rather than raised, to allow
        partial population on blob-size failures.

        Parameters
        ----------
        _session_keys : dict
            Nested {animal_id: [session_key_dicts]} for this phase.
        phase : str
            Experimental phase label.
        params : dict
            Decoding hyperparameters.
        populate : bool
            If True (default), insert results into the DB.
        """
        # print('\n Gathering session keys...',flush=True)
        start__run = time()

        # ── Extract window boundaries from the common time axis ───────────────
        window        = params['window']
        window_length = params['window_length']
        sessions_info = sessions_info_to_decode(_session_keys, params)
        # nr_sessions = len(session_ids)
        available_set_trials = sessions_info['available_set_trials']

        event_align    = params['event_align']
        variables      = params['variables']
        nr_cores       = params['nr_cores']
        timepoint_axis = sessions_info['timepoint_axis']
        # Frame indices corresponding to [t=0, t=window_length]
        window_idx = [np.where(timepoint_axis>0)[0][0],
                      np.where(timepoint_axis>window_length)[0][0]]

        # Pre-generate shared subsampling and trial-shuffle indices
        subsampling_idx = [get_subsampling_idx(sessions_info, n_neurons=params['nr_neurons_session']) for n in range(params['nr_crossvalidations'])]
        available_sets  = [shuffle_available_set(available_set_trials) for n in range(params['nr_crossvalidations'])]

        # ── Run all-variable joint decoder in parallel (real labels only) ──────
        # null distribution in pool parallelization
        with Pool(nr_cores) as pool:
            # print('computing accuracy ...',flush=True)
            nr_cores = params['nr_cores']
            # Cross-validation: shuffle=False → real trial labels
            args    = [(sessions_info, params, window_idx, available_sets[n], subsampling_idx[n], False) for n in range(params['nr_crossvalidations'])]
            results = pool.starmap(window_decode_allvars, args)

        # ── Average accuracy metrics across all CV runs ────────────────────────
        # pick 10 cross validation runs at random
        results  = np.array(results, dtype=object)
        # Mean over CV runs: shape (nr_vars,)
        accuracy = np.mean([result['acc']  for result in results], axis=0)
        print('accuracy ', accuracy)
        ccgp = np.mean([result['ccgp'] for result in results], axis=0)
        print('ccgp ', ccgp)
        ps   = np.mean([result['ps']   for result in results], axis=0)
        print('ps ', ps)

        # ── Subsample 20 CV runs for storing decoder vectors + matrices ────────
        idx      = np.random.randint(0, len(results), 20)
        selected = results[idx]

        # nr selected x nr vars x nr neurons
        decoder_vec  = np.array([result['decoder_vec'] for result in selected])
        combinations = results[0]['combinations']   # same across all runs
        # X_train nr selected x nr combinations*nr trials x nr neurons
        X_train = np.array([result['X_train'] for result in selected])
        # print('x train ',len(X_train),X_train[0].shape,flush=True)
        X_test  = np.array([result['X_test']  for result in selected])
        # print('x test',len(X_test),X_test[0].shape,flush=True)

        # Flatten nested session keys into ordered lists
        session_ids = [key['session_id'] for animal_id in _session_keys for key in _session_keys[animal_id]]
        animal_ids  = [key['animal_id']  for animal_id in _session_keys for key in _session_keys[animal_id]]

        # ── Insert results (catch DataJoint errors without aborting the run) ───
        print('inserting into table')
        entry = {
            'experimental_timepoint': phase,
            'window':                 window,
            'window_length':          window_length,
            'variables':              variables,
            'event_align':            event_align,
            'session_ids':            session_ids,
            'animal_ids':             animal_ids,
            'accuracy':               accuracy,
            'ccgp':                   ccgp,
            'ps':                     ps,
            'decoder_vec':            decoder_vec,
            'combinations':           combinations,
            'x_train':                X_train,
            'x_test':                 X_test,
            }
        try:
            self.insert1(entry, skip_duplicates=True)
        except dj.DataJointError as e:
            # Log the error but continue processing remaining phases
            print('error ', e)

        print('TIME run : ', (time()-start__run))
        print('times 12')

    def decode_cohort(self, params):
        """
        Outer loop: run ``decoder_in_phase`` for every combination of window
        and phase defined in ``params`` (first 4 phases only).

        Parameters
        ----------
        params : dict
            Experiment-wide hyperparameters.
        """
        print('Gathering sessions in experiment...', flush=True)

        # ── Unpack loop-level parameters ──────────────────────────────────────
        variables       = params['variables']
        min_nr_trials   = params['min_nr_trials_comb']
        nr_combinations = params['nr_combinations']

        for i, window in enumerate(params['windows']):
            # Set window-specific params for this iteration
            params['window']        = window
            params['window_length'] = params['window_lengths'][i]
            params['event_align']   = params['event_alignments'][i]

            # ── Gather and validate session keys ──────────────────────────────
            session_keys_dict = get_session_keys(params, params['event_align'])
            # print(session_keys_dict['discrimination']['BK4933_LR'])
            get_valid_sessions(session_keys_dict, variables, min_nr_trials, nr_combinations)
            # print(session_keys_dict['discrimination']['BK4933_LR'])

            # Compute per-phase neuron subsampling counts
            nr_neurons_session, n_datapoints = get_nr_subsample_neurons(
                session_keys_dict,
                params['subsampling_fr'],
                params['nr_sessions_phase'],
                across_axis='phases',
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

            # ── Iterate over first 4 phases; skip existing DB entries ──────────
            for phase in phases[:4]:
                print('\n phase: %s...' % phase, flush=True)
                params['zero_left']          = zero_left[phase]
                params['nr_neurons_session'] = nr_neurons_session[phase]
                # Check if entry already exists
                entries = self \
                    & 'experimental_timepoint="%s"' % phase \
                    & 'window="%s"'                 % params['window'] \
                    & 'window_length="%s"'          % params['window_length']

                if len(entries) == 0:
                    self.decoder_in_phase(session_keys_dict[phase], phase,
                                          params, True)
                pbar.update(1)


# ===========================================================================
# Experiment-wide hyperparameters
# ===========================================================================

params = {}

# ── Experimental phases to decode ─────────────────────────────────────────────
params['phases']             = ['gentest_2', 'gentest_1', 'discrimination', 'categorization_4']

# ── Time-axis binning ─────────────────────────────────────────────────────────
params['bin_window']         = 5    # width of each time bin (frames)
params['overlap_window']     = 1    # step size between consecutive bins (frames)

# ── Session / neuron subsampling ──────────────────────────────────────────────
params['nr_sessions_phase']  = 2    # minimum sessions required per phase
params['subsampling_fr']     = 0.9  # fraction of neurons to subsample per session

# ── Decoder training / validation ─────────────────────────────────────────────
params['training_fr']        = 0.7  # fraction of trials used for training
params['nr_crossvalidations'] = 100  # number of cross-validation folds

# ── Variables to decode ───────────────────────────────────────────────────────
params['variables']          = ['category', 'choice', 'outcome']

# ── Null distribution ─────────────────────────────────────────────────────────
params['nr_iter']            = 500  # shuffle iterations for null model

# ── Trial filtering ───────────────────────────────────────────────────────────
params['min_nr_trials_comb'] = 2    # minimum trials per condition combination
params['nr_combinations']    = 4    # number of trial condition combinations

# ── Parallelisation ───────────────────────────────────────────────────────────
params['nr_cores']           = 4    # worker processes (run `nproc` in terminal to check)

# ── Pseudopopulation mode ─────────────────────────────────────────────────────
params['pseudopopulation']   = True  # pool neurons across sessions into a pseudopopulation

# ── Window definitions (one entry per window; lists must be same length) ──────
params['event_alignments']   = ['choice', 'choice', 'choice']   # alignment event per window
params['window_lengths']     = [0.7, 1, 1.5]                    # window duration in seconds
params['windows']            = ['post_choice', 'post_choice', 'post_choice']  # window label


