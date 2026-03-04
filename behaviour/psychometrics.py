#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba

# =============================================================================
# psychometrics.py
# Defines the Psychometric_curves DataJoint table and helpers for fitting
# psychometric curves to 1-D tone categorisation behavioural data.
# =============================================================================
"""
import numpy as np
import os, sys, inspect
import datajoint as dj
from sklearn.metrics import mean_squared_error

print("Calling psychometrics.py from module script: ", __name__)

# ---------------------------------------------------------------------------
# Conditional imports – module behaves differently when run as a script vs
# imported from a parent package.
# ---------------------------------------------------------------------------
if __name__ == '__main__' or __name__ == 'psychometrics':
    from training import get_stimuli_space
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir  = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utilities import mle_fit_psycho, erf_psycho, trainingpoint_dict, get_property
    from data_import import Session, Trial
else:
    print('not implemented')

# ---------------------------------------------------------------------------
# DataJoint configuration
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema = dj.schema('psychometrics_hpc_cat_2025', locals(), create_tables=True)


# =============================================================================
# Stimulus preprocessing
# =============================================================================

def scale_transform_stimulus(experiment, stimuli):
    """
    Map raw stimulus values onto a normalised log scale.

    Steps:
      1. Retrieve the full stimulus space for *experiment*.
      2. Log-transform and z-score the space (mean=0, std=1).
      3. Return both the full normalised space and the values corresponding
         to the requested *stimuli*.

    Parameters
    ----------
    experiment : str
        Experiment identifier used to look up the stimulus space.
    stimuli : array-like of int
        Raw stimulus values to be scaled.

    Returns
    -------
    norm_log_space : np.ndarray
        Full normalised log-transformed stimulus space.
    scaled_stimuli : np.ndarray
        Normalised values for the requested *stimuli*.
    """
    stimuli_space = get_stimuli_space(experiment)

    # Find the index of each requested stimulus in the full space
    idx = [np.argwhere(stimuli_space == int(s))[0][0] for s in stimuli]

    # Log-transform then z-score the full space
    log_space      = np.log(stimuli_space)
    norm_log_space = (log_space - np.mean(log_space)) / np.std(log_space)

    return norm_log_space, norm_log_space[idx]


# =============================================================================
# Goodness-of-fit helper
# =============================================================================

def compute_rmse(pars, x, y_true):
    """
    Compute the root-mean-squared error between the psychometric model
    prediction and the observed proportion at a single stimulus value.

    Parameters
    ----------
    pars : array-like
        Fitted parameters for `erf_psycho`.
    x : float
        Normalised stimulus value.
    y_true : float
        Observed proportion of 'high' choices for this stimulus.

    Returns
    -------
    float
        RMSE between model prediction and observation.
    """
    y_pred = erf_psycho(pars, x)
    return mean_squared_error([y_true], [y_pred], squared=False)


# =============================================================================
# Data assembly
# =============================================================================

def get_psy_data(experiment, stimuli_list, trials, port_layout):
    """
    Build the (3 × n_stimuli) data matrix required by `mle_fit_psycho`.

    Rows:
      0 – normalised log stimulus values (vv)
      1 – trial counts per stimulus     (nn)
      2 – proportion of 'high' choices  (pp)

    Parameters
    ----------
    experiment : str
        Experiment identifier (passed to `scale_transform_stimulus`).
    stimuli_list : list of str
        Unique stimulus IDs presented in the session. The '0' catch stimulus
        is removed before processing.
    trials : DataJoint table expression
        Filtered trial rows for the session of interest.
    port_layout : int / bool
        If falsy, the high-frequency category is assigned to port 1 (right);
        if truthy, it is assigned to port 0 (left).

    Returns
    -------
    stimuli : list of int
        Sorted raw stimulus values (excluding '0').
    data : np.ndarray, shape (3, n_stimuli)
        Stacked [vv, nn, pp] matrix.
    """
    # Remove the catch / no-stimulus condition
    if '0' in stimuli_list:
        stimuli_list.remove('0')

    stimuli = sorted(int(s) for s in stimuli_list)
    _, vv   = scale_transform_stimulus(experiment, stimuli)

    # Port assigned to the high-frequency tone category
    high_port = 0 if port_layout else 1

    nn, pp = [], []
    for stimulus in stimuli:
        trial_set = trials & f'stimulus_id="{stimulus}"'
        count     = len(trial_set)
        nn.append(count)
        pp.append(round(len(trial_set & f'response="{high_port}"') / count, 1))

    return stimuli, np.vstack((vv, nn, pp))


# =============================================================================
# Psychometric_curves – Manual DataJoint table
# =============================================================================

@schema
class Psychometric_curves(dj.Manual):
    definition = """ # psychometric curves for generalisation tests in 1-D tone categorisation task

    animal_id                  : varchar(128)   # Mouse unique identifier
    experimental_timepoint     : varchar(256)   # Experimental stage (e.g. discrimination, generalisation)
    ---
    session_ids     : blob       # Session IDs used to compute the curve
    parameters      : tinyblob  # Fitted parameter vector
    data            : blob       # (3 x n) matrix: stimulus values, trial counts, high-choice proportions
    likelihood      : float      # Log-likelihood of the fit
    error_stimulus  : blob       # Per-stimulus RMSE
    stimuli         : blob       # Raw stimulus values included in the fit
    """

    def psychometric_curves_subject(self, animal_id, session_id, experimental_timepoint):
        """
        Fit and store a psychometric curve for one subject / session.

        Fetches trials and port layout for *session_id*, assembles the data
        matrix, fits an erf_psycho model via MLE, computes per-stimulus RMSE,
        and inserts the result into the table.

        Parameters
        ----------
        animal_id : str
            Mouse identifier.
        session_id : int
            Session to process.
        experimental_timepoint : str
            Label for the experimental phase (e.g. 'gentest_1').
        """
        # Fetch trials and port layout for this session
        trials      = Trial()   & f'animal_id="{animal_id}"' & f'session_id="{session_id}"'
        port_layout = (Session() & f'animal_id="{animal_id}"' & f'session_id="{session_id}"') \
                        .fetch('port_layout')[0]

        # Collect unique stimulus IDs presented in this session
        stimuli_list = list(dict.fromkeys(trials.fetch('stimulus_id')))

        # Build (3 x n_stimuli) data matrix
        stimuli, data = get_psy_data(animal_id, stimuli_list, trials, port_layout)
        assert data.shape[0] == 3,                    "Data must have exactly 3 rows (vv, nn, pp)."
        assert data.shape[1] == len(stimuli_list) - ('0' in stimuli_list), \
                                                       "Column count must match number of valid stimuli."
        assert len(stimuli) > 2,                      "At least 3 stimuli are required to fit a curve."

        # Maximum-likelihood fit
        pars, L = mle_fit_psycho(data, 'erf_psycho')

        # Per-stimulus RMSE
        error = [compute_rmse(pars, data[0][i], data[2][i]) for i in range(data.shape[1])]

        self.insert1(dict(
            animal_id              = animal_id,
            experimental_timepoint = experimental_timepoint,
            session_ids            = [session_id],
            parameters             = pars,
            data                   = data,
            likelihood             = L,
            error_stimulus         = error,
            stimuli                = stimuli,
        ), skip_duplicates=True)

    def psychometric_gt_curves(self):
        """
        Fit psychometric curves for all subjects across generalisation test phases.

        Iterates over 'gentest_1' and 'gentest_2', finds the most recent
        session in each phase for every subject, and calls
        `psychometric_curves_subject` for each.
        """
        phases = ['gentest_1', 'gentest_2']

        for phase in phases:
            mode, stage, _ = trainingpoint_dict[phase]
            sessions  = Session() & f'mode="{mode}"' & f'stage="{stage}"'
            animal_ids = get_property(sessions, 'animal_id')

            for animal_id in animal_ids:
                # Use the most recent session for this phase
                session_id = (sessions & f'animal_id="{animal_id}"').fetch('session_id')[-1]
                print(f'\n phase      : {phase}')
                print(f'animal_id  : {animal_id}')
                print(f'session_id : {session_id}')
                self.psychometric_curves_subject(animal_id, session_id, phase)


# =============================================================================
# Module-level analysis helper
# =============================================================================

def goodness_fit():
    """
    Summarise log-likelihood across generalisation test phases for all subjects.

    Returns a (n_subjects × 2) array where columns correspond to 'gentest_1'
    and 'gentest_2' and values are mean log-likelihoods per subject.

    Returns
    -------
    data : np.ndarray, shape (n_subjects, 2)
        Mean log-likelihood for each subject × phase combination.
    """
    phases     = ['gentest_1', 'gentest_2']
    fits       = Psychometric_curves() & 'experimental_timepoint="gentest_2"'
    animal_ids = get_property(fits, 'animal_id')

    data = np.zeros((len(animal_ids), len(phases)))

    for i, animal_id in enumerate(animal_ids):
        fits_subject = Psychometric_curves() & f'animal_id="{animal_id}"'
        for j, phase in enumerate(phases):
            likelihoods  = (fits_subject & f'experimental_timepoint="{phase}"').fetch('likelihood')
            data[i, j]   = np.mean(likelihoods)

    return data



    