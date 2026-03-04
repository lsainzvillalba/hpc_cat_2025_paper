#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba

# =============================================================================
# helper_logreg.py
# Logistic regression helpers for behavioural data: regressor construction,
# design-matrix building, model fitting, and bootstrap significance testing.
#
# Public API:
#   get_basic_regressors    – fetch raw trial variables from DataJoint
#   get_regressors          – build the full aligned regressor dictionary
#   get_X_matrix            – assemble a Patsy design matrix
#   format_regressor_columns – clean Patsy column names for display
#   logistic_regression     – fit model and compute bootstrap p-values
#   get_deltastim           – log-scale stimulus change between trials
#   get_actions             – stay/shift action sequence from choices
# =============================================================================
"""

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats, linalg
from patsy import dmatrix

print("Calling helper_logreg.py from module script: ", __name__)

# ---------------------------------------------------------------------------
# Patsy formula strings for each predicted variable
# ---------------------------------------------------------------------------
_DESIGN_CHOICE = (
    'stimuli + C(previous_choices) + C(previous_outcomes) '
    '+ C(previous_outcomes)*C(previous_choices)'
)


# Maps each predicted variable to its regressor list and Patsy formula
predicted_variable_dict = {
    'choices': {
        'regressors':    ['stimuli', 'previous_outcomes', 'previous_choices'],
        'design_matrix': _DESIGN_CHOICE,
    }
}

# Stimulus IDs used as category prototypes (boundary anchors on the log scale)
_INIT_A = 4652
_INIT_B = 9358

# Number of previous trials used as history regressors
_N_BACK = 1

# Number of bootstrap resamples for p-value estimation
_N_BOOTSTRAP = 1000


# =============================================================================
# Raw regressor extraction
# =============================================================================

def get_basic_regressors(session_trials, port_layout):
    """
    Fetch and align the fundamental trial variables for one session.

    Applies port-layout correction when the category–choice assignment is
    crossed (i.e. left port → category B rather than the default A).

    Parameters
    ----------
    session_trials : DataJoint table expression
        Committed, active trial rows for one session.
    port_layout : int or bool
        If truthy, flip binary choice and category values.

    Returns
    -------
    stimuli : np.ndarray of str
        Stimulus IDs per trial.
    choices : np.ndarray of int
        Binary choice per trial (0 = left, 1 = right; corrected for layout).
    rewards : np.ndarray of float
        1 for correct trials, 0 otherwise (control trials forced to 0).
    errors : np.ndarray of float
        1 for incorrect trials, 0 otherwise (control trials forced to 0).
    category : np.ndarray of int
        Binary category label per trial (corrected for layout).
    """
    choices       = session_trials.fetch('response')
    category      = session_trials.fetch('baited_port')
    responsetypes = session_trials.fetch('responsetype')
    trialtypes    = session_trials.fetch('trialtype')
    stimuli       = session_trials.fetch('stimulus_id')

    if port_layout:
        # Crossed assignment: flip both choice and category labels
        choices  = np.array([1 if c == 0 else 0 for c in choices])
        category = np.array([1 if c == 0 else 0 for c in category])

    # Reward vector: 1 = correct, 0 = incorrect or control
    rewards = np.zeros(len(responsetypes))
    rewards[responsetypes == 'correct'] = 1
    rewards[trialtypes == 'control_nostimulus'] = 0

    # Error vector: 1 = incorrect, 0 = correct or control
    errors = np.zeros(len(responsetypes))
    errors[responsetypes == 'incorrect'] = 1
    errors[trialtypes == 'control_nostimulus'] = 0

    return stimuli, choices, rewards, errors, category


# =============================================================================
# Regressor construction
# =============================================================================

def get_deltastim(stimuli, space, norm_log_space):
    """
    Compute the absolute log-scale stimulus change between consecutive trials.

    Returns None for transitions involving a catch stimulus (ID == '0').

    Parameters
    ----------
    stimuli : array-like of str
        Stimulus IDs in trial order.
    space : list of int
        Ordered stimulus space (integer IDs).
    norm_log_space : list of float
        Normalised log-transformed values aligned with *space*.

    Returns
    -------
    np.ndarray
        Delta-stimulus values of length ``len(stimuli) - 1``; entries are
        None where either the current or previous trial was a catch trial.
    """
    deltastim = []
    for i, st in enumerate(stimuli[1:]):
        if st == '0' or stimuli[i] == '0':
            deltastim.append(None)
        else:
            idx_current = space.index(int(st))
            idx_past    = space.index(int(stimuli[i]))
            deltastim.append(
                abs(norm_log_space[idx_current] - norm_log_space[idx_past])
            )
    return np.array(deltastim)


def get_actions(stimuli, choices):
    """
    Convert a choice sequence into a stay (0) / shift (1) action sequence.

    Returns None for transitions involving a catch stimulus (ID == '0').

    Parameters
    ----------
    stimuli : array-like of str
        Stimulus IDs in trial order.
    choices : array-like of int
        Binary choice per trial.

    Returns
    -------
    np.ndarray
        Action values of length ``len(choices) - 1``; entries are None where
        either the current or previous trial was a catch trial.
    """
    actions = []
    for i, c in enumerate(choices[1:]):
        if stimuli[i + 1] == '0' or stimuli[i] == '0':
            actions.append(None)
        else:
            # 0 = stay (same choice as previous), 1 = shift
            actions.append(0 if c == choices[i] else 1)
    return np.array(actions)


def get_regressors(stimuli, choices, rewards, errors, category, stimuli_space):
    """
    Build the full aligned regressor dictionary for logistic regression.

    Constructs history regressors (previous outcomes and choices via Toeplitz),
    delta-stimulus and action sequences, and normalised log-scale stimulus
    values. All arrays are trimmed to the same length and catch trials are
    removed.

    Parameters
    ----------
    stimuli : array-like of str
        Stimulus IDs per trial.
    choices : array-like of int
        Binary choice per trial.
    rewards : array-like of float
        Binary reward per trial.
    errors : array-like of float
        Binary error per trial.
    category : array-like of int
        Binary category label per trial.
    stimuli_space : array-like of int
        Full ordered stimulus space.

    Returns
    -------
    dict
        Keys: 'stimuli', 'dist_init_A', 'dist_init_B', 'choices',
              'previous_choices', 'previous_outcomes', 'actions', 'errors',
              'delta_stimuli', 'category'.
        All values are np.ndarrays of equal length.
    """
    # Normalise the stimulus space to log scale (z-score)
    stimuli_space    = list(stimuli_space)
    log_space        = np.log(stimuli_space)
    norm_log_space   = list((log_space - np.mean(log_space)) / np.std(log_space))

    # Distance from each trial's stimulus to the two prototype stimuli
    init_ids  = [_INIT_A, _INIT_B]
    init_log  = [
        np.abs(np.array(norm_log_space)) - norm_log_space[stimuli_space.index(i)]
        for i in init_ids
    ]

    # Build history regressors using Toeplitz structure
    previous_outcomes = np.ravel(
        linalg.toeplitz(rewards,  np.zeros((1, _N_BACK)))[_N_BACK - 1:-1]
    )
    previous_choices  = np.ravel(
        linalg.toeplitz(choices,  np.zeros((1, _N_BACK)))[_N_BACK - 1:-1]
    )

    # Compute trial-to-trial delta and action sequences (length = n_trials - 1)
    delta_stimuli = get_deltastim(stimuli, stimuli_space, norm_log_space)[_N_BACK - 1:]
    actions       = get_actions(stimuli, choices)[_N_BACK - 1:]

    # Trim all arrays to the same length (drop the first _N_BACK trials)
    stimuli  = stimuli[_N_BACK:]
    choices  = choices[_N_BACK:]
    errors   = errors[_N_BACK:]
    category = category[_N_BACK:]

    # Remove catch trials (actions == None)
    valid            = actions != None   # noqa: E711 – intentional None comparison
    previous_outcomes = previous_outcomes[valid]
    previous_choices  = previous_choices[valid]
    choices           = choices[valid]
    stimuli           = stimuli[valid]
    errors            = errors[valid]
    delta_stimuli     = delta_stimuli[valid]
    category          = category[valid]
    actions           = actions[valid]

    # Convert stimulus IDs to normalised log-scale values and prototype distances
    stimuli_int = np.array([int(st) for st in stimuli])
    dist_init_A = np.array([init_log[0][stimuli_space.index(st)] for st in stimuli_int])
    dist_init_B = np.array([init_log[1][stimuli_space.index(st)] for st in stimuli_int])
    stimuli_norm = np.array([norm_log_space[stimuli_space.index(st)] for st in stimuli_int])

    # Sanity checks: all regressors must be the same length
    assert len(choices) == len(stimuli_norm)
    assert len(stimuli_norm) == len(actions)
    assert len(actions) == len(delta_stimuli)
    assert len(delta_stimuli) == len(errors)
    assert len(errors) == len(previous_choices)
    assert len(previous_choices) == len(previous_outcomes)
    assert len(previous_choices) == len(dist_init_A)
    assert len(dist_init_A) == len(dist_init_B)

    return {
        'stimuli':           stimuli_norm,
        'dist_init_A':       dist_init_A,
        'dist_init_B':       dist_init_B,
        'choices':           choices,
        'previous_choices':  previous_choices,
        'previous_outcomes': previous_outcomes,
        'actions':           actions,
        'errors':            errors,
        'delta_stimuli':     delta_stimuli,
        'category':          category,
    }


# =============================================================================
# Design-matrix helpers
# =============================================================================

def get_X_matrix(regressor_dict, regressors, design_matrix):
    """
    Assemble a Patsy design matrix from selected regressors.

    Patsy adds an intercept column by default; it is dropped here because
    sklearn's LogisticRegression adds its own intercept.

    Parameters
    ----------
    regressor_dict : dict
        Full regressor dictionary from `get_regressors`.
    regressors : list of str
        Keys in *regressor_dict* to include in the design matrix.
    design_matrix : str
        Patsy formula string (right-hand side only).

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_regressors)
        Design matrix with the intercept column removed.
    columns : tuple of str
        Column names from Patsy (including the dropped intercept at index 0).
    """
    data = {r: list(regressor_dict[r]) for r in regressors}
    X    = dmatrix(design_matrix, data)

    columns = X.design_info.column_names
    X = X[:, 1:]   # drop Patsy's default intercept column (index 0)
    return X, columns


def format_regressor_columns(columns):
    """
    Convert Patsy-generated column names to readable short labels.

    Patsy uses notation like ``C(previous_choices)[T.1]`` for categorical
    terms and ``C(a):C(b)`` for interactions. This function strips brackets
    and parentheses and joins interaction terms with ' x '.

    Parameters
    ----------
    columns : tuple or list of str
        Raw Patsy column names (as returned by `get_X_matrix`).

    Returns
    -------
    list of str
        Human-readable regressor labels.
    """
    formatted = []
    for i, col in enumerate(columns):
        if i == 0:
            # First column is the intercept – keep as-is
            formatted.append(col)
            continue

        parts = col.split(':')
        if len(parts) > 1:
            # Interaction term: e.g. 'C(a)[T.1]:C(b)[T.1]' → 'a x b'
            name = ''
            for j, term in enumerate(parts):
                clean = term.split('[')[0].split('(')
                name += clean[-1].replace(')', '')
                if j < len(parts) - 1:
                    name += ' x '
        else:
            # Simple term: e.g. 'C(previous_choices)[T.1]' → 'previous_choices'
            clean = parts[0].split('[')[0].split('(')
            name  = clean[-1].replace(')', '')

        formatted.append(name)

    return formatted


# =============================================================================
# Model fitting and significance testing
# =============================================================================

def _bootstrap_fit(df):
    """
    Fit a logistic regression on one bootstrap resample of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'y' column (labels) and feature columns.

    Returns
    -------
    list of float or None
        [intercept, coef_1, …, coef_k], or None if the fit fails.
    """
    sample = df.sample(df.shape[0], replace=True)
    X_bs   = sample[[c for c in sample.columns if c != 'y']].values
    y_bs   = sample['y'].values

    model = LogisticRegression(penalty='l2', solver='liblinear')
    try:
        model.fit(X_bs, y_bs)
    except ValueError:
        return None

    return [model.intercept_[0]] + list(model.coef_[0])


def logistic_regression(X, y):
    """
    Fit a logistic regression and estimate coefficient significance via
    bootstrap resampling (Wald-style test on bootstrap standard errors).

    Fits the model on the full data, then runs `_N_BOOTSTRAP` bootstrap
    resamples to estimate the standard error of each coefficient. Uses a
    t-distribution with ``n - p - 1`` degrees of freedom to compute p-values.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_features)
        Feature matrix (no intercept column; model adds it internally).
    y : array-like, shape (n_trials,)
        Binary trial labels.

    Returns
    -------
    pd.DataFrame or None
        Columns: 'coef', 'z', '.025', '.975', 'df', 'pvalue' for each
        coefficient (intercept first). Returns None if the fit fails or if
        any bootstrap resample fails to converge.
    """
    model = LogisticRegression(penalty='l2', solver='liblinear')
    try:
        model.fit(X, y)
    except ValueError:
        return None

    coef = [model.intercept_[0]] + list(model.coef_[0])

    # Bootstrap resampling to estimate standard errors
    df_boot    = pd.DataFrame(X)
    df_boot['y'] = y
    boot_fits  = [_bootstrap_fit(df_boot) for _ in range(_N_BOOTSTRAP)]

    if None in boot_fits:
        return None

    se  = pd.DataFrame(boot_fits).std()
    dof = X.shape[0] - X.shape[1] - 1

    results = pd.DataFrame({
        'coef':  coef,
        'z':     coef / se,
        '.025':  coef - se,
        '.975':  coef + se,
        'df':    dof,
    })

    # One-sided p-value from the t-distribution (upper tail)
    results['pvalue'] = stats.t.sf(np.abs(results['z']), df=results['df'])

    return results
