#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:54:29 2023

@author: Laura Sainz Villalba

# =============================================================================
# logreg.py
# Defines the Logreg_coefs DataJoint table and helpers for fitting logistic
# regression models to behavioural trial data across training phases.
# =============================================================================
"""
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import os, sys, inspect
import datajoint as dj
import numpy as np
from time import time
from tqdm import tqdm

print("Calling logreg.py from module script: ", __name__)

# ---------------------------------------------------------------------------
# Conditional imports – supports three execution contexts:
#   1. Run directly as a script (__main__)
#   2. Imported as a top-level module ('logreg')
#   3. Imported as part of the 'behaviour' sub-package ('behaviour.logreg')
# ---------------------------------------------------------------------------
if __name__ == '__main__' or __name__ == 'logreg':
    from training import get_stimuli_space
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir  = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utilities import (predicted_variable_dict, get_regressors,
                           get_X_matrix, logistic_regression,
                           format_regressor_columns, get_basic_regressors,
                           get_property, trainingpoint_dict)
    from data_import import Session, Trial

elif __name__ == 'behaviour.logreg':
    from .training import get_stimuli_space
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir  = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utilities import (predicted_variable_dict, get_regressors,
                           get_X_matrix, logistic_regression,
                           format_regressor_columns, get_basic_regressors,
                           get_property, trainingpoint_dict)
    from data_import import Session, Trial

# ---------------------------------------------------------------------------
# DataJoint configuration
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema = dj.schema('analysis_logreg_hpc_cat_2025', locals(), create_tables=True)

HOME_DIRECTORY = '/home/lsainz/Doctorado/DoctoradoDatos/experimentdata/'

# Boolean flags (one per training timepoint in trainingpoint_dict order)
# indicating whether a 'trend' analysis applies to each phase (1) or not (0).
TREND_ANALYSIS = [1, 0, 1, 1, 0, 0]


# =============================================================================
# Module-level helpers
# =============================================================================

def get_performance(trials):
    """
    Return percent correct on active trials (0 if no active trials exist).

    Parameters
    ----------
    trials : DataJoint table expression
        Trial rows to evaluate.

    Returns
    -------
    float
        Percent correct in [0, 100].
    """
    active  = trials & 'trialtype="active"'
    correct = active  & 'responsetype="correct"'
    if len(active) == 0:
        return 0
    return round((len(correct) / len(active)) * 100, 2)


def log_regression(regressor_dict, predicted_var):
    """
    Fit a logistic regression model for *predicted_var* and return results.

    Retrieves the appropriate regressors and design-matrix spec from
    `predicted_variable_dict`, builds the feature matrix, fits the model,
    and applies variable-specific post-processing to the output columns.

    Parameters
    ----------
    regressor_dict : dict
        Dictionary mapping regressor names to value arrays.
    predicted_var : str
        Target variable key ('choices' or 'errors').

    Returns
    -------
    columns : list of str or None
        Regressor names after formatting (None if fit failed).
    coef : list of float or None
        Fitted coefficients (None if fit failed).
    pvalue : list of float or None
        Coefficient p-values (None if fit failed).
    """
    regressors    = predicted_variable_dict[predicted_var]['regressors']
    design_matrix = predicted_variable_dict[predicted_var]['design_matrix']

    X, cols = get_X_matrix(regressor_dict, regressors, design_matrix)
    y       = list(regressor_dict[predicted_var])
    results = logistic_regression(X, y)

    if results is None:
        return None, None, None

    coef    = results['coef']
    pvalue  = results['pvalue']
    columns = format_regressor_columns(cols)

    # Variable-specific column/coefficient handling
    if predicted_var == 'choices':
        columns[0] = 'bias'          # rename intercept to 'bias'
    elif predicted_var == 'errors':
        columns = columns[1:]        # drop intercept for error model
        coef    = coef[1:]
        pvalue  = pvalue[1:]

    return columns, coef.tolist(), pvalue.tolist()


# =============================================================================
# Logreg_coefs – Manual DataJoint table
# =============================================================================

@schema
class Logreg_coefs(dj.Manual):
    definition = """ # logistic regression coefficients across training timepoints

    animal_id          : varchar(128)   # Mouse unique identifier
    predicted_var      : varchar(128)   # Predicted variable (choices, errors)
    training_timepoint : varchar(256)   # Training phase (discrimination, generalisation, etc.)
    analysis_type      : varchar(128)   # 'trend' (all dates) or 'point' (best-performance date)
    ---
    coef_names    : blob   # Regressor names
    coef_values   : blob   # Fitted logistic regression coefficients
    coef_pvalues  : blob   # Coefficient p-values
    dates         : blob   # Session date(s) used for this entry
    """

    # ------------------------------------------------------------------
    # Internal population helpers
    # ------------------------------------------------------------------

    def _fit_sessions(self, sessions, trials, port_layout, stimuli_space):
        """
        Fit logistic regression for every session date and return per-date results.

        Iterates over sorted session dates, computes performance and per-session
        regressors, and collects columns, coefficients, and p-values for both
        predicted variables ('choices' and 'errors').

        Parameters
        ----------
        sessions : DataJoint table expression
            Session rows to process.
        trials : DataJoint table expression
            Trial rows matching the sessions.
        port_layout : int
            Port layout flag (passed to `get_basic_regressors`).
        stimuli_space : np.ndarray
            Full stimulus space for the experiment.

        Returns
        -------
        str_dates : list of str
            Sorted session dates as strings.
        fr_correct : np.ndarray
            Percent correct per date.
        columns : list
            Per-date, per-predicted-var regressor name lists.
        coefs : list
            Per-date, per-predicted-var coefficient lists.
        pvalues : list
            Per-date, per-predicted-var p-value lists.
        """
        predicted_vars = ['choices', 'errors']
        dates = sorted(get_property(sessions, 'date'))
        nr_dates   = len(dates)
        str_dates  = []
        fr_correct = np.zeros(nr_dates)
        columns, coefs, pvalues = [], [], []

        for i in tqdm(range(nr_dates)):
            d          = str(dates[i])
            str_dates.append(d)
            session_id = (sessions & f'date="{d}"').fetch('session_id')[0]

            # Restrict to responding trials only
            session_trials = trials & f'session_id="{session_id}"' & 'response!="-1"'
            fr_correct[i]  = get_performance(session_trials)

            stimuli, choices, rewards, errors, category = get_basic_regressors(
                session_trials, port_layout
            )

            cols_day, coefs_day, pvals_day = [], [], []
            for predicted_var in predicted_vars:
                print(f'  predicted var: {predicted_var}')
                regressor_dict = get_regressors(
                    stimuli, choices, rewards, errors, category, stimuli_space
                )
                cols, coef, pvalue = log_regression(regressor_dict, predicted_var)
                cols_day.append(cols)
                coefs_day.append(coef)
                pvals_day.append(pvalue)

            columns.append(cols_day)
            coefs.append(coefs_day)
            pvalues.append(pvals_day)

        return str_dates, fr_correct, columns, coefs, pvalues

    def _insert_trend(self, animal_id, train_point, predicted_vars,
                      columns, coefs, pvalues, str_dates):
        """
        Insert 'trend' entries (all dates) for each predicted variable.

        Filters out dates where the regression failed (None columns), then
        inserts one row per predicted variable into the table.

        Parameters
        ----------
        animal_id : str
            Mouse identifier.
        train_point : str
            Current training timepoint key.
        predicted_vars : list of str
            Variables to store ('choices', 'errors').
        columns, coefs, pvalues : np.ndarray
            Arrays shaped (nr_dates, nr_predicted_vars, ...).
        str_dates : np.ndarray of str
            Session date strings aligned with the first array axis.
        """
        # Normalise training timepoint label for categorisation phases
        training_timepoint = (
            'categorization' if train_point.startswith('categ') else train_point
        )

        for k, predicted_var in enumerate(predicted_vars):
            c_names = columns[:, k]
            valid   = c_names != None                   # noqa: E711 – intentional None comparison

            c_names_valid      = c_names[valid]
            coef_values_valid  = coefs[:, k][valid]
            coef_pvalues_valid = pvalues[:, k][valid]
            ds                 = str_dates[valid]

            if len(c_names_valid) == 0:
                continue  # no valid fits for this variable – skip

            entry = dict(
                animal_id          = animal_id,
                predicted_var      = predicted_var,
                training_timepoint = training_timepoint,
                analysis_type      = 'trend',
                coef_names         = c_names_valid[0],      # names are identical across dates
                coef_values        = np.array(coef_values_valid.tolist()),
                coef_pvalues       = np.array(coef_pvalues_valid.tolist()),
                dates              = ds,
            )

            assert len(entry['dates'])       == len(entry['coef_values'])
            assert len(entry['coef_names'])  == len(entry['coef_values'][0])

            self.insert1(entry, skip_duplicates=True)

    def _insert_point(self, animal_id, train_point, predicted_vars,
                      columns, coefs, pvalues, str_dates, fr_correct):
        """
        Insert 'point' entry for the best-performance date.

        Parameters
        ----------
        animal_id : str
            Mouse identifier.
        train_point : str
            Current training timepoint key.
        predicted_vars : list of str
            Variables to store ('choices', 'errors').
        columns, coefs, pvalues : np.ndarray
            Arrays shaped (nr_dates, nr_predicted_vars, ...).
        str_dates : np.ndarray of str
            Session date strings.
        fr_correct : np.ndarray
            Percent correct per date used to identify the best session.
        """
        best_idx = np.argmax(fr_correct)

        for k, predicted_var in enumerate(predicted_vars):
            c_names = columns[best_idx, k]
            if c_names is None:
                continue  # regression failed for this date – skip

            entry = dict(
                animal_id          = animal_id,
                predicted_var      = predicted_var,
                training_timepoint = train_point,
                analysis_type      = 'point',
                coef_names         = c_names,
                coef_values        = np.array(coefs[best_idx, k]),
                coef_pvalues       = np.array(pvalues[best_idx, k]),
                dates              = str_dates[best_idx],
            )

            assert len(entry['coef_names']) == len(entry['coef_values'])
            self.insert1(entry, skip_duplicates=True)

    # ------------------------------------------------------------------
    # Public population interface
    # ------------------------------------------------------------------

    def populate_trainingtimepoint(self, sessions, raw_trials, train_point):
        """
        Fit and store logistic regression coefficients for one training phase.

        For each analysis type ('trend' and 'point'), fits logistic regression
        across session dates and inserts results. 'trend' stores coefficients
        for every date; 'point' stores only the best-performance date.

        The TREND_ANALYSIS flag for this training point gates whether a 'trend'
        entry is inserted (some phases are too short to be meaningful as trends).

        Parameters
        ----------
        sessions : DataJoint table expression
            Session rows for this animal at this training phase.
        raw_trials : DataJoint table expression
            Matching trial rows.
        train_point : str
            Training phase key into `trainingpoint_dict`.
        """
        animal_id    = get_property(sessions, 'animal_id')[0]
        port_layout  = sessions.fetch('port_layout')[0]
        stimuli_space = get_stimuli_space()

        print(f'animal_id: {animal_id}')

        predicted_vars      = ['choices', 'errors']
        analysis_types      = ['trend', 'point']
        training_timepoints = list(trainingpoint_dict.keys())
        use_trend           = TREND_ANALYSIS[training_timepoints.index(train_point)]

        for analysis in analysis_types:
            print(f'analysis type: {analysis}')
            sessions_filtered = sessions
            trials_filtered   = raw_trials

            # For 'point' analysis on categorisation phases, restrict to the
            # specific categoryset_id encoded in the train_point name.
            if analysis == 'point' and train_point.startswith('categorization'):
                categoryset_id    = int(train_point.split('_')[-1])
                sessions_filtered = sessions & f'categoryset_id="{categoryset_id}"'
                trials_filtered   = raw_trials & f'categoryset_id="{categoryset_id}"'

            nr_dates = len(get_property(sessions_filtered, 'date'))
            print(f'nr_dates: {nr_dates}')

            if nr_dates == 0:
                continue  # nothing to process for this phase

            str_dates, fr_correct, columns, coefs, pvalues = self._fit_sessions(
                sessions_filtered, trials_filtered, port_layout, stimuli_space
            )

            # Convert to object arrays for consistent indexing
            columns   = np.array(columns,   dtype=object)
            coefs     = np.array(coefs,     dtype=object)
            pvalues   = np.array(pvalues,   dtype=object)
            str_dates = np.array(str_dates, dtype=object)

            assert len(predicted_vars) == columns.shape[1]

            if analysis == 'trend' and use_trend:
                self._insert_trend(animal_id, train_point, predicted_vars,
                                   columns, coefs, pvalues, str_dates)

            elif analysis == 'point':
                self._insert_point(animal_id, train_point, predicted_vars,
                                   columns, coefs, pvalues, str_dates, fr_correct)

    def populate_subject(self, animal_id):
        """
        Populate all training timepoints for a single subject.

        Skips timepoints that already have entries in the table.

        Parameters
        ----------
        animal_id : str
            Mouse unique identifier.
        """
        subject_trials   = Trial()   & f'animal_id="{animal_id}"'
        subject_sessions = Session() & f'animal_id="{animal_id}"'

        for train_point in trainingpoint_dict:
            mode, stage, _ = trainingpoint_dict[train_point]
            sessions = subject_sessions & f'mode="{mode}"' & f'stage="{stage}"'
            trials   = subject_trials   & f'mode="{mode}"' & f'stage="{stage}"'

            already_done = self & f'animal_id="{animal_id}"' & f'training_timepoint="{train_point}"'

            if len(sessions) != 0 and len(already_done) == 0:
                print(f'animal_id: {animal_id}  |  train_point: {train_point}')
                self.populate_trainingtimepoint(sessions, trials, train_point)

    def update(self):
        """
        Populate the table for all subjects in the experiment.

        Iterates over every unique animal_id found in the Session table and
        calls `populate_subject` for each. Prints total processing time on
        completion.
        """
        start      = time()
        animal_ids = get_property(Session(), 'animal_id')

        for animal_id in animal_ids:
            self.populate_subject(animal_id)

        print(f'PROCESSING TIME: {round((time() - start) / 60, 2)} min')