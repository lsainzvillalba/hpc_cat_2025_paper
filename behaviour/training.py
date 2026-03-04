#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:11:26 2023

@author: Laura Sainz Villalba

# =============================================================================
# training.py
# Utility functions for retrieving stimuli spaces, neural recordings, and
# behavioural performance across experimental training phases.
# =============================================================================
"""

import numpy as np
import os, sys, inspect

print("Calling training.py from module script: ", __name__)

# ---------------------------------------------------------------------------
# Conditional imports – supports three execution contexts:
#   1. Run directly as a script (__main__)
#   2. Imported as a top-level module ('training')
#   3. Imported as part of the 'behaviour' sub-package ('behaviour.training')
# ---------------------------------------------------------------------------
if __name__ == '__main__' or __name__ == 'training':
    from day_statistics import Day
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir  = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from design import categories
    from data_import import Session, Raw_trialtensor_data
    from utilities import get_property, trainingpoint_dict

elif __name__ == 'behaviour.training':
    from .day_statistics import Day
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir  = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from design import categories
    from data_import import Session, Raw_trialtensor_data
    from utilities import get_property, trainingpoint_dict


# =============================================================================
# Stimulus space helpers
# =============================================================================

def get_grid_id():
    """
    Return the grid_id associated with *experiment*'s curriculum.

    Looks up the most recent non-zero curriculum_id for the experiment,
    then retrieves the corresponding grid_id from the Curriculum table.

    Parameters
    ----------

    Returns
    -------
    int
        Grid ID for the experiment's category structure.
    """
    curriculum_id = (
        Session()  & 'curriculum_id!="0"'
    ).fetch('curriculum_id')[-1]

    return (categories.Curriculum() & f'curriculum_id={curriculum_id}').fetch('grid_id')[0]


def get_stimuli_space():
    """
    Return the stimulus grid (array of raw stimulus values) for *experiment*.

    Parameters
    ----------

    Returns
    -------
    np.ndarray
        Array of stimulus values defining the experiment's stimulus space.
    """
    grid_id = get_grid_id()
    return (categories.Stimuligrid() & f'grid_id="{grid_id}"').fetch('stimuligrid')[0]


# =============================================================================
# Neural recording helpers
# =============================================================================

def get_recordings(trainingpoint, output_date=False):
    """
    Return a per-subject dictionary of session IDs (or dates) for sessions
    at *trainingpoint* that have associated neural recordings.

    Parameters
    ----------
    trainingpoint : str
        Key into `trainingpoint_dict` specifying (mode, stage, categoryset_id).
    output_date : bool, optional
        If True, values are recording dates (str); if False (default), values
        are session IDs (int).

    Returns
    -------
    dict
        {animal_id: [session_id_or_date, ...]} for subjects with recordings.
    """
    mode, stage, categoryset_id = trainingpoint_dict[trainingpoint]

    sessions = (
        Session()
        & f'mode="{mode}"'
        & f'stage="{stage}"'
        & f'categoryset_id="{categoryset_id}"'
    )
    animal_ids   = get_property(sessions, 'animal_id')
    session_dict = {}

    for animal_id in animal_ids:
        animal_sessions = sessions & f'animal_id="{animal_id}"'

        for session_id in animal_sessions.fetch('session_id'):
            recordings = (
                Raw_trialtensor_data()
                & f'animal_id="{animal_id}"'
                & f'session_id="{session_id}"'
            )
            if len(recordings) == 0:
                continue  # skip sessions without neural data

            value = str(recordings.fetch('date')[0]) if output_date else session_id
            session_dict.setdefault(animal_id, []).append(value)

    return session_dict


def get_neurons_in_phase(phase):
    """
    Return the mean and standard error of accepted neuron counts across
    subjects at a given training *phase*.

    For each subject, the mean neuron count is computed across recording
    sessions; these per-subject means are then averaged across subjects.

    Parameters
    ----------
    experiment : str
        Experiment name used to filter Raw_trialtensor_data.
    phase : str
        Training phase key (passed to `get_recordings`).

    Returns
    -------
    mean_neurons : float
        Grand mean of accepted neurons across subjects.
    sem_neurons : float
        Standard error of the mean across subjects.
    """
    session_dict = get_recordings(phase, output_date=True)
    animal_ids   = list(session_dict.keys())
    nr_subjects  = len(animal_ids)

    neurons_per_subject = []
    for animal_id in animal_ids:
        counts = [
            (Raw_trialtensor_data()
             & f'animal_id="{animal_id}"'
             & f'date="{date}"').fetch('nr_accepted')
            for date in session_dict[animal_id]
        ]
        neurons_per_subject.append(np.mean(counts))

    mean_neurons = np.mean(neurons_per_subject)
    sem_neurons  = np.std(neurons_per_subject) / np.sqrt(nr_subjects)
    return mean_neurons, sem_neurons


def get_neuron_subsampling_phase_comparison():
    """
    Return the number of neurons to subsample when comparing phases.

    Uses 90 % of the minimum accepted-neuron count across all sessions in
    *experiment* (excluding the 'categorization_3' timepoint).

    Parameters
    ----------
    experiment : str
        Experiment name used to filter Raw_trialtensor_data.

    Returns
    -------
    int
        Number of neurons to subsample.
    """
    nr_accepted = (
        Raw_trialtensor_data()
        & 'experimental_timepoint!="categorization_3"'
    ).fetch('nr_accepted')

    return int(0.9 * min(nr_accepted))


# =============================================================================
# Behavioural performance helpers
# =============================================================================

def get_performance_subject(phase, animal_id):
    """
    Return the mean performance (% correct) for *animal_id* at *phase*,
    averaged over the two most recent sessions in that phase.

    Parameters
    ----------
    phase : str
        Training phase key into `trainingpoint_dict`.
    animal_id : str
        Mouse unique identifier.

    Returns
    -------
    float
        Mean percent correct across the two most recent session dates.
    """
    mode, stage, categoryset_id = trainingpoint_dict[phase]

    sessions = (
        Session()
        & f'mode="{mode}"'
        & f'stage="{stage}"'
        & f'categoryset_id="{categoryset_id}"'
        & f'animal_id="{animal_id}"'
    )
    # Use only the two most recent dates for this phase
    recent_dates = sessions.fetch('date')[-2:]

    performance = [
        (Day() & f'animal_id="{animal_id}"' & f'date="{d}"').fetch('performance')[0]
        for d in recent_dates
    ]
    return np.mean(performance)
