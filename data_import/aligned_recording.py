#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:04:50 2023

@author: Laura Sainz Villalba

Aligns per-session neural tensor data (neurons × trials × timepoints) to
specific behavioural events (choice, stimulus on/off, port on/off) and stores
the resulting tensors in the DataJoint table ``Aligned_trialtensor_data``.

Pipeline overview
-----------------
1.  Load raw trial-tensor data from ``Raw_trialtensor_data``
2.  Filter to accepted neuron masks only
3.  For each alignment event:
    a. Select only trials that contain that event
    b. Reframe trial timepoints relative to the event timestamp
    c. Crop/align the tensor to a common time window
    d. Reframe all other event timestamps to the same reference
    e. Insert the aligned tensor into ``Aligned_trialtensor_data``
4.  Optionally run the above in parallel across animals
"""

import os
import sys
import math
import inspect
import multiprocessing as mp
from time import time

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np

print("Calling aligned_recording.py from module script:", __name__)

# ---------------------------------------------------------------------------
# Conditional imports depending on execution context
# ---------------------------------------------------------------------------
if __name__ == '__main__' or __name__ == 'aligned_recording':
    from recording_import import Raw_trialtensor_data

elif __name__ == 'data_import.aligned_recording':
    from .recording_import import Raw_trialtensor_data

# Add parent directory to path for shared utilities
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utilities import (
    select_tensor_by_axis,
    reframe_trial_timepoints,
    get_trial_bool,
    align_tensor,
    reframe_all_events,
    get_property,
    split_by,
)

# ---------------------------------------------------------------------------
# DataJoint schema
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema = dj.schema('alignment_recording_hpc_cat_2025', locals(), create_tables=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# All behavioural event names and their index in the event array
ALL_EVENTS = ['choice', 'stimulus_on', 'stimulus_off', 'ports_on', 'ports_off']

# Events to which tensors will be aligned (subset of ALL_EVENTS)
EVENTS_TO_ALIGN = ['choice', 'stimulus_on', 'ports_on', 'ports_off']

# Expected tensor axis ordering
EXPECTED_TENSOR_FORMAT = ['neurons', 'trials', 'timepoints']

# Minimum number of DB entries considered "complete" for a session
# events_to_align : 4 events 
COMPLETE_ENTRY_COUNT = 4


# ===========================================================================
# DataJoint table
# ===========================================================================

@schema
class Aligned_trialtensor_data(dj.Manual):
    """
    Per-session neural tensor aligned to a behavioural event.

    Shape: neurons × trials × timepoints, where timepoints are expressed
    relative to the alignment event (t=0 at the event).
    """

    definition = """
    animal_id              : varchar(128)   # Mouse ID (unique)
    session_id             : int            # Session counter (1-based, chronological)
    event_to_align         : varchar(128)   # Behavioural event used as t=0
    experimental_timepoint : varchar(256)   # Training stage (discrimination / generalisation / …)
    ------
    date               : date       # Session date (YYYY-MM-DD)
    neuron_ids         : blob       # Neuron IDs corresponding to accepted mask components
    nr_accepted        : int        # Number of accepted cell-mask components in this session
    tensor_format      : tinyblob   # Axis order — default ['neurons', 'trials', 'timepoints']
    tensor_data        : longblob   # Aligned tensor (neurons × trials × timepoints)
    tensor_dim_vars    : blob       # Axis labels: ['mask_id', 'trial_id', 'timepoints']
    tensor_dim_values  : longblob   # Axis values: [mask_ids, trial_ids, timepoints_array]
    event_timestamps   : longblob   # Per-trial timestamps for all events, reframed to t=0
    """

    # -----------------------------------------------------------------------
    def align_tensors_session(self, animal_id, session_id):
        """
        Align the neural tensor for one session to every event in EVENTS_TO_ALIGN
        and insert the results into the database.

        For each alignment event the method:
          1. Filters to trials that have that event present
          2. Reframes trial timepoints relative to the event timestamp
          3. Crops the tensor to a common aligned window
          4. Reframes all other event timestamps to the same reference
          5. Inserts the aligned tensor entry (skips if already present)

        Parameters
        ----------
        animal_id  : str
        session_id : int
        """
        start = time()

        # ── Load raw tensor from DB ───────────────────────────────────────────
        print('Loading tensor...')
        tensor_entry = (Raw_trialtensor_data()
                        & f'animal_id="{animal_id}"'
                        & f'session_id="{session_id}"')

        date                   = str(tensor_entry.fetch('date')[0])
        neuron_ids             = tensor_entry.fetch('neuron_ids')[0]
        tensor_dff             = np.array(tensor_entry.fetch('tensor_data_dff')[0])
        tensor_format          = tensor_entry.fetch('tensor_format')[0]
        mask_ids, trial_ids, trials_timepoints = tensor_entry.fetch('tensor_dim_values')[0]
        trial_starts           = tensor_entry.fetch('trial_starts')[0]
        event_array            = np.array(tensor_entry.fetch('event_timestamps')[0])
        accepted               = tensor_entry.fetch('accepted')[0]
        experimental_timepoint = tensor_entry.fetch('experimental_timepoint')[0]
        print('Done loading.')

        # ── Validate raw tensor dimensions ────────────────────────────────────
        assert tensor_format == EXPECTED_TENSOR_FORMAT
        assert tensor_dff.ndim == 3
        assert tensor_dff.shape[0] == len(mask_ids),         "Neuron dim mismatch"
        assert tensor_dff.shape[1] == len(trial_ids),        "Trial dim mismatch"
        assert tensor_dff.shape[2] == len(trials_timepoints[0]), "Timepoint dim mismatch"

        # ── Filter to accepted neuron masks ───────────────────────────────────
        accepted_bool        = [bool(a) for a in accepted]
        nr_accepted_masks    = sum(accepted_bool)
        selected_mask_ids    = np.array(mask_ids)[accepted_bool]
        selected_neuron_ids  = np.array(neuron_ids)[accepted_bool]

        selected_mask_tensor_dff = select_tensor_by_axis(
            tensor_dff, tensor_format, mask_ids, selected_mask_ids, 'neurons'
        )

        # Validate shape after neuron selection
        assert selected_mask_tensor_dff.ndim == 3
        assert selected_mask_tensor_dff.shape[0] == nr_accepted_masks
        assert len(selected_mask_ids)   == nr_accepted_masks
        assert len(selected_neuron_ids) == nr_accepted_masks
        assert selected_mask_tensor_dff.shape[1] == len(trial_ids)

        # ── Align to each target event ─────────────────────────────────────────
        for event in EVENTS_TO_ALIGN:

            # Skip if this event × session entry already exists
            existing = (self
                        & f'animal_id="{animal_id}"'
                        & f'session_id="{session_id}"'
                        & f'event_to_align="{event}"')
            if len(existing) > 0:
                continue

            print(f'Aligning for event: {event}...')
            event_idx        = ALL_EVENTS.index(event)
            event_timestamps = event_array[event_idx, :]   # shape: (n_trials,)

            # ── Select only trials where this event is present ────────────────
            selected_trial_bool = get_trial_bool(event_timestamps)
            selected_trial_ids  = np.array(trial_ids)[selected_trial_bool]

            selected_tensor_dff = select_tensor_by_axis(
                selected_mask_tensor_dff, tensor_format,
                trial_ids, selected_trial_ids, 'trials'
            )

            assert selected_tensor_dff.ndim == 3
            assert selected_tensor_dff.shape[1] == len(selected_trial_ids)

            # Subset the per-trial metadata to matching trials
            selected_timepoints        = np.array(trials_timepoints)[selected_trial_bool]
            selected_trial_starts      = np.array(trial_starts)[selected_trial_bool]
            selected_event_timestamps  = np.array(event_timestamps)[selected_trial_bool]

            assert selected_tensor_dff.shape[1] == len(selected_timepoints)
            assert selected_tensor_dff.shape[1] == len(selected_trial_starts)
            assert selected_tensor_dff.shape[1] == len(selected_event_timestamps)

            # ── Reframe timepoints to t=0 at the alignment event ──────────────
            onset_timepoints = reframe_trial_timepoints(
                selected_timepoints,
                selected_trial_starts,
                selected_event_timestamps,
            )
            assert selected_tensor_dff.shape[1] == len(onset_timepoints)

            # ── Crop tensor to a common aligned window ────────────────────────
            aligned_tensor_dff, aligned_timepoints_dff = align_tensor(
                selected_tensor_dff, tensor_format, onset_timepoints
            )

            # Use mean timepoints across trials as the canonical time axis
            aligned_timepoints = np.mean(aligned_timepoints_dff, axis=0)

            assert aligned_tensor_dff.ndim == 3
            assert selected_tensor_dff.shape[1] == len(aligned_timepoints_dff)
            assert aligned_tensor_dff.shape[2]  == len(aligned_timepoints)

            # ── Reframe all event timestamps to the alignment reference ────────
            # event_array has shape (events × trials); select matching trials first
            selected_event_array = np.array(event_array)[:, selected_trial_bool]
            reframed_events      = reframe_all_events(
                selected_event_array, selected_event_timestamps
            )

            # ── Insert aligned tensor into DB ──────────────────────────────────
            print(f'  Inserting aligned tensor for event={event}...')
            aligned_entry = {
                'animal_id':              animal_id,
                'session_id':             session_id,
                'event_to_align':         event,
                'experimental_timepoint': experimental_timepoint,
                'date':                   date,
                'neuron_ids':             selected_neuron_ids,
                'nr_accepted':            nr_accepted_masks,
                'tensor_format':          tensor_format,
                'tensor_data':            aligned_tensor_dff,
                'tensor_dim_vars':        ['mask_id', 'trial_id', 'timepoints'],
                'tensor_dim_values':      [selected_mask_ids,
                                           selected_trial_ids,
                                           aligned_timepoints],
                'event_timestamps':       reframed_events,   # (events × trials), t=0 at alignment event
            }
            Aligned_trialtensor_data().insert1(aligned_entry, skip_duplicates=True)

        print(f'Session done. Elapsed: {(time() - start) / 60:.2f} mins')

    # -----------------------------------------------------------------------
    def run_parallel(self, animal_ids):
        """
        Process all sessions for a given list of animals sequentially.

        Intended to be the target function of a ``multiprocessing.Process``;
        each parallel worker receives a non-overlapping subset of animal IDs.

        Parameters
        ----------
        animal_ids : list of str
        """
        for animal_id in animal_ids:
            session_ids = get_property(
                Raw_trialtensor_data() & f'animal_id="{animal_id}"',
                'session_id',
            )

            for session_id in session_ids:
                # Check whether this session is already fully processed
                existing = (Aligned_trialtensor_data()
                            & f'animal_id="{animal_id}"'
                            & f'session_id="{session_id}"')

                if len(existing) < COMPLETE_ENTRY_COUNT:
                    continue

                try:
                    self.align_tensors_session(animal_id, session_id)
                except AssertionError:
                    print(f'AssertionError — animal_id: {animal_id}, session_id: {session_id}')
                    raise RuntimeError('Assertion condition not satisfied')
                        
    # -----------------------------------------------------------------------
    def align_tensors_experiment(self, nr_parallel_processes=1):
        """
        Align tensors for every session in the experiment, optionally using
        multiple parallel worker processes.

        The full list of animals is split into ``nr_parallel_processes`` roughly
        equal subsets; each subset is handled by a dedicated ``mp.Process``.

        Parameters
        ----------
        nr_parallel_processes : int
            Number of parallel worker processes (default: 1 = sequential).
        """
        # ── Gather all unique animal IDs with raw tensor data ─────────────────
        all_animals = get_property(Raw_trialtensor_data(), 'animal_id')
        nr_animals  = len(all_animals)

        # ── Partition animals across workers ──────────────────────────────────
        subsets  = split_by(nr_animals, nr_parallel_processes)
        animals  = np.array(all_animals)
        workers  = []

        for worker_idx, subset in enumerate(subsets):
            animal_subset = list(animals[subset])
            print(f'Spawning worker {worker_idx} for animals: {animal_subset}')

            proc = mp.Process(target=self.run_parallel, args=(animal_subset,))
            proc.start()
            workers.append(proc)

        # ── Wait for all workers to finish ────────────────────────────────────
        for proc in workers:
            proc.join()
            print(f'Worker {proc.pid} finished.')

        print('All sessions aligned successfully.')