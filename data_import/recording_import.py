#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba

Recording Import Module

This module handles importing neural recording data into a DataJoint database.
It processes neural activity tensors (neurons x trials x timepoints) and stores them
alongside behavioral data. Supports backup storage as JSON files.

Data Format: One file per session day and subject
Structure: Pandas dataframe for behavior + tensor for neural activity (dff traces)
"""

# Import libraries
import json
from tqdm import tqdm
from time import time
import os
import sys
import inspect
import datajoint as dj
import numpy as np
import random
from scipy import stats

# Import modules
print("Calling recording_import.py from module script: ", __name__)

# Handle different import contexts (direct run vs module import)
if __name__ == '__main__' or __name__ == 'recording_import':
    from behaviour_import import Trial
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utilities import get_property, get_bool_on_lengths, get_squared_timepoints, \
        get_squared_tensor
elif __name__ == 'data_import.recording_import':
    from .behaviour_import import Trial
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utilities import get_property, get_bool_on_lengths, get_squared_timepoints, \
        get_squared_tensor

# Enable Python native blobs for storing complex data structures
dj.config["enable_python_native_blobs"] = True

# Create or connect to the recording_import schema
schema = dj.schema('recording_import_hpc_cat_2025', locals(), create_tables=True)


def get_experimental_timepoint(mode, stage, categoryset_id):
    """
    Maps experimental parameters to a standardized timepoint label.
    
    Args:
        mode (str): Training mode (e.g., 'periodshaping', 'experimental')
        stage (int): Stage number within the training mode
        categoryset_id (int): Identifier for the category set used
        
    Returns:
        str: Experimental timepoint label (e.g., 'discrimination', 'categorization_2')
    """
    # Define mapping between (mode, stage, categoryset_id) and experimental timepoint
    mode_stage_dict = {
        ('periodshaping', 1, 1): 'discrimination',
        ('experimental', 1, 0): 'gentest_1',
        ('experimental', 2, 2): 'categorization_2',
        ('experimental', 2, 3): 'categorization_3',
        ('experimental', 2, 4): 'categorization_4',
        ('experimental', 3, 0): 'gentest_2'
    }
    
    experimental_timepoint = mode_stage_dict[(mode, stage, categoryset_id)]
    
    return experimental_timepoint


def get_tensor_from_session(recording_session_entries, cell_validation_entries):
    """
    Constructs 3D neural activity tensors from session recordings.
    
    This function aggregates all neural activity data for a session into
    standardized 3D arrays with dimensions: neurons x trials x timepoints
    for both dff (delta F/F) traces.
    
    Args:
        recording_session_entries: DataJoint query result with neural recordings
        cell_validation_entries: DataJoint query result with cell validation data
        
    Returns:
        dict: Contains tensors and metadata including:
            - accepted: boolean list of validated cells
            - tensor_neuron_ids: unique neuron identifiers
            - tensor_mask_ids: mask identifiers for each neuron
            - tensor_trial_ids: trial identifiers
            - baseline_dff: baseline recording for dff
            - tensor_dff: main dff tensor (neurons x trials x timepoints)
            - tensor_timepoints: timestamps for each trial
    """
    # Fetch mask and neuron IDs from recordings
    mask_ids = recording_session_entries.fetch('mask_id')
    neuron_ids = recording_session_entries.fetch('neuron_id')
    mask_ids = list(map(int, mask_ids))
    neuron_ids = list(map(int, neuron_ids))
    
    # Get trial segmentation info from first mask (reference)
    segment_trials = (recording_session_entries & 'mask_id="1"').fetch('segment_trials')[0]
    
    # Extract trial IDs (skip first element which is baseline segment with ID 0)
    trial_ids = segment_trials[1:]
    trial_ids = list(map(int, trial_ids))
    
    # Validate that first segment is baseline (trial_id = 0)
    if segment_trials[0] != 0:
        print(recording_session_entries & 'mask_id="1"')
        print('segment_trials[0]', segment_trials)
        raise Exception('first segment')
        
    # Get number of frames for segmentation
    nr_frames_segmentation = (recording_session_entries & 'mask_id="1"').fetch('nr_frames')[0]
    
    # Get timepoints for all trials (exclude baseline segment)
    nonparsed_timepoints = (recording_session_entries & 'mask_id="1"').fetch('timepoints')[0]
    alltrial_timepoints = nonparsed_timepoints[1:]  # Skip baseline
    
    # Initialize tensor lists
    tensor_dff = []  # Will be: neurons x trials x timepoints
    baseline_dff = []  # Will be: neurons x timepoints
    
    # Sort mask IDs for consistent ordering
    mask_ids.sort()
    accepted = []
    
    # Process each neuron/mask
    for i in tqdm(range(len(mask_ids))):
        mask_id = mask_ids[i]
        
        # Check if this cell was accepted in validation
        accepted_bool = (cell_validation_entries & 'mask_id="%i"' % mask_id).fetch('accepted')[0]
        accepted.append(int(accepted_bool))
        
        # Fetch full dff traces for this neuron
        nonparsed_dff = (recording_session_entries & 'mask_id="%i"' % mask_id).fetch('dff')[0]
        
        # Extract baseline segment (first segment before trials)
        baseline_dff_mask = list(nonparsed_dff[:nr_frames_segmentation[0]])
        baseline_dff.append(baseline_dff_mask)
        
        # Initialize trial-wise data for this neuron
        tensor_dff_mask = []
        start_frame = 0
        
        # Process each trial segment
        for j in range(len(segment_trials)):
            trial_id = segment_trials[j]
            
            if trial_id == 0:
                # Trial ID 0 = baseline recording at beginning of session
                start_frame = nr_frames_segmentation[0]
                # Baseline does not correspond to any behavioral trial
            else:
                # Extract frames for this trial
                nr_frames = nr_frames_segmentation[j]
                end_frame = start_frame + nr_frames
                
                # Slice traces for this trial
                tensor_dff_trial = list(nonparsed_dff[start_frame:end_frame])
                
                tensor_dff_mask.append(tensor_dff_trial)
                
                # Validate that number of timepoints matches number of frames
                nr_timepoints = len(nonparsed_timepoints[j])
                if nr_timepoints != len(tensor_dff_trial):
                    print(recording_session_entries & 'mask_id="1"')
                    print('nr_timepoints: ', nr_timepoints)
                    print('len(tensor_dff_trial): ', len(tensor_dff_trial))
                    print('mask_id: ', mask_id)
                    print('trial_id: ', trial_id)
                    raise Exception('Nr frames and nr timepoints do not align')
                
                # Update start position for next trial
                start_frame = end_frame
        
        # Add this neuron's data to the full tensor
        tensor_dff.append(tensor_dff_mask)

    # Handle cases where trials have unequal timepoint lengths
    if len(np.array(tensor_dff).shape) != 3:
        # Check if there's variability in trial lengths
        unequal_bool, min_nr_timepoints, trial_lengths = get_bool_on_lengths(tensor_dff, alltrial_timepoints)
        
        if unequal_bool:
            if min_nr_timepoints > 260:
                # If minimum is still reasonable, trim all trials to minimum length
                alltrial_timepoints = get_squared_timepoints(alltrial_timepoints, min_nr_timepoints)
                tensor_dff = get_squared_tensor(tensor_dff, min_nr_timepoints)
                print('min nr timepoints: ', min_nr_timepoints)
            else:
                # If minimum is too short, it's likely a bad trial at the end
                trial_idx = trial_lengths.index(min_nr_timepoints)
                if trial_idx > 90:
                    # Remove trials after the problematic one
                    alltrial_timepoints = alltrial_timepoints[:trial_idx - 1]
                    trial_ids = trial_ids[:trial_idx - 1]
                    
                    tensor_dff = np.array(tensor_dff)[:, :trial_idx - 1].tolist()
                    print('trial_idx: ', trial_idx)
                else:
                    # Problem occurred early in session - this is unusual
                    print('min nr timepoints: ', min_nr_timepoints)
                    raise Exception('Affected early trial: ', trial_idx)
        else:
            # Tensor is not 3D but has equal lengths - unexpected case
            print('min nr timepoints: ', min_nr_timepoints)
            raise Exception('Tensor is not 3d and has equal timepoint lengths')
    
    # Validate tensor dimensions
    assert len(trial_ids) == len(alltrial_timepoints)
    assert len(trial_ids) == np.array(tensor_dff).shape[1]
    
    # Package results into dictionary
    resultdict = dict()
    resultdict['accepted'] = accepted
    resultdict['tensor_neuron_ids'] = neuron_ids
    resultdict['tensor_mask_ids'] = mask_ids
    resultdict['tensor_trial_ids'] = trial_ids
    resultdict['baseline_dff'] = baseline_dff
    resultdict['tensor_dff'] = tensor_dff
    resultdict['tensor_timepoints'] = alltrial_timepoints
    
    return resultdict


@schema
class Raw_trialtensor_data(dj.Manual):
    """
    Table storing neural recording datasets as tensors.
    
    Format: neurons x trials x timepoints for both dff and spike-inferred traces.
    Includes behavioral event timestamps aligned to neural data.
    """
    
    definition = """ # Session dataset recordings - neurons x trials x timepoints 
    animal_id          : varchar(128)   # Mouse id (unique id)
    session_id         : int            # Overall session counter for mouse (base1) in chronological order
    experimental_timepoint : varchar(256)   # discrimination, generalization_1, categorization, generalization_2, exception
    ------
    date               : date           # Date of the experimental session recordings (YYYY-MM-DD)
    neuron_ids         : blob           # neuron ids (unique across sessions) corresponding to mask ids in tensor as neurons
    accepted           : blob           # bool for accepted masks in validation extraction of components
    nr_accepted        : int            # number of accepted cell mask components in session
    trial_starts       : blob           # all trial start timestamps
    baseline_dff           : longblob       # baseline trace for dff, neurons x timepoints
    tensor_format          : tinyblob       # by default ['neurons', 'trials', 'timepoints']
    tensor_data_dff          : longblob   # tensor data of neurons x trials x timepoints
    tensor_dim_vars        : blob       # list of variables corresponding to dimensions trial ids x mask ids x timepoints
    tensor_dim_values      : longblob   # list of values for each tensor dimension, ['mask_id', 'trial_id', trial timepoints']
    event_timestamps       : longblob   # list of lists of timestamps of events ['choice', 'stimulus_on', 'stimulus_off', 'ports_on', 'ports_off']
    """
    
    def validate_entries(self, animal_id, session_id, nr_checks=15):
        """
        Validates tensor data integrity by cross-checking with source recordings.
        
        Randomly samples trials and neurons to verify that:
        1. Event timestamps match behavioral data
        2. Trial start times are consistent
        3. Neural traces match original recordings (correlation > 0.99)
        
        Args:
            experiment (str): Experiment name
            animal_id (str): Animal identifier
            session_id (int): Session number
            nr_checks (int): Number of random samples to validate (default: 15)
        """
        start = time()
        print('****check****')
        print('animal id: ', animal_id)
        print('session id: ', session_id)
        
        from data_process import neuralactivity
        
        # Fetch tensor entry from database
        tensor_entry = self  & 'animal_id="%s"' % animal_id \
                & 'session_id="%s"' % session_id
                
        print('Loading tensor...')
        tensor_data_dff = tensor_entry.fetch('tensor_data_dff')[0]
        baseline_dff = tensor_entry.fetch('baseline_dff')[0]
        mask_ids, trial_ids, all_timepoints = tensor_entry.fetch('tensor_dim_values')[0]
        
        # Define events to validate
        events = ['choice', 'stimulus_on', 'stimulus_off', 'ports_on', 'ports_off']
        event_timestamps = tensor_entry.fetch('event_timestamps')[0]
        trial_starts = tensor_entry.fetch('trial_starts')[0]
        
        # Get frame counts
        frames_trial = len(tensor_data_dff[0][0])
        frames_baseline = len(baseline_dff[0])
        assert frames_baseline == 2000  # Expected baseline length
        
        # Select random trials and masks for validation
        selected_trials = random.choices(trial_ids, k=nr_checks)
        selected_masks = random.choices(mask_ids, k=nr_checks)
        
        # Validate each selected trial
        for j, trial_id in enumerate(selected_trials):
            print(' trial %i/%i' % (j, len(selected_trials)))
            trial_id_idx = trial_ids.index(trial_id)
            
            # Get behavioral information from trial table
            trial_entry = Trial() \
                & 'animal_id="%s"' % animal_id & 'session_id="%s"' % session_id \
                & 'trial_id="%i"' % trial_id
            
            # Check event timestamps against behavioral table
            for e in tqdm(range(len(events))):
                event = events[e]
                ts = event_timestamps[e][trial_id_idx]
                
                # Extract ground truth timestamp from behavioral data
                if event != 'choice':
                    # Standard events: stimulus_on, stimulus_off, ports_on, ports_off
                    if trial_entry.fetch(event)[0] is not None:
                        ts_gr = trial_entry.fetch(event)[0][0]
                    else:
                        ts_gr = None
                else:
                    # Choice event: calculated from reaction time
                    response = trial_entry.fetch('response')[0]
                    if response != -1:
                        reaction_time = trial_entry.fetch('reaction_time')[0]
                        ports_on = trial_entry.fetch('ports_on')[0][0]
                        ts_gr = round(reaction_time + ports_on, 3)
                    else:
                        ts_gr = None

                # Validate timestamp match
                if ts != ts_gr:
                    print('event: ', event)
                    print(ts)
                    print(ts_gr)
                    raise Exception('timestamps do not match')
            
            # Check trial start times
            trial_start = trial_entry.fetch('trial_start')[0]
            if trial_start != trial_starts[trial_id_idx]:
                raise Exception('trial start timestamps do not match')
            
            # Validate neural traces for each selected mask
            for i in tqdm(range(len(selected_masks))):
                mask_id = selected_masks[i]
                mask_id_idx = mask_ids.index(mask_id)
                
                # Fetch original neural recording
                mask_entry = neuralactivity.Neuron_session_recording() \
                    & 'animal_id="%s"' % animal_id \
                    & 'session_id="%s"' % session_id & 'mask_id="%i"' % mask_id
                dff_trace_session = mask_entry.fetch('dff')[0]
                
                # Get trace from tensor
                dff_trial_trace = tensor_data_dff[mask_id_idx][trial_id_idx]
                
                # Get expected trace from session recording
                idx_trial = frames_baseline + (trial_id - 1) * frames_trial
                dff_trial_trace_gt = dff_trace_session[idx_trial:idx_trial + frames_trial]
                
                # Calculate correlation between tensor and original trace
                r = stats.pearsonr(dff_trial_trace, dff_trial_trace_gt)[0]
                if r < 0.99:
                    print('animal_id: ', animal_id, 'for session_id: ', session_id)
                    raise Exception('Error: Trial parsing not correct')
        
        print('time (mins) for backup and table: ', (time() - start) / 60)
    
    
    def populate_from_file(self, session_recording_file):
        """
        Populates tensor table from a JSON backup file.
        
        Args:
            session_recording_file (str): Path to JSON session file
        """
        start = time()
        
        # Load session data from JSON file
        with open(session_recording_file) as f:
            session_dict = json.load(f)
        
        # Reconstruct tensor dimension values
        tensor_dim_values = [session_dict['tensor_mask_ids'],
                           session_dict['tensor_trial_ids'],
                           session_dict['tensor_timepoints']]
        
        # Populate table with information from file
        print('\n Populating tables...')
        tensor_entry = {
            'name': session_dict['name'],
            'animal_id': session_dict['animal_id'],
            'session_id': session_dict['session_id'],
            'date': session_dict['date'],
            'neuron_ids': session_dict['tensor_neuron_ids'],
            'accepted': session_dict['accepted'],
            'nr_accepted': sum(session_dict['accepted']),
            'trial_starts': session_dict['trial_start'],
            'baseline_dff': session_dict['baseline_dff'],
            'tensor_format': session_dict['tensor_format'],
            'tensor_data_dff': session_dict['tensor_dff'],
            'tensor_dim_vars': session_dict['tensor_dim_vars'],
            'tensor_dim_values': tensor_dim_values,
            'event_timestamps': session_dict['events_array']
        }
        
        self.insert1(tensor_entry, skip_duplicates=True)
        
        print('Time in min: ', (time() - start) / 60)
    
    def create_dataset(self, dataset_folder, source='table'):
        """
        Creates complete dataset by populating tensor table from either local tables or files.
        
        Args:
            experiment (str): Experiment name
            dataset_folder (str): Path to dataset storage folder
            source (str): Data source - 'table' for local DataJoint tables, 'file' for JSON files
        """
        from data_process.recording import component_extraction as comp_extr
        
        if source == 'table':
            # Build dataset from local tables
            directory = dataset_folder + 'data-hpc_cat_2025/'
            all_recordings = comp_extr.Segmentation() 
            segmentation_ids = all_recordings.fetch('segmentation_id')
            
            for segmentation_id in segmentation_ids:
                # Get session information
                recording_entry = all_recordings & 'segmentation_id="%i"' % segmentation_id
                animal_id = recording_entry.fetch('animal_id')[0]
                session_id = int(recording_entry.fetch('session_id')[0])
                
                # Check if backup file already exists
                session_str = animal_id.replace('_', '-') + '_' + str(session_id)
                filename = directory + session_str + '.json'
                
                # Only process if backup doesn't exist
                # if not os.path.isfile(filename):
                #     # Exclude specific problematic sessions
                #     if not (animal_id == 'BK4936_L' and session_id == 33):  # Excessive movement in z
                #         if not (animal_id == 'BK4956_LR' and session_id == 16):
                #             self.populate_from_local_table(animal_id, session_id, dataset_folder)
        
        else:
            # Build dataset from JSON files
            recording_files = os.listdir(dataset_folder)
            recording_files = [session for session in recording_files if session.endswith('.json')]
            
            for session_recording_file in recording_files:
                # Parse filename to get animal_id and session_id
                filename = session_recording_file.split('.')[0]
                animal_id, session_id = filename.split('_')
                animal_id = animal_id.replace('-', '_')
                
                # Check if entry already exists in table
                entry = self & 'animal_id="%i"' % animal_id & 'session_id="%i"' % session_id
                if len(entry) == 0:
                    file_directory = dataset_folder + session_recording_file
                    print('file directory', file_directory)
                    self.populate_from_file(file_directory)
    
    def validate_experiment(self):
        """
        Validates data integrity for random sessions across all subjects in experiment.
        
        For each subject, randomly selects 5 sessions and validates their tensor data
        against source recordings.

        """
        animal_ids = get_property(self, 'animal_id')
        
        for animal_id in animal_ids:
            # Get all sessions for this animal
            entries_subject = self & 'animal_id="%s"' % animal_id
            session_ids = get_property(entries_subject, 'session_id')
            
            # Randomly select 5 sessions to validate
            selected_session_ids = random.choices(session_ids, k=5)
            
            for session_id in selected_session_ids:
                # Validate 15 random samples per session
                self.validate_entries(animal_id, session_id, nr_checks=15)