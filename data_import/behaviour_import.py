#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:31:45 2022

@author: Laura Sainz Villalba

Behavior Data Import Module
This module handles importing experimental behavior session data into a DataJoint database.
It processes JSON session files and organizes them into Session and Trial tables.
"""

# Import libraries
import datajoint as dj
import os
import json
from tqdm import tqdm

print("Calling behaviour_import.py from module script: ", __name__)

# Enable Python native blobs for storing complex data structures
dj.config["enable_python_native_blobs"] = True

# Create or connect to the behaviour_import schema
schema = dj.schema('behaviour_import_hpc_cat_2025', locals(), create_tables=True)

# Base directory for behavior data files
directory = './data/behaviour_files/'


@schema
class Session(dj.Manual):
    """
    Table that captures all events for each experimental session.
    Each session contains metadata and event timestamps from behavioral experiments.
    """
    
    definition = """# table that captures events for each session within experiment from event file
    
    session_id         : int   # Overall session counter for mouse (base1) in chronological order
    animal_id          : varchar(128)    # Mouse id (unique id)
    ---
    date               : date            # Date of the experimental session (YYYY-MM-DD)
    tos                : time            # Time of session entrance (HH:MM:SS)
    condition          : varchar(128)    # Experimental condition
    mode               : varchar(128)    # mode of training
    stage              : int             # stage of training within mode
    session_time       : float           # time in secs spent in session
    grace_period       : float           # time in secs for grace period
    delay_period       : float           # time in secs for delay period
    nr_triggers = NULL         : longblob        # vector with number of ttl triggers 
    ttl_timestamps = NULL      : longblob        # timestamp vector for ttl triggers for microscope
    new_trial          : longblob        # timestamp vector for new_trial events
    baited_port        : longblob        # vector of baited side for each trial presented in cronological order
    curriculum_id      : longblob        # integer identifier for curriculum applied in session
    categoryset_id     : int             # Unique identifier for category set used in session
    stimulus_on        : longblob        # timestamp vector for stimuli_on events
    stimulus_off       : longblob        # timestamp vector for stimuli_off events
    stimulus_id        : longblob        # vector of stimuli presented in cronological order
    sound_on           : longblob        # timestamp vector for sound_on events
    sound_off          : longblob        # timestamp vector for sound_off events
    sound_id           : longblob        # vector of sounds presented in cronological order
    response_window    : longblob        # vector of timestamps of response window onsets
    ports_on           : longblob        # timestamp vector for onset of response window events
    ports_off          : longblob        # timestamp vector for offset of response window events
    ports_id           : longblob        # vector of ports available in each trial, in chronological order
    port_layout        : int             # boolean integer for one of two possible port layouts
    left_licks = NULL  : longblob        # timestamp vector for left lick events
    right_licks = NULL : longblob        # timestamp vector for right lick events
    response           : longblob        # vector of side responses in cronological order
    reaction_time      : longblob        # vector of reaction time , for decisive lick response from port onset
    water = NULL       : longblob        # timestamp vector for water delivery events          
    punish = NULL      : longblob        # timestamp vector for punish events
    trialtype          : longblob        # vector for trial type for all trials in session
    responsetype       : longblob        # vector for response type for all trials in session
    trial_end          : longblob        # timestamp vector for trial_end events
    file               : varchar(256)    # file path to session file
    """
    
    def get_lastcounter(self, animal_id):
        """
        Returns the last session counter value for a specific animal.
        
        Args:
            animal_id (str): Unique identifier for the animal
            
        Returns:
            int: Last session counter, or 0 if no sessions exist for this animal
        """
        # Query all sessions for this animal
        all_e = self & 'animal_id="%s"' % animal_id
        
        if len(all_e) != 0:
            # Get the maximum session_id (most recent session)
            session_counter = max(list(all_e.fetch('session_id')))
            return session_counter
        else:
            # No sessions found, return 0
            return 0
        
    def load_session(self, session_file):
        """
        Loads a single session from a JSON file into the database.
        
        Args:
            session_file (str): Path to the session JSON file
        """
        # Define all event variables to extract from the session file
        event_vars = ['condition', 'mode', 'session_time',
                      'stage', 'grace_period', 'delay_period', 'nr_triggers',
                      'ttl_triggers', 'new_trial', 'baited_port', 'curriculum_id', 'categoryset_id',
                      'stimulus_on', 'stimulus_off', 'stimulus_id', 'response_window',
                      'sound_on', 'sound_off', 'sound_id',
                      'ports_on', 'ports_off', 'ports_id', 'left_licks', 'port_layout',
                      'right_licks', 'response', 'reaction_time', 'trialtype',
                      'responsetype', 'water', 'punish', 'trial_end'] 

        # Load the JSON session file
        with open(session_file) as f:
            data = json.load(f)
            
        num_trials = len(data['new_trial'])
        
        # Only process sessions that contain at least one trial
        if num_trials != 0:
            # Parse animal_id and session metadata from filename
            # Expected format: <animal_id>_<date>_<time>.json
            filename = session_file.split('_')
            animal_id = filename[0] + '_' + filename[1]
            
            # Get the next session counter for this animal
            lastcounter = self.get_lastcounter(animal_id)
            session_id = lastcounter + 1
            
            # Extract date and time from filename
            date = filename[-3]
            tos = filename[-2].replace('-', ':') + ':00'  # Convert time format
            
            # Build entry dictionary with session metadata
            entry_dict = {
                'session_id': session_id,
                'animal_id': animal_id,
                'date': date,
                'tos': tos
            }
            
            # Add all event data from the JSON file
            for event in event_vars:
                try:
                    entry_dict[event] = data[event]
                except KeyError:
                    # If event doesn't exist in file, set to None
                    entry_dict[event] = None
                    
            print('entry ', entry_dict)
            
            # Insert into database, skip if duplicate exists
            self.insert1(entry_dict, skip_duplicates=True)
            
    def load_experiment(self, directory):
        """
        Populates the Session table with all JSON session files in the directory.
        
        Args:
            directory (str): Path to directory containing session files
        """
        # Construct path to sessions subdirectory
        datapath = directory + '/sessions/'
        session_files = os.listdir(datapath)
        
        # Filter for JSON files only, exclude hidden files
        session_files = [session for session in session_files 
                        if session.endswith('.json') and not session.startswith('._')]
        
        j = 0  # Counter for newly added sessions
        
        # Process each session file with progress bar
        for i in tqdm(range(len(session_files))):
            session_file = session_files[i]
            
            # Check if this session already exists in database
            filename = session_file.split('_')
            animal_id = filename[0] + '_' + filename[1]
            date = filename[-3]
            
            entry = self & 'animal_id="%s"' % animal_id & 'date="%s"' % date
            
            # Only load if session doesn't already exist
            if len(entry) == 0:
                j += 1
                self.load_session(session_file)

        print('\n DONE: %i sessions added' % j)


@schema
class Trial(dj.Manual):
    """
    Table containing individual trial data extracted from sessions.
    Each trial represents a single behavioral event within a session.
    """
    
    definition = """ # Trials within sessions

    trial_id                  : int                    # Unique identifier, counter of trial within session (base 1)
    -> Session
    stimulus_id               : varchar(128)           # stimulus id presented in trial
    ---
    date                      : date                  # Date of the session trial (YYYY-MM-DD)
    condition                 : varchar(128)           # Experimental condition
    mode                      : varchar(128)           # mode of training
    stage                     : int                    # stage of training within mode
    trial_start               : float                  # timestamp within session for trial start
    trial_end                 : float                  # timestamp within session for trial end
    grace_period = NULL       : float                  # time in secs for grace period
    delay_period = NULL       : float                  # time in secs for delay period
    baited_port               : int                    # Baited port for current trial 1-right, 0-left
    stimulus_on = NULL        : tinyblob               # Timestamp for stimuli onset
    stimulus_off = NULL       : tinyblob               # Timestamp for stimuli offset
    curriculum_id             : int                    # integer identifier for curriculum applied in session
    categoryset_id            : int                    # Unique identifier for category set used in session
    sound_on = NULL           : tinyblob               # Timestamp for stimuli onset
    sound_off = NULL          : tinyblob               # Timestamp for stimuli offset
    sound_id = NULL           : tinyblob               # Vector of sounds id's presented for the task
    ports_on = NULL           : tinyblob               # timestamp for onset of response window
    ports_off = NULL          : tinyblob               # timestamp for offset of response window
    ports_id  = NULL          : varchar(128)           # vector of ports displayed in this trial
    port_layout               : int                    # boolean integer for one of two possible port layouts
    left_licks = NULL         : longblob               # Vector of timestamps when left port is licked
    right_licks = NULL        : longblob               # Vector of timestamps when right port is licked
    lick_disparity = NULL     : int                    # Difference of lick responses between ports
    reaction_time = NULL      : float                  # Timestamp for first decisive lick after image onset
    response                  : int                    # Response given for current trial 1-right, 0-left, -1-no response
    trialtype                 : varchar(128)           # Type of trial depending on licking response 
    responsetype              : varchar(128)           # vector for response type for all trials in session
    water = NULL              : tinyblob               # Vector of timestamps when water was delivered
    punish = NULL             : tinyblob               # Timestamp for punish timeout onset
    """
    
    def get_lastdate(self):
        """
        Gets the most recent date of entries for a given experiment.
        
        Args:
            experimentname (str): Name of the experiment
            
        Returns:
            str: Most recent date as string, or 0 if no entries exist
        """
        all_e = self 
        
        if len(all_e) == 0:
            return 0
        else:
            # Get all dates and sort them
            date_list = list(all_e.fetch('date'))
            date_list.sort()
            lastdate = date_list[-1]
            
            return str(lastdate)
        
    def reframe_time(self, vec, timeref):
        """
        Reframes timestamps relative to a reference time point.
        
        Args:
            vec (list): Vector of timestamps
            timeref (float): Reference time to subtract from all timestamps
            
        Returns:
            list: Reframed timestamps, or None if input is empty
        """
        if vec is None or vec == []:
            return None
        else:
            # Subtract reference time and round to 3 decimal places
            reframed = [round(x - timeref, 3) for x in vec]
            return reframed
    
    def extract_ids(self, sound_on, stimulus_on, ports_on, session):
        """
        Extracts sound and stimulus IDs corresponding to onset timestamps.
        
        Args:
            sound_on (list): Sound onset timestamps
            stimulus_on (list): Stimulus onset timestamps
            ports_on (list): Port onset timestamps
            session (dict): Session data dictionary
            
        Returns:
            tuple: (sound_id, stimulus_id) lists or None if not present
        """
        sound_ids = session['sound_id']
        stimulus_ids = session['stimulus_id']
        
        sound_id = []
        stimulus_id = []
        
        # Extract sound IDs matching onset timestamps
        if sound_on is not None:
            for s in sound_on:
                index = session['sound_on'].index(s)
                id_s = sound_ids[index]
                sound_id.append(id_s)
        else:
            sound_id = None
            
        # Extract stimulus IDs matching onset timestamps
        if stimulus_on is not None:
            for s in stimulus_on:
                index = session['stimulus_on'].index(s)
                id_st = stimulus_ids[index]
                stimulus_id.append(id_st)
        else:
            stimulus_id = None

        return sound_id, stimulus_id
    
    def parse_by_time(self, varvector, precut, postcut):
        """
        Extracts timestamps that fall within a specified time window.
        
        Args:
            varvector (list): Vector of timestamps
            precut (float): Start of time window
            postcut (float): End of time window
            
        Returns:
            list: Timestamps within the window, or None if no timestamps match
        """
        targetvector = varvector
        
        # Handle empty or None input
        if varvector is None or len(varvector) == 0:
            return None
        else:
            # If all timestamps are before window, return None
            if varvector[-1] < precut:
                return None
        
        # Find first timestamp after precut
        initialcut = 0
        for i in range(len(varvector)):
            initialcut = i
            if varvector[i] > precut:
                break

        # Handle edge case: only one element after precut
        if initialcut == len(varvector) - 1:
            if varvector[i] < postcut:
                return [varvector[i]]
            else:
                return None
        
        # Slice from first valid timestamp
        targetvector = varvector[initialcut:]
        
        # Find last timestamp before postcut
        finalcut = 0
        for j in range(len(targetvector)):
            finalcut = j
            if targetvector[j] > postcut:
                break
        
        # Slice to last valid timestamp
        if finalcut != len(targetvector) - 1:
            targetvector = targetvector[:finalcut]
        else:
            if targetvector[j] > postcut:
                targetvector = targetvector[:finalcut]
        
        # Return None if no timestamps in window
        if targetvector == []:
            return None
        else:
            return targetvector
            
    def extract_from_session(self, session_file):
        """
        Extracts individual trial data from a session entry.
        
        Args:
            experiment (str): Experiment name
            session_file (str): Session file identifier
        """
        # Fetch session data from database
        session = (Session() \
                  & 'file="%s"' % session_file).fetch(as_dict=True)
            
        print('Session: ', session['file'])
        
        # Define variable categories for extraction
        shared_vars = ['session_id', 'animal_id', 'name', 'condition', 'mode',
                       'stage', 'curriculum_id', 'categoryset_id',
                       'port_layout', 'date']
        
        vars_by_numtrial = ['new_trial', 'trial_end', 'baited_port',
                            'response', 'reaction_time', 'trialtype',
                            'responsetype', 'ports_id'] 
        
        vars_by_time = ['left_licks', 'right_licks', 'water',
                        'punish', 'stimulus_on', 'stimulus_off',
                        'sound_on', 'sound_off', 'ports_on', 'ports_off']
        
        # Extract shared variables (same for all trials)
        for sharedv in shared_vars:
            globals()[sharedv] = session[sharedv]

        num_trials = len(session['trial_end'])
        
        # Validate that trial-indexed variables have correct length
        for var_numtrial in vars_by_numtrial:
            if len(session[var_numtrial]) != num_trials:
                raise Exception('%s with different number of elements in session entry: %i instead of num_trials: %i' 
                              % (var_numtrial, len(session[var_numtrial]), num_trials))
        
        # Process each trial in the session
        for i in range(num_trials):
            trial_id = i + 1  # Trial IDs are 1-indexed
            
            # Extract trial-specific values
            for numv in vars_by_numtrial:
                if len(session[numv]) != 0:
                    globals()[numv] = session[numv][i]
                else:
                    globals()[numv] = None

            # Parse time-based events for this trial window
            for timev in vars_by_time:
                globals()[timev] = self.parse_by_time(session[timev], new_trial, trial_end)

            # Extract stimulus and sound IDs for this trial
            sound_id, stimulus_id = self.extract_ids(sound_on,
                                                     stimulus_on,
                                                     ports_on,
                                                     session)
            
            # Handle stimulus_id (take first if multiple)
            if stimulus_id is None:
                stimulus_id = '0'
            else:
                stimulus_id = stimulus_id[0]
                
            # Calculate grace and delay periods if stimulus was presented
            if stimulus_on is not None: 
                stim_dur = stimulus_off[-1] - stimulus_on[0]
                
                if ports_on is not None:
                    # Grace period: time from stimulus onset to port availability
                    grace_period = round(ports_on[0] - stimulus_on[0], 3)
                    if grace_period > stim_dur:
                        grace_period = round(stim_dur, 3)
                    
                    # Delay period: time from stimulus offset to port availability
                    delay_period = round(ports_on[0] - stimulus_off[-1], 3)
                    if delay_period <= 0:
                        delay_period = 0
                else:
                    grace_period = None
                    delay_period = None
            else:
                grace_period = None
                delay_period = None
                
            # Reframe all timestamps relative to trial start
            for v in vars_by_time:
                globals()[v] = self.reframe_time(globals()[v], new_trial)
                    
            # Calculate lick disparity (difference in lick counts between ports)
            if left_licks is None and right_licks is None:
                lick_disparity = None
            elif left_licks is not None and right_licks is not None:
                lick_disparity = abs(len(left_licks) - len(right_licks))
            else:
                # Only one port has licks
                if right_licks is None:
                    lick_disparity = len(left_licks)
                else:
                    lick_disparity = len(right_licks)
                    
            # Convert ports_id to readable format
            if ports_id == [1, 0]:
                id_ports = 'left'
            elif ports_id == [0, 1]:
                id_ports = 'right'
            elif ports_id == [1, 1]:
                id_ports = 'both'
            else:
                id_ports = 'none'
                
            # Insert trial into database
            self.insert1((trial_id, session_id, animal_id, name, stimulus_id,
                          str(date), condition,
                          mode, stage, new_trial,
                          trial_end, grace_period, delay_period, baited_port,
                          stimulus_on, stimulus_off, curriculum_id,
                          categoryset_id, sound_on,
                          sound_off, sound_id, ports_on, ports_off, id_ports,
                          port_layout, left_licks, right_licks,
                          lick_disparity, reaction_time, response, trialtype,
                          responsetype, water, punish), skip_duplicates=True)

    def load_experiment(self):
        """
        Populates the Trial table with data from all sessions in the Session table.
        Extracts individual trials from each session.
        """
        # Get all sessions from database
        all_sessions = Session() 
        j = 0  # Counter for newly added trials
        
        # Process each session with progress bar
        for i in tqdm(range(len(all_sessions))):
            session = all_sessions[i]
            animal_id = session['animal_id']
            session_id = session['session_id']
            
            # Check if trials for this session already exist
            entry = self & 'animal_id="%s"' % animal_id & 'session_id="%i"' % session_id
            
            if len(entry) == 0:
                j += 1
                self.extract_from_session(animal_id, session_id)


# ====== Convenience Functions ======

def load_session_file(session_file):
    """
    Load a single session file and extract its trials.
    
    Args:
        session_file (str): Path to session JSON file
    """
    Session().load_session(session_file)
    Trial().extract_from_session(session_file)


def load_allfiles(directory):
    """
    Load all session files from a directory and extract all trials.
    
    Args:
        directory (str): Path to directory containing session files
    """
    Session().load_experiment(directory)
    Trial().load_experiment()