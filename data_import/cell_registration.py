#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba
Registers uniquely identified neurons across recording sessions by reading
CaImAn / PCAICA component-matching JSON files and inserting the resulting
neuron-to-mask-ID mapping into the DataJoint table ``Neuron_registration``.

Pipeline overview
-----------------
1.  Load a JSON matching file produced by the cross-session registration tool
2.  Iterate over the ``assignments`` array:
      assignments[i][j] = k  means that mask k from session j was identified
      as the same physical neuron i across all sessions in the comparison
3.  Insert only fully-matched neurons (no None entries) with an auto-incremented
    neuron_id into ``Neuron_registration``
4.  ``register_experiment_from_files`` batch-processes every JSON file found in
    the experiment backup folder
"""

# import libraries
import datajoint as dj
import json
import os
from tqdm import tqdm

print("Calling validation_alignment.py from module script: ",__name__)

dj.config["enable_python_native_blobs"] = True
schema = dj.schema('cell_registration_hpc_cat_2025', locals(), create_tables = True)
print("Calling cell_registration.py from module script: ",__name__)


@schema
class Neuron_registration(dj.Manual):
    definition = """ # Neuron registration for uniquely identified components
    animal_id          : varchar(128)   # Mouse id (unique id)
    neuron_id          : int            # id for uniquely identified neuron across days of recording
    comparison         : varchar(128)   # starting id for experimental timepoint registration performed 
    ------
    session_ids        : longblob    # list of session ids registered
    mask_ids           : longblob    # list of mask ids in session ids corresponding to neuron id
    """

    # -----------------------------------------------------------------------
    def register_subject_from_file(self,matching_file):
        """
        Parse one JSON matching file and insert all fully-matched neurons.

        The JSON is expected to contain:
          - 'assignments' : list of lists
                assignments[i][j] = k  →  mask k in session j is neuron i
                A None entry means the neuron was not detected in that session.
          - 'animal_id'   : mouse identifier
          - 'comparison'  : label for the registration comparison performed
          - 'session_ids' : ordered list of session IDs covered by this file

        Only neurons that were detected in *every* session (no None values) are
        inserted; each receives a sequential neuron_id starting from 1.

        Parameters
        ----------
        matching_file : str
            Path to the JSON registration output file.
        """
        # ── Load JSON registration data ───────────────────────────────────────
        with open(matching_file) as f:
            matching_dict = json.load(f)

        # assignments[i] is a list of mask IDs (one per session) for neuron i
        assignments = matching_dict['assignments']

        # ── Iterate over all candidate neurons and insert matched ones ─────────
        neuron_id = 1   # sequential ID assigned only to fully-matched neurons
        for i in tqdm(range(len(assignments))):
            mask_ids = assignments[i] # components corresponding to unique component across sessions matched

            # Skip neurons that were not detected in at least one session
            if None not in mask_ids:
                entry_dict = {
                            'animal_id': matching_dict['animal_id'],
                            'neuron_id': neuron_id,
                            'comparison': matching_dict['comparison'],
                            'session_ids':matching_dict['session_ids'],
                            'mask_ids': mask_ids
                            }
                self.insert1(entry_dict,skip_duplicates=True)
                neuron_id +=1   # only increment for successfully inserted neurons

    # -----------------------------------------------------------------------
    def register_experiment_from_files(self,data_folder):
        """
        Batch-register all subjects in an experiment by processing every JSON
        matching file found in the experiment data directory.

        Files whose names start with 'B' (e.g. backup copies) are skipped.

        Parameters
        ----------
        data_folder : str
            Root path of the data storage.  The method looks for JSON files
            inside ``<data_folder>/data-hpc_cat_2025/``.
        """
        # ── Locate all valid registration JSON files ──────────────────────────
        path = data_folder + '/data-hpc_cat_2025'
        registration_files = os.listdir(path)
        # Exclude files starting with 'B' (backup / system artefacts)
        registration_files = [f for f in registration_files if not f.startswith('B')]

        # ── Process each file ─────────────────────────────────────────────────
        for file in registration_files:
            self.register_subject_from_file(file)
