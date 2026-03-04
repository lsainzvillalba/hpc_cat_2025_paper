#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:07:48 2024

@author: Laura Sainz Villalba

# =============================================================================
# simulation.py
# Defines two DataJoint tables for simulating and summarising neural space
# decoding properties across a grid of geometry parameters.
#
# Main components:
#   - Simulation_neuralspace : runs full parallel decoding simulations
#   - Simulation_forstats    : stores simplified, stats-ready summaries
#   - simplify_entry         : converts a raw simulation row to a flat dict
# =============================================================================
"""

import os, sys, inspect
import numpy as np
from tqdm import tqdm
import datajoint as dj
from time import time
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# DataJoint configuration
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema = dj.schema('simulation_results_hpc_cat_2025', locals(), create_tables=True)

# ---------------------------------------------------------------------------
# Path setup – add parent directory so utility modules can be imported
# ---------------------------------------------------------------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utilities import (generate_geometry, get_pointcloud, enclosed_volume,
                       generate_cloud, decoding_var_parallel)

# ---------------------------------------------------------------------------
# Simulation parameter defaults
# (overridden per grid sweep inside simulate_all)
# ---------------------------------------------------------------------------
DEFAULT_GEOM_PARAMS = {
    'dim':           1,
    'r':             4,
    'sigma_state':   0.05,
    'entropy_r':     1,
    'entropy_sigma': 1,
    'n_dims':        3,
    'n_vars':        3,
    'n_states':      4,
    'sigma_noise':   0.02,
    'n_trials':      100,
}

DEFAULT_DECODING_PARAMS = {
    'training_fraction':  0.7,
    'decode_var':         'outcome',
    'nr_crossvalidations': 100,
    'cross_var':          None,
    'shuffle':            False,
    'state_vars':         ['category', 'choice', 'outcome'],
}


# =============================================================================
# Simulation_neuralspace – Manual DataJoint table
# Stores one row per unique geometry configuration with full decoding results.
# =============================================================================

@schema
class Simulation_neuralspace(dj.Manual):
    definition = """ # simulation of neural space decoding properties

    geometry_id       : int       # Unique geometry identifier
    ---
    dim               : int       # Number of task-relevant dimensions
    sigma_state       : float     # Mean state sigma relative to mean centroid distance
    r                 : float     # Mean centroid distance between states
    entropy_r         : float     # Entropy of centroid distance distribution
    entropy_sigma     : float     # Entropy of state sigma distribution
    state_clouds      : longblob  # Trial point clouds: (n_states x n_trials x n_dims)
    volume            : float     # Convex-hull volume spanned by state centroids
    cat_acc           : float     # Category decoding accuracy
    cat_cross_cho     : blob      # Category accuracy split by choice
    cat_cross_out     : blob      # Category accuracy split by outcome
    cho_acc           : float     # Choice decoding accuracy
    cho_cross_cat     : blob      # Choice accuracy split by category
    cho_cross_out     : blob      # Choice accuracy split by outcome
    out_acc           : float     # Outcome decoding accuracy
    out_cross_cat     : blob      # Outcome accuracy split by category
    out_cross_cho     : blob      # Outcome accuracy split by choice
    decoding_vectors  : longblob  # Mean decoder vectors and intercepts per variable
    cross_vectors     : longblob  # Mean cross-decoding vectors and intercepts
    cross_cosine      : longblob  # Cosine angles between decoder vectors (n_vars x n_cross)
    """

    # ------------------------------------------------------------------
    # Internal decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_parallel_decoding(geom_params, decoding_params, state_clouds,
                                nr_crossvalidations, nr_cores):
        """
        Run `nr_crossvalidations` parallel decoding iterations via multiprocessing.

        Parameters
        ----------
        geom_params : dict
            Geometry configuration passed to `decoding_var_parallel`.
        decoding_params : dict
            Decoding configuration (decode_var, cross_var, etc.).
        state_clouds : np.ndarray
            Trial point clouds of shape (n_states, n_trials, n_dims).
        nr_crossvalidations : int
            Number of cross-validation folds to run in parallel.
        nr_cores : int
            Number of worker processes for the multiprocessing pool.

        Returns
        -------
        list of dict
            One result dictionary per cross-validation fold.
        """
        args = [
            (geom_params, decoding_params, state_clouds, False)
            for _ in range(nr_crossvalidations)
        ]
        with Pool(nr_cores) as pool:
            return pool.starmap(decoding_var_parallel, args)

    @staticmethod
    def _aggregate_simple(results):
        """
        Aggregate simple-decoding results across cross-validation folds.

        Parameters
        ----------
        results : list of dict
            Output of `_run_parallel_decoding` for simple decoding.

        Returns
        -------
        mean_accuracy : float
            Mean decoding accuracy across folds.
        decoder_entry : list
            [mean_weight_vector, mean_intercept] averaged over folds.
        """
        accuracies   = [r['accuracy']     for r in results]
        decoder_vecs = [r['decoder_vec']  for r in results]
        intercepts   = [r['intercept']    for r in results]
        mean_accuracy = float(np.mean(accuracies))
        decoder_entry = [list(np.mean(decoder_vecs, axis=0)), float(np.mean(intercepts))]
        return mean_accuracy, decoder_entry

    @staticmethod
    def _aggregate_cross(results):
        """
        Aggregate cross-decoding results across cross-validation folds.

        Parameters
        ----------
        results : list of dict
            Output of `_run_parallel_decoding` for cross decoding.

        Returns
        -------
        mean_accs : np.ndarray, shape (2,)
            Mean accuracy for both decoding directions.
        mean_cosine : float
            Mean cosine angle between the two decoder vectors.
        cross_entry : list
            Nested list [[vec1, intercept1], [vec2, intercept2]] averaged over folds.
        """
        accs_1  = [r['accuracy']     for r in results]
        accs_2  = [r['accuracy_2']   for r in results]
        vecs_1  = [r['decoder_vec']  for r in results]
        vecs_2  = [r['decoder_vec_2'] for r in results]
        ints_1  = [r['intercept']    for r in results]
        ints_2  = [r['intercept_2']  for r in results]
        cosines = [r['cosine']       for r in results]

        mean_accs   = np.mean([accs_1, accs_2], axis=1)
        mean_cosine = float(np.mean(cosines))
        cross_entry = [
            [list(np.mean(vecs_1, axis=0)), float(np.mean(ints_1))],
            [list(np.mean(vecs_2, axis=0)), float(np.mean(ints_2))],
        ]
        return mean_accs, mean_cosine, cross_entry

    def _decode_all_vars(self, geom_params, decoding_params, state_clouds,
                         nr_crossvalidations, nr_cores):
        """
        Decode all state variables (category, choice, outcome) and their
        cross-splits, collecting accuracies, decoder vectors, and cosines.

        Parameters
        ----------
        geom_params : dict
            Geometry configuration.
        decoding_params : dict
            Decoding configuration; 'state_vars' controls which variables
            are decoded.
        state_clouds : np.ndarray
            Trial point clouds.
        nr_crossvalidations : int
            Number of CV folds per decode run.
        nr_cores : int
            Number of parallel worker processes.

        Returns
        -------
        single_decoding : np.ndarray, shape (n_vars,)
            Simple decoding accuracy per variable.
        cross_decoding : np.ndarray, shape (n_vars, n_vars-1, 2)
            Cross-decoding accuracy for each variable × cross-split × direction.
        decoding_vectors : list
            Mean decoder [vector, intercept] per variable.
        cross_vectors : list
            Nested mean cross-decoder vectors per variable × cross-split.
        cross_cosine : np.ndarray, shape (n_vars, n_vars-1)
            Mean cosine angles between decoder pairs.
        """
        state_vars = decoding_params['state_vars']
        n_vars     = len(state_vars)

        single_decoding  = np.zeros(n_vars)
        cross_decoding   = np.zeros((n_vars, n_vars - 1, 2))
        decoding_vectors = []
        cross_vectors    = []
        cross_cosine     = np.zeros((n_vars, n_vars - 1))

        for j, var in enumerate(state_vars):
            # Simple decoding for this variable
            decoding_params['decode_var'] = var
            decoding_params['cross_var']  = None
            results = self._run_parallel_decoding(
                geom_params, decoding_params, state_clouds, nr_crossvalidations, nr_cores
            )
            acc, dec_entry              = self._aggregate_simple(results)
            single_decoding[j]          = acc
            decoding_vectors.append(dec_entry)

            # Cross-decoding against each of the remaining variables
            cross_vars      = [v for v in state_vars if v != var]
            cross_vecs_var  = []
            for k, cross_var in enumerate(cross_vars):
                decoding_params['cross_var'] = cross_var
                results = self._run_parallel_decoding(
                    geom_params, decoding_params, state_clouds, nr_crossvalidations, nr_cores
                )
                mean_accs, mean_cosine, cross_entry = self._aggregate_cross(results)
                cross_decoding[j, k]  = mean_accs
                cross_cosine[j, k]    = mean_cosine
                cross_vecs_var.append(cross_entry)

            cross_vectors.append(cross_vecs_var)

        return single_decoding, cross_decoding, decoding_vectors, cross_vectors, cross_cosine

    # ------------------------------------------------------------------
    # Public simulation interface
    # ------------------------------------------------------------------

    def simulate_all(self):
        """
        Run the full parameter-grid simulation and populate the table.

        Sweeps over all combinations of dimensionality, noise sigma, centroid
        entropy, and sigma entropy. For each combination, generates a neural
        geometry, computes the convex-hull volume, and runs parallel decoding
        for all state variables and their cross-splits.

        Each unique (geometry_id) row encodes one parameter setting and stores
        the raw trial clouds, decoding accuracies, decoder vectors, and cosines.
        """
        geom_params     = DEFAULT_GEOM_PARAMS.copy()
        decoding_params = DEFAULT_DECODING_PARAMS.copy()

        # Sweep grid
        nr_cores            = 7
        nr_iter             = 500
        nr_crossvalidations = 100
        dims                = [1, 2, 3]
        sigmas              = np.linspace(0.005, 1.5, 4, endpoint=True)
        entropy_dist        = np.linspace(0, 1,   4, endpoint=True)
        entropy_sigma_vals  = np.linspace(0, 1,   4, endpoint=True)

        total_entries = (nr_iter * len(dims) * len(sigmas)
                         * len(entropy_dist) * len(entropy_sigma_vals))
        print(f'Total entries to simulate: {total_entries}')

        pbar          = tqdm(total=total_entries)
        entry_counter = len(Simulation_neuralspace())

        for _ in range(nr_iter):
            for d in dims:
                geom_params['dim'] = d

                for e_r in entropy_dist:
                    geom_params['entropy_r'] = e_r

                    for e_s in entropy_sigma_vals:
                        geom_params['entropy_sigma'] = e_s

                        # Generate state centroids and compute convex-hull volume
                        state_points = generate_geometry(geom_params)
                        points       = get_pointcloud(state_points, plot=False)
                        volume       = enclosed_volume(points)

                        for sigma in sigmas:
                            geom_params['sigma_state'] = sigma

                            # Sample trial clouds around the state centroids
                            state_clouds = generate_cloud(state_points, geom_params)

                            # Decode all variables and cross-splits
                            (single_decoding, cross_decoding,
                             decoding_vectors, cross_vectors,
                             cross_cosine) = self._decode_all_vars(
                                geom_params, decoding_params, state_clouds,
                                nr_crossvalidations, nr_cores
                            )

                            entry = {
                                'geometry_id':    entry_counter,
                                'dim':            d,
                                'sigma_state':    sigma,
                                'r':              geom_params['r'],
                                'entropy_r':      e_r,
                                'entropy_sigma':  e_s,
                                'state_clouds':   state_clouds,
                                'volume':         volume,
                                # Category decoding
                                'cat_acc':        single_decoding[0],
                                'cat_cross_cho':  cross_decoding[0, 0],
                                'cat_cross_out':  cross_decoding[0, 1],
                                # Choice decoding
                                'cho_acc':        single_decoding[1],
                                'cho_cross_cat':  cross_decoding[1, 0],
                                'cho_cross_out':  cross_decoding[1, 1],
                                # Outcome decoding
                                'out_acc':        single_decoding[2],
                                'out_cross_cat':  cross_decoding[2, 0],
                                'out_cross_cho':  cross_decoding[2, 1],
                                # Decoder geometry
                                'decoding_vectors': decoding_vectors,
                                'cross_vectors':    cross_vectors,
                                'cross_cosine':     cross_cosine,
                            }

                            start = time()
                            Simulation_neuralspace().insert1(entry, skip_duplicates=True)
                            print(f'TIME PER ENTRY: {(time() - start) / 60:.3f} min')

                            pbar.update(1)
                            entry_counter += 1


# =============================================================================
# simplify_entry – module-level helper
# =============================================================================

def simplify_entry(key):
    """
    Convert a raw Simulation_neuralspace row into a flat, stats-ready dict.

    Removes large blob fields, averages cross-decoding arrays to scalars,
    and extracts the two relevant cosine angles as named scalar fields.

    Parameters
    ----------
    key : dict
        DataJoint primary-key dict identifying the row to simplify.

    Returns
    -------
    dict
        Flattened entry suitable for insertion into Simulation_forstats.
    """
    # Fetch the full row and drop large storage blobs not needed for stats
    entry = (Simulation_neuralspace() & key).fetch(as_dict=True)[0]
    for blob_key in ['state_clouds', 'decoding_vectors', 'cross_vectors']:
        del entry[blob_key]

    state_vars = ['category', 'choice', 'outcome']

    # Replace each cross-decoding array with its scalar mean
    for var in state_vars:
        cross_vars = [v for v in state_vars if v != var]
        for cross_var in cross_vars:
            col = f'{var[:3]}_cross_{cross_var[:3]}'
            entry[col] = float(np.mean(entry[col]))

    # Extract the two cosine angles used in downstream statistics
    cross_cosine = entry.pop('cross_cosine')
    entry['cos_cho_cross_out'] = float(cross_cosine[1, 1])
    entry['cos_out_cross_cho'] = float(cross_cosine[2, 1])

    return entry


# =============================================================================
# Simulation_forstats – Manual DataJoint table
# Stores flattened, scalar summaries derived from Simulation_neuralspace.
# =============================================================================

@schema
class Simulation_forstats(dj.Manual):
    definition = """ # simplified simulation statistics (scalar summaries per geometry)

    geometry_id       : int     # Unique geometry identifier (matches Simulation_neuralspace)
    ---
    dim               : int     # Number of task-relevant dimensions
    sigma_state       : float   # Mean state sigma relative to mean centroid distance
    r                 : float   # Mean centroid distance between states
    entropy_r         : float   # Entropy of centroid distance distribution
    entropy_sigma     : float   # Entropy of state sigma distribution
    volume            : float   # Convex-hull volume spanned by state centroids
    cat_acc           : float   # Category decoding accuracy
    cat_cross_cho     : float   # Mean category accuracy split by choice
    cat_cross_out     : float   # Mean category accuracy split by outcome
    cho_acc           : float   # Choice decoding accuracy
    cho_cross_cat     : float   # Mean choice accuracy split by category
    cho_cross_out     : float   # Mean choice accuracy split by outcome
    out_acc           : float   # Outcome decoding accuracy
    out_cross_cat     : float   # Mean outcome accuracy split by category
    out_cross_cho     : float   # Mean outcome accuracy split by choice
    cos_cho_cross_out : float   # Cosine angle: choice decoder vs outcome cross-decoder
    cos_out_cross_cho : float   # Cosine angle: outcome decoder vs choice cross-decoder
    """

    def compute_all(self):
        """
        Populate this table by simplifying every row in Simulation_neuralspace.

        Iterates over all primary keys in Simulation_neuralspace, calls
        `simplify_entry` for each, and bulk-inserts the results.
        """
        entry_keys = Simulation_neuralspace().fetch(dj.key)
        entries    = []

        for key in tqdm(entry_keys):
            entries.append(simplify_entry(key))

        self.insert(entries, skip_duplicates=True)