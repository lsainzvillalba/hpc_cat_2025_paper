#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:43:03 2024

@author: Laura Sainz Villalba

# =============================================================================

# Simulates neural population geometry with multiple discrete states and
# evaluates linear decodability of task variables using an SVM classifier.
#
# States represent combinations of task variables (category, choice, outcome):
#   BL -> [1, 0, 0], AR -> [0, 1, 0], AL -> [0, 0, 1], BR -> [1, 1, 1]
# =============================================================================
"""

import copy
import numpy as np
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matplotlib colors for each task state
COLORS_DICT = {
    'BL': 'tab:cyan',
    'AR': 'tab:olive',
    'AL': 'tab:gray',
    'BR': 'tab:orange',
}

# Binary labels per state: [category, choice, outcome]
LABELS_DICT = {
    'BL': [1, 0, 0],
    'AR': [0, 1, 0],
    'AL': [0, 0, 1],
    'BR': [1, 1, 1],
}


# ---------------------------------------------------------------------------
# Geometry Construction
# ---------------------------------------------------------------------------

def create_centroid_directions(dim, n_dims, n_states):
    """
    Generate unit vectors pointing from a shared centroid toward each state.

    The first `dim` vectors are sampled independently and orthonormalised via
    normalisation. The remaining (n_states - dim) vectors are linear
    combinations of the independent set, ensuring the full matrix has rank
    exactly `dim` (i.e., states live in a `dim`-dimensional subspace).

    Parameters
    ----------
    dim : int
        Intrinsic dimensionality of the state geometry.
    n_dims : int
        Ambient dimensionality of the neural space.
    n_states : int
        Total number of discrete states.

    Returns
    -------
    centroid_dirs : np.ndarray, shape (n_states, n_dims)
        Unit vectors, one per state.
    """
    # Sample and normalise `dim` independent direction vectors
    centroid_vec_ind = np.random.uniform(-1, 1, size=(dim, n_dims))
    for i, vec in enumerate(centroid_vec_ind):
        centroid_vec_ind[i] = vec / np.linalg.norm(vec)

    centroid_dirs = centroid_vec_ind  # shape: (dim, n_dims)

    # Build the remaining directions as random linear combinations
    for _ in range(n_states - dim):
        coeffs = np.random.uniform(-1, 1, size=dim)
        new_vec = centroid_vec_ind.T @ coeffs          # linear combination
        new_vec = new_vec / np.linalg.norm(new_vec)    # re-normalise
        centroid_dirs = np.vstack((centroid_dirs, new_vec))

    assert np.linalg.matrix_rank(centroid_dirs) == dim, (
        "Rank of centroid_dirs does not match requested intrinsic dim."
    )

    return centroid_dirs  # shape: (n_states, n_dims)


def get_entropy_distribution(n_states, entropy_rel):
    """
    Sample a probability distribution whose entropy approximates a target.

    Iteratively resamples a uniform distribution until the Shannon entropy is
    within 0.1 bits of the desired level.

    Parameters
    ----------
    n_states : int
        Number of probability mass points (states).
    entropy_rel : float
        Desired entropy expressed as a fraction of maximum entropy log2(n_states).

    Returns
    -------
    probabilities : np.ndarray, shape (n_states,)
        Non-negative values whose entropy ≈ entropy_rel * log2(n_states).
        Note: values are not normalised to sum to 1.
    """
    max_entropy = np.log2(n_states)
    target_entropy = entropy_rel * max_entropy

    probabilities = np.random.uniform(0, 1, size=n_states)
    current_entropy = -np.sum(probabilities * np.log2(probabilities))

    # Rejection-sample until entropy is close enough to target
    while abs(current_entropy - target_entropy) > 1e-1:
        probabilities = np.random.uniform(0, 1, size=n_states)
        current_entropy = -np.sum(probabilities * np.log2(probabilities))

    return probabilities


def get_state_points(state_distances, centroid_dirs, geom_params):
    """
    Place state centroids in the ambient space at controlled distances.

    Each state is positioned along its centroid direction at a distance
    proportional to `state_distances`, scaled so the mean distance equals `r`.
    Gaussian noise is added afterwards.

    Parameters
    ----------
    state_distances : array-like, shape (n_states,)
        Relative distances from a shared centroid (e.g. from entropy distribution).
    centroid_dirs : np.ndarray, shape (n_states, n_dims)
        Unit direction vectors for each state.
    geom_params : dict
        Must contain: 'n_dims', 'r', 'n_states', 'sigma_noise'.

    Returns
    -------
    state_points : np.ndarray, shape (n_states, n_dims)
        Coordinates of each state's noisified centroid.
    """
    n_dims = geom_params['n_dims']
    r = geom_params['r']
    n_states = geom_params['n_states']
    sigma_noise = geom_params['sigma_noise']

    # Common origin from which all states radiate
    centroid = np.random.uniform(-1, 1, size=n_dims)

    assert len(state_distances) == n_states

    mean_dist = np.mean(state_distances)
    state_points = np.zeros((n_states, n_dims))

    # Scale each state so that the mean distance across states equals r
    for i, dist in enumerate(state_distances):
        magnitude = r * dist / mean_dist
        state_points[i] = magnitude * centroid_dirs[i] + centroid

    # Perturb state centroids with isotropic Gaussian noise
    noise = np.random.normal(0, sigma_noise, (n_states, n_dims))
    state_points += noise

    return state_points


def generate_geometry(geom_params):
    """
    Full pipeline: create centroid directions → distances → state centroids.

    Parameters
    ----------
    geom_params : dict
        Keys used: 'n_dims', 'dim', 'n_states', 'entropy_r',
                   'r', 'sigma_noise'.

    Returns
    -------
    state_points : np.ndarray, shape (n_states, n_dims)
    """
    n_dims = geom_params['n_dims']
    dim = geom_params['dim']
    n_states = geom_params['n_states']
    entropy_r = geom_params['entropy_r']

    centroid_dirs = create_centroid_directions(dim, n_dims, n_states)
    state_distances = get_entropy_distribution(n_states, entropy_r)
    state_points = get_state_points(state_distances, centroid_dirs, geom_params)

    return state_points


# ---------------------------------------------------------------------------
# Trial Cloud Generation
# ---------------------------------------------------------------------------

def generate_state_clouds(geom_params, state_points, state_sigmas):
    """
    Sample trial-level point clouds around each state centroid.

    Each state's cloud is drawn from a multivariate Gaussian centred on that
    state's point. The spread (sigma) of each cloud is derived from
    `state_sigmas` rescaled so its mean equals `sigma_state * r`.

    Parameters
    ----------
    geom_params : dict
        Keys used: 'n_trials', 'sigma_state', 'r', 'n_states', 'n_dims'.
    state_points : np.ndarray, shape (n_states, n_dims)
    state_sigmas : np.ndarray, shape (n_states,)
        Relative spread values (e.g. from entropy distribution).

    Returns
    -------
    X : np.ndarray, shape (n_states, n_trials, n_dims)
        Simulated trial data.
    """
    n_trials = geom_params['n_trials']
    sigma_state = geom_params['sigma_state']
    r = geom_params['r']
    n_states = geom_params['n_states']
    n_dims = geom_params['n_dims']

    # Target sigma is a fraction of the global scale r
    sigma = r * sigma_state
    mean_sigma = np.mean(state_sigmas)

    # Rescale so the mean state sigma equals the target
    state_sigmas = state_sigmas * sigma / mean_sigma

    X = []
    rs = np.random.RandomState(1234)  # fixed seed for reproducibility

    for i in range(n_states):
        # Isotropic covariance with state-specific spread
        cov = state_sigmas[i] * np.eye(n_dims)
        trials = rs.multivariate_normal(state_points[i], cov, size=n_trials)
        X.append(trials)

    return np.array(X)  # (n_states, n_trials, n_dims)


def generate_cloud(state_points, geom_params):
    """
    Convenience wrapper: sample trial clouds with entropy-distributed sigmas.

    Parameters
    ----------
    state_points : np.ndarray, shape (n_states, n_dims)
    geom_params : dict
        Keys used: 'entropy_sigma', 'n_states', plus all keys for
        generate_state_clouds.

    Returns
    -------
    state_clouds : np.ndarray, shape (n_states, n_trials, n_dims)
    """
    entropy_sigma = geom_params['entropy_sigma']
    n_states = geom_params['n_states']

    state_sigmas = get_entropy_distribution(n_states, entropy_sigma)
    return generate_state_clouds(geom_params, state_points, state_sigmas)


# ---------------------------------------------------------------------------
# Geometry Utilities
# ---------------------------------------------------------------------------

def get_centroid_vectors(state_points):
    """
    Compute the centroid of all state points and vectors from it to each state.

    Parameters
    ----------
    state_points : np.ndarray, shape (n_states, n_dims)

    Returns
    -------
    centroid : np.ndarray, shape (n_dims,)
    centroid_vectors : np.ndarray, shape (n_states, n_dims)
        Vectors from centroid to each state point.
    """
    centroid = np.mean(state_points, axis=0)
    centroid_vectors = state_points - centroid  # broadcasting over states
    return centroid, centroid_vectors


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_state_points(state_points, ax):
    """
    Scatter-plot the state centroids in 3-D with per-state colours.

    Parameters
    ----------
    state_points : np.ndarray, shape (n_states, 3)
    ax : matplotlib Axes3D

    Returns
    -------
    ax : matplotlib Axes3D (modified in-place)
    """
    state_labels = list(COLORS_DICT.keys())

    for i, label in enumerate(state_labels):
        ax.scatter3D(
            state_points[i, 0], state_points[i, 1], state_points[i, 2],
            c=COLORS_DICT[label], label=label, s=100,
        )

    ax.legend()
    # Hide tick labels for a cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    return ax


def plot_state_cloud(n_states, X, ax, colors_dict, state_labels):
    """
    Scatter-plot the trial clouds for each state in 3-D.

    Parameters
    ----------
    n_states : int
    X : np.ndarray, shape (n_states, n_trials, 3)
    ax : matplotlib Axes3D
    colors_dict : dict  {label: color}
    state_labels : list of str

    Returns
    -------
    ax : matplotlib Axes3D (modified in-place)
    """
    for i in range(n_states):
        ax.scatter3D(
            X[i, :, 0], X[i, :, 1], X[i, :, 2],
            c=colors_dict[state_labels[i]],
            s=10, alpha=0.3,
        )

    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_zlabel('p3')

    return ax


def plot_hyperplane(state_clouds, decoder_vector, decoder_intercept, ax):
    """
    Overlay the SVM decision hyperplane on a 3-D state-cloud plot.

    The plane is defined by:
        decoder_vector · x + decoder_intercept = 0
    solved for z given a grid of (x, y) values centred on the data.

    Parameters
    ----------
    state_clouds : np.ndarray, shape (n_states, n_trials, 3)
    decoder_vector : np.ndarray, shape (3,)
    decoder_intercept : float
    ax : matplotlib Axes3D

    Returns
    -------
    ax : matplotlib Axes3D (modified in-place)
    """
    # z as a function of x, y from the plane equation
    def z_from_plane(x, y):
        return (-decoder_intercept
                - decoder_vector[0] * x
                - decoder_vector[1] * y) / decoder_vector[2]

    # Centre the grid on the mean of all state centroids
    state_points = np.mean(state_clouds, axis=1)
    centroid, centroid_vectors = get_centroid_vectors(state_points)

    # Use mean distance to state centroids as a proxy for axis range
    norms = np.linalg.norm(centroid_vectors, axis=1)
    r = np.mean(norms)
    px, py, _ = centroid

    # Build a square grid around the centroid
    grid_x = np.linspace(px - r / 3, px + r / 3, 51)
    grid_y = np.linspace(py - r / 3, py + r / 3, 51)
    x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)

    ax.plot_surface(x_mesh, y_mesh, z_from_plane(x_mesh, y_mesh), alpha=0.4)

    return ax


# ---------------------------------------------------------------------------
# SVM Decoding
# ---------------------------------------------------------------------------

def prepare_dataset(geom_params, decoding_params, X):
    """
    Build train / test splits for decoding a single task variable.

    Two modes:
    - Standard (cross_var=None): split each state's trials by fraction.
    - Cross-variable: train on states where cross_var=1, test where cross_var=0.

    Parameters
    ----------
    geom_params : dict
        Keys: 'n_states', 'n_trials'.
    decoding_params : dict
        Keys: 'state_vars', 'decode_var', 'training_fraction', 'cross_var'.
    X : np.ndarray, shape (n_states, n_trials, n_dims)

    Returns
    -------
    train_tensor : np.ndarray, shape (n_train, n_dims)
    y_train      : np.ndarray, shape (n_train,)
    test_tensor  : np.ndarray, shape (n_test, n_dims)
    y_test       : np.ndarray, shape (n_test,)
    """
    n_states = geom_params['n_states']
    n_trials = geom_params['n_trials']
    state_vars = decoding_params['state_vars']
    decode_var = decoding_params['decode_var']
    training_fraction = decoding_params['training_fraction']
    cross_var = decoding_params['cross_var']

    nr_training_trials = int(training_fraction * n_trials)

    # Index of the variable to decode within each state's label vector
    label_idx = state_vars.index(decode_var)
    state_labels = list(LABELS_DICT.keys())

    assert X.shape == (n_states, n_trials, X.shape[2])

    train_tensor, test_tensor = [], []
    y_s_train, y_s_test = [], []

    if cross_var is None:
        # ------------------------------------------------------------------
        # Standard split: first `training_fraction` of each state → train
        # ------------------------------------------------------------------
        for i in range(n_states):
            label_val = LABELS_DICT[state_labels[i]][label_idx]
            np.random.shuffle(X[i])

            train_data = X[i, :nr_training_trials, :]
            test_data = X[i, nr_training_trials:, :]

            train_tensor.append(train_data)
            test_tensor.append(test_data)
            y_s_train.append(np.repeat(label_val, len(train_data)))
            y_s_test.append(np.repeat(label_val, len(test_data)))

    else:
        # ------------------------------------------------------------------
        # Cross-variable split: states where cross_var=1 → train,
        #                       states where cross_var=0 → test
        # ------------------------------------------------------------------
        cross_var_idx = state_vars.index(cross_var)

        for i in range(n_states):
            trial_label = LABELS_DICT[state_labels[i]]
            label_val = trial_label[label_idx]
            np.random.shuffle(X[i])

            # Subsample to nr_training_trials, then oversample 2× for balance
            data = X[i, :nr_training_trials, :]
            idx = np.random.choice(nr_training_trials, 2 * nr_training_trials)
            data = data[idx]

            if trial_label[cross_var_idx]:
                train_tensor.append(data)
                y_s_train.append(np.repeat(label_val, len(data)))
            else:
                test_tensor.append(data)
                y_s_test.append(np.repeat(label_val, len(data)))

    train_tensor = np.vstack(train_tensor)
    test_tensor = np.vstack(test_tensor)
    y_train = np.hstack(y_s_train)
    y_test = np.hstack(y_s_test)

    assert train_tensor.shape[0] == n_states * nr_training_trials

    return train_tensor, y_train, test_tensor, y_test


def svm_train(train_tensor, y_train, shuffle=False):
    """
    Fit a linear SVM classifier.

    Parameters
    ----------
    train_tensor : np.ndarray, shape (n_train, n_dims)
    y_train : np.ndarray, shape (n_train,)
    shuffle : bool
        If True, permute labels before fitting (null-distribution baseline).

    Returns
    -------
    svm_classifier : fitted LinearSVC instance
    """
    if shuffle:
        # Permute labels to generate a chance-level baseline
        y_train_labels = copy.deepcopy(y_train)
        np.random.shuffle(y_train_labels)
    else:
        y_train_labels = y_train

    svm_classifier = LinearSVC(
        dual=False, C=1.0, class_weight='balanced', max_iter=5000
    )
    svm_classifier.fit(train_tensor, y_train_labels)

    return svm_classifier


def svm_test(svm_classifier, test_tensor, y_test):
    """
    Evaluate a fitted SVM on held-out data.

    Parameters
    ----------
    svm_classifier : fitted LinearSVC
    test_tensor : np.ndarray, shape (n_test, n_dims)
    y_test : np.ndarray, shape (n_test,)

    Returns
    -------
    accuracy : float  (fraction of correct classifications)
    """
    return svm_classifier.score(test_tensor, y_test)


def decoding_var_parallel(geom_params, decoding_params, X, shuffle=False):
    """
    Train and evaluate an SVM decoder for one task variable.

    Optionally performs cross-variable decoding: trains on states where the
    cross variable equals 1 and tests on states where it equals 0 (and vice
    versa), then computes the cosine similarity between the two decoder axes.

    Parameters
    ----------
    geom_params : dict
    decoding_params : dict
        Keys: 'state_vars', 'decode_var', 'training_fraction', 'cross_var'.
    X : np.ndarray, shape (n_states, n_trials, n_dims)
    shuffle : bool
        Permute labels for a chance-level control.

    Returns
    -------
    results : dict with keys:
        'decoder_vec'  - SVM weight vector (primary decoder)
        'intercept'    - SVM bias (primary decoder)
        'accuracy'     - test accuracy (primary decoder)
        [if cross_var:]
        'decoder_vec_2', 'intercept_2', 'accuracy_2'
        'cosine'       - cosine similarity of the two decoder axes
    """
    cross_var = decoding_params['cross_var']

    train_tensor, y_train, test_tensor, y_test = prepare_dataset(
        geom_params, decoding_params, X
    )

    # Primary decoder: trained on train split, tested on test split
    svm = svm_train(train_tensor, y_train, shuffle=shuffle)
    accuracy = svm_test(svm, test_tensor, y_test)

    results = {
        'decoder_vec': svm.coef_[0],
        'intercept': svm.intercept_[0],
        'accuracy': accuracy,
    }

    if cross_var is not None:
        # Secondary decoder: trained on the complementary split
        svm_2 = svm_train(test_tensor, y_test, shuffle=shuffle)
        accuracy_2 = svm_test(svm_2, train_tensor, y_train)

        # Cosine similarity between the two (normalised) decoder axes
        v1 = svm.coef_[0] / np.linalg.norm(svm.coef_[0])
        v2 = svm_2.coef_[0] / np.linalg.norm(svm_2.coef_[0])

        results.update({
            'decoder_vec_2': svm_2.coef_[0],
            'intercept_2': svm_2.intercept_[0],
            'accuracy_2': accuracy_2,
            'cosine': np.dot(v1, v2),
        })

    return results