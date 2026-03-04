#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba

# =============================================================================

# Statistical utility functions for behavioural / neural data analysis.
# Covers: sphericity testing, normality testing, bootstrap p-values,
# significance annotation, and entropy/time-series helpers.
# =============================================================================
"""
import numpy as np
from scipy.stats import chi2, shapiro
from statistics import mean, stdev


# ---------------------------------------------------------------------------
# Parametric Assumption Tests
# ---------------------------------------------------------------------------

def sphericity_test(data):
    """
    Mauchly's test for sphericity on repeated-measures data.

    Tests whether the variances of all pairwise differences between
    conditions are equal — a key assumption of repeated-measures ANOVA.

    Parameters
    ----------
    data : np.ndarray, shape (n_subjects, n_conditions)

    Returns
    -------
    sphericity : bool
        True if the sphericity assumption is not violated (p >= 0.05).
    """
    nr_subjects, nr_conditions = data.shape

    # Step 1: Build pairwise difference vectors for all condition pairs
    diffs = [
        data[:, i] - data[:, j]
        for i in range(nr_conditions)
        for j in range(i + 1, nr_conditions)
    ]
    cov_matrix = np.cov(diffs, rowvar=False)

    # Step 2: Mauchly's W — ratio of geometric mean to arithmetic mean of eigenvalues
    eigenvalues = np.linalg.eigvals(cov_matrix)
    n_eig = len(eigenvalues)
    W = np.prod(eigenvalues) / (np.sum(eigenvalues) / n_eig) ** n_eig

    # Step 3: Chi-square approximation of the W statistic
    k = nr_conditions
    chi_square = -(nr_subjects - 1) * np.log(W)
    dof = k * (k - 1) // 2 - 1

    # Step 4: Two-tailed p-value from chi-square distribution
    p_value = 1 - chi2.cdf(chi_square, dof)

    return p_value >= 0.05


def normality_test(datapoints, pval=False):
    """
    Test whether a sample is drawn from a normal distribution (Shapiro-Wilk).

    Parameters
    ----------
    datapoints : array-like, shape (n,)
        1-D vector of observations.
    pval : bool
        If True, also return the Shapiro-Wilk p-value.

    Returns
    -------
    normality : bool
        True if normality is not rejected (p >= 0.05).
    p_value : float  (only returned when pval=True)
    """
    result = shapiro(datapoints)
    normality = result.pvalue >= 0.05

    if pval:
        return normality, result.pvalue
    return normality


# ---------------------------------------------------------------------------
# Bootstrap Inference
# ---------------------------------------------------------------------------

def bootstrap_pval(x, null, stat_type='one-side larger'):
    """
    Compute a bootstrap p-value by comparing an observed statistic to a
    null distribution.

    The precision of the returned p-value is automatically matched to the
    resolution of the null distribution (1 / n_null_samples).

    Parameters
    ----------
    x : float
        Observed test statistic.
    null : array-like
        Null distribution of the same statistic under H0.
    stat_type : {'one-side larger', 'one-side smaller', 'two-sided'}

    Returns
    -------
    p_value : float
    """
    null = np.asarray(null)
    nr_null = len(null)

    # Determine decimal precision from null distribution size
    nr_decimals = len(str(nr_null)) - 1  # e.g. 1000 samples → 3 decimals
    value = round(x, nr_decimals)
    null_rounded = np.round(null, decimals=nr_decimals)

    if stat_type == 'one-side larger':
        p_value = np.mean(null_rounded > value)

    elif stat_type == 'one-side smaller':
        p_value = np.mean(null_rounded < value)

    elif stat_type == 'two-sided':
        # Centre the null distribution and compare absolute deviations
        shifted_null = null - np.mean(null)
        p_value = np.mean(np.abs(shifted_null) >= np.abs(x - np.mean(null)))

    else:
        raise ValueError(f"Unknown stat_type: '{stat_type}'")

    return round(p_value, nr_decimals)


def null_two_conditions(condition1, condition2, nr_iter=1000):
    """
    Generate a permutation null distribution for a two-condition comparison.

    Concatenates both conditions and repeatedly shuffles samples into two
    pseudo-groups of the original sizes to estimate the null distribution of
    any difference statistic.

    Parameters
    ----------
    condition1 : np.ndarray, shape (n, ...)
    condition2 : np.ndarray, shape (n, ...)
        Must have the same number of samples as condition1.
    nr_iter : int
        Number of permutation iterations.

    Returns
    -------
    null_dist : list of [pseudo_group1, pseudo_group2]
        Each element is a list of two arrays split from a shuffled pool.
    """
    assert len(condition1) == len(condition2), (
        "Both conditions must have the same number of samples."
    )

    n = len(condition1)
    all_samples = np.vstack((condition1, condition2))
    null_dist = []

    for _ in range(nr_iter):
        shuffled = all_samples.copy()
        np.random.shuffle(shuffled)
        null_dist.append([shuffled[:n], shuffled[n:]])

    return null_dist


# ---------------------------------------------------------------------------
# Significance Annotation
# ---------------------------------------------------------------------------

def p_to_ast(p):
    """
    Convert a p-value to an asterisk significance string.

    Parameters
    ----------
    p : float

    Returns
    -------
    str : '***', '**', '*', or 'ns'
    """
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def statistical_annotation(axs, comparisons, pvalues, ymax, ymin):
    """
    Draw bracket-and-asterisk significance annotations on a matplotlib axis.

    Annotations are stacked vertically above `ymax`, spaced by 10 % of the
    data range.

    Parameters
    ----------
    axs : matplotlib Axes
    comparisons : list of (x1, x2)
        Pairs of x-coordinates to connect with a bracket.
    pvalues : list of float
        One p-value per comparison.
    ymax : float
        Top of the data range (first bracket is drawn just above this).
    ymin : float
        Bottom of the data range (used to compute bracket spacing).

    Returns
    -------
    axs : matplotlib Axes  (modified in-place)
    ycoords : float
        y-coordinate of the last bracket drawn.
    """
    bracket_step = (ymax - ymin) * 0.1
    ycoords = ymax  # start just above the data

    for i, (comparison, pvalue) in enumerate(zip(comparisons, pvalues)):
        # Stack each successive bracket one step higher
        ycoords = ymax + bracket_step if i == 0 else ycoords + bracket_step

        # Map p-value to annotation string
        if pvalue >= 0.05:
            annot = 'n.s.'
        elif pvalue >= 0.01:
            annot = '*'
        elif pvalue >= 0.001:
            annot = '**'
        else:
            annot = '***'

        x1, x2 = comparison
        x_mid = (x1 + x2) * 0.5

        # Horizontal bracket line
        axs.plot([x1, x2], [ycoords, ycoords], lw=1.5, c='k')
        # Centred annotation text, slightly below the bracket
        axs.text(x_mid, ycoords - 0.2, annot,
                 ha='center', va='bottom', size=16, color='k')

    return axs, ycoords


# ---------------------------------------------------------------------------
# Time-Series Helpers
# ---------------------------------------------------------------------------

def get_asym_transpose(matrix):
    """
    Transpose a ragged (jagged) 2-D list without padding.

    In the input, each row is a variable's time series (possibly different
    lengths). In the output, each row collects all values at one timestep,
    skipping rows that don't reach that timestep.

    Parameters
    ----------
    matrix : list of lists
        Rows may have different lengths.

    Returns
    -------
    transpose : list of lists
        Length equals the longest row in the input.
    """
    n_timesteps = max(len(row) for row in matrix)
    transpose = []

    for t in range(n_timesteps):
        # Collect values from every row that has a t-th element
        timestep_vals = []
        for row in matrix:
            if t < len(row):
                timestep_vals.append(row[t])
        transpose.append(timestep_vals)

    return transpose


def get_stats_by_timepoints(matrix):
    """
    Compute mean and standard deviation across subjects at each timepoint.

    Parameters
    ----------
    matrix : list of lists
        Rows are subjects/samples; columns are timepoints (may be ragged).

    Returns
    -------
    ym  : list of float  — per-timepoint mean  (rounded to 2 d.p.)
    yst : list of float  — per-timepoint stdev (rounded to 2 d.p.)
    """
    transposed = get_asym_transpose(matrix)
    ym, yst = [], []

    for timepoint_vals in transposed:
        if len(timepoint_vals) > 1:
            ym.append(round(mean(timepoint_vals), 2))
            yst.append(round(stdev(timepoint_vals), 2))
        else:
            # Single observation: stdev is undefined, use 0
            ym.append(timepoint_vals[0])
            yst.append(0.0)

    return ym, yst


def get_stats_by_timecolumn(datapoints, alignment='last'):
    """
    Compute column-wise mean and std for ragged time series after NaN-padding.

    Rows of unequal length are padded with NaN so they align either to the
    first or last timestep before computing statistics.

    Parameters
    ----------
    datapoints : list of lists
        Each row is one sample's time series (variable length).
    alignment : {'last', 'first'}
        'last'  — pad the beginning so all rows end at the same timestep.
        'first' — pad the end so all rows start at the same timestep.

    Returns
    -------
    m   : list of float  — per-column mean  (NaN-safe)
    std : list of float  — per-column stdev (NaN-safe)
    """
    max_length = max(len(row) for row in datapoints)

    padded = []
    for row in datapoints:
        n_pad = max_length - len(row)
        if alignment == 'last':
            padded.append([np.nan] * n_pad + row)
        else:  # 'first'
            padded.append(row + [np.nan] * n_pad)

    arr = np.array(padded)
    m = np.nanmean(arr, axis=0).tolist()
    std = np.nanstd(arr, axis=0).tolist()

    return m, std


# ---------------------------------------------------------------------------
# Entropy Over Time
# ---------------------------------------------------------------------------

def entropy_in_time(timestampvector, timebin):
    """
    Compute the Shannon entropy of a timestamp distribution within a trial.

    Timestamps are binned at the given resolution and entropy is computed
    over the non-zero bins, giving a measure of how spread out (high entropy)
    or clustered (low entropy) events are across the trial.

    Parameters
    ----------
    timestampvector : list / array of float, or None
        Event timestamps (e.g. lick times) within a trial.
    timebin : float
        Bin width in the same time units as the timestamps.

    Returns
    -------
    entropy : float or None
        Shannon entropy (nats). Returns 0 for trivial cases (≤1 event),
        None if all timestamps are NaN.
    """
    # Trivial cases: no events or only one event → entropy = 0
    if timestampvector is None or len(timestampvector) <= 1:
        return 0

    # Remove NaN values and sort
    timestamps = sorted(t for t in timestampvector if not np.isnan(t))

    if len(timestamps) == 0:
        return None
    if len(timestamps) == 1:
        return 0

    # Build bin edges spanning the range of timestamps
    timespan = timestamps[-1] - timestamps[0]
    firstbintime = max(0.0, round(timestamps[0] - timebin / 2, 2))
    lastbintime = firstbintime + round(((timespan / timebin) + 1) * timebin, 2)

    bins = list(np.arange(firstbintime, lastbintime, timebin))
    bins.append(bins[-1] + timebin)  # ensure the last timestamp is included

    # Histogram and probability over non-zero bins
    counts, _ = np.histogram(timestampvector, bins=bins)
    non_zero_counts = counts[counts != 0]
    probabilities = non_zero_counts / non_zero_counts.sum()

    # Shannon entropy (nats)
    entropy = -(probabilities * np.log(np.abs(probabilities))).sum()

    return round(float(entropy), 2) if entropy != 0 else 0