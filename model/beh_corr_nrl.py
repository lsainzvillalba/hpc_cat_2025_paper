#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:28:35 2025

@author: Laura Sainz Villalba

https://gist.github.com/sachinsdate/5fae9fb94053ecef44426d026c471620

# =============================================================================
# beh_corr_nrl.py
# Binomial regression linking neural geometry parameters to behavioural
# performance across training phases.
#
# Main components:
#   - fit_binmodel        : fits a GLM binomial model with full diagnostics
#   - get_bin_nrl_dataset : assembles the neural + behavioural data matrix
#   - Bin_nrl_to_beh      : DataJoint table storing all model fits
#   - select_model        : post-hoc model selection helper
# =============================================================================
"""

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import os, sys, inspect
import datajoint as dj
import numpy as np
import pandas as pd
import math
import warnings

from patsy import dmatrices
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor, GLMInfluence

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup (only needed when run as a script)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir  = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

from decoding_population import Window_subject_decoder
from data_import import Trial
from utilities import trainingpoint_dict, get_property

# ---------------------------------------------------------------------------
# DataJoint configuration
# ---------------------------------------------------------------------------
schema = dj.schema('beh_corr_nrl_hpc_cat_2025', locals(), create_tables=True)

# Subjects included in the neural–behavioural dataset
ANIMAL_IDS = ['BK4926_L', 'BK4933_LR', 'BK4936_L',
              'BK4937_R', 'BK4947_R',  'BK4956_LR']

# Training phases and their reference phase for 'training' neural parameters
PHASES = ['discrimination', 'gentest_1', 'categorization_4', 'gentest_2']

# Regression formulas indexed by formula_id
FORMULAS = {
    0: 'Corr + Incorr ~ Cc_cho + Cc_out',
    1: 'Corr + Incorr ~ Cc_cho + Cc_out + Cc_cho:Cc_out',
    2: 'Corr + Incorr ~ Cc_cho_raw + Cc_out_raw',
    3: 'Corr + Incorr ~ Cc_cho_raw + Cc_out_raw + Cc_cho_raw:Cc_out_raw',
    4: 'Corr + Incorr ~ C(Animal_id) + Cc_cho + Cc_out + Cc_cho:Cc_out',
    5: 'Corr + Incorr ~ C(St_id) + Cc_cho + Cc_out + Cc_cho:Cc_out',
    6: 'Corr + Incorr ~ C(Animal_id) + Cc_cho_raw + Cc_out_raw + Cc_cho_raw:Cc_out_raw',
    7: 'Corr + Incorr ~ C(St_id) + Cc_cho_raw + Cc_out_raw + Cc_cho_raw:Cc_out_raw',
    8: 'Corr + Incorr ~ C(Animal_id) + C(St_id) + Cc_out + Cc_cho + Cc_cho:Cc_out',
    9: 'Corr + Incorr ~ C(Animal_id) + C(St_id) + Cc_cho_raw + Cc_out_raw + Cc_cho_raw:Cc_out_raw',
}

# =============================================================================
# Statistical helpers
# =============================================================================

def likelihood_ratio_test(diff_deviances, diff_params):
    """
    Compute the Wilks likelihood ratio test p-value between two nested models.

    Parameters
    ----------
    diff_deviances : float
        Deviance difference: reduced_deviance – full_deviance.
    diff_params : int
        Difference in number of parameters: full_params – reduced_params.

    Returns
    -------
    float
        p-value; values below 0.05 indicate the full model fits significantly
        better than the reduced model.
    """
    p_value = 1 - stats.chi2.cdf(diff_deviances, diff_params)
    return p_value


def outlier_detection(results):
    """
    Identify datapoints flagged as outliers on all three influence criteria.

    A point is considered an outlier only when it simultaneously exceeds the
    threshold for leverage, standardised residuals (> 2), and Cook's distance.

    Parameters
    ----------
    results : dict
        Output dictionary from `fit_binmodel` containing 'leverage',
        'std_residuals', 'cooks_d', 'df_model', and 'nr_datapoints'.

    Returns
    -------
    list of int
        Indices of outlier datapoints.
    """
    lev_threshold   = (2 * results['df_model']) / results['nr_datapoints']
    cooks_threshold = stats.chi2.ppf(0.5, results['df_model']) / results['df_model']

    leverage_flags  = [l > lev_threshold   for l in results['leverage']]
    std_res_flags   = [s > 2               for s in results['std_residuals']]
    cooks_flags     = [c > cooks_threshold for c in results['cooks_d']]

    # Outlier only when all three criteria are exceeded simultaneously
    return [i for i, (lev, std, cooks) in enumerate(zip(leverage_flags, std_res_flags, cooks_flags))
            if all([lev, std, cooks])]


def deviance_regressor(formula, data, null_deviance, cov_type):
    """
    Compute per-regressor deviance contribution via sequential term removal.

    For each term in the formula, fits a reduced model without that term and
    records null_deviance – reduced_deviance as the term's deviance contribution.
    Interaction terms whose component regressors are absent in the reduced model
    are automatically removed.

    Parameters
    ----------
    formula : str
        Patsy formula string of the full model.
    data : pd.DataFrame
        Data used to fit the models.
    null_deviance : float
        Deviance of the intercept-only (null) model.
    cov_type : str or None
        Covariance type passed to `GLM.fit`.

    Returns
    -------
    list of float
        Deviance contribution for each term in the formula (same order as terms).
    """
    y_response = formula.split(' ~ ')[0]
    terms      = formula.split(' ~ ')[1].split(' + ')
    deviance_diff = []

    for term in terms:
        reduced_terms = terms.copy()
        reduced_terms.remove(term)

        # Remove dangling interaction terms whose main effects are no longer present
        for interaction in [t for t in reduced_terms if ':' in t]:
            components = interaction.split(':')
            if not all(c in reduced_terms for c in components):
                reduced_terms.remove(interaction)

        if not reduced_terms:
            deviance_diff.append(0)
            continue

        reduced_formula = y_response + ' ~ ' + ' + '.join(reduced_terms)
        y_red, X_red    = dmatrices(reduced_formula, data, return_type='dataframe')
        reduced_results = sm.GLM(y_red, X_red, family=sm.families.Binomial()).fit()
        deviance_diff.append(null_deviance - reduced_results.deviance)

    return deviance_diff


# =============================================================================
# Model fitting
# =============================================================================

def _sanitise_inf(results, keys):
    """Replace inf values with None for a list of result keys."""
    for key in keys:
        if key in results and results[key] is not None and math.isinf(results[key]):
            results[key] = None


def fit_binmodel(formula, df_data, cov_type, debug=False):
    """
    Fit a binomial GLM and return a comprehensive diagnostics dictionary.

    Computes model coefficients, goodness-of-fit statistics, influence
    diagnostics, in-sample MSE, and leave-one-out cross-validation (LOOCV)
    MSE. Optionally fits a reduced model (without the interaction term) and
    performs a likelihood ratio test.

    Parameters
    ----------
    formula : str
        Patsy formula string (e.g. 'Corr + Incorr ~ Cc_cho + Cc_out').
    df_data : pd.DataFrame
        Dataset; must contain columns referenced in *formula* plus
        'Fr_corr', 'Cc_cho_raw', and 'Cc_out_raw'.
    cov_type : str or None
        Covariance estimator for `GLM.fit` (e.g. 'HC3', or None for default).
    debug : bool, optional
        If True, prints the full model summary.

    Returns
    -------
    dict
        Dictionary containing all model statistics and diagnostics.
    """
    results       = {}
    nr_datapoints = len(df_data)

    # ------------------------------------------------------------------
    # Descriptive statistics of the response and key regressors
    # ------------------------------------------------------------------
    results['response_mean']  = float(np.mean(df_data['Fr_corr']))
    results['response_var']   = float(np.var(df_data['Fr_corr']))
    results['cc_cho_mean']    = float(np.mean(df_data['Cc_cho_raw']))
    results['cc_cho_std']     = float(np.std(df_data['Cc_cho_raw']))
    results['cc_out_mean']    = float(np.mean(df_data['Cc_out_raw']))
    results['cc_out_std']     = float(np.std(df_data['Cc_out_raw']))
    results['nr_datapoints']  = nr_datapoints

    # Intraclass correlation coefficient (ICC) – measures clustering by animal
    grouped             = df_data.groupby('Animal_id')['Fr_corr']
    between_var         = grouped.mean().var()
    within_var          = grouped.var().mean()
    results['icc_approx'] = float(between_var / (between_var + within_var))

    # ------------------------------------------------------------------
    # Fit the full model
    # ------------------------------------------------------------------
    y, X = dmatrices(formula, df_data, return_type='dataframe')
    bin_mod = sm.GLM(y, X, family=sm.families.Binomial())
    fit_kwargs = {'cov_type': cov_type} if cov_type is not None else {}
    model_fit  = bin_mod.fit(**fit_kwargs)

    if debug:
        print(model_fit.summary())

    # ------------------------------------------------------------------
    # Coefficient and fit-quality statistics
    # ------------------------------------------------------------------
    results['reg_names']       = list(model_fit.params.keys())
    results['coefs']           = list(model_fit.params)
    results['pvalues']         = list(model_fit.pvalues)
    results['conf_int']        = [list(model_fit.conf_int()[0]),
                                   list(model_fit.conf_int()[1])]
    results['pseudo_rsquared'] = float(model_fit.pseudo_rsquared(kind='cs'))

    llf      = float(model_fit.llf)
    df_model = int(model_fit.df_model)
    df_resid = int(model_fit.df_resid)
    deviance      = float(model_fit.deviance)
    null_deviance = float(model_fit.null_deviance)

    results['llf']              = llf
    results['df_model']         = df_model
    results['df_resid']         = df_resid
    results['aic']              = float(model_fit.aic)
    results['bic']              = -2 * llf + (df_model + 1) * np.log(nr_datapoints)
    results['deviance']         = deviance
    results['null_deviance']    = null_deviance
    results['dev_residuals']    = list(model_fit.resid_deviance)
    results['fitted_values']    = list(model_fit.fittedvalues)

    deviance_explained          = null_deviance - deviance
    results['deviance_explained'] = deviance_explained
    results['deviance_param']     = deviance_explained / df_model

    # Deviance ratio: ~ 1 is ideal; > 1 = overdispersed, < 1 = underdispersed
    deviance_ratio              = deviance / df_resid
    results['deviance_ratio']   = deviance_ratio
    if deviance_ratio > 1:
        results['deviance_ratio_pvalue'] = 1 - stats.chi2.cdf(deviance, df_resid)
    else:
        results['deviance_ratio_pvalue'] = stats.chi2.cdf(deviance, df_resid)

    results['deviance_reg'] = deviance_regressor(formula, df_data, null_deviance, cov_type)
    results['lrt_null']     = likelihood_ratio_test(deviance_explained, df_model - 1)

    # ------------------------------------------------------------------
    # Reduced model (drop interaction term if present)
    # ------------------------------------------------------------------
    if ':' in formula:
        y_response = formula.split(' ~ ')[0]
        terms      = formula.split(' ~ ')[1].split(' + ')
        interaction_terms = [[t, i] for i, t in enumerate(terms) if ':' in t]
        assert len(interaction_terms) == 1, "Only one interaction term is supported."

        interaction, idx = interaction_terms[0]
        reg1, reg2       = interaction.split(':')

        if reg1 in terms and reg2 in terms:
            # Build reduced formula without the interaction term
            reduced_terms   = [t for i, t in enumerate(terms) if i != idx]
            reduced_formula = y_response + ' ~ ' + ' + '.join(reduced_terms)

            y_red, X_red      = dmatrices(reduced_formula, df_data, return_type='dataframe')
            reduced_fit_kwargs = {'cov_type': cov_type} if cov_type is not None else {}
            reduced_fit       = sm.GLM(y_red, X_red, family=sm.families.Binomial()).fit(**reduced_fit_kwargs)

            reduced_deviance        = float(reduced_fit.deviance)
            results['reduced_deviance'] = reduced_deviance
            results['lrt_reduced']      = likelihood_ratio_test(
                reduced_deviance - deviance, df_model - int(reduced_fit.df_model)
            )
        else:
            results['reduced_deviance'] = None
            results['lrt_reduced']      = None
    else:
        results['reduced_deviance'] = None
        results['lrt_reduced']      = None

    # ------------------------------------------------------------------
    # Influence diagnostics
    # ------------------------------------------------------------------
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    results['vif']            = vif
    results['collinear_bool'] = int(any(v > 10 for v in vif))

    influence      = GLMInfluence(model_fit)
    leverage       = influence.results.get_hat_matrix_diag()
    lev_threshold  = (2 * df_model) / nr_datapoints
    results['leverage']      = leverage
    results['leverage_bool'] = int(any(l > lev_threshold for l in leverage))

    std_residuals            = influence.resid / np.sqrt(influence.scale * (1 - leverage))
    results['std_residuals'] = std_residuals
    results['std_bool']      = int(any(s > 2 for s in std_residuals))

    cooks_d                  = (std_residuals ** 2 / influence.k_vars) * (leverage / (1 - leverage))
    cooks_threshold          = stats.chi2.ppf(0.5, df_model) / df_model
    results['cooks_d']         = cooks_d
    results['influence_bool']  = int(any(c > cooks_threshold for c in cooks_d))

    # ------------------------------------------------------------------
    # Replace inf values with None (DataJoint cannot store inf)
    # ------------------------------------------------------------------
    _sanitise_inf(results, ['pseudo_rsquared', 'aic', 'bic', 'llf'])
    if results.get('deviance') is None or math.isinf(results['deviance']):
        for key in ['deviance', 'deviance_explained', 'deviance_param',
                    'deviance_ratio', 'deviance_reg']:
            results[key] = None

    # ------------------------------------------------------------------
    # In-sample MSE
    # ------------------------------------------------------------------
    fr_corr        = np.array(df_data['Fr_corr'])
    fitted_values  = np.array(results['fitted_values'])
    results['mse_in_sample'] = float(np.mean((fr_corr - fitted_values) ** 2))

    # ------------------------------------------------------------------
    # Leave-one-out cross-validation (LOOCV) MSE
    # ------------------------------------------------------------------
    loocv_predictions = []
    for i in range(nr_datapoints):
        train_data = df_data.drop(df_data.index[i])
        test_data  = df_data.iloc[[i]]

        y_train, X_train = dmatrices(formula, train_data, return_type='dataframe')
        loocv_fit        = sm.GLM(y_train, X_train,
                                  family=sm.families.Binomial()).fit(**fit_kwargs)

        # Align test columns to training columns (fill missing with 0)
        _, X_test       = dmatrices(formula, test_data, return_type='dataframe')
        X_test_aligned  = pd.DataFrame(0, index=X_test.index, columns=X.columns)
        for col in X_test.columns:
            if col in X_test_aligned.columns:
                X_test_aligned[col] = X_test[col]

        loocv_predictions.append(loocv_fit.predict(X_test_aligned).iloc[0])

    errors                    = fr_corr - np.array(loocv_predictions)
    mse_loocv                 = float(np.mean(errors ** 2))
    mse_loocv_std             = float(np.std(errors ** 2))
    mse_gap                   = mse_loocv - results['mse_in_sample']
    results['mse_loocv']      = mse_loocv
    results['mse_loocv_std']  = mse_loocv_std
    results['mse_ratio']      = mse_gap / results['mse_in_sample']
    results['mse_ratio_std']  = mse_gap / mse_loocv_std

    return results


# =============================================================================
# Dataset assembly
# =============================================================================

def _fetch_neural_params(entry):
    """
    Extract choice- and outcome-aligned CCGP and PS values from a decoder entry.

    Parameters
    ----------
    entry : DataJoint table expression
        Single-row Window_subject_decoder entry.

    Returns
    -------
    tuple of float
        (cc_cho, cc_out, ps_cho, ps_out)
    """
    ccgp = np.mean(np.array(entry.fetch('ccgp')[0]), axis=0)
    ps   = np.mean(np.array(entry.fetch('ps')[0]),   axis=0)
    return (
        ccgp[1, 1],
        ccgp[2, 1],
        np.cos(np.deg2rad(ps[1, 1])),
        np.cos(np.deg2rad(ps[2, 1])),
    )


def get_bin_nrl_dataset(window, window_length):
    """
    Assemble the neural–behavioural dataset for binomial regression.

    For each subject and training phase, collects:
      - CCGP and PS neural geometry parameters (current and training-phase)
      - Correct / incorrect trial counts across sessions

    Then standardises CCGP columns and computes derived interaction terms.

    Parameters
    ----------
    window : str
        Decoder window label (e.g. 'post_choice', 'post_outcome').
    window_length : float
        Window duration in seconds; matched with ±0.1 tolerance.

    Returns
    -------
    pd.DataFrame
        Assembled and standardised dataset ready for `fit_binmodel`.
    """
    # Verify a unique event-alignment label exists for this window
    events = get_property(Window_subject_decoder() & f'window="{window}"', 'event_align')
    assert len(events) == 1, "Expected exactly one event_align value for this window."
    event_align = events[0]

    # Convenience filter for window_length with floating-point tolerance
    wl_filter = (f'window_length>="{window_length - 0.1}"'
                 f' AND window_length<"{window_length + 0.1}"')

    columns = ['Animal_id', 'St_id', 'Corr', 'Incorr',
               'Cc_out', 'Cc_cho', 'Cc_out_tr', 'Cc_cho_tr',
               'Ps_cho', 'Ps_out', 'Ps_cho_tr', 'Ps_out_tr']
    data = {col: [] for col in columns}

    for i, phase in enumerate(PHASES):
        # For generalisation test phases use the preceding phase as 'training' reference
        training_phase = PHASES[i - 1] if phase in ('gentest_1', 'gentest_2') else None
        mode, stage, categoryset_id = trainingpoint_dict[phase]

        for animal_id in ANIMAL_IDS:
            # Query current-phase decoder entry
            entry = (
                Window_subject_decoder()
                & f'experimental_timepoint="{phase}"'
                & f'event_align="{event_align}"'
                & f'window="{window}"'
                & wl_filter
                & f'animal_id="{animal_id}"'
            )
            if len(entry) == 0:
                continue

            session_ids              = entry.fetch('session_ids')[0]
            cc_cho, cc_out, ps_cho, ps_out = _fetch_neural_params(entry)

            # Query training-phase neural parameters (NaN if not applicable)
            if training_phase is not None:
                entry_tr = (
                    Window_subject_decoder()
                    & f'experimental_timepoint="{training_phase}"'
                    & f'event_align="{event_align}"'
                    & f'window="{window}"'
                    & wl_filter
                    & f'animal_id="{animal_id}"'
                )
                if len(entry_tr) != 0:
                    cc_cho_tr, cc_out_tr, ps_cho_tr, ps_out_tr = _fetch_neural_params(entry_tr)
                else:
                    cc_cho_tr = cc_out_tr = ps_cho_tr = ps_out_tr = np.nan
            else:
                cc_cho_tr = cc_out_tr = ps_cho_tr = ps_out_tr = np.nan

            # Accumulate correct / incorrect trials across all sessions in this phase
            total_trials = corr_trials = 0
            for session_id in session_ids:
                valid = (
                    Trial()
                    & f'mode="{mode}"' & f'stage="{stage}"'
                    & f'categoryset_id="{categoryset_id}"'
                    & f'animal_id="{animal_id}"'
                    & f'session_id="{session_id}"'
                    & 'trialtype="active"'
                    & 'responsetype!="no response"'
                )
                total_trials += len(valid)
                corr_trials  += len(valid & 'responsetype="correct"')

            data['Animal_id'].append(animal_id)
            data['St_id'].append(phase)
            data['Corr'].append(corr_trials)
            data['Incorr'].append(total_trials - corr_trials)
            data['Cc_out'].append(cc_out);   data['Cc_cho'].append(cc_cho)
            data['Ps_out'].append(ps_out);   data['Ps_cho'].append(ps_cho)
            data['Cc_out_tr'].append(cc_out_tr); data['Cc_cho_tr'].append(cc_cho_tr)
            data['Ps_out_tr'].append(ps_out_tr); data['Ps_cho_tr'].append(ps_cho_tr)

    df = pd.DataFrame(data)
    df['Total']   = df['Corr'] + df['Incorr']
    df['Fr_corr'] = df['Corr'] / df['Total']

    # Preserve raw (unstandardised) copies for formula variants
    df['Cc_out_raw'] = df['Cc_out']
    df['Cc_cho_raw'] = df['Cc_cho']
    df['Cc_int_raw'] = df['Cc_cho_raw'] * df['Cc_out_raw']

    # Z-score CCGP columns
    df['Cc_out'] = (df['Cc_out'] - df['Cc_out'].mean()) / df['Cc_out'].std()
    df['Cc_cho'] = (df['Cc_cho'] - df['Cc_cho'].mean()) / df['Cc_cho'].std()
    df['Cc_int'] = df['Cc_cho'] * df['Cc_out']

    return df


# =============================================================================
# Bin_nrl_to_beh – Manual DataJoint table
# =============================================================================

@schema
class Bin_nrl_to_beh(dj.Manual):
    definition = """ # binomial regression predicting performance from neural geometry parameters

    run_id          : int            # Unique run identifier
    window          : varchar(128)   # Decoder window (post_choice, post_outcome)
    window_length   : float          # Window duration (seconds)
    formula_id      : int            # Regression formula ID (see FORMULAS)
    ---
    cov_type=NULL   : varchar(128)   # Covariance type (HC3 or NULL for default)
    formula         : varchar(256)   # Full Patsy formula string
    nr_datapoints   : int            # Number of observations
    response_mean   : float          # Mean fraction correct
    response_var    : float          # Variance of fraction correct
    cc_cho_mean     : float          # Raw mean of CCGP (choice split)
    cc_out_mean     : float          # Raw mean of CCGP (outcome split)
    cc_cho_std      : float          # Raw std of CCGP (choice split)
    cc_out_std      : float          # Raw std of CCGP (outcome split)
    icc_approx      : float          # Approximate ICC (clustering by animal; target < 0.2–0.3)
    reg_names       : blob           # Regressor names
    coefs           : blob           # Fitted log-odds coefficients
    pvalues         : blob           # Coefficient p-values
    conf_int        : blob           # 95% confidence intervals [lower, upper]
    pseudo_rsquared=NULL : float     # Cox–Snell pseudo R²
    aic=NULL             : float     # Akaike Information Criterion
    bic=NULL             : float     # Bayesian Information Criterion
    llf=NULL             : float     # Log-likelihood
    df_resid        : int            # Residual degrees of freedom
    df_model        : int            # Model degrees of freedom
    deviance=NULL          : float   # Model deviance
    null_deviance          : float   # Null (intercept-only) model deviance
    lrt_null               : float   # LRT p-value vs null model
    reduced_deviance=NULL  : float   # Deviance of reduced model (no interaction)
    lrt_reduced=NULL       : float   # LRT p-value vs reduced model
    deviance_explained=NULL : float  # null_deviance – deviance
    deviance_param=NULL     : float  # deviance_explained / df_model
    deviance_reg=NULL       : blob   # Per-regressor deviance contribution
    deviance_ratio=NULL     : float  # deviance / df_resid (~1 is ideal)
    deviance_ratio_pvalue=NULL : float  # p-value of deviance ratio departure from 1
    dev_residuals   : blob           # Per-observation deviance residuals
    fitted_values   : blob           # Per-observation fitted values
    vif             : blob           # Variance inflation factors
    leverage        : blob           # Leverage values
    std_residuals   : blob           # Standardised residuals
    cooks_d         : blob           # Cook's distances
    collinear_bool  : int            # Any VIF > 10?
    leverage_bool   : int            # Any leverage > 2k/n?
    std_bool        : int            # Any |std_residual| > 2?
    influence_bool  : int            # Any Cook's d above threshold?
    outliers=NULL   : blob           # Indices of removed outliers
    mse_in_sample   : float          # In-sample MSE
    mse_loocv       : float          # LOOCV MSE
    mse_loocv_std   : float          # Std of LOOCV MSE
    mse_ratio       : float          # (LOOCV – in-sample) / in-sample
    mse_ratio_std   : float          # mse_gap / mse_loocv_std
    """

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _next_run_id(self):
        """Return the next available run_id (0 if table is empty)."""
        return int(max(self.fetch('run_id'))) + 1 if len(self) else 0

    def _insert_fit(self, run_id, entry_key_dict, results):
        """
        Merge key fields with results dict and insert one row.

        Parameters
        ----------
        run_id : int
            Unique run identifier for this entry.
        entry_key_dict : dict
            Primary-key fields (window, window_length, formula_id, etc.).
        results : dict
            Output of `fit_binmodel`.

        Returns
        -------
        int
            Incremented run_id ready for the next insertion.
        """
        entry = {**entry_key_dict, **results, 'run_id': run_id}
        self.insert1(entry, skip_duplicates=True)
        return run_id + 1

    def populate_phase(self, window, window_length, df):
        """
        Fit all formula × covariance-type combinations and store results.

        For each combination, detects outliers and inserts both a version with
        outliers removed and (subsequently) the full-data version.

        Parameters
        ----------
        window : str
            Decoder window label.
        window_length : float
            Window duration in seconds.
        df : pd.DataFrame
            Full assembled dataset from `get_bin_nrl_dataset`.
        """
        cov_types = [None, 'HC3']
        run_id    = self._next_run_id()

        for cov_type in cov_types:
            print(f'cov_type: {cov_type}')
            for formula_id, formula in FORMULAS.items():
                print(f'formula_id: {formula_id}')

                entry_key = dict(
                    window        = window,
                    window_length = window_length,
                    formula_id    = formula_id,
                    cov_type      = cov_type,
                    formula       = formula,
                )

                # For formulas referencing training-phase regressors, keep only
                # the generalisation test rows (which have non-NaN training data)
                if 'Cc_out_tr' in formula:
                    df_data = pd.concat([
                        df[df['St_id'] == 'gentest_1'],
                        df[df['St_id'] == 'gentest_2'],
                    ], axis=0)
                else:
                    df_data = df

                results  = fit_binmodel(formula, df_data, cov_type)
                results['nr_datapoints'] = len(df_data)
                outliers = outlier_detection(results)

                if outliers:
                    # Insert a clean model fit without the detected outliers
                    df_clean         = df_data.drop(df_data.index[outliers])
                    results_clean    = fit_binmodel(formula, df_clean, cov_type)
                    results_clean['outliers']      = None
                    results_clean['nr_datapoints'] = len(df_clean)
                    run_id = self._insert_fit(run_id, entry_key, results_clean)

                # Insert the full-data model (outliers field records their indices)
                results['outliers'] = outliers if outliers else None
                run_id = self._insert_fit(run_id, entry_key, results)

    def update(self):
        """
        Populate the table for all windows and window lengths not yet processed.

        Iterates over 'post_choice' and 'post_outcome' windows, finds all
        available window lengths, and calls `populate_phase` for any
        (window, window_length) pair not already in the table.
        """
        for window in ['post_choice', 'post_outcome']:
            print(f'window: {window}')
            window_lengths = get_property(
                Window_subject_decoder() & f'window="{window}"', 'window_length'
            )
            for window_length in window_lengths:
                print(f'window_length: {window_length}')
                existing = self & f'window="{window}"' & f'window_length={window_length}'
                if len(existing) == 0:
                    df = get_bin_nrl_dataset(window, window_length)
                    self.populate_phase(window, window_length, df)


# =============================================================================
# Model selection
# =============================================================================

def select_model(window_length, window='post_choice', cov_type='HC3'):
    """
    Filter and rank fitted models by BIC after applying quality criteria.

    Filters to models that:
      - Pass the collinearity check (VIF < 10 for all regressors)
      - Show significant improvement over the null model (LRT p < 0.05)
      - Show significant improvement over the reduced model (LRT p < 0.05)

    Parameters
    ----------
    window_length : float
        Window length matched with ±0.1 tolerance.
    window : str, optional
        Decoder window label (default 'post_choice').
    cov_type : str, optional
        Covariance type to filter on (default 'HC3').

    Returns
    -------
    pd.DataFrame
        Selected models sorted by BIC (ascending).
    """
    wl_filter = (f'window_length>"{window_length - 0.1}" '
                 f'AND window_length<"{window_length + 0.1}"')

    candidates = (
        Bin_nrl_to_beh()
        & f'cov_type="{cov_type}"'
        & f'window="{window}"'
        & wl_filter
        & 'collinear_bool="0"'     # no multicollinearity
        & 'lrt_null<"0.05"'        # significantly better than null
        & 'lrt_reduced<"0.05"'     # interaction term adds value
    )

    properties = ['run_id', 'bic', 'deviance_explained', 'deviance_ratio',
                  'mse_ratio', 'mse_ratio_std', 'formula', 'nr_outliers']
    model_selection = {prop: [] for prop in properties}

    for model in candidates:
        for prop in properties:
            if prop in model:
                model_selection[prop].append(model[prop])
            elif prop == 'nr_outliers':
                outliers = model.get('outliers')
                model_selection[prop].append(len(outliers) if outliers is not None else 0)

    df = pd.DataFrame(model_selection).sort_values(by='bic')
    return df