#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:31:45 2022

@author: Laura Sainz Villalba

Processing of  behavioral files and data extraction for AUDITORY categorization paradigm
"""
import numpy as np
import datajoint as dj
import os, sys, inspect
import scipy.stats as stats

print("Calling day_statistics.py from module script: ", __name__)

# ---------------------------------------------------------------------------
# Path setup – add parent and grandparent directories to sys.path so that
# local utility and design modules can be imported.
# ---------------------------------------------------------------------------
currentdir     = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir      = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

from utilities import get_property, entropy_in_time
from data_import import Session, Trial
from design import categories, helper_design

# ---------------------------------------------------------------------------
# DataJoint configuration
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema = dj.schema('day_statistics_hpc_cat_2025', locals(), create_tables=True)

HOME_DIRECTORY = '/home/lsainz/Doctorado/DoctoradoDatos/experimentdata/'


@schema
class Day(dj.Manual):
    definition = """ # variables in day dynamics

    day_id             : int            # Overall day counter for mouse (1-based)
    animal_id          : varchar(128)   # Mouse unique identifier
    ---
    date               : date           # Experimental date (YYYY-MM-DD)
    weight = NULL      : float          # Body weight in grams
    condition          : varchar(128)   # Experimental condition
    mode               : varchar(128)   # Training mode
    stage              : int            # Training stage within mode
    categorymask_type         : varchar(128)  # Category mask condition
    categorystructure_type    : varchar(128)  # Category structure condition
    categoryset_id            : int           # Unique ID for category set used
    curriculum_type           : varchar(128)  # Curriculum condition
    nr_entries         : int            # Number of sessions in the day
    nr_trials          : int            # Total number of trials
    engagement         : float          # Percent of trials with committed response
    session_time       : float          # Mean session duration (minutes)
    total_time         : float          # Total session time in day (minutes)
    grace_period       : float          # Grace period duration (seconds)
    delay_period       : float          # Delay period duration (seconds)
    wateramount        : int            # Total water intake (uL)
    watertrials        : float          # Percent of rewarded trials
    bias               : float          # Lick bias toward right port (-50 to +50)
    performance        : float          # Overall correct-response rate (%)
    performance_left = NULL   : float   # Correct rate for left-port trials (%)
    performance_right = NULL  : float   # Correct rate for right-port trials (%)
    perf_switch = NULL        : float   # Performance on trials after a port switch
    perf_switch_corr = NULL   : float   # Performance after switch following correct
    perf_switch_corr_bias = NULL : float  # Performance after switch, correct, against bias
    discrimination_left = NULL   : float  # Hit - False alarm for left-port category
    discrimination_right = NULL  : float  # Hit - False alarm for right-port category
    discrimination               : float  # Mean discrimination across ports
    dprime_discrimination = NULL : float  # d-prime derived from discrimination quantile
    reactiontime_mean = NULL     : float  # Mean reaction time (seconds)
    reactiontime_entropy = NULL  : float  # Entropy of reaction-time distribution
    lick_disparity = NULL        : float  # Mean lick difference between ports
    lick_control = NULL          : float  # Fraction of no-lick control trials
    """

    # ------------------------------------------------------------------
    # Table query helpers
    # ------------------------------------------------------------------

    def get_lastdate(self, animal_id=None):
        """Return the most recent date present in the table (or 0 if empty)."""
        subset = (self & f'animal_id="{animal_id}"') if animal_id else self
        if len(subset) == 0:
            return 0
        return str(sorted(subset.fetch('date'))[-1])

    def get_lastcounter(self, mouse):
        """Return the highest day_id already recorded for *mouse* (or 0)."""
        subset = self & f'animal_id="{mouse}"'
        return max(subset.fetch('day_id')) if len(subset) else 0

    def get_status(self, mouse, date):
        """Return (mode, stage, categoryset_id) for *mouse* on *date*."""
        sessions = Session() & f'animal_id="{mouse}"' & f'date="{date}"'
        return (
            sessions.fetch('mode')[-1],
            int(sessions.fetch('stage')[-1]),
            int(sessions.fetch('categoryset_id')[-1]),
        )

    # ------------------------------------------------------------------
    # Behavioural metrics
    # ------------------------------------------------------------------

    def get_bias(self, trials):
        """
        Compute lick bias toward the right port (%).
        Returns a value in [-50, +50]; positive = preference for right.
        Rescue and passive trial types are excluded.
        """
        active    = trials & 'trialtype!="rescue"' & 'trialtype!="passive"'
        responses = active.fetch('response')
        committed = [r for r in responses if r != -1]
        if not committed:
            return 0
        return round((sum(committed) / len(committed)) * 100 - 50, 2)

    def get_engagement(self, trials):
        """Return percent of trials with a committed (non-'no response') answer."""
        committed = len(trials & 'responsetype!="no response"')
        return round((committed / len(trials)) * 100, 2)

    def get_water_trials_amount(self, trials):
        """
        Return (total_water_uL, nr_rewarded_trials).
        Reward = 8 uL per drop; passive trials count as rewarded.
        """
        passive  = len(trials & 'trialtype="passive"')
        correct  = len(trials & 'trialtype!="passive"' & 'responsetype="correct"')
        rewarded = passive + correct
        return rewarded * 8, rewarded

    def get_reactiontime_info(self, trials):
        """Return (mean_RT, RT_entropy) for all trials with a committed response."""
        rts = (trials & 'responsetype!="no response"').fetch('reaction_time')
        return np.mean(rts), entropy_in_time(list(rts), 0.025)

    def get_withholdlick(self, trials):
        """
        Return fraction of 'control_nostimulus' trials with no lick (response == -1).
        Returns None if no such trials exist.
        """
        no_stim = trials & 'trialtype="control_nostimulus"'
        if len(no_stim) == 0:
            return None
        return round(len(no_stim & 'response="-1"') / len(no_stim), 2)

    def get_licking_info(self, trials):
        """Return mean lick disparity (port difference) across responding trials."""
        return np.mean((trials & 'response!="-1"').fetch('lick_disparity'))

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def get_performance(self, trials):
        """Return percent correct on active trials (0 if no active trials)."""
        active = trials & 'trialtype="active"'
        if len(active) == 0:
            return 0
        return round((len(active & 'responsetype="correct"') / len(active)) * 100, 2)

    def get_portperformance(self, trials):
        """Return (left_performance%, right_performance%) for active trials."""
        active = trials & 'trialtype="active"'
        return (
            self.get_performance(active & 'response="0"'),
            self.get_performance(active & 'response="1"'),
        )

    def get_discriminationindex(self, trials, target):
        """
        Compute hit rate, false-alarm rate, and discrimination index for *target* port.
        discrimination = (hit_rate - false_alarm_rate) * 100  (%).
        """
        active = trials & 'trialtype="active"'

        # Hit rate: fraction of target-port stimuli correctly identified
        target_trials = active & f'baited_port="{target}"'
        Rhit = (len(target_trials & 'responsetype="correct"') / len(target_trials)
                if len(target_trials) else 0)

        # False-alarm rate: fraction of non-target stimuli incorrectly identified
        nontarget_trials = active & f'baited_port="{1 - target}"'
        Rfalsealarm = (len(nontarget_trials & 'responsetype="incorrect"') / len(nontarget_trials)
                       if len(nontarget_trials) else 0)

        return Rhit, Rfalsealarm, round((Rhit - Rfalsealarm) * 100, 2)

    # ------------------------------------------------------------------
    # Trial-selection helpers (used for switch-performance metrics)
    # ------------------------------------------------------------------

    def select_switch(self, trials):
        """Return trials where the baited port changed relative to the previous trial."""
        baited     = trials.fetch('baited_port')
        trial_keys = trials.fetch(dj.key)
        keep = {i for i in range(1, len(baited)) if baited[i] != baited[i - 1]}
        return self._filter_trials(trials, trial_keys, keep)

    def select_aftercorr(self, trials):
        """Return trials that follow a correct response AND a port switch."""
        responses  = trials.fetch('responsetype')
        baited     = trials.fetch('baited_port')
        trial_keys = trials.fetch(dj.key)
        keep = {i for i in range(1, len(responses))
                if responses[i - 1] == 'correct' and baited[i] != baited[i - 1]}
        return self._filter_trials(trials, trial_keys, keep)

    def select_switchagainstbias(self, trials):
        """
        Return trials where a post-correct port switch goes against the
        animal's dominant lick bias.
        """
        responses      = trials.fetch('responsetype')
        response_ports = trials.fetch('response')
        baited         = trials.fetch('baited_port')
        trial_keys     = trials.fetch(dj.key)

        # Dominant port: 1 (right) if mean response > 0.5, else 0 (left)
        bias_port = 1 if round(sum(response_ports) / len(responses), 2) > 0.5 else 0

        keep = {i for i in range(1, len(responses))
                if (responses[i - 1] == 'correct'
                    and baited[i] != baited[i - 1]
                    and baited[i] == 1 - bias_port)}
        return self._filter_trials(trials, trial_keys, keep)

    @staticmethod
    def _filter_trials(trials, trial_keys, keep_indices):
        """Remove all rows whose index is NOT in *keep_indices*."""
        result = trials
        for i, key in enumerate(trial_keys):
            if i not in keep_indices:
                result = result - key
        return result

    # ------------------------------------------------------------------
    # Accumulated-correct helper (utility; not stored in table)
    # ------------------------------------------------------------------

    def get_accumulatedcorrect(self, responsetype, baited, trial_ends):
        """
        Return a time-binned (200 ms) cumulative correct-response vector
        over the course of a session.
        """
        correct_times = sorted(
            trial_ends[i] for i in range(len(baited)) if responsetype[i] == 'correct'
        )
        timebin  = 0.2
        bins     = list(np.arange(0, ((trial_ends[-1] + 1) / timebin + 2) * timebin, timebin))
        counts, _ = np.histogram(correct_times, bins)
        accum = [sum(counts[:i]) for i in range(1, len(counts))]
        accum.append(accum[-1] + counts[-1])
        return accum

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_day(self, matrix, date, animal_id):
        """
        Aggregate all sessions and trials for *animal_id* on *date* into a
        single row tuple and append it to *matrix*.
        """
        day_id = self.get_lastcounter(animal_id) + 1
        weight = helper_design.get_relative_weight(animal_id, date)

        all_sessions = Session() & f'animal_id="{animal_id}"' & f'date="{date}"'
        all_trials   = Trial()   & f'animal_id="{animal_id}"' & f'date="{date}"'

        # Session metadata
        condition     = get_property(all_sessions, 'condition')[0]
        curriculum_id = get_property(all_sessions, 'curriculum_id')[0]
        nr_entries    = len(all_sessions)
        curriculum_type = (
            'discrimination' if curriculum_id == 0
            else (categories.Curriculum() & f'curriculum_id="{curriculum_id}"')
                  .fetch('curriculum_type')[0]
        )
        mode, stage, categoryset_id = self.get_status(animal_id, date)
        categorystructure_type, categorymask_type = categories.get_category_types(curriculum_id)

        # Timing
        session_times = get_property(all_sessions, 'session_time')
        session_time  = round(float(np.mean(session_times)), 2)
        total_time    = round(sum(session_times), 2)
        grace_period  = get_property(all_sessions, 'grace_period')[0]
        delay_period  = get_property(all_sessions, 'delay_period')[0]

        # Trials and water
        nr_trials                = len(all_trials)
        wateramount, watertrials = self.get_water_trials_amount(all_trials)

        # Behavioural metrics
        engagement               = self.get_engagement(all_trials)
        bias                     = self.get_bias(all_trials)
        performance              = self.get_performance(all_trials)
        performance_left, performance_right = self.get_portperformance(all_trials)

        # Switch-trial performance
        perf_switch           = self.get_performance(self.select_switch(all_trials))
        perf_switch_corr      = self.get_performance(self.select_aftercorr(all_trials))
        perf_switch_corr_bias = self.get_performance(self.select_switchagainstbias(all_trials))

        # Discrimination and d-prime
        Rhit_L, Rfalsealarm_L, discrimination_left  = self.get_discriminationindex(all_trials, 0)
        Rhit_R, Rfalsealarm_R, discrimination_right = self.get_discriminationindex(all_trials, 1)
        discrimination   = round((discrimination_left + discrimination_right) / 2, 2)
        Rhit_mean        = round((Rhit_L + Rhit_R) / 2, 2)
        Rfalsealarm_mean = round((Rfalsealarm_L + Rfalsealarm_R) / 2, 2)
        dprime_discrimination = stats.norm.ppf(Rhit_mean) - stats.norm.ppf(Rfalsealarm_mean)

        # Reaction time and licking
        reactiontime_mean, reactiontime_entropy = self.get_reactiontime_info(all_trials)
        lick_disparity = self.get_licking_info(all_trials)
        lick_control   = self.get_withholdlick(all_trials)

        matrix.append((
            day_id, animal_id, date, weight,
            condition, mode, stage, categorymask_type,
            categorystructure_type, categoryset_id,
            curriculum_type, nr_entries, nr_trials,
            engagement, session_time, total_time,
            grace_period, delay_period, wateramount,
            watertrials, bias, performance, performance_left,
            performance_right, perf_switch, perf_switch_corr,
            perf_switch_corr_bias, discrimination_left,
            discrimination_right, discrimination, dprime_discrimination,
            reactiontime_mean, reactiontime_entropy,
            lick_disparity, lick_control,
        ))
        return matrix

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def update(self):
        """
        Populate the table with data from all sessions recorded after the
        last date already stored. Iterates over all subjects and their dates.
        """
        all_sessions = Session()
        lastdate     = str(self.get_lastdate())

        # Skip dates already in the table
        if lastdate != '0':
            all_sessions = all_sessions & f'date>"{lastdate}"'

        for j, animal_id in enumerate(get_property(all_sessions, 'animal_id'), start=1):
            print(f'\n EXTRACTING: {animal_id}  (subject {j})')
            matrix = []
            sessions_sub = all_sessions & f'animal_id="{animal_id}"'
            for date in get_property(sessions_sub, 'date'):
                print(f'  processing day: {date}')
                matrix = self.extract_day(matrix, str(date), animal_id)
            self.insert(matrix, skip_duplicates=True)

        print('\n DONE: finished updating day statistics')
         