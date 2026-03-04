#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:35:01 2020


@author: Laura Sainz Villalba
Processes DeepLabCut (DLC) pose-estimation CSV outputs alongside behavioural session
data to extract per-frame licking, movement, and trial-alignment variables, then
stores the results in a DataJoint database table (Dlc_videoinfo).

Pipeline overview
-----------------
1.  Read DLC CSV  →  dict of body-part trajectories
2.  Detect TTL onset frames in the video (for corrupted sessions)
3.  Align video frame timestamps to the behavioural session clock
4.  Extract per-frame lick side (left / right)
5.  Aggregate lick variables per trial (response window, anticipatory window)
6.  Detect movement bouts via frame-differencing
7.  Cross-check video-derived lick responses with logged session responses
8.  Insert everything into the DataJoint schema
"""

import os
import sys
import json
import inspect
from time import time

import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

print("CALLING dlc_video.py from module script:", __name__)

# ---------------------------------------------------------------------------
# Conditional imports depending on execution context
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from behaviour_import import Session
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utilities import (
        get_trainingpoint, parse_filename, norm_diff_frame,
        video_array, get_port_pos, get_lick_transitions, get_port_params,
        videoframe_to_time, time_to_videoframe, get_video_info,
    )
    import behaviour

elif __name__ == 'data_import.dlc_video':
    from .behaviour_import import Session
    from utilities import (
        get_trainingpoint, parse_filename, norm_diff_frame,
        video_array, get_port_pos, get_lick_transitions, get_port_params,
        videoframe_to_time, time_to_videoframe, get_video_info,
    )
    import behaviour

# ---------------------------------------------------------------------------
# DataJoint schema
# ---------------------------------------------------------------------------
dj.config["enable_python_native_blobs"] = True
schema = dj.schema('dlc_video_hpc_cat_2025', locals(), create_tables=True)


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_dlc_csv(filename):
    """
    Load a DeepLabCut output CSV into a dict of numpy arrays.

    DLC CSVs have a two-row header (body-part name, coordinate name); data
    starts on row 3.  The function also strips trailing all-zero rows that DLC
    sometimes appends when the video ends mid-batch.

    Parameters
    ----------
    filename : str
        Path to the DLC .csv file.

    Returns
    -------
    data : dict  {str -> np.ndarray}
        Keys are '<BodyPart>_<coord>'  e.g. 'Nose_x', 'Tongue_likelihood'.
        Values are float arrays of length == number of valid frames.
    """
    # Some DLC files have inconsistent column counts; fall back to explicit
    # column numbering in that case.
    try:
        df = pd.read_csv(filename)
    except pd.errors.ParserError as err:
        num_cols = int(str(err).split("Expected ")[1].split(",")[0])
        df = pd.read_csv(filename, names=range(num_cols))

    # Build the data dict from the multi-level header
    data = {}
    for col in list(df.keys())[1:]:                   # skip the unnamed index col
        key = df[col][0] + '_' + df[col][1]           # e.g. 'Nose_x'
        data[key] = np.array([float(v) for v in df[col][2:]])

    # Find the last row that is NOT entirely zero (DLC padding artefact)
    last_valid_row = 0
    for i in reversed(range(len(data[key]))):
        if any(data[k][i] != 0 for k in data):
            last_valid_row = i
            break

    # Trim all arrays to the last valid frame
    for key in data:
        data[key] = data[key][:last_valid_row + 1]

    return data


# ===========================================================================
# TTL onset detection (for video-to-session alignment)
# ===========================================================================

def get_video_ttl_onsets(data, csv_filename, ttl_roi):
    """
    Detect TTL pulse onset frames directly from the video pixel signal.

    A small region-of-interest (ROI) near the imaging TTL LED is monitored.
    A sudden large positive jump in pixel-norm between consecutive frames
    indicates a TTL onset.

    Parameters
    ----------
    data : dict
        DLC trajectory dict (used to locate the animal's back / nose to
        define the ROI when ttl_roi is None).
    csv_filename : str
        Path to DLC CSV; the corresponding .mp4 is inferred by stripping the
        'DLC...' suffix.
    ttl_roi : tuple (y, x) or None
        If provided, use this pixel coordinate directly as the ROI centre.
        If None, estimate the ROI from the animal's average nose/back position.

    Returns
    -------
    ttl_video_onsets : list of int
        Frame indices where a TTL onset was detected.
    """
    clip_name = csv_filename.split('DLC')[0] + '.mp4'

    # ── Determine ROI centre ────────────────────────────────────────────────
    x = data['Nose_x'].copy()
    valid_x = data['Nose_likelihood']
    x[valid_x < 0.98] = np.nan
    mu_x = int(np.nanmean(x))

    if ttl_roi is not None:
        roi_coords = ttl_roi
        mu_y, mu_x = ttl_roi
        print('ttl_roi:', ttl_roi)
    else:
        y = data['Back_y'].copy()
        valid = data['Back_likelihood']
        y[valid < 0.98] = np.nan
        mu_y = int(np.nanmean(y)) + 30      # offset below the animal's back
        roi_coords = mu_y, mu_x

    # ── Extract pixel-norm time series from the ROI ─────────────────────────
    frames_gray, nr_frames, _ = video_array(clip_name, roi_coords)
    assert nr_frames == len(valid_x), "Frame count mismatch between video and DLC CSV"

    imaging = np.linalg.norm(frames_gray, axis=(1, 2))

    # Diagnostic plots
    fig, axs = plt.subplots()
    axs.plot(imaging)
    axs.set_title('imaging')
    plt.show()

    # ── Compute frame-to-frame difference ───────────────────────────────────
    diff = np.diff(imaging)
    diff_tot = np.zeros(len(diff) + 1)
    diff_tot[1:] = diff                     # prepend a zero so indices align

    fig, axs = plt.subplots()
    axs.plot(diff_tot)
    axs.set_title('diff_tot')
    plt.show()

    # ── Detect onsets: large positive jump after a quiet period ─────────────
    # Rules:
    #   • diff_tot[i] > 450  (large brightness jump)
    #   • Previous frame was quiet  (|diff| < 450, or NaN)
    #   • Pre-onset mean < 100  (background was dark / stable)
    #   • Minimum 240-frame gap between successive TTL onsets (~5.3 s at 45 Hz)
    JUMP_THRESHOLD   = 450
    QUIET_THRESHOLD  = 450
    PREONSET_WINDOW  = 45   # frames to look back for pre-onset baseline
    MIN_INTER_ONSET  = 240  # minimum frames between consecutive TTL events

    ttl_video_onsets = []
    j = 0
    for i in range(1, len(diff_tot)):
        prev_quiet = (-QUIET_THRESHOLD < diff_tot[i - 1] < QUIET_THRESHOLD)
        prev_nan   = np.isnan(diff_tot[i - 1])

        if diff_tot[i] > JUMP_THRESHOLD and (prev_quiet or prev_nan):
            preonset_mean = np.nanmean(diff_tot[i - PREONSET_WINDOW:i])
            if preonset_mean >= 100:
                continue                    # pre-onset was too bright → skip

            if j == 0 or abs(ttl_video_onsets[j - 1] - i) > MIN_INTER_ONSET:
                ttl_video_onsets.append(i)
                j += 1

    return ttl_video_onsets


# ===========================================================================
# Movement detection
# ===========================================================================

def get_video_mov_params(data, csv_filename, delta, frame_timestamps,
                         session_entry, plot=False):
    """
    Detect movement bouts from frame-differencing of a body-part ROI.

    Uses the animal's back position as the ROI centre, computes frame-to-frame
    difference norms, finds peaks that exceed 3 standard deviations above the
    mean, and removes peaks that co-occur with imaging TTL pulses (artefacts).

    Parameters
    ----------
    data : dict
        DLC trajectory dict.
    csv_filename : str
        Path to DLC CSV; .mp4 is inferred automatically.
    delta : float
        Time offset (seconds) to convert video timestamps to session clock.
    frame_timestamps : np.ndarray
        Session-aligned timestamp for every video frame.
    session_entry : DataJoint query result
        Behavioural session entry (provides TTL timestamps).
    plot : bool
        If True, show a diagnostic figure of the movement signal.

    Returns
    -------
    movement_bool : np.ndarray
        Binary array (length == nr_frames); 1 at movement onset peaks, 0 elsewhere.
    nr_frames : int
        Total number of frames in the video.
    fps : float
        Frames per second of the video.
    """
    # ── Build TTL reference timestamps in video time ─────────────────────────
    ttl_onsets  = np.array(session_entry.fetch('ttl_timestamps')[0])
    ttl_offsets = ttl_onsets + 9 * 60             # 9 s after each TTL offset
    ttl_all_ts  = np.round(np.concatenate([ttl_onsets, ttl_offsets]) - delta)

    # Human-readable versions for coincidence checking
    def _frames_to_str(t):
        mins = int(t / 60)
        return '%i mins and %i seconds' % (mins, t - mins * 60)

    ttl_ts_str  = [_frames_to_str(t) for t in ttl_all_ts]
    ttl_peaks   = [time_to_videoframe(t, frame_timestamps) for t in ttl_onsets + delta]

    clip_name = csv_filename.split('DLC')[0] + '.mp4'

    # ── Validate back-tracking quality ──────────────────────────────────────
    print('getting roi gray frames')
    valid = data['Back_likelihood']
    x = data['Back_x'].copy()
    x[valid < 0.98] = np.nan

    if np.isnan(np.nanmean(x)):
        # Back not tracked at all → return NaN movement array
        nr_frames, fps = get_video_info(clip_name)
        return np.array([np.nan] * int(nr_frames)), int(nr_frames), fps

    # ── Extract grayscale ROI frames centred on the animal's back ────────────
    mu_x = int(np.nanmean(x))
    y = data['Back_y'].copy()
    y[valid < 0.98] = np.nan
    mu_y = int(np.nanmean(y))

    frames_gray, nr_frames, fps = video_array(clip_name, (mu_y, mu_x))
    assert nr_frames == len(valid), "Frame count mismatch"

    # ── Compute movement signal via frame-differencing norm ──────────────────
    print('computing norm')
    mov = norm_diff_frame(frames_gray)

    # Mask frames with poor likelihood or implausible large values
    mov[valid < 0.98] = np.nan
    mov[mov > 0.5]    = np.nan          # clip extreme artefacts

    # ── Find movement peaks (3 SD above mean, ≥1 s apart at 45 Hz) ──────────
    PEAK_MIN_DISTANCE = 45              # frames (~1 s)
    height_threshold  = 3 * np.nanstd(mov) + np.nanmean(mov)

    print('finding peaks')
    roi_peaks, _ = find_peaks(mov.copy(), distance=PEAK_MIN_DISTANCE,
                               height=height_threshold)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(mov)
        ax.scatter(ttl_peaks,  mov[ttl_peaks],  marker='o', s=60, color='green',
                   label='TTL onset')
        ax.scatter(roi_peaks,  mov[roi_peaks],  marker='+', s=60, color='red',
                   alpha=0.6, label='movement peak')
        ax.set_xlabel('video frames')
        ax.set_ylabel('Frame differencing norm')
        ax.legend()
        plt.show()

    # ── Remove peaks that coincide with imaging TTL pulses ───────────────────
    movement_bool = np.zeros(len(mov), dtype=int)
    movement_bool[roi_peaks] = 1

    coincident_indices = []
    for i in range(1, len(movement_bool)):
        if movement_bool[i] and not movement_bool[i - 1]:
            # onset of a movement bout
            tot_secs   = i / 45
            peak_label = _frames_to_str(tot_secs)
            if peak_label in ttl_ts_str:
                coincident_indices.append(i)

    ARTIFACT_HALFWIN = 10
    for idx in coincident_indices:
        movement_bool[idx - ARTIFACT_HALFWIN: idx + ARTIFACT_HALFWIN] = 0

    print('coincidences with TTL removed:', len(coincident_indices))
    return movement_bool, nr_frames, fps


def check_movement_bouts(movement_bool):
    """
    Return human-readable timestamps of all movement bout onsets.

    Parameters
    ----------
    movement_bool : array-like
        Per-frame binary movement indicator.

    Returns
    -------
    timings : list of str
        Unique time-strings for each detected movement onset.
    """
    timings = []
    for i in range(1, len(movement_bool)):
        if movement_bool[i] and not movement_bool[i - 1]:
            t = videoframe_to_time(i)
            if t not in timings:
                timings.append(t)
    return timings


# ===========================================================================
# Trial-alignment helpers
# ===========================================================================

def sideport_by_portonsets(data):
    """
    Classify each detected port-contact onset as Left ('L'), Right ('R'),
    or Both ('B') based on the DLC port-position data.

    When both ports are active within ±30 frames of each other, the event is
    labelled 'B' and assigned to the earlier onset frame.

    Parameters
    ----------
    data : dict
        DLC trajectory dict.

    Returns
    -------
    side_port : list of str
        Sorted sequence of port-side labels (one per detected trial onset).
    response_window_onsets : list of int
        Corresponding video frame indices, sorted chronologically.
    """
    port_dict = get_port_params(data)
    ports = list(port_dict.keys())      # expected to be [left_key, right_key]

    COINCIDENCE_WINDOW = 30             # frames (±0.67 s at 45 Hz)

    response_window_onsets = []
    side_port = []

    for i, port in enumerate(ports):
        contralateral_onsets = port_dict[ports[1 - i]]

        for onset_idx in port_dict[port]:
            fs_min = onset_idx - COINCIDENCE_WINDOW
            fs_max = onset_idx + COINCIDENCE_WINDOW

            # Check whether the opposite port fired within the coincidence window
            bothport = False
            other_onset_idx = None
            for other_idx in contralateral_onsets:
                if fs_min < other_idx < fs_max:
                    bothport = True
                    other_onset_idx = other_idx
                    break

            if bothport:
                # Use the earlier of the two port onsets as the trial frame
                idx = min(other_onset_idx, onset_idx)
                if idx not in response_window_onsets:
                    response_window_onsets.append(idx)
                    side_port.append('B')
            else:
                response_window_onsets.append(onset_idx)
                side_port.append(port[0])   # 'L' or 'R'

    # Sort both lists chronologically
    pairs = sorted(zip(response_window_onsets, side_port))
    response_window_onsets = [p[0] for p in pairs]
    side_port              = [p[1] for p in pairs]

    return side_port, response_window_onsets


def get_align_indices(data, session_entry, csv_filename):
    """
    Find the response-window frame indices for all trials and identify the
    first 'both-ports' trial (used as the primary alignment anchor).

    Also validates the pre-anchor port sequence against the logged behavioural
    session data.

    Parameters
    ----------
    data : dict
        DLC trajectory dict.
    session_entry : DataJoint query result
        Behavioural session entry.
    csv_filename : str
        Path to DLC CSV (used only for error messages).

    Returns
    -------
    response_window_onsets : list of int
        Frame indices for every detected trial's response window onset.
    first_bothport_idx : int
        Frame index of the first 'both-ports' trial (0 if not found).
    side_port : list of str
        Port-side label sequence for detected trials.
    """
    side_port, response_window_onsets = sideport_by_portonsets(data)

    # ── Locate the first 'both-ports' trial ─────────────────────────────────
    first_bothport_idx = 0
    bothport_trial_pos = None

    if side_port[0] != 'B':
        for i in range(15):
            if side_port[i] == 'B':
                first_bothport_idx  = response_window_onsets[i]
                bothport_trial_pos  = i
                break

    # ── Validate pre-anchor port sequence against session log ────────────────
    if first_bothport_idx != 0:
        seq_video = side_port[:bothport_trial_pos]

        ports_id  = session_entry.fetch('ports_id')[0][:15]
        seq_session = []
        for j, ports in enumerate(ports_id):
            if   ports == [1, 0]: seq_session.append('L')
            elif ports == [0, 1]: seq_session.append('R')
            elif ports == [1, 1]: break   # first both-ports trial in session

        # Align sequence lengths before comparison
        seq_session = seq_session[-len(seq_video):]
        assert seq_video == seq_session, (
            f"Port sequence mismatch: video={seq_video}  session={seq_session}"
        )
    else:
        assert side_port[0] == 'B', \
            f"Expected first trial to be 'B' but got: {side_port[0]}"

    # ── Trim any extra trials appended beyond the session end ────────────────
    if len(side_port) < 15:
        # Corrupted video with very few detected trials — return as-is
        return response_window_onsets, first_bothport_idx, side_port

    last_bothport_idx = -1
    for i in list(np.arange(-15, -1, 1)):
        if side_port[i] == 'B' and side_port[i + 1] != 'B':
            last_bothport_idx = i
            break

    if last_bothport_idx != -1:
        print('extra trials at the end — trimming')
        response_window_onsets = response_window_onsets[:last_bothport_idx + 1]
        side_port              = side_port[:last_bothport_idx + 1]

    return response_window_onsets, first_bothport_idx, side_port


# ===========================================================================
# Lick extraction
# ===========================================================================

def get_lick_vars_frames(data, inreach_pos=None):
    """
    Classify each video frame as containing a lick and, if so, assign a side.

    Side assignment uses two criteria:
      1. Tongue is to the right of the nose  (relative nose rule)
      2. Tongue is closer to the right port  (relative port rule)
    Only frames where both criteria agree are labelled left (0) or right (1);
    disagreements are labelled –1.

    Parameters
    ----------
    data : dict
        DLC trajectory dict.
    inreach_pos : tuple (left_x, right_x) or None
        Pre-computed in-reach x-positions of the two water ports.  If None,
        they are detected automatically from the DLC data.

    Returns
    -------
    lick_bool : list of int   (0 / 1)
    lick_side : list          (0=left, 1=right, –1=ambiguous, nan=no lick)
    """
    # ── Per-frame lick detection (DLC likelihood threshold) ──────────────────
    lick_bool = [1 if v > 0.98 else 0 for v in data['Tongue_likelihood']]

    # ── Port positions ───────────────────────────────────────────────────────
    if inreach_pos is None:
        inreach_leftport,  _ = get_port_pos('WaterportLeft',  data)
        inreach_rightport, _ = get_port_pos('WaterportRight', data)
    else:
        inreach_leftport, inreach_rightport = inreach_pos

    tongue_pos = data['Tongue_x']
    nose_pos   = data['Nose_x']

    # ── Assign lick side per frame ───────────────────────────────────────────
    lick_side = []
    for i, lick_frame in enumerate(lick_bool):
        if not lick_frame:
            lick_side.append(np.nan)
            continue

        tongue_right_of_nose = tongue_pos[i] < nose_pos[i]         # x increases rightward
        tongue_closer_right  = (abs(inreach_rightport - tongue_pos[i])
                                < abs(inreach_leftport  - tongue_pos[i]))

        if tongue_right_of_nose and tongue_closer_right:
            lick_side.append(1)     # right lick
        elif not tongue_right_of_nose and not tongue_closer_right:
            lick_side.append(0)     # left lick
        else:
            lick_side.append(-1)    # ambiguous

    return lick_bool, lick_side


def get_licks_vars_trial(data, response_window_onset_idx, lick_side):
    """
    Aggregate per-frame lick information into per-trial summaries.

    For each trial the function inspects a 2-second response window (90 frames
    at 45 Hz) and a 1.5-second anticipatory window (67 frames) preceding the
    port onset.  The inferred response side follows a priority rule:
      – If the first two transitions agree → use that side.
      – If they disagree → check anticipatory licks to disambiguate.
      – If still ambiguous → use the first pair of consecutive identical licks.

    Parameters
    ----------
    data : dict
        DLC trajectory dict (unused here but kept for API consistency).
    response_window_onset_idx : list of int
        Frame index of each trial's response window onset.
    lick_side : list
        Per-frame lick side label.

    Returns
    -------
    lick_response       : list  – inferred response side per trial
    trial_licks         : list of lists  – lick transitions in response window
    anticipatory_licks  : list of lists  – lick transitions in anticipatory window
    """
    RESPONSE_WINDOW_FRAMES     = 90     # 2 s at 45 Hz
    ANTICIPATORY_WINDOW_FRAMES = 67     # ~1.5 s at 45 Hz

    lick_response      = []
    trial_licks        = []
    anticipatory_licks = []

    for idx in response_window_onset_idx:
        # ── Extract transitions in each window ───────────────────────────────
        resp_window  = lick_side[idx: idx + RESPONSE_WINDOW_FRAMES]
        antcp_window = lick_side[idx - ANTICIPATORY_WINDOW_FRAMES: idx]

        resp_trans  = get_lick_transitions(resp_window)
        antcp_trans = get_lick_transitions(antcp_window)

        trial_licks.append(resp_trans)
        anticipatory_licks.append(antcp_trans)

        # ── Infer response side ──────────────────────────────────────────────
        if len(resp_trans) < 2:
            lick_response.append(np.nan)
            continue

        # Find first pair of identical consecutive transitions
        first_consec = resp_trans[0]
        for n in range(1, len(resp_trans)):
            if resp_trans[n] == resp_trans[n - 1]:
                first_consec = resp_trans[n]
                break

        if resp_trans[0] == resp_trans[1]:
            # First two transitions agree → clear response
            lick_response.append(resp_trans[0])
        elif antcp_trans:
            # Use anticipatory licks to resolve the ambiguity
            if antcp_trans[-1] == resp_trans[0]:
                lick_response.append(resp_trans[0])
            else:
                lick_response.append(first_consec)
        else:
            lick_response.append(first_consec)

    return lick_response, trial_licks, anticipatory_licks


# ===========================================================================
# Video-to-session timestamp alignment
# ===========================================================================

# ---------------------------------------------------------------------------
# Hard-coded manual parameters for corrupted / problematic sessions
# ---------------------------------------------------------------------------
# Key: (animal_id, session_id)
# Value: dict with keys 'side_port', 'inreach_pos', 'ttl_roi'
# 'side_port'   – manually verified trial-type sequence
# 'inreach_pos' – (left_x, right_x) pixel coords of water ports
# 'ttl_roi'     – (y, x) pixel coord of TTL ROI, or None

MANUAL_SESSION_PARAMS = {
    ('BK4937_R',  24): {'side_port': ['L','R','B','B','B','B'],              'inreach_pos': (152, 102),  'ttl_roi': None},
    ('BK4926_L',  23): {'side_port': ['R','L','L','L','B','B'],              'inreach_pos': (277, 225),  'ttl_roi': None},
    ('BK4934_LL', 51): {'side_port': ['N','R','L','L','R','R','B'],          'inreach_pos': (277, 212),  'ttl_roi': [208, 240]},
    ('BK4940_RR', 23): {'side_port': ['N','L','R','R','L','R','B'],          'inreach_pos': (269, 204),  'ttl_roi': [222, 255]},
    ('BK4933_LR', 37): {'side_port': ['L','L','L','B','B'],                  'inreach_pos': (258, 191),  'ttl_roi': [233, 234]},
    ('BK4940_RR', 37): {'side_port': ['N','L','R','R','L','R','B'],          'inreach_pos': (270, 204),  'ttl_roi': [205, 229]},
    ('BK4936_L',  37): {'side_port': ['N','R','R','L','R','L','B'],          'inreach_pos': (254, 199),  'ttl_roi': None},
    ('BK4933_LR', 52): {'side_port': ['N','R','L','R','L','L','B'],          'inreach_pos': (256, 197),  'ttl_roi': None},
    ('BK4949_L',  19): {'side_port': ['N','L','L','L','R','L','B'],          'inreach_pos': (252, 199),  'ttl_roi': [221, 226]},
    ('BK4949_L',  20): {'side_port': ['R','L','L','L','R','B'],              'inreach_pos': (221, 163),  'ttl_roi': [131, 186]},
    ('BK4949_L',  21): {'side_port': ['N','L','L','R','L','L','B'],          'inreach_pos': (260, 204),  'ttl_roi': [232, 235]},
    ('BK4936_L',  24): {'side_port': ['L','L','R','L','R','B'],              'inreach_pos': (298, 237),  'ttl_roi': [222, 255]},
    ('BK4926_L',  43): {'side_port': ['N','L','L','L','L','R'],              'inreach_pos': (265, 202),  'ttl_roi': [239, 239]},
    ('BK4926_L',  49): {'side_port': ['N','R','L','L','R','R','B'],          'inreach_pos': (265, 202),  'ttl_roi': [272, 206]},
    ('BK4926_L',  51): {'side_port': ['N','R','L','L','R','L','B'],          'inreach_pos': (287, 216),  'ttl_roi': [218, 236]},
    ('BK4934_LL', 43): {'side_port': ['N','R','L','R','L','L','B'],          'inreach_pos': (273, 211),  'ttl_roi': [244, 234]},
    ('BK4934_LL', 54): {'side_port': ['N','R','L','L','L','R','B'],          'inreach_pos': (270, 208),  'ttl_roi': [221, 226]},
}


def get_video_frame_times_ttls(session_entry, csv_filename, data,
                                nr_frames, ttl_roi):
    """
    Align video frame timestamps to the session clock using TTL pulses
    detected directly in the video frames.

    Used for sessions where DLC port detection is unreliable (corrupted files).
    Manually verified side_port sequences are looked up from MANUAL_SESSION_PARAMS.

    Parameters
    ----------
    session_entry : DataJoint query
    csv_filename  : str
    data          : dict  – DLC trajectory dict
    nr_frames     : int   – total video frames
    ttl_roi       : tuple or None  – ROI for TTL detection

    Returns
    -------
    response_window_onsets : list of int  – frame indices per trial
    videoframe_ts          : np.ndarray  – session-aligned timestamps per frame
    delta                  : float       – time offset applied (seconds)
    All three are None if the session is not in MANUAL_SESSION_PARAMS.
    """
    animal_id, session_id, _ = parse_filename(csv_filename)
    key = (animal_id, session_id)

    # ── Look up manually verified parameters ─────────────────────────────────
    if key not in MANUAL_SESSION_PARAMS:
        ans = input(f'Session not in manual params: {csv_filename}\n'
                    'Press Enter to skip or anything else to raise: ')
        if ans == '':
            return None, None, None
        raise RuntimeError('Stopped manually')

    params    = MANUAL_SESSION_PARAMS[key]
    side_port = params['side_port'][:]     # copy to avoid mutation

    ttl_onsets      = session_entry.fetch('ttl_timestamps')[0][1:]  # skip baseline TTL
    ports_onsets    = session_entry.fetch('ports_on')[0]
    ttl_video_onsets = get_video_ttl_onsets(data, csv_filename, ttl_roi)

    # ── Strip 'N' (no-port / baseline) entries from the start ────────────────
    if side_port[0] == 'N':
        side_port        = side_port[1:]
        ttl_video_onsets = ttl_video_onsets[1:]

    # ── Identify first 'both-ports' trial as the alignment anchor ────────────
    first_bothport_idx   = 0
    firstbothport_trial  = 0

    if side_port[0] != 'B':
        for i, s in enumerate(side_port):
            if s == 'B':
                first_bothport_idx = ttl_video_onsets[i]
                break

    # Derive the session trial index of the first both-ports event
    ports_id = session_entry.fetch('ports_id')[0][:15]
    for j, ports in enumerate(ports_id):
        if   ports == [1, 0]: pass          # L
        elif ports == [0, 1]: pass          # R
        elif ports == [1, 1]: break         # first both-ports trial
    firstbothport_trial = j

    # Special case: align to first trial TTL for a specific corrupted session
    if animal_id == 'BK4926_L' and session_id == 43:
        print('align to first trial TTL')
        first_bothport_idx  = ttl_video_onsets[0]
        firstbothport_trial = 0

    # ── Compute time offset (delta) ──────────────────────────────────────────
    VIDEO_FPS  = 45
    video_ts   = np.arange(0, nr_frames / VIDEO_FPS, 1 / VIDEO_FPS)

    ref_time          = ttl_onsets[firstbothport_trial]         # session clock
    non_reframed_time = video_ts[first_bothport_idx]            # raw video clock
    delta             = ref_time - non_reframed_time

    print('aligning with first both-port TTL')
    videoframe_ts = video_ts + delta

    # Convert detected TTL video frames to session-clock timestamps
    ttl_video_onset_ts = [videoframe_ts[f] for f in ttl_video_onsets]

    # ── Match video TTL frames to session TTL onsets (±150 ms tolerance) ─────
    ALIGN_TOLERANCE_SEC = 0.15
    response_window_onsets = []
    last    = 0
    matches = 0

    for i, ttl_onset in enumerate(ttl_onsets):
        for j in range(last, len(ttl_video_onsets)):
            if abs(ttl_onset - ttl_video_onset_ts[j]) < ALIGN_TOLERANCE_SEC:
                # Compute port-onset frame offset from TTL onset
                delta_frames = round((ports_onsets[i] - ttl_onset) * VIDEO_FPS)
                assert delta_frames > 0, "Port onset should come after TTL onset"
                response_window_onsets.append(ttl_video_onsets[j] + delta_frames)
                last = j
                matches += 1
                break

    print(f'TTL-to-session matches: {matches}')
    return response_window_onsets, videoframe_ts, delta


def get_video_frame_times_ports(session_entry, csv_filename, data, nr_frames):
    """
    Align video frame timestamps to the session clock using DLC-detected
    port-contact transients (standard, non-corrupted sessions).

    Parameters
    ----------
    session_entry : DataJoint query
    csv_filename  : str
    data          : dict  – DLC trajectory dict
    nr_frames     : int

    Returns
    -------
    response_window_onsets : list of int
    videoframe_ts          : np.ndarray
    delta                  : float
    """
    VIDEO_FPS = 45
    response_window_onsets, first_bothport_idx, _ = get_align_indices(
        data, session_entry, csv_filename
    )

    video_ts = np.arange(0, nr_frames / VIDEO_FPS, 1 / VIDEO_FPS)

    # Find the session trial index of the first both-ports event
    ports_id = session_entry.fetch('ports_id')[0][:15]
    for j, ports in enumerate(ports_id):
        if   ports == [1, 0]: pass
        elif ports == [0, 1]: pass
        elif ports == [1, 1]: break
    firstbothport_trial = j

    if first_bothport_idx != 0:
        print('aligned with first both-port trial')
        ref_time          = session_entry.fetch('ports_on')[0][firstbothport_trial]
        non_reframed_time = video_ts[first_bothport_idx]
    else:
        # Fallback: align using the last detected trial
        print('aligned with last both-port trial')
        ref_time          = session_entry.fetch('ports_on')[0][-1]
        non_reframed_time = video_ts[response_window_onsets[-1]]

    delta         = ref_time - non_reframed_time
    videoframe_ts = video_ts + delta

    return response_window_onsets, videoframe_ts, delta


# ===========================================================================
# Quality-control: cross-check video licks with behavioural log
# ===========================================================================

def check_with_behaviour(session_entry, videoframe_ts,
                          response_window_onsets, lick_info):
    """
    Align video-derived trial lick data to the session trial list.

    Handles missed detections by inserting NaN placeholders so that every
    entry in the returned structure corresponds to the same session trial index.
    Also verifies that lick responses agree with the logged session responses
    within a 33 % discrepancy threshold.

    Parameters
    ----------
    session_entry          : DataJoint query
    videoframe_ts          : np.ndarray  – session-aligned frame timestamps
    response_window_onsets : list of int
    lick_info              : tuple  – (lick_response, trial_licks, anticipatory_licks)

    Returns
    -------
    lick_info_dict : dict  – per-trial lick variables keyed for DB insertion
    missed_trials  : list of int  – 1-based trial IDs not detected in video
    """
    lick_response, trial_licks, anticipatory_licks = lick_info
    ports_onsets      = session_entry.fetch('ports_on')[0]
    session_responses = session_entry.fetch('response')[0]

    # Convert frame indices to timestamps
    port_video_onsets = [videoframe_ts[idx] for idx in response_window_onsets]

    ALIGN_TOLERANCE_SEC = 0.15

    # ── Re-align detected trials to session trial list ────────────────────────
    # Insert NaN for any session trial whose port onset was not detected.
    lick_response_n         = []
    trial_licks_n           = []
    anticipatory_licks_n    = []
    port_video_onsets_n     = []
    response_window_onsets_n = []

    last = 0
    for i, port_onset in enumerate(ports_onsets):
        prev_last = last
        for j in range(last, len(port_video_onsets)):
            if abs(port_onset - port_video_onsets[j]) < ALIGN_TOLERANCE_SEC:
                lick_response_n.append(lick_response[j])
                trial_licks_n.append(trial_licks[j])
                anticipatory_licks_n.append(anticipatory_licks[j])
                port_video_onsets_n.append(port_video_onsets[j])
                response_window_onsets_n.append(response_window_onsets[j])
                last = j + 1
                break
        if prev_last == last:
            # This session trial was not detected in the video
            lick_response_n.append(np.nan)
            trial_licks_n.append(np.nan)
            anticipatory_licks_n.append(np.nan)
            port_video_onsets_n.append(np.nan)
            response_window_onsets_n.append(np.nan)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert len(response_window_onsets_n) == len(ports_onsets), \
        "Trial count mismatch after re-alignment"

    # Verify timestamps are close (< 150 ms) for detected trials
    for trial_id, pvo in enumerate(port_video_onsets_n):
        if not np.isnan(pvo):
            diff = abs(ports_onsets[trial_id] - pvo)
            assert diff < ALIGN_TOLERANCE_SEC, (
                f"Trial {trial_id}: port onset timestamps too far apart "
                f"({ports_onsets[trial_id]:.3f} vs {pvo:.3f} s)"
            )

    # ── Compute response discrepancy flag per trial ───────────────────────────
    resp_discrepancy_bool = np.array([np.nan] * len(ports_onsets))
    missed_trials         = []
    discrepancies         = 0

    for trial_id in range(len(ports_onsets)):
        lick_resp  = lick_response_n[trial_id]
        sess_resp  = session_responses[trial_id]

        if np.isnan(port_video_onsets_n[trial_id]):
            missed_trials.append(trial_id + 1)     # 1-based trial ID
            continue

        if lick_resp != sess_resp and sess_resp != -1 and not np.isnan(lick_resp):
            resp_discrepancy_bool[trial_id] = 1
            discrepancies += 1

    discrepancy_rate = (discrepancies / len(resp_discrepancy_bool)) * 100
    print(f'Nr missed trials: {len(missed_trials)}')
    assert discrepancy_rate < 33, (
        f"Lick-response discrepancy rate too high: {discrepancy_rate:.1f}%"
    )

    lick_info_dict = {
        'lick_response':       lick_response_n,
        'trial_licks':         trial_licks_n,
        'anticipatory_licks':  anticipatory_licks_n,
        'resp_discrepancy_bool': list(resp_discrepancy_bool),
        'resp_window_idx':     response_window_onsets_n,
    }
    return lick_info_dict, missed_trials


# ===========================================================================
# DataJoint table definition
# ===========================================================================

@schema
class Dlc_videoinfo(dj.Manual):
    """DataJoint table storing per-session DLC video analysis results."""

    definition = """
    animal_id              : varchar(128)   # Mouse ID (unique)
    session_id             : int            # Session counter (1-based, chronological)
    experimental_timepoint : varchar(256)   # Training stage (discrimination / generalization / …)
    ---
    date                   : date           # Session date (YYYY-MM-DD)
    lick_bool              : longblob       # Per-frame lick indicator (0/1)
    lick_side              : longblob       # Per-frame lick side (0=L, 1=R, –1=ambiguous, NaN=no lick)
    movement_bool          : longblob       # Per-frame movement indicator (0/1)
    nr_missed_trials       : int            # Number of trials not detected in video
    missed_trials          : blob           # 1-based IDs of missed trials
    resp_window_idx        : blob           # Frame index of each trial's response window onset
    video_lick_response    : blob           # Video-derived lick response per trial
    resp_discrepancy_bool  : blob           # 1 where video and session responses disagree
    trial_licks            : longblob       # Lick transitions per trial (list of lists)
    anticipatory_licks     : longblob       # Anticipatory lick transitions per trial
    videoframe_times       : longblob       # Session-aligned timestamp for every frame
    """

    # -----------------------------------------------------------------------
    def dlc_video_session(self, csv_filename, debug=False):
        """
        Process one DLC session CSV and insert results into the database.

        Parameters
        ----------
        csv_filename : str    Path to the DLC .csv output file.
        debug        : bool   If True, print movement-bout timings.
        """
        print('Processing:', csv_filename)
        start = time()

        # ── Parse metadata from filename ──────────────────────────────────────
        animal_id, session_id, date = parse_filename(csv_filename)

        # ── Load DLC data ──────────────────────────────────────────────────────
        data = read_dlc_csv(csv_filename)

        # ── Load JSON sidecar (frame-drop log) ────────────────────────────────
        json_path = csv_filename.split('.')[0] + '.json'
        with open(json_path, 'r') as fh:
            log_data = json.load(fh)

        assert log_data['missed_frames'] == [], \
            f"Missed frames detected in {csv_filename}"
        nr_frames = log_data['nframes']

        # ── Fetch session metadata from DataJoint ─────────────────────────────
        session_entry = (Session()
                         & f'animal_id="{animal_id}"'
                         & f'session_id="{session_id}"')

        nr_trials              = len(session_entry.fetch('new_trial')[0])
        mode                   = session_entry.fetch('mode')[0]
        stage                  = session_entry.fetch('stage')[0]
        categoryset_id         = session_entry.fetch('categoryset_id')[0]
        experimental_timepoint = get_trainingpoint([mode, stage, categoryset_id])
        session_date           = str(session_entry.fetch('date')[0])

        assert date == session_date, \
            f"Date mismatch: filename={date}, session={session_date}"

        # ── Detect whether the video is corrupted ─────────────────────────────
        # A session is treated as corrupted if port in-reach and out-reach
        # positions are too close (< 100 px), indicating DLC failed to track
        # the port consistently.
        corrupted = False
        for port_name in ('Left', 'Right'):
            inreach, outreach = get_port_pos(f'Waterport{port_name}', data)
            if abs(inreach - outreach) < 100:
                corrupted = True

        # Override corrupted flag from manual parameter table if needed
        key = (animal_id, session_id)
        if key in MANUAL_SESSION_PARAMS:
            corrupted    = True     # all manually-specified sessions are treated as corrupted
            ttl_roi      = MANUAL_SESSION_PARAMS[key].get('ttl_roi')
            inreach_pos  = MANUAL_SESSION_PARAMS[key].get('inreach_pos')
        else:
            ttl_roi      = None
            inreach_pos  = None

        # ── Timestamp alignment ───────────────────────────────────────────────
        print('Getting video frame timestamps...')
        if corrupted:
            response_window_onsets, videoframe_ts, delta = get_video_frame_times_ttls(
                session_entry, csv_filename, data, nr_frames, ttl_roi
            )
            if response_window_onsets is None:
                return      # user chose to skip this session

            print('Getting licking vars (corrupted path)...')
            lick_bool, lick_side = get_lick_vars_frames(data, inreach_pos)
        else:
            response_window_onsets, videoframe_ts, delta = get_video_frame_times_ports(
                session_entry, csv_filename, data, nr_frames
            )
            print('Getting licking vars...')
            lick_bool, lick_side = get_lick_vars_frames(data)

        # ── Per-trial lick extraction ─────────────────────────────────────────
        lick_info = get_licks_vars_trial(data, response_window_onsets, lick_side)

        # ── Cross-check with behavioural session ──────────────────────────────
        print('Checking trial onsets...')
        lick_info, missed_trials = check_with_behaviour(
            session_entry, videoframe_ts, response_window_onsets, lick_info
        )

        # ── Movement detection ────────────────────────────────────────────────
        print('Getting movement bool on frames...')
        movement_bool, nr_frames_vid, fps = get_video_mov_params(
            data, csv_filename, delta, videoframe_ts, session_entry
        )
        assert nr_frames     == nr_frames_vid, "Frame count mismatch (movement)"
        assert int(fps)      == 45,            "Unexpected FPS"

        if debug:
            timings = check_movement_bouts(movement_bool)
            print('movement_timings:', timings[:150])

        # ── Final integrity checks ────────────────────────────────────────────
        for field in ('resp_window_idx', 'lick_response',
                      'resp_discrepancy_bool', 'trial_licks', 'anticipatory_licks'):
            assert nr_trials == len(lick_info[field]), \
                f"Trial count mismatch in field '{field}'"

        # ── Insert into DataJoint table ───────────────────────────────────────
        entry = {
            'animal_id':               animal_id,
            'session_id':              session_id,
            'experimental_timepoint':  experimental_timepoint,
            'date':                    date,
            'lick_bool':               lick_bool,
            'lick_side':               lick_side,
            'movement_bool':           movement_bool,
            'nr_missed_trials':        len(missed_trials),
            'missed_trials':           missed_trials,
            'resp_window_idx':         lick_info['resp_window_idx'],
            'video_lick_response':     lick_info['lick_response'],
            'resp_discrepancy_bool':   lick_info['resp_discrepancy_bool'],
            'trial_licks':             lick_info['trial_licks'],
            'anticipatory_licks':      lick_info['anticipatory_licks'],
            'videoframe_times':        videoframe_ts,
        }
        Dlc_videoinfo().insert1(entry, skip_duplicates=True)

        print(f'Done. Computation time: {(time() - start) / 60:.2f} mins')

    # -----------------------------------------------------------------------
    def dlc_video_experiment(self, dlc_directory):
        """
        Batch-process all DLC CSV files in a directory for a given experiment.

        Skips sessions that are already in the database or that do not appear
        in the recording dictionary for the corresponding animal.

        Parameters
        ----------
        dlc_directory : str   Path to the folder containing DLC .csv files.
        """
        all_files = os.listdir(dlc_directory)
        csv_files = [f for f in all_files if f.endswith('.csv')]

        for file in tqdm(csv_files, desc='Processing sessions'):
            csv_file  = os.path.join(dlc_directory, file)
            animal_id, session_id, _ = parse_filename(csv_file)

            # Skip the known-bad session BK4934_LL session 41
            if animal_id == 'BK4934_LL' and session_id == 41:
                continue

            # Check if already processed
            entry = (self
                     & f'animal_id="{animal_id}"'
                     & f'session_id="{session_id}"')
            if len(entry) > 0:
                continue

            # Only process sessions that are part of the recording list
            rec_dict = behaviour.get_recordings_subject(animal_id)
            all_sessions = [s for sublist in rec_dict.values() for s in sublist]
            if session_id in all_sessions:
                self.dlc_video_session(csv_file)