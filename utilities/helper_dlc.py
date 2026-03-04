#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:06:02 2022

@author: Laura Sainz Villalba

# =============================================================================
#
# Utilities for video loading, frame-level motion analysis, and
# DeepLabCut (DLC) pose-based detection of lick-port reach events.
#
# Assumed camera frame rate: 45 Hz (used for frame ↔ time conversions).
# DLC likelihood threshold: 0.98 (used to filter low-confidence keypoints).
# =============================================================================
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_RATE = 45          # Camera acquisition rate in Hz
DLC_LIKELIHOOD_THRESH = 0.98  # Minimum DLC keypoint confidence to accept


# ---------------------------------------------------------------------------
# Time / Frame Conversion
# ---------------------------------------------------------------------------

def videoframe_to_time(videoframe):
    """
    Convert a video frame index to a human-readable time string.

    Parameters
    ----------
    videoframe : int
        Zero-based frame index.

    Returns
    -------
    str : e.g. '2 mins and 13 seconds'
    """
    total_secs = videoframe / FRAME_RATE
    mins = int(total_secs / 60)
    secs = total_secs - mins * 60
    return '%i mins and %i seconds' % (mins, secs)


def time_to_videoframe(time, frame_timestamps):
    """
    Find the video frame index that corresponds to a given timestamp.

    Iterates through `frame_timestamps` and returns the index of the last
    frame whose timestamp is still less than `time`.

    Parameters
    ----------
    time : float
        Target time (same units as `frame_timestamps`).
    frame_timestamps : array-like
        Monotonically increasing timestamp for each frame.

    Returns
    -------
    int : Frame index (0-based).
    """
    for i, timestamp in enumerate(frame_timestamps):
        if time < timestamp:
            break
    return i - 1


# ---------------------------------------------------------------------------
# Filename Parsing
# ---------------------------------------------------------------------------

def parse_filename(dlc_csv_filename):
    """
    Extract animal ID, session ID, and date from a DLC CSV filename.

    Expected filename convention:
        .../YYYY-MM-DD_<animal_id>_<session_id>DLC....csv

    Parameters
    ----------
    dlc_csv_filename : str
        Full or relative path to the DLC output CSV file.

    Returns
    -------
    animal_id : str   (hyphens replaced with underscores)
    session_id : int
    date : str        (YYYY-MM-DD, taken from the first 10 chars of the basename)
    """
    prefix = dlc_csv_filename.split('DLC')[0]  # everything before 'DLC'
    parts = prefix.split('_')

    session_id = int(parts[-1])
    animal_id = parts[-2].replace('-', '_')
    date = dlc_csv_filename.split('/')[-1][:10]

    return animal_id, session_id, date


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def get_video_info(video_file):
    """
    Return basic metadata for a video file without loading any frames.

    Parameters
    ----------
    video_file : str  Path to the video file.

    Returns
    -------
    nr_frames : float  Total frame count (as reported by OpenCV).
    fps : float        Frames per second.
    """
    cap = cv2.VideoCapture(video_file)
    nr_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return nr_frames, fps


def video_array(video_file, roi_coords=None):
    """
    Load all frames from a video file into a NumPy array.

    If `roi_coords` is provided, each frame is cropped to a 20×20-pixel
    region of interest (ROI) centred on that point and converted to
    grayscale using standard luminance weights.

    Parameters
    ----------
    video_file : str
        Path to the video file.
    roi_coords : (int, int) or None
        (y, x) centre of the ROI in pixels. If None, full colour frames
        are returned unchanged.

    Returns
    -------
    frames : np.ndarray
        Shape (n_frames, H, W[, 3]) — grayscale if roi_coords given,
        colour otherwise.
    nr_frames : int
    fps : float
    """
    cap = cv2.VideoCapture(video_file)
    frames = []

    while True:
        frame_ready, frame = cap.read()
        if not frame_ready:
            break

        if roi_coords is not None:
            mu_y, mu_x = roi_coords
            # Crop a 20×20 ROI around the specified centre
            frame = frame[mu_y - 10: mu_y + 10, mu_x - 10: mu_x + 10, :]
            # Convert RGB to grayscale using luminance weights
            frame = np.dot(frame, [0.299, 0.587, 0.114])

        frames.append(frame)

    frames = np.array(frames)
    nr_frames = len(frames)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sanity check: OpenCV frame count should match what we actually read
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == nr_frames
    cap.release()

    return frames, nr_frames, fps


# ---------------------------------------------------------------------------
# Motion Analysis
# ---------------------------------------------------------------------------

def convert_to_gray(frames):
    """
    Convert an RGB frame array to grayscale using luminance weighting.

    Parameters
    ----------
    frames : np.ndarray, shape (n_frames, H, W, 3)

    Returns
    -------
    frames_gray : np.ndarray, shape (n_frames, H, W)
    """
    frames_gray = np.zeros(frames.shape[:3])
    for i, frame in enumerate(frames):
        frames_gray[i] = np.dot(frame[:, :, :3], [0.299, 0.587, 0.114])
    return frames_gray


def norm_diff_frame(frames_gray):
    """
    Compute normalised frame-to-frame pixel change as a motion signal.

    Each value is the Frobenius norm of the pixel difference between
    consecutive frames, divided by the norm of the previous frame.
    The first entry is NaN (no previous frame to compare against).

    Parameters
    ----------
    frames_gray : np.ndarray, shape (n_frames, H, W)

    Returns
    -------
    change : np.ndarray, shape (n_frames,)
        Normalised inter-frame change; change[0] = NaN.
    """
    diffs = [
        np.linalg.norm(frames_gray[i] - frames_gray[i - 1])
        / np.linalg.norm(frames_gray[i - 1])
        for i in range(1, len(frames_gray))
    ]
    return np.array([np.nan] + diffs)


# ---------------------------------------------------------------------------
# DLC Port-Reach Detection
# ---------------------------------------------------------------------------

def get_port_pos(portkey, data):
    """
    Estimate the inreach and outreach x-positions of a lick port from DLC data.

    Only high-confidence detections (likelihood > DLC_LIKELIHOOD_THRESH) are
    used. Stable positions (small frame-to-frame displacement) are selected,
    and the 10th / 90th percentiles define the two extremes.

    Convention (based on port side):
    - Left port:  inreach = lower x (10th pct),  outreach = upper x (90th pct)
    - Right port: inreach = upper x (90th pct),  outreach = lower x (10th pct)

    Parameters
    ----------
    portkey : str
        DLC body-part key, e.g. 'WaterportLeft' or 'WaterportRight'.
    data : dict-like
        DLC tracking data with columns '<portkey>_x' and '<portkey>_likelihood'.

    Returns
    -------
    inreach : float   x-coordinate when the animal is at the port.
    outreach : float  x-coordinate when the animal is away from the port.
    """
    x_raw = data[portkey + '_x']

    # Keep only high-confidence detections
    x_valid = x_raw[np.array(data[portkey + '_likelihood']) > DLC_LIKELIHOOD_THRESH]

    # Remove frames with large position jumps (tracking noise / outliers)
    diff = np.diff(x_valid)
    diff_with_lead = np.zeros(len(diff) + 1)
    diff_with_lead[1:] = diff
    x_stable = x_valid[diff_with_lead < 20]

    # Use robust percentiles as position extremes
    min_x = np.percentile(x_stable, 10)
    max_x = np.percentile(x_stable, 90)

    # Inreach is the x-value closest to the port spout (side-dependent)
    if portkey.endswith('Right'):
        inreach, outreach = max_x, min_x
    else:
        inreach, outreach = min_x, max_x

    return inreach, outreach


def get_lick_transitions(lick_window):
    """
    Detect onset events in a lick signal by finding transitions from invalid
    (−1 or NaN) to valid lick-position values.

    Parameters
    ----------
    lick_window : array-like
        Sequence of lick x-positions; −1 or NaN indicates no lick detected.

    Returns
    -------
    transitions : list of int/float
        Values at frames where a new lick bout begins.
    """
    transitions = []
    for j in range(1, len(lick_window)):
        prev_val = lick_window[j - 1]
        val = lick_window[j]
        # A transition is a valid frame (int, not −1) preceded by an invalid frame
        if isinstance(val, int) and val != -1:
            if prev_val == -1 or np.isnan(prev_val):
                transitions.append(val)
    return transitions


def get_port_params(data):
    """
    Detect inreach onset frames for both lick ports from DLC tracking data.

    For each port the function:
    1. Estimates inreach / outreach x-positions via `get_port_pos`.
    2. Classifies each frame as inreach (1), outreach (0), or uncertain (NaN).
    3. Detects inreach onsets — transitions from outreach to inreach — while
       requiring at least 90 frames of sustained outreach immediately before
       the onset (preonset mean ≈ 0) to suppress spurious detections.

    Parameters
    ----------
    data : dict-like
        DLC tracking data with columns 'WaterportLeft_x', 'WaterportLeft_likelihood',
        'WaterportRight_x', 'WaterportRight_likelihood'.

    Returns
    -------
    port_dict : dict  {'Left': [onset_frame, ...], 'Right': [onset_frame, ...]}
        Frame indices of confirmed inreach onsets for each port.
    """
    ports = ['Left', 'Right']
    port_dict = {}

    for port in ports:
        print('port:', port)
        portkey = 'Waterport%s' % port

        inreach, outreach = get_port_pos(portkey, data)
        print('inreach:', inreach)
        print('outreach:', outreach)

        # Mask low-confidence frames with NaN
        x_raw = data[portkey + '_x'].copy()
        x_raw[data[portkey + '_likelihood'] < DLC_LIKELIHOOD_THRESH] = np.nan

        # Classify each frame as inreach (1), outreach (0), or NaN
        inreach_bool = []
        for pos in x_raw:
            if np.isnan(pos):
                inreach_bool.append(np.nan)
            elif abs(pos - outreach) < abs(pos - inreach):
                inreach_bool.append(0)  # closer to outreach position
            else:
                inreach_bool.append(1)  # closer to inreach position

        inreach_bool = np.array(inreach_bool)

        # Detect inreach onsets: frame transitions into inreach after sustained outreach
        onset_idx = []
        for i in range(1, len(inreach_bool)):
            current = inreach_bool[i]
            previous = inreach_bool[i - 1]

            if current != 1:
                continue  # only interested in frames where animal is at port

            # Confirm the preceding 90-frame window was fully outreach
            preonset_mean = np.nanmean(inreach_bool[i - 90: i])

            if previous == 0 and preonset_mean == 0:
                # Clean 0→1 transition with confirmed pre-onset outreach
                onset_idx.append(i)
            elif np.isnan(previous) and preonset_mean < 0.01:
                # Transition from uncertain frame, but pre-onset window is clean
                onset_idx.append(i)

        port_dict[port] = onset_idx

    return port_dict