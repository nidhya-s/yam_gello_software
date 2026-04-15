import pickle
from pathlib import Path
import time
from typing import Dict

import numpy as np


def save_frame(
    file: Path,
    timestamp: time.perf_counter_ns,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs_copy = dict(obs)
    # Only keep: joint_positions, control, control_timestamp, follower_joints, follower_timestamp
    filtered_obs = {}
    
    # Keep joint_positions just in case
    if "joint_positions" in obs_copy:
        filtered_obs["joint_positions"] = obs_copy["joint_positions"]
    
    # Store control and control timestamp
    filtered_obs["control"] = action
    if "control_data_timestamp" in obs_copy:
        filtered_obs["control_timestamp"] = obs_copy["control_data_timestamp"]
    
    # Store follower joints and follower timestamp
    if "follower_joint_positions" in obs_copy:
        filtered_obs["follower_joints"] = obs_copy["follower_joint_positions"]
    if "follower_joint_timestamp" in obs_copy:
        filtered_obs["follower_timestamp"] = obs_copy["follower_joint_timestamp"]
    # make folder if it doesn't exist
    folder = file.parent
    folder.mkdir(exist_ok=True, parents=True)
    # Convert ns timestamp to ISO string (if desired, or save as str(ns))
    # If timestamp is int (from perf_counter_ns), store it as string
    timestamp_str = str(timestamp)
    recorded_file = folder / (timestamp_str + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(filtered_obs, f)
