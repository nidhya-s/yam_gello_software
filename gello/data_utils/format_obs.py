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
    # Only keep specific keys: joint_positions, joint_velocities, gripper_position, and add control
    filtered_obs = {}
    for key in ["joint_positions", "joint_velocities", "gripper_position"]:
        if key in obs_copy:
            filtered_obs[key] = obs_copy[key]
    filtered_obs["control"] = action
    # make folder if it doesn't exist
    folder = file.parent
    folder.mkdir(exist_ok=True, parents=True)
    # Convert ns timestamp to ISO string (if desired, or save as str(ns))
    # If timestamp is int (from perf_counter_ns), store it as string
    timestamp_str = str(timestamp)
    recorded_file = folder / (timestamp_str + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(filtered_obs, f)
