import pickle
from pathlib import Path
from typing import Any, Dict

from gello.agents.agent import Action, action_pos


def save_frame(
    file: Path,
    timestamp: int,
    obs: Dict[str, Any],
    action: Action,
) -> None:
    obs_copy = dict(obs)
    # Keep the saved control schema numeric even though live actions are dicts.
    filtered_obs = {}

    # Keep joint_positions just in case
    if "joint_positions" in obs_copy:
        filtered_obs["joint_positions"] = obs_copy["joint_positions"]

    filtered_obs["control"] = action_pos(action)
    if "vel" in action and action["vel"] is not None:
        filtered_obs["control_vel"] = action["vel"]
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
