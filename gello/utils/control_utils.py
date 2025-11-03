"""Shared utilities for robot control loops."""

from datetime import datetime
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from gello.agents.agent import Agent
from gello.env import RobotEnv

DEFAULT_MAX_JOINT_DELTA = 1.0


def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually.

    Args:
        env: Robot environment
        agent: Agent that provides target position
        max_delta: Maximum joint delta per step
        steps: Number of steps for gradual movement

    Returns:
        bool: True if successful, False if position too far
    """
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = DEFAULT_MAX_JOINT_DELTA
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return False

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    return True


class SaveInterface:
    """Handles keyboard-based data saving interface."""

    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        """Initialize save interface.

        Args:
            data_dir: Base directory for saving data
            agent_name: Name of agent (used for subdirectory)
            expand_user: Whether to expand ~ in data_dir path
        """
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        # Use data_dir directly without creating timestamp subdirectory
        self.save_path: Optional[Path] = self.data_dir
        # self.save_path.mkdir(parents=True, exist_ok=True)
        print(f"Automatic save mode enabled. Saving to {self.save_path} every step.")

    def update(self, obs: Dict[str, Any], action: np.ndarray, control_data_timestamp: Optional[int] = None) -> Optional[str]:
        """Update save interface and handle saving.

        Args:
            obs: Current observations
            action: Current action
            control_data_timestamp: Timestamp (nanoseconds) when control data was gathered

        Returns:
            Optional[str]: "quit" if user wants to exit, None otherwise
        """
        from gello.data_utils.format_obs import save_frame

        cur_time = time.perf_counter_ns()

        if self.save_path is not None:
            # Pass a dummy file path - save_frame will use parent directory and create timestamped files
            save_file = self.save_path / "data.pkl"
            # Store control_data_timestamp in obs if provided
            if control_data_timestamp is not None:
                obs["control_data_timestamp"] = control_data_timestamp
            save_frame(save_file, cur_time, obs, action)

        return None


def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
    get_follower_joints_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
) -> None:
    """Run the main control loop with exponential smoothing.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
        get_follower_joints_fn: Optional function to get follower joint positions
        smoothing_alpha: Exponential smoothing factor (0.0=no smoothing, 0.9=very smooth)
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

    start_time = time.time()
    obs = env.get_obs()

    while True:
        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "

            if colors_available:
                print(
                    colored(message, color="white", attrs=["bold"]), end="", flush=True
                )
            else:
                print(message, end="", flush=True)

        # Track timestamp when control action is being retrieved
        control_data_timestamp = time.perf_counter()
        action = agent.act(obs)
        
        # Get follower joint positions and add to observation for saving
        if get_follower_joints_fn is not None:
            result = get_follower_joints_fn()
            if result is not None:
                follower_joints, follower_timestamp = result
                if follower_joints is not None:
                    obs["follower_joint_positions"] = follower_joints
                    obs["follower_joint_timestamp"] = follower_timestamp

        # Handle save interface
        if save_interface is not None:
            result = save_interface.update(obs, action, control_data_timestamp)
            if result == "quit":
                break
        
        obs = env.step(action)
