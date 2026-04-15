"""Shared utilities for robot control loops."""

import copy
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from gello.agents.agent import Agent, action_pos
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
    # Unwrap dict actions to position-only — move_to_start_position is a pure
    # position ramp and has no use for the feedforward velocity field.
    start_pos = action_pos(agent.act(env.get_obs()))
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
        command_joints = action_pos(agent.act(obs))
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

    def update(
        self,
        obs: Dict[str, Any],
        action: Any,
        control_data_timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Update save interface and handle saving.

        Args:
            obs: Current observations
            action: Current action — either a legacy position-only ndarray or
                a dict {"pos": ndarray, "vel"?: ndarray, "timestamp"?: float}
                emitted by agents that compute feedforward velocity.
            control_data_timestamp: Float seconds from time.perf_counter(),
                same monotonic clock as follower_joint_timestamp. For dict
                actions this may be the leader hardware sample timestamp
                from action["timestamp"]; otherwise it is captured right
                after agent.act(). Used to synchronize GELLO commands with
                follower and camera streams.

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


def _copy_record_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy_record_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_record_value(v) for v in value)
    return copy.deepcopy(value)


def _minimal_record_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Copy only fields that save_frame can persist."""
    keep = ("joint_positions",)
    return {k: _copy_record_value(obs[k]) for k in keep if k in obs}


class AsyncRecordingWorker:
    """Background follower-read and disk-save worker for recording runs."""

    def __init__(
        self,
        save_interface: SaveInterface,
        get_follower_joints_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
        max_queue_size: int = 512,
    ):
        self._save_interface = save_interface
        self._get_follower_joints_fn = get_follower_joints_fn
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._dropped = 0
        self._saved = 0
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="gello_async_recorder", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        obs: Dict[str, Any],
        action: Any,
        control_data_timestamp: float,
    ) -> None:
        if self._error is not None:
            raise RuntimeError("Async recording worker failed") from self._error

        record = {
            "obs": _minimal_record_obs(obs),
            "action": _copy_record_value(action),
            "control_data_timestamp": float(control_data_timestamp),
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            print("[recording] warning: async recorder did not stop within timeout")
        if self._dropped:
            print(f"[recording] warning: dropped {self._dropped} frames")
        if self._saved:
            print(f"[recording] async recorder saved {self._saved} frames")
        if self._error is not None:
            raise RuntimeError("Async recording worker failed") from self._error

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                self._write_record(record)
                self._saved += 1
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self._queue.task_done()

    def _write_record(self, record: Dict[str, Any]) -> None:
        obs = record["obs"]
        action = record["action"]
        control_data_timestamp = record["control_data_timestamp"]

        if self._get_follower_joints_fn is not None:
            result = self._get_follower_joints_fn()
            if result is not None:
                follower_joints, follower_timestamp = result
                if follower_joints is not None:
                    obs["follower_joint_positions"] = _copy_record_value(
                        follower_joints
                    )
                    obs["follower_joint_timestamp"] = follower_timestamp

        result = self._save_interface.update(obs, action, control_data_timestamp)
        if result == "quit":
            raise KeyboardInterrupt("async recorder requested quit")


def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
    get_follower_joints_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
) -> None:
    """Run the main control loop.

    In recording mode, the critical path is:
        agent.act() -> env.begin_step(action) -> async recorder enqueue -> rate sleep

    The async recorder performs follower reads and disk writes off the command
    path, while preserving exact timestamps for lag analysis.
    """
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
    async_recorder = (
        AsyncRecordingWorker(save_interface, get_follower_joints_fn)
        if save_interface is not None
        else None
    )

    try:
        while True:
            if print_timing:
                num = time.time() - start_time
                message = f"\rTime passed: {round(num, 2)}          "
                if colors_available:
                    print(
                        colored(message, color="white", attrs=["bold"]),
                        end="",
                        flush=True,
                    )
                else:
                    print(message, end="", flush=True)

            action = agent.act(obs)

            # Prefer the agent's internally captured timestamp when present.
            # For live GelloAgent this is the Dynamixel sample time.
            if isinstance(action, dict) and "timestamp" in action:
                control_data_timestamp = float(action["timestamp"])
            else:
                control_data_timestamp = time.perf_counter()

            if async_recorder is not None:
                step_start = env.begin_step(action)
                async_recorder.submit(obs, action, control_data_timestamp)
                obs = env.finish_step(step_start)
                continue

            # Non-recording path. Keep this lean; launch_yaml only passes a
            # follower reader when recording, so this path is just command+sleep.
            obs = env.step(action)
    finally:
        if async_recorder is not None:
            async_recorder.close()
