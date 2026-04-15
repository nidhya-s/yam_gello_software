from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Protocol, Union

import numpy as np


# Actions are either legacy position-only ndarrays or dicts of the form
# {"pos": ndarray, "vel"?: ndarray, "timestamp"?: float, ...}.
# The "timestamp" key, when present, is a perf_counter()-clock timestamp
# captured inside the agent. For live GELLO it is the Dynamixel sample time.
Action = Union[np.ndarray, Dict[str, Any]]


def action_pos(action: Action) -> np.ndarray:
    """Extract the position ndarray from either action encoding."""
    if isinstance(action, dict):
        return action["pos"]
    return action


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> Action:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment. Either a position-only
                ndarray or a dict {"pos": ndarray, "vel"?: ndarray,
                "timestamp"?: float}.
        """
        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="bimanual_agent"
        )

    def act(self, obs: Dict[str, Any]) -> Action:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]

        # The left/right GELLO devices are independent serial chains. Reading
        # them in parallel avoids making the first arm's sample older by the
        # duration of the second arm read.
        left_future = self._executor.submit(self.agent_left.act, left_obs)
        right_future = self._executor.submit(self.agent_right.act, right_obs)
        left_action = left_future.result()
        right_action = right_future.result()

        left_is_dict = isinstance(left_action, dict)
        right_is_dict = isinstance(right_action, dict)
        if left_is_dict != right_is_dict:
            raise TypeError(
                "BimanualAgent: both sub-agents must return the same action "
                f"type; got left={type(left_action).__name__}, "
                f"right={type(right_action).__name__}. Check that both "
                "arms use the same agent class."
            )

        if not left_is_dict:
            return np.concatenate([left_action, right_action])

        combined: Dict[str, Any] = {
            "pos": np.concatenate([left_action["pos"], right_action["pos"]]),
        }
        if "vel" in left_action or "vel" in right_action:
            left_vel = left_action.get("vel")
            right_vel = right_action.get("vel")
            if left_vel is None:
                left_vel = np.zeros_like(left_action["pos"])
            if right_vel is None:
                right_vel = np.zeros_like(right_action["pos"])
            combined["vel"] = np.concatenate([left_vel, right_vel])

        # The current saved joint schema has one control timestamp per bimanual
        # command. Use the left timestamp for backward compatibility.
        ts = left_action.get("timestamp", right_action.get("timestamp"))
        if ts is not None:
            combined["timestamp"] = ts

        return combined
