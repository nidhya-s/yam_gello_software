from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Protocol

import numpy as np


# Actions are always dictionaries. "pos" is required; "vel" and "timestamp"
# are optional metadata used by velocity-aware robots and recording.
Action = Dict[str, Any]


def action_pos(action: Action) -> np.ndarray:
    """Extract the required position command from an action dictionary."""
    return action["pos"]


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> Action:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: dict containing at least {"pos": ndarray}.
        """
        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> Action:
        return {"pos": np.zeros(self.num_dofs)}


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

        combined: Action = {
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

        # The saved joint schema has one control timestamp per bimanual command.
        # Use the left timestamp as the representative leader sample time.
        timestamp = left_action.get("timestamp", right_action.get("timestamp"))
        if timestamp is not None:
            combined["timestamp"] = timestamp

        return combined

    def close(self) -> None:
        for agent in (self.agent_left, self.agent_right):
            if hasattr(agent, "close"):
                agent.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
