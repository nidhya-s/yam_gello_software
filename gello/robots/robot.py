from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Protocol

import numpy as np

from gello.agents.agent import Action, action_pos


class Robot(Protocol):
    """Robot protocol.

    A protocol for a robot that can be controlled.
    """

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        raise NotImplementedError

    @abstractmethod
    def command_joint_state(self, action: Action) -> None:
        """Command the leader robot to a given state.

        Args:
            action: Action dict containing at least {"pos": ndarray}.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        raise NotImplementedError


class PrintRobot(Robot):
    """A robot that prints the commanded joint state."""

    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        self._joint_state = np.zeros(num_dofs)
        self._dont_print = dont_print

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, action: Action) -> None:
        joint_state = action_pos(action)
        assert len(joint_state) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, "
            f"got {len(joint_state)}."
        )
        self._joint_state = joint_state
        if not self._dont_print:
            print(self._joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()
        pos_quat = np.zeros(7)
        return {
            "joint_positions": joint_state,
            "joint_velocities": joint_state,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(0),
        }


class BimanualRobot(Robot):
    def __init__(
        self,
        robot_l: Robot,
        robot_r: Robot,
        parallel_commands: bool = False,
    ):
        self._robot_l = robot_l
        self._robot_r = robot_r
        self._parallel_commands = bool(parallel_commands)
        self._executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="bimanual_robot")
            if self._parallel_commands
            else None
        )

    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    def command_joint_state(self, action: Action) -> None:
        joint_state = action_pos(action)
        split = self._robot_l.num_dofs()
        left_action: Action = {"pos": joint_state[:split]}
        right_action: Action = {"pos": joint_state[split:]}
        if "vel" in action:
            left_action["vel"] = action["vel"][:split]
            right_action["vel"] = action["vel"][split:]

        if self._executor is not None:
            left_future = self._executor.submit(
                self._robot_l.command_joint_state, left_action
            )
            right_future = self._executor.submit(
                self._robot_r.command_joint_state, right_action
            )
            left_error = None
            right_error = None
            try:
                left_future.result()
            except BaseException as exc:
                left_error = exc
            try:
                right_future.result()
            except BaseException as exc:
                right_error = exc
            if left_error is not None:
                raise left_error
            if right_error is not None:
                raise right_error
            return

        self._robot_l.command_joint_state(left_action)
        self._robot_r.command_joint_state(right_action)

    def get_observations(self) -> Dict[str, np.ndarray]:
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs

    def close(self) -> None:
        for robot in (self._robot_l, self._robot_r):
            if hasattr(robot, "close"):
                robot.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)


def main():
    pass


if __name__ == "__main__":
    main()
