import time
from typing import Any, Dict, Optional
import numpy as np
from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot
class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate
    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()
class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._cycle_time = 1.0 / control_rate_hz
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def _command_action(self, action: Any) -> None:
        """Validate and send an action without rate sleeping."""
        num_dofs = self._robot.num_dofs()
        if isinstance(action, dict):
            pos = action["pos"]
            assert len(pos) == num_dofs, (
                f"input pos:{len(pos)}, robot:{num_dofs}"
            )
            assert np.all(np.isfinite(pos)), "action pos must be finite"
            # Build a new dict so we never mutate the caller's action (it is
            # still used for save/debug) and so timestamp/extra keys don't
            # flow into the robot layer.
            robot_action: Dict[str, Any] = {"pos": pos}
            if "vel" in action and action["vel"] is not None:
                vel = action["vel"]
                assert len(vel) == num_dofs, (
                    f"input vel:{len(vel)}, robot:{num_dofs}"
                )
                assert np.all(np.isfinite(vel)), "action vel must be finite"
                robot_action["vel"] = vel
            self._robot.command_joint_state(robot_action)
        else:
            assert len(action) == num_dofs, (
                f"input:{len(action)}, robot:{num_dofs}"
            )
            self._robot.command_joint_state(action)

    def begin_step(self, action: Any) -> float:
        """Command an action immediately and return the step start time."""
        t_step_start = time.perf_counter()
        self._command_action(action)
        return t_step_start

    def finish_step(self, _step_start: float) -> Dict[str, Any]:
        """Rate sleep and return observations."""
        self._rate.sleep()
        return self.get_obs()

    def robot(self) -> Robot:
        """Get the robot object.
        Returns:
            robot: the robot object.
        """
        return self._robot
    def __len__(self):
        return 0
    def step(self, action: Any) -> Dict[str, Any]:
        """Step the environment forward.
        Args:
            action: either a legacy position-only ndarray of length num_dofs,
                or a dict {"pos": ndarray, "vel"?: ndarray, "timestamp"?: float}.
                The "timestamp" key is metadata for lag analysis and is not
                forwarded to the robot.
        Returns:
            obs: observation from the environment.
        """
        return self.finish_step(self.begin_step(action))
    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.
        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth
        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        return observations
def main() -> None:
    pass
if __name__ == "__main__":
    main()
