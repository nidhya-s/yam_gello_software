import atexit
from typing import Any, Dict, Union

import numpy as np

from gello.robots.robot import Robot


class YAMRobot(Robot):
    """A YAM robot (6 arm joints + 1 gripper)."""

    def __init__(
        self,
        channel="can0",
        velocity_feedforward_eta: float = 0.9,
        velocity_feedforward_max_vel: float = 6.0,
    ):
        from i2rt.robots.get_robot import get_yam_robot

        self.robot = get_yam_robot(channel=channel)

        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = np.zeros(7)
        self._joint_velocities = np.zeros(7)
        self._gripper_state = 0.0

        # ExtremControl-style velocity feedforward. eta scales the agent's
        # desired velocity; max_vel is a motor-side safety clamp applied AFTER
        # the eta scale ("max desired FF velocity actually sent to motor").
        # Default eta=0.9 is the tuned GELLO/YAM setting. Explicit eta=0.0
        # keeps a position-only path for latency baselines and safety tests.
        self._channel = channel
        self._ff_eta = float(velocity_feedforward_eta)
        self._ff_max_vel = float(velocity_feedforward_max_vel)
        self._ff_frame_count = 0
        self._ff_clip_count = 0
        self._ff_max_abs_seen = 0.0
        atexit.register(self._print_ff_summary)

    def num_dofs(self) -> int:
        return 7

    def _to_7(self, x: np.ndarray) -> np.ndarray:
        if len(x) > 7:
            return x[:7]
        if len(x) < 7:
            return np.pad(x, (0, 7 - len(x)), "constant")
        return x

    def get_joint_state(self) -> np.ndarray:
        joint_pos = self._to_7(self.robot.get_joint_pos())
        self._joint_state = joint_pos
        return self._joint_state

    def command_joint_state(
        self, joint_state: Union[np.ndarray, Dict[str, Any]]
    ) -> None:
        # ndarray actions (legacy position-only path: move_to_start_position
        # ramps, and live teleop before Pass 3) still route through
        # command_joint_pos so they cannot accidentally engage the vel path.
        if not isinstance(joint_state, dict):
            pos = np.asarray(joint_state, dtype=np.float64)
            assert (
                len(pos) == self.num_dofs()
            ), f"Expected {self.num_dofs()} joint values, got {len(pos)}"
            target_pos = self._to_7(pos.copy())
            self._joint_state = target_pos
            self.robot.command_joint_pos(target_pos.copy())
            return

        # Dict actions may carry timestamp-only metadata for lag analysis.
        # If no velocity target is present, or eta is explicitly disabled,
        # keep the legacy position-only motor path.
        pos = np.asarray(joint_state["pos"], dtype=np.float64)
        assert (
            len(pos) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(pos)}"
        target_pos = self._to_7(pos.copy())

        raw_vel = joint_state.get("vel")
        if raw_vel is None or self._ff_eta == 0.0:
            self._joint_state = target_pos
            self.robot.command_joint_pos(target_pos.copy())
            return

        # Velocity-bearing dict actions go through i2rt's velocity-aware path,
        # which implements the ExtremControl PD law
        #     tau = kp*(q_target - q) + kd*(eta*vel_target - qdot)
        # by dispatching {"pos", "vel"} with the already-configured kp/kd.
        vel_ff = self._ff_eta * np.asarray(raw_vel, dtype=np.float64)
        vel_ff = self._to_7(vel_ff)

        # Boundary safety: re-zero the gripper vel even though the agent
        # already does it — if a future agent forgets, the motor-side still
        # won't get a velocity target on the indirect-drive joint.
        if vel_ff.size > 0:
            vel_ff[-1] = 0.0

        # Clamp to the motor-side safety limit AFTER the eta scale, so the
        # knob means "max desired FF velocity actually sent to motor".
        n_clipped = int(np.count_nonzero(np.abs(vel_ff) > self._ff_max_vel))
        if n_clipped > 0:
            vel_ff = np.clip(vel_ff, -self._ff_max_vel, self._ff_max_vel)
        self._ff_frame_count += 1
        self._ff_clip_count += n_clipped
        max_this_frame = float(np.max(np.abs(vel_ff))) if vel_ff.size else 0.0
        if max_this_frame > self._ff_max_abs_seen:
            self._ff_max_abs_seen = max_this_frame

        # Motor-boundary defense. RobotEnv.step already filters non-finite pos
        # and vel, but this is the last chance before i2rt, and an accidental
        # NaN in the PD law produces unbounded torque.
        assert np.all(np.isfinite(target_pos)), (
            f"[YAMRobot {self._channel}] non-finite target_pos: {target_pos}"
        )
        assert np.all(np.isfinite(vel_ff)), (
            f"[YAMRobot {self._channel}] non-finite vel_ff: {vel_ff}"
        )

        self._joint_state = target_pos
        self.robot.command_joint_state(
            {"pos": target_pos.copy(), "vel": vel_ff.copy()}
        )

    def _print_ff_summary(self) -> None:
        """Printed at interpreter shutdown via atexit. No hardware calls."""
        if self._ff_frame_count == 0:
            return
        # Denominator excludes the gripper slot because gripper vel is force-
        # zeroed and can never be clipped — including it would dilute the
        # percentage without adding information.
        ff_enabled_joints = max(self.num_dofs() - 1, 1)
        total_elems = self._ff_frame_count * ff_enabled_joints
        clip_rate = self._ff_clip_count / total_elems if total_elems > 0 else 0.0
        print(
            f"[YAMRobot {self._channel}] velocity FF summary: "
            f"eta={self._ff_eta:.3f}  max_vel={self._ff_max_vel:.2f} rad/s  "
            f"frames={self._ff_frame_count}  "
            f"clipped_elements={self._ff_clip_count}/{total_elems} "
            f"(over {ff_enabled_joints} FF-enabled joints, "
            f"{clip_rate * 100:.3f}%)  "
            f"max_abs_vel_sent={self._ff_max_abs_seen:.3f} rad/s"
        )

    def get_observations(self) -> Dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }

    def get_joint_pos(self):
        return self._to_7(self.robot.get_joint_pos())

    def command_joint_pos(self, target_pos):
        target_pos = self._to_7(np.asarray(target_pos, dtype=np.float64))
        self._joint_state = target_pos.copy()
        self.robot.command_joint_pos(target_pos.copy())


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
