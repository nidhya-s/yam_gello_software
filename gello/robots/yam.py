import atexit
from typing import Dict

import numpy as np

from gello.agents.agent import Action, action_pos
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

        self._channel = channel
        self._ff_eta = float(velocity_feedforward_eta)
        self._ff_max_vel = float(velocity_feedforward_max_vel)
        self._ff_frame_count = 0
        self._ff_clip_count = 0
        self._ff_max_abs_seen = 0.0
        self._closed = False
        atexit.register(self._print_ff_summary)

    def num_dofs(self) -> int:
        return 7

    def _to_7(self, values: np.ndarray) -> np.ndarray:
        if len(values) > 7:
            return values[:7]
        if len(values) < 7:
            return np.pad(values, (0, 7 - len(values)), "constant")
        return values

    def get_joint_state(self) -> np.ndarray:
        joint_pos = self._to_7(np.asarray(self.robot.get_joint_pos()))
        self._joint_state = joint_pos
        return self._joint_state

    def command_joint_state(self, action: Action) -> None:
        joint_state = np.asarray(action_pos(action), dtype=np.float64)
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        target_pos = self._to_7(joint_state.copy())
        raw_vel = action.get("vel")
        if raw_vel is None or self._ff_eta == 0.0:
            dt = 0.01
            self._joint_velocities = (target_pos - self._joint_state) / dt
            self._joint_state = target_pos
            self.robot.command_joint_pos(target_pos.copy())
            return

        vel_ff = self._ff_eta * np.asarray(raw_vel, dtype=np.float64)
        vel_ff = self._to_7(vel_ff)
        if vel_ff.size > 0:
            vel_ff[-1] = 0.0

        n_clipped = int(np.count_nonzero(np.abs(vel_ff) > self._ff_max_vel))
        if n_clipped > 0:
            vel_ff = np.clip(vel_ff, -self._ff_max_vel, self._ff_max_vel)
        self._ff_frame_count += 1
        self._ff_clip_count += n_clipped
        max_this_frame = float(np.max(np.abs(vel_ff))) if vel_ff.size else 0.0
        if max_this_frame > self._ff_max_abs_seen:
            self._ff_max_abs_seen = max_this_frame

        assert np.all(np.isfinite(target_pos)), (
            f"[YAMRobot {self._channel}] non-finite target_pos: {target_pos}"
        )
        assert np.all(np.isfinite(vel_ff)), (
            f"[YAMRobot {self._channel}] non-finite vel_ff: {vel_ff}"
        )

        self._joint_state = target_pos
        self._joint_velocities = vel_ff
        self.robot.command_joint_state(
            {"pos": target_pos.copy(), "vel": vel_ff.copy()}
        )

    def _print_ff_summary(self) -> None:
        """Printed at interpreter shutdown via atexit. No hardware calls."""
        if self._ff_frame_count == 0:
            return
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
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }

    def get_joint_pos(self):
        return self._to_7(np.asarray(self.robot.get_joint_pos()))

    def command_joint_pos(self, target_pos):
        target_pos = self._to_7(np.asarray(target_pos, dtype=np.float64))
        self._joint_state = target_pos.copy()
        self.robot.command_joint_pos(target_pos.copy())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        stop_event = getattr(self.robot, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        motor_chain = getattr(self.robot, "motor_chain", None)
        if motor_chain is not None:
            try:
                motor_chain.close()
            except Exception as exc:
                print(f"[YAMRobot {self._channel}] error closing motor chain: {exc}")
            motor_interface = (
                None
                if hasattr(motor_chain, "close")
                else getattr(motor_chain, "motor_interface", None)
            )
            if motor_interface is not None and hasattr(motor_interface, "close"):
                try:
                    motor_interface.close()
                except Exception:
                    pass

        server_thread = getattr(self.robot, "_server_thread", None)
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=2.0)
            if server_thread.is_alive():
                print(
                    f"[YAMRobot {self._channel}] warning: i2rt robot_server "
                    "thread did not stop cleanly"
                )


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
