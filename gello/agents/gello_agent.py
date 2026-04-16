import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gello.agents.agent import Action, Agent
from gello.robots.dynamixel import DynamixelRobot


def _check_ftdi_latency_timer(port: str) -> None:
    """Warn if a GELLO FTDI adapter is not in Linux low-latency mode."""
    try:
        resolved = Path(port).resolve(strict=True)
    except (OSError, RuntimeError):
        return

    tty = resolved.name
    latency_path = Path("/sys/bus/usb-serial/devices") / tty / "latency_timer"
    if not latency_path.exists():
        return

    try:
        latency_ms = int(latency_path.read_text().strip())
    except (OSError, ValueError) as exc:
        print(f"[GELLO] Could not read FTDI latency timer for {tty}: {exc}")
        return

    if latency_ms == 1:
        print(f"[GELLO] FTDI latency timer {tty}=1 ms")
        return

    installer = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "install_ftdi_latency_udev.sh"
    )
    print(
        f"[GELLO] Warning: FTDI latency timer {tty}={latency_ms} ms; "
        "expected 1 ms for GELLO."
    )
    print(f"[GELLO] Runtime fix: echo 1 | sudo tee {latency_path}")
    print(f"[GELLO] Persistent fix: sudo {installer}")


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, float, float]
    """The gripper config of GELLO.

    Tuple of (gripper_joint_id, degrees_at_normalized_zero,
    degrees_at_normalized_one). For the YAM leader, normalized zero is closed
    and normalized one is open.
    """

    gripper_zero_deadband: float = 0.0
    """Snap normalized gripper readings <= this value to exactly 0.0."""

    baudrate: int = 57600
    """The serial baudrate configured in the Dynamixel servos' EEPROM."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self,
        port: str = "/dev/ttyUSB0",
        start_joints: Optional[np.ndarray] = None,
        baudrate: Optional[int] = None,
    ) -> DynamixelRobot:
        if baudrate is None:
            baudrate = self.baudrate
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            baudrate=baudrate,
            gripper_config=self.gripper_config,
            gripper_zero_deadband=self.gripper_zero_deadband,
            start_joints=start_joints,
        )


PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    # xArm
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            3 * np.pi / 2,
            2 * np.pi / 2,
            1 * np.pi / 2,
            4 * np.pi / 2,
            -2 * np.pi / 2 + 2 * np.pi,
            3 * np.pi / 2,
            4 * np.pi / 2,
        ),
        joint_signs=(1, -1, 1, 1, 1, -1, 1),
        gripper_config=(8, 195, 152),
    ),
    # yam
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=[
            0 * np.pi,
            2 * np.pi / 2,
            4 * np.pi / 2,
            6 * np.pi / 6,
            5 * np.pi / 3,
            2 * np.pi / 2,
        ],
        joint_signs=(1, -1, -1, -1, 1, 1),
        gripper_config=(
            7,
            -30,
            24,
        ),  # Reversed: now starts open (-30) and closes on press (24)
    ),
    # Left UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            0,
            1 * np.pi / 2 + np.pi,
            np.pi / 2 + 0 * np.pi,
            0 * np.pi + np.pi / 2,
            np.pi - 2 * np.pi / 2,
            -1 * np.pi / 2 + 2 * np.pi,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 20, -22),
    ),
    # Right UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            np.pi + 0 * np.pi,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            1 * np.pi,
            3 * np.pi / 2,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 286, 248),
    ),
}


class GelloAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        baudrate: Optional[int] = None,
        return_timestamp: bool = False,
        return_velocity: bool = True,
    ):
        self._return_velocity = bool(return_velocity)
        self._return_timestamp = bool(return_timestamp) or self._return_velocity
        _check_ftdi_latency_timer(port)
        # Ensure start_joints is a numpy array if provided
        if start_joints is not None and not isinstance(start_joints, np.ndarray):
            start_joints = np.array(start_joints)
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints, baudrate=baudrate
            )
        else:
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(
                port=port, start_joints=start_joints, baudrate=baudrate
            )

    def act(self, obs: Dict[str, np.ndarray]) -> Action:
        if self._return_velocity:
            pos, vel, timestamp = (
                self._robot.get_joint_state_and_velocity_with_timestamp()
            )
            return {"pos": pos, "vel": vel, "timestamp": timestamp}
        if self._return_timestamp:
            pos, timestamp = self._robot.get_joint_state_with_timestamp()
            return {"pos": pos, "timestamp": timestamp}
        return {"pos": self._robot.get_joint_state()}

    def close(self) -> None:
        if hasattr(self._robot, "close"):
            self._robot.close()
