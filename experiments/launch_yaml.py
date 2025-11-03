import atexit
import signal
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict

import numpy as np
from gello.robots.yam import YAMRobot

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


def get_follower_joint_positions(robot):
    """Get follower joint positions from YAM robot.
    
    Args:
        robot: Robot object (could be YAMRobot or wrapped in BimanualRobot)
        
    Returns:
        tuple: (joint_positions, timestamp) where joint_positions is np.ndarray or None, 
               timestamp is int (nanoseconds from perf_counter), or None if not available
    """
    import time
    timestamp = time.perf_counter()
    
    if hasattr(robot, '_robot_l') and hasattr(robot, '_robot_r'):
        left_robot = robot._robot_l
        right_robot = robot._robot_r
        
        left_joints = None
        right_joints = None
        
        if isinstance(left_robot, YAMRobot):
            i2rt_robot_l = left_robot.robot
            left_joints = np.array(i2rt_robot_l.get_joint_pos())
        
        if isinstance(right_robot, YAMRobot):
            i2rt_robot_r = right_robot.robot
            right_joints = np.array(i2rt_robot_r.get_joint_pos())
        
        if left_joints is not None and right_joints is not None:
            joints = np.concatenate((left_joints, right_joints))
            return joints, timestamp
        elif left_joints is not None:
            return left_joints, timestamp
        elif right_joints is not None:
            return right_joints, timestamp
        else:
            return None, timestamp
    
    # No robot or not bimanual - return None with timestamp
    return None, timestamp


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    use_save_interface: bool = False
    """Enable saving data with keyboard interface."""

    output_dir: str = "data/"
    """Path to the data directory."""

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    # Don't call cleanup here - let it run in the finally block
    # Raise KeyboardInterrupt to exit the control loop gracefully
    raise KeyboardInterrupt("Received shutdown signal")


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])
        # agent = None

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        # print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        # print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        # print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        # print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    from gello.utils.control_utils import SaveInterface, run_control_loop

    # Use output_dir directly - caller handles directory structure
    save_dir = args.output_dir

    # Initialize save interface if requested
    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=save_dir,
            agent_name=agent.__class__.__name__,
            expand_user=True,
        )

    # Run main control loop
    try:
        run_control_loop(
            env, 
            agent, 
            save_interface, 
            get_follower_joints_fn=lambda: get_follower_joint_positions(robot)
        )
    except KeyboardInterrupt:
        print("\nControl loop interrupted by user")
    except Exception as e:
        print(f"\nError in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup runs even on normal exit or exceptions
        cleanup()
        print("Exiting...")


if __name__ == "__main__":
    main()
