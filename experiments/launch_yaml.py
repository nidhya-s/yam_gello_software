import atexit
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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
active_executors = []
active_agents = []
cleanup_in_progress = False


def dump_alive_threads(context: str) -> None:
    current_thread = threading.current_thread()
    frames = sys._current_frames()
    alive_threads = [
        thread
        for thread in threading.enumerate()
        if thread is not current_thread and thread.is_alive()
    ]
    if not alive_threads:
        return

    print(f"[shutdown] alive threads after {context}:")
    for thread in alive_threads:
        print(
            f"[shutdown] thread name={thread.name!r} "
            f"daemon={thread.daemon} ident={thread.ident}"
        )
        frame = frames.get(thread.ident)
        if frame is not None:
            stack = "".join(traceback.format_stack(frame))
            print(stack.rstrip())


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if not (active_servers or active_threads or active_agents or active_executors):
        return

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
        if thread.is_alive():
            print(f"Warning: thread {thread.name} did not stop cleanly")

    for agent in active_agents:
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception as e:
            print(f"Error closing agent: {e}")

    for executor in active_executors:
        executor.shutdown(wait=True, cancel_futures=True)

    dump_alive_threads("cleanup")
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


def _read_yam_joint_positions(robot):
    if isinstance(robot, YAMRobot):
        return np.array(robot.robot.get_joint_pos())
    return None


def get_follower_joint_positions(robot, executor=None):
    """Get follower joint positions from YAM robot.
    
    Args:
        robot: Robot object (could be YAMRobot or wrapped in BimanualRobot)
        
    Returns:
        tuple: (joint_positions, timestamp) where timestamp is float seconds from
               time.perf_counter() captured after the follower reads complete.
    """
    import time
    
    if hasattr(robot, '_robot_l') and hasattr(robot, '_robot_r'):
        left_robot = robot._robot_l
        right_robot = robot._robot_r

        if executor is None:
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="follower_joint_reader"
            ) as local_executor:
                left_future = local_executor.submit(
                    _read_yam_joint_positions, left_robot
                )
                right_future = local_executor.submit(
                    _read_yam_joint_positions, right_robot
                )
                left_joints = left_future.result()
                right_joints = right_future.result()
        else:
            left_future = executor.submit(_read_yam_joint_positions, left_robot)
            right_future = executor.submit(_read_yam_joint_positions, right_robot)
            left_joints = left_future.result()
            right_joints = right_future.result()

        timestamp = time.perf_counter()
        if left_joints is not None and right_joints is not None:
            joints = np.concatenate((left_joints, right_joints))
            return joints, timestamp
        elif left_joints is not None:
            return left_joints, timestamp
        elif right_joints is not None:
            return right_joints, timestamp
        else:
            return None, timestamp

    if isinstance(robot, YAMRobot):
        joints = _read_yam_joint_positions(robot)
        return joints, time.perf_counter()

    # No YAM robot - return None with timestamp
    return None, time.perf_counter()


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
    active_agents.append(agent)

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
        parallel_commands = isinstance(left_robot, YAMRobot) and isinstance(
            right_robot, YAMRobot
        )
        robot = BimanualRobot(
            left_robot,
            right_robot,
            parallel_commands=parallel_commands,
        )
        if parallel_commands:
            print("Bimanual YAM command dispatch: parallel")

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

        # Keep this non-daemon so cleanup exposes shutdown regressions instead
        # of silently abandoning the server thread.
        server_thread = threading.Thread(
            target=robot.serve,
            name="gello_robot_server",
            daemon=False,
        )
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
        server_thread = threading.Thread(
            target=server.serve,
            name="gello_hardware_server",
            daemon=False,
        )
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

    follower_reader = None
    if save_interface is not None:
        follower_executor = None
        if bimanual:
            follower_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="follower_joint_reader"
            )
            active_executors.append(follower_executor)
        follower_reader = lambda: get_follower_joint_positions(
            robot, executor=follower_executor
        )

    # Run main control loop
    try:
        run_control_loop(
            env,
            agent,
            save_interface,
            get_follower_joints_fn=follower_reader,
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
