#!/usr/bin/env python3
"""One-shot baud-rate migration for a GELLO Dynamixel chain.

Dry-run by default. Add --apply to write EEPROM.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SDK_SRC = Path(__file__).resolve().parents[1] / "third_party" / "DynamixelSDK" / "python" / "src"
if SDK_SRC.exists():
    sys.path.insert(0, str(SDK_SRC))


ADDR_BAUD_RATE = 8
ADDR_RETURN_DELAY_TIME = 9
ADDR_TORQUE_ENABLE = 64

BAUD_TO_CODE = {
    9600: 0,
    57600: 1,
    115200: 2,
    1_000_000: 3,
    2_000_000: 4,
    3_000_000: 5,
    4_000_000: 6,
    4_500_000: 7,
}
CODE_TO_BAUD = {code: baud for baud, code in BAUD_TO_CODE.items()}


def load_sdk() -> tuple[Any, Any, Any, int]:
    try:
        from dynamixel_sdk.group_sync_write import GroupSyncWrite
        from dynamixel_sdk.packet_handler import PacketHandler
        from dynamixel_sdk.port_handler import PortHandler
        from dynamixel_sdk.robotis_def import COMM_SUCCESS
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Dynamixel SDK dependency. Run this from the GELLO environment "
            "with dynamixel_sdk/pyserial installed, or add the vendored SDK to "
            "PYTHONPATH."
        ) from exc
    return GroupSyncWrite, PacketHandler, PortHandler, COMM_SUCCESS


def open_port(device: str, baudrate: int, port_handler_cls: Any) -> Any:
    port = port_handler_cls(device)
    if not port.openPort():
        raise RuntimeError(f"failed to open {device}")
    if not port.setBaudRate(baudrate):
        port.closePort()
        raise RuntimeError(f"failed to set {device} baudrate to {baudrate}")
    return port


def comm_error(
    packet: Any, result: int, error: int, context: str, comm_success: int
) -> str:
    parts = []
    if result != comm_success:
        parts.append(packet.getTxRxResult(result))
    if error != 0:
        parts.append(packet.getRxPacketError(error))
    detail = "; ".join(parts) if parts else "unknown communication error"
    return f"{context}: {detail}"


def read_byte(
    port: Any, packet: Any, comm_success: int, dxl_id: int, address: int
) -> int:
    value, result, error = packet.read1ByteTxRx(port, dxl_id, address)
    if result != comm_success or error != 0:
        raise RuntimeError(
            comm_error(
                packet, result, error, f"id {dxl_id} read {address}", comm_success
            )
        )
    return int(value)


def sync_write_byte(
    group_sync_write_cls: Any,
    port: Any,
    packet: Any,
    comm_success: int,
    ids: Sequence[int],
    address: int,
    value: int,
) -> None:
    writer = group_sync_write_cls(port, packet, address, 1)
    for dxl_id in ids:
        if not writer.addParam(int(dxl_id), [int(value) & 0xFF]):
            raise RuntimeError(f"failed to add id {dxl_id} to sync write at {address}")
    result = writer.txPacket()
    writer.clearParam()
    if result != comm_success:
        raise RuntimeError(
            comm_error(packet, result, 0, f"sync write {address}", comm_success)
        )


def set_usb_latency(device: str, value: int) -> None:
    tty = Path(device).resolve().name
    latency_path = Path("/sys/bus/usb-serial/devices") / tty / "latency_timer"
    if not latency_path.exists():
        print(f"[latency] {latency_path} not found; skipping")
        return

    old = latency_path.read_text().strip()
    try:
        latency_path.write_text(f"{int(value)}\n")
        new = latency_path.read_text().strip()
        print(f"[latency] {tty}: {old} -> {new} ms")
    except PermissionError:
        print(
            f"[latency] {tty}: currently {old} ms. Run:\n"
            f"  echo {int(value)} | sudo tee {latency_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set Dynamixel X-series baud EEPROM via Protocol 2.0. "
            "Uses sync-write for the baud register so all selected IDs switch together."
        )
    )
    parser.add_argument("--port", required=True, help="Dynamixel serial device")
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=list(range(1, 8)),
        help="servo IDs on this chain; default: 1 2 3 4 5 6 7",
    )
    parser.add_argument(
        "--old-baud",
        type=int,
        default=57_600,
        choices=sorted(BAUD_TO_CODE),
        help="current servo baudrate",
    )
    parser.add_argument(
        "--new-baud",
        type=int,
        default=1_000_000,
        choices=sorted(BAUD_TO_CODE),
        help="target servo baudrate",
    )
    parser.add_argument(
        "--return-delay",
        type=int,
        default=0,
        help="set Return Delay Time(9), units are 2 us; use -1 to leave unchanged",
    )
    parser.add_argument(
        "--set-usb-latency",
        type=int,
        default=None,
        help="also set Linux FTDI latency_timer to this many ms, usually 1",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write EEPROM; without this the script only reads and prints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = [int(x) for x in args.ids]
    new_code = BAUD_TO_CODE[args.new_baud]
    group_sync_write_cls, packet_handler_cls, port_handler_cls, comm_success = load_sdk()

    print(
        f"Opening {args.port} at {args.old_baud} baud for IDs {ids}. "
        "Torque will be disabled before any EEPROM write."
    )
    packet = packet_handler_cls(2.0)
    port = open_port(args.port, args.old_baud, port_handler_cls)
    try:
        print("Current baud EEPROM values:")
        for dxl_id in ids:
            code = read_byte(port, packet, comm_success, dxl_id, ADDR_BAUD_RATE)
            baud = CODE_TO_BAUD.get(code, f"unknown code {code}")
            print(f"  id {dxl_id}: code={code} baud={baud}")

        if not args.apply:
            print("\nDry run only. Re-run with --apply to write EEPROM.")
            return

        print("\nDisabling torque on selected IDs...")
        sync_write_byte(
            group_sync_write_cls, port, packet, comm_success, ids, ADDR_TORQUE_ENABLE, 0
        )

        if args.return_delay >= 0:
            print(f"Setting Return Delay Time(9) to {args.return_delay}...")
            sync_write_byte(
                group_sync_write_cls,
                port,
                packet,
                comm_success,
                ids,
                ADDR_RETURN_DELAY_TIME,
                args.return_delay,
            )

        print(f"Setting Baud Rate(8) to code {new_code} ({args.new_baud} baud)...")
        sync_write_byte(
            group_sync_write_cls,
            port,
            packet,
            comm_success,
            ids,
            ADDR_BAUD_RATE,
            new_code,
        )
    finally:
        port.closePort()

    time.sleep(0.25)

    if args.set_usb_latency is not None:
        set_usb_latency(args.port, args.set_usb_latency)

    print(f"\nVerifying at {args.new_baud} baud...")
    verify_port = open_port(args.port, args.new_baud, port_handler_cls)
    try:
        for dxl_id in ids:
            code = read_byte(
                verify_port, packet, comm_success, dxl_id, ADDR_BAUD_RATE
            )
            status = "OK" if code == new_code else f"expected {new_code}"
            print(f"  id {dxl_id}: code={code} ({status})")
    finally:
        verify_port.closePort()

    print(
        "\nDone. Update the GELLO config to use the same baudrate, e.g. "
        f"`baudrate: {args.new_baud}` under the agent or Dynamixel config."
    )


if __name__ == "__main__":
    main()
