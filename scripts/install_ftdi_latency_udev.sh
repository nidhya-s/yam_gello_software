#!/usr/bin/env bash
set -euo pipefail

# Keep GELLO FTDI USB-Serial converters in low-latency mode across unplug/reboot.
# Defaults are the two current GELLO adapters; pass serials as args to override.

RULE_FILE="/etc/udev/rules.d/99-gello-ftdi-latency.rules"
SERIALS=("FT3M9NVB" "FTAKRMFX")

if [ "$#" -gt 0 ]; then
  SERIALS=("$@")
fi

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

serial_enabled() {
  local needle="$1"
  local serial
  for serial in "${SERIALS[@]}"; do
    if [ "$serial" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

write_timer() {
  local timer="$1"
  if [ "$(id -u)" -eq 0 ]; then
    printf '1\n' > "$timer"
  else
    printf '1\n' | sudo tee "$timer" >/dev/null
  fi
}

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

{
  echo '# GELLO FTDI USB-Serial low-latency mode.'
  echo '# Installed by dependencies/gello_software/scripts/install_ftdi_latency_udev.sh'
  for serial in "${SERIALS[@]}"; do
    printf 'ACTION=="add", SUBSYSTEM=="usb-serial", DRIVERS=="ftdi_sio", ATTRS{serial}=="%s", ATTR{latency_timer}="1"\n' "$serial"
    printf 'ACTION=="change", SUBSYSTEM=="usb-serial", DRIVERS=="ftdi_sio", ATTRS{serial}=="%s", ATTR{latency_timer}="1"\n' "$serial"
  done
} > "$tmp"

as_root install -m 0644 "$tmp" "$RULE_FILE"
as_root udevadm control --reload-rules

# Apply immediately to currently plugged-in matching adapters too.
for timer in /sys/bus/usb-serial/devices/ttyUSB*/latency_timer; do
  [ -e "$timer" ] || continue
  dev="/dev/$(basename "$(dirname "$timer")")"
  serial="$(
    udevadm info -q property -n "$dev" 2>/dev/null |
      awk -F= '$1 == "ID_SERIAL_SHORT" { print $2; exit }'
  )"
  if serial_enabled "$serial"; then
    write_timer "$timer"
  fi
done

as_root udevadm trigger --subsystem-match=usb-serial --action=change || true

echo "Installed $RULE_FILE for serials: ${SERIALS[*]}"
for timer in /sys/bus/usb-serial/devices/ttyUSB*/latency_timer; do
  [ -e "$timer" ] || continue
  dev="/dev/$(basename "$(dirname "$timer")")"
  serial="$(
    udevadm info -q property -n "$dev" 2>/dev/null |
      awk -F= '$1 == "ID_SERIAL_SHORT" { print $2; exit }'
  )"
  printf '%s serial=%s latency_timer=%s\n' "$dev" "${serial:-unknown}" "$(cat "$timer")"
done
