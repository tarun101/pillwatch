"""Portable board-power readers used by the paper statistics utility.

Three sources, tried in order of trust:

1. **External meter** (``PILLBOX_POWER_METER_CMD``) — a command that prints the
   current board draw in watts. This is the *only* option on hardware without
   software power telemetry, notably the **Jetson Orin Nano / Orin NX**, whose
   modules drop the onboard INA3221 rail sensors older Jetsons had, so
   ``tegrastats`` reports no power there. Point it at a USB power-meter reader,
   a smart-plug API call, or a bench-PSU query.
2. **Raspberry Pi 5 PMIC** (``vcgencmd pmic_read_adc``) — sum of V×I per rail.
3. **Jetson tegrastats** — total input rail (VDD_IN / POM_5V_IN) on the Jetsons
   that still expose it (AGX/older NX). Returns None on Orin Nano.

Every reader returns watts (float) or None; None simply means "no power for
this run", and latency is still reported.
"""

import os
import re
import shutil
import subprocess


def classify_device_model(model):
    """Classify a Linux device-tree model into a supported hardware family."""
    normalized = (model or "").lower()
    if "nvidia" in normalized or "jetson" in normalized:
        return "jetson"
    if "raspberry pi 5" in normalized:
        return "raspberry_pi_5"
    if "raspberry pi 4" in normalized:
        return "raspberry_pi_4"
    if "raspberry pi" in normalized:
        return "raspberry_pi"
    return "other"


def _read_pi_power_watts():
    """Return total Pi 5 PMIC rail power, or None when vcgencmd is unavailable."""
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    volts, amps = {}, {}
    for line in out.stdout.splitlines():
        match = re.match(r"\s*(\S+)_([AV])\s+\S+=([0-9.]+)[AV]\s*$", line)
        if match:
            target = amps if match.group(2) == "A" else volts
            target[match.group(1)] = float(match.group(3))
    power = sum(rail_volts * amps[name]
                for name, rail_volts in volts.items() if name in amps)
    return round(power, 2) if power > 0 else None


_JETSON_TOTAL_RAILS = ("VDD_IN", "POM_5V_IN", "VIN_SYS_5V0", "VDD_TOTAL")


def _parse_tegrastats_power(text):
    """Extract total board input power from one or more tegrastats lines."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    for rail in _JETSON_TOTAL_RAILS:
        # Examples vary by JetPack generation:
        #   VDD_IN 2532mW/2532mW
        #   POM_5V_IN 4286/4286
        match = re.search(
            rf"(?:^|\s){re.escape(rail)}\s+([0-9.]+)\s*(mW|W)?(?:/|\s|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            unit = (match.group(2) or "mW").lower()
            watts = value if unit == "w" else value / 1000.0
            return round(watts, 3) if watts > 0 else None
    return None


def _read_jetson_power_watts():
    """Return Jetson input power from tegrastats, or None off Jetson.

    tegrastats has no portable one-shot flag across JetPack releases, so start
    it briefly, collect its first sample, and terminate it.
    """
    command = shutil.which("tegrastats")
    if command is None:
        return None
    try:
        proc = subprocess.Popen(
            [command, "--interval", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=0.35)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout, _ = proc.communicate(timeout=0.2)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, _ = proc.communicate()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return _parse_tegrastats_power(stdout)


def _read_external_meter_watts():
    """Read watts from a user-supplied command (``PILLBOX_POWER_METER_CMD``).

    The command's stdout must contain the current board draw in watts (a bare
    number; a trailing 'W' or extra text is tolerated — the first number wins).
    This is how you get power on boards without a software rail, e.g. the Orin
    Nano. Runs through the shell, so it is the operator's own trusted command.
    """
    cmd = os.environ.get("PILLBOX_POWER_METER_CMD")
    if not cmd:
        return None
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", out.stdout)
    if not match:
        return None
    watts = float(match.group(0))
    return round(watts, 3) if watts > 0 else None


def read_power_watts(with_source=False):
    """Read board input power from the first available source.

    Order of trust: an explicit external meter (``PILLBOX_POWER_METER_CMD``),
    then the Pi 5 PMIC, then Jetson tegrastats. Returns watts or None (None =
    no telemetry; latency is still measured). ``with_source`` additionally
    returns which interface supplied the number, so the paper output can record
    the method honestly. Pi 4 and Orin Nano have no software rail, so without an
    external meter they are latency-only.
    """
    for source, reader in (
        ("external_meter", _read_external_meter_watts),
        ("raspberry_pi_pmic", _read_pi_power_watts),
        ("jetson_tegrastats", _read_jetson_power_watts),
    ):
        watts = reader()
        if watts is not None:
            return (watts, source) if with_source else watts
    return (None, None) if with_source else None
