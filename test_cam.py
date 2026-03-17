#!/usr/bin/env python3
"""
RealSense Diagnostics
Checks USB visibility, pyrealsense2 detection, and udev permissions.
Run this first to diagnose why the camera isn't found.

    python3 rs_diag.py
"""

import subprocess
import sys
import os

SEP = "─" * 55

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        return e.output.decode().strip()

print(f"\n{'═'*55}")
print("  RealSense Diagnostics")
print(f"{'═'*55}\n")

# 1. USB visibility
print(f"[1] USB — lsusb (Intel devices)\n{SEP}")
usb = run("lsusb | grep -i 'intel\\|realsense\\|8086'")
if usb:
    print(usb)
else:
    print("  ⚠️  No Intel/RealSense USB device found in lsusb")
    print("     Check the cable and try a USB 3.0 (blue) port.\n")
print()

# 2. /dev/video* nodes
print(f"[2] Video devices (/dev/video*)\n{SEP}")
vdevs = run("ls -la /dev/video* 2>/dev/null || echo 'none'")
print(vdevs)
print()

# 3. Current user & groups
print(f"[3] User / groups\n{SEP}")
print(f"  User : {run('whoami')}")
print(f"  Groups: {run('groups')}")
in_video    = 'video'    in run('groups')
in_plugdev  = 'plugdev'  in run('groups')
print(f"  In 'video'   group: {'✅' if in_video   else '❌'}")
print(f"  In 'plugdev' group: {'✅' if in_plugdev  else '❌'}")
print()

# 4. udev rules
print(f"[4] RealSense udev rules\n{SEP}")
udev = run("ls /etc/udev/rules.d/ | grep -i real")
if udev:
    print(f"  Found: {udev}")
else:
    print("  ❌  No RealSense udev rules found")
    print("     This is the most common cause of 'device not found'")
print()

# 5. pyrealsense2 version
print(f"[5] pyrealsense2 version\n{SEP}")
try:
    import pyrealsense2 as rs
    print(f"  pyrealsense2 : {rs.__version__}")
except ImportError:
    print("  ❌  pyrealsense2 not installed")
    sys.exit(1)

# 6. Low-level context query
print(f"\n[6] rs.context device scan\n{SEP}")
ctx = rs.context()
devs = ctx.query_devices()
print(f"  Devices found: {len(devs)}")
for i, d in enumerate(devs):
    try:
        print(f"    [{i}] {d.get_info(rs.camera_info.name)}"
              f"  S/N:{d.get_info(rs.camera_info.serial_number)}")
    except Exception as e:
        print(f"    [{i}] (error reading info: {e})")

print()
if len(devs) == 0:
    print("══════════════════════════════════════════════════════")
    print("  FIX — run these commands then REPLUG the camera:")
    print("══════════════════════════════════════════════════════")
    print()
    print("  # 1. Install udev rules")
    print("  sudo apt-get install -y librealsense2-utils")
    print()
    print("  # Or manually download and install the rules:")
    print("  wget https://raw.githubusercontent.com/IntelRealSense/"
          "librealsense/master/config/99-realsense-libusb.rules")
    print("  sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/")
    print()
    print("  # 2. Reload udev & add yourself to 'plugdev'")
    print("  sudo udevadm control --reload-rules")
    print("  sudo udevadm trigger")
    print(f"  sudo usermod -aG plugdev $(whoami)")
    print(f"  sudo usermod -aG video  $(whoami)")
    print()
    print("  # 3. Log out and log back in (groups only apply after re-login)")
    print("  #    OR run this to apply immediately in the current shell:")
    print("  newgrp plugdev")
    print()
    print("  # 4. Replug the camera and rerun:")
    print("  python3 rs_diag.py")
else:
    print("  ✅  Camera is visible — you can run live_realsense.py")
