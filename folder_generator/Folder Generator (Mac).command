#!/bin/bash
# Double-click launcher for the Flashdrive Folder Generator on macOS.
# Runs the GUI using the system Python 3. If it is missing, use the
# standalone .app built by build_folder_generator.sh instead.
#
# First time only: right-click this file > Open (to get past Gatekeeper),
# or run  chmod +x "Folder Generator (Mac).command"  in Terminal.
cd "$(dirname "$0")"
python3 folder_generator.py
