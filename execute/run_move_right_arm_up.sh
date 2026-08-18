#!/bin/bash
# run_move_right_arm_up.sh — 设置GDK环境并执行右手末端抬升
set -e
DIR=/home/agi/app
SO_DIRS=$(find "$DIR/lib" -type f -name "*.so*" -printf "%h\n" 2>/dev/null | sort -u)
SO_DIRS="$SO_DIRS $(find $DIR/build_dep -type f -name "*.so*" -printf "%h\n" 2>/dev/null | sort -u)"
NEW_LIB_PATH=$(echo "$SO_DIRS" | tr " " ":" | tr "\n" ":")
export LD_LIBRARY_PATH="${NEW_LIB_PATH%:}:$LD_LIBRARY_PATH"
export PYTHONPATH="$DIR/gdk/lib:$PYTHONPATH"
export LOCATOR_IP=$(ip -o -4 addr list | grep '10.42.1.' | awk '{print $4}' | cut -d/ -f1 | head -1)
export AORTA_DISCOVERY_URI=http://10.42.1.101:2379
export AORTA_DISPATCHER_THREAD_NUM=6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$SCRIPT_DIR/move_right_arm_up.py" "$@"
