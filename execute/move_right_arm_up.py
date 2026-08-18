#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""move_right_arm_up.py — 右手末端相对移动 (需配合 run_move_right_arm_up.sh 运行)

坐标系: X+向前, Y+向左, Z+向上 (单位: 米)

用法:
  bash run_move_right_arm_up.sh                    # 抬升 30mm (默认)
  bash run_move_right_arm_up.sh --dz 0.05          # 抬升 50mm
  bash run_move_right_arm_up.sh --dz -0.02         # 下降 20mm
  bash run_move_right_arm_up.sh --dx 0.01          # 向前10mm + 抬升30mm
  bash run_move_right_arm_up.sh --dry-run          # 只打印信息
"""

import argparse
import os
import sys
import time

GDK_LIB = "/home/agi/app/gdk/lib"
if GDK_LIB not in sys.path:
    sys.path.insert(0, GDK_LIB)

import agibot_gdk
sys.path.insert(0, "/data/wxf/wxf0721/services")
from offset_move_common import EndEffectorController, LEFT_NAME, RIGHT_NAME


def main():
    parser = argparse.ArgumentParser(description="右手末端相对移动")
    parser.add_argument("--dx", type=float, default=0.0, help="X方向偏移(米), 正=向前")
    parser.add_argument("--dy", type=float, default=0.0, help="Y方向偏移(米), 正=向左")
    parser.add_argument("--dz", type=float, default=0.03, help="Z方向偏移(米), 正=向上, 默认0.03(30mm)")
    parser.add_argument("--dry-run", action="store_true", help="只打印当前位姿，不执行移动")
    args = parser.parse_args()

    offset_r = (args.dx, args.dy, args.dz)
    total = abs(args.dx) + abs(args.dy) + abs(args.dz)

    print("=" * 55)
    print(f"右手末端相对移动: dx={args.dx*1000:.1f}mm dy={args.dy*1000:.1f}mm dz={args.dz*1000:.1f}mm")
    if total < 1e-6:
        print("偏移量为零，无需移动")
        return 0
    print("=" * 55)

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("[错误] GDK 初始化失败")
        return 1
    print("[GDK] 初始化成功")

    robot = agibot_gdk.Robot()
    time.sleep(2)

    try:
        controller = EndEffectorController(robot)

        status = robot.get_motion_control_status()
        start_r = controller._find_pose(status, RIGHT_NAME)
        print(f"[右臂起始] 位置: ({start_r['position'][0]:.4f}, {start_r['position'][1]:.4f}, {start_r['position'][2]:.4f}) m")

        target_r = {
            "position": [
                start_r["position"][0] + offset_r[0],
                start_r["position"][1] + offset_r[1],
                start_r["position"][2] + offset_r[2],
            ],
            "orientation": list(start_r["orientation"]),
        }
        print(f"[右臂目标] 位置: ({target_r['position'][0]:.4f}, {target_r['position'][1]:.4f}, {target_r['position'][2]:.4f}) m")

        if args.dry_run:
            print("[dry-run] 未执行移动")
            return 0

        print("[运动] 正在执行右手末端直线移动...")
        ok = controller.adjust_arms_relative(offset_l=(0, 0, 0), offset_r=offset_r)

        if ok:
            time.sleep(1)
            status2 = robot.get_motion_control_status()
            end_r = controller._find_pose(status2, RIGHT_NAME)
            actual_dx = end_r["position"][0] - start_r["position"][0]
            actual_dy = end_r["position"][1] - start_r["position"][1]
            actual_dz = end_r["position"][2] - start_r["position"][2]
            print(f"[右臂完成] 位置: ({end_r['position'][0]:.4f}, {end_r['position'][1]:.4f}, {end_r['position'][2]:.4f}) m")
            print(f"[右臂完成] 实际位移: dx={actual_dx*1000:.1f}mm dy={actual_dy*1000:.1f}mm dz={actual_dz*1000:.1f}mm")
            print(f"[右臂完成] 期望位移: dx={args.dx*1000:.1f}mm dy={args.dy*1000:.1f}mm dz={args.dz*1000:.1f}mm")
            print("[完成] ✓ 右手移动执行成功")
            return 0
        else:
            print("[错误] 右手移动失败")
            return 2

    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
        return 3
    finally:
        if agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            print("[警告] GDK 释放失败")
        else:
            print("[GDK] 资源已释放")


if __name__ == "__main__":
    sys.exit(main())
