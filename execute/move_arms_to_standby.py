#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_arms_to_standby.py — 双臂运动到【待机】姿态

使用方法:
  1. 修改下方 POSE_FILE 为要调用的位姿 JSON 文件路径
  2. 运行: python move_arms_to_standby.py

JSON 文件格式 (由 get_current_pose.py 生成):
{
  "timestamp": "...",
  "left_ee": {...},
  "right_ee": {...},
  "joints": {
    "idx21_arm_l_joint1": 1.73,
    "idx22_arm_l_joint2": -1.15,
    ...
    "idx61_arm_r_joint1": -1.73,
    ...
  }
}

工作原理:
  - 读取 JSON 中的 joints 字段 (左右臂共 14 个关节角)
  - 通过 MQTT 发送 {"command":"arms", "data":{...}} 到 /humanoid/joints/control
  - 服务端调用 common.robot.move_arm_joint(positions, velocities, 2) 执行
  - 订阅 /humanoid/commands/done 等待执行完成

命令行参数:
  --pose PATH     覆盖默认 POSE_FILE 路径
  --timeout N     等待完成超时(秒), 默认 15
  --dry-run       只读取并打印, 不发送运动指令
  --speed V       关节速度 (0~1), 仅用于信息显示, 实际由服务端 ARM_SPEED 决定
"""

import argparse
import json
import os
import sys
import time

import paho.mqtt.client as mqtt

# ═══════════════════════════════════════════════════════════
#  ★ 可替换路径 - 调用的位姿 JSON 文件 ★
# ═══════════════════════════════════════════════════════════
POSE_FILE = "/data/wzd/position/pose_standby.json"
# ═══════════════════════════════════════════════════════════

# ===================== MQTT 配置 =====================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_JOINTS_CONTROL = "/humanoid/joints/control"   # 发送关节运动命令
TOPIC_COMMANDS_DONE  = "/humanoid/commands/done"    # 接收完成通知

# ===================== 关节定义 (与服务端一致) =====================
LEFT_ARM_JOINT_KEYS = [
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]
RIGHT_ARM_JOINT_KEYS = [
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]


# ═══════════════════════════════════════════════════════════
#  位姿文件读取
# ═══════════════════════════════════════════════════════════

def load_arm_joints(pose_path):
    """从位姿 JSON 文件中读取手臂关节角

    Returns
    -------
    dict
        {关节名: 角度} 字典 (仅手臂关节)
    """
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"位姿文件不存在: {pose_path}")

    with open(pose_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    joints_all = data.get("joints", {})
    if not joints_all:
        raise ValueError(f"位姿文件中未找到 joints 字段: {pose_path}")

    # 仅保留手臂关节
    arm_keys = set(LEFT_ARM_JOINT_KEYS) | set(RIGHT_ARM_JOINT_KEYS)
    arm_joints = {k: float(v) for k, v in joints_all.items() if k in arm_keys}

    if not arm_joints:
        raise ValueError(f"位姿文件中未找到手臂关节: {pose_path}")

    return arm_joints, data.get("timestamp", "N/A")


def validate_joints(joints):
    """校验关节角完整性, 返回 (left_list, right_list, missing_keys)"""
    left  = [joints.get(k, None) for k in LEFT_ARM_JOINT_KEYS]
    right = [joints.get(k, None) for k in RIGHT_ARM_JOINT_KEYS]

    missing = []
    for k, v in zip(LEFT_ARM_JOINT_KEYS, left):
        if v is None:
            missing.append(k)
    for k, v in zip(RIGHT_ARM_JOINT_KEYS, right):
        if v is None:
            missing.append(k)

    left  = [v if v is not None else 0.0 for v in left]
    right = [v if v is not None else 0.0 for v in right]
    return left, right, missing


# ═══════════════════════════════════════════════════════════
#  MQTT 运动命令
# ═══════════════════════════════════════════════════════════

def send_arms_command(joints_dict, broker=MQTT_BROKER, port=MQTT_PORT, timeout=15.0):
    """发送 arms 关节运动命令, 等待完成通知"""
    payload = json.dumps({"command": "arms", "data": joints_dict})
    done = {"ok": False, "error": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_COMMANDS_DONE, qos=0)
            client.publish(TOPIC_JOINTS_CONTROL, payload, qos=2)
            print(f"[运动] 已发送 arms 命令 ({len(joints_dict)} 个关节)")
        else:
            print(f"[运动] MQTT 连接失败, rc={rc}")
            done["error"] = True

    def on_message(client, userdata, msg):
        if msg.topic == TOPIC_COMMANDS_DONE:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                cmd = data.get("cmd")
                if cmd == "arms":
                    done["ok"] = True
                    client.disconnect()
            except Exception:
                pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        print(f"[运动] 连接 MQTT 失败: {e}")
        return False

    t_start = time.time()
    while not done["ok"] and not done["error"] and time.time() - t_start < timeout:
        client.loop(timeout=0.1)

    try:
        client.disconnect()
    except Exception:
        pass

    if done["error"]:
        return False
    if not done["ok"]:
        print(f"[运动] ⚠ 等待完成超时 ({timeout}s)")
        return False
    return True


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="双臂运动到【待机】姿态")
    parser.add_argument("--pose", type=str, default=POSE_FILE,
                        help=f"位姿 JSON 文件路径, 默认 {POSE_FILE}")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="等待运动完成超时(秒), 默认 15")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读取并打印, 不发送运动指令")
    args = parser.parse_args()

    pose_path = args.pose
    print(f"[待机] 位姿文件: {pose_path}")

    # 1. 读取位姿
    try:
        joints, ts = load_arm_joints(pose_path)
    except Exception as e:
        print(f"[待机] ✗ 读取位姿失败: {e}")
        return 1

    print(f"[待机] ✓ 已加载位姿 (采样时间: {ts})")
    print(f"[待机] 关节数: {len(joints)}")

    # 2. 校验完整性
    left, right, missing = validate_joints(joints)
    print(f"\n[位姿] 左臂 → {[f'{p:.3f}' for p in left]}")
    print(f"[位姿] 右臂 → {[f'{p:.3f}' for p in right]}")
    if missing:
        print(f"\n⚠ 缺失关节 (将补 0): {missing}")

    # 3. dry-run 退出
    if args.dry_run:
        print("\n[dry-run] 未发送运动指令")
        return 0

    # 4. 发送运动命令
    print(f"\n[运动] 正在发送命令并等待完成 (超时 {args.timeout}s)...")
    ok = send_arms_command(joints, timeout=args.timeout)
    if ok:
        print("[运动] ✓ 双臂已到达【待机】位姿")
        return 0
    else:
        print("[运动] ✗ 运动未完成")
        return 2


if __name__ == "__main__":
    sys.exit(main())
