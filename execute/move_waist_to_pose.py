#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_waist_to_pose.py — 腰部运动到 JSON 文件记录的关节位姿 (模板)

使用方法:
  1. 修改下方 POSE_FILE 为要调用的位姿 JSON 文件路径
  2. 运行: python move_waist_to_pose.py

JSON 文件格式 (由 get_waist_pose.py 生成):
{
  "timestamp": "...",
  "joints": {
    "idx01_body_joint1": 0.0,
    "idx02_body_joint2": 0.0,
    ...
  }
}

工作原理:
  - 读取 JSON 中的 joints 字段 (腰部 5 个关节角)
  - 通过 MQTT 发送 {"command":"waist", "data":{...}} 到 /humanoid/joints/control
  - 服务端调用 common.robot.move_waist_joint(pos, vel) 执行
  - 订阅 /humanoid/commands/done 等待执行完成

命令行参数:
  --pose PATH     覆盖默认 POSE_FILE 路径
  --timeout N     等待完成超时(秒), 默认 15
  --dry-run       只读取并打印, 不发送运动指令
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
POSE_FILE = "/data/wzd/position/waist_20260807_100059.json"
# ═══════════════════════════════════════════════════════════

# ===================== MQTT 配置 =====================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_JOINTS_CONTROL = "/humanoid/joints/control"   # 发送关节运动命令
TOPIC_COMMANDS_DONE  = "/humanoid/commands/done"    # 接收完成通知

# ===================== 关节定义 (与服务端一致) =====================
WAIST_JOINT_KEYS = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
]


# ═══════════════════════════════════════════════════════════
#  位姿文件读取
# ═══════════════════════════════════════════════════════════

def load_waist_joints(pose_path):
    """从位姿 JSON 文件中读取腰部关节角

    Returns
    -------
    dict
        {关节名: 角度} 字典 (仅腰部关节)
    """
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"位姿文件不存在: {pose_path}")

    with open(pose_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    joints_all = data.get("joints", {})
    if not joints_all:
        raise ValueError(f"位姿文件中未找到 joints 字段: {pose_path}")

    # 仅保留腰部关节
    waist_keys = set(WAIST_JOINT_KEYS)
    waist_joints = {k: float(v) for k, v in joints_all.items() if k in waist_keys}

    if not waist_joints:
        raise ValueError(f"位姿文件中未找到腰部关节: {pose_path}")

    return waist_joints, data.get("timestamp", "N/A")


def validate_joints(joints):
    """校验关节角完整性, 返回 (waist_list, missing_keys)"""
    waist = [joints.get(k, None) for k in WAIST_JOINT_KEYS]

    missing = []
    for k, v in zip(WAIST_JOINT_KEYS, waist):
        if v is None:
            missing.append(k)

    waist = [v if v is not None else 0.0 for v in waist]
    return waist, missing


# ═══════════════════════════════════════════════════════════
#  MQTT 运动命令
# ═══════════════════════════════════════════════════════════

def send_waist_command(joints_dict, broker=MQTT_BROKER, port=MQTT_PORT, timeout=15.0):
    """发送 waist 关节运动命令, 等待完成通知

    Parameters
    ----------
    joints_dict : dict
        {关节名: 角度} 字典
    timeout : float
        等待完成超时(秒)

    Returns
    -------
    bool
        True 表示收到完成通知, False 表示超时
    """
    payload = json.dumps({"command": "waist", "data": joints_dict})
    done = {"ok": False, "error": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_COMMANDS_DONE, qos=0)
            # 订阅成功后再发布命令
            client.publish(TOPIC_JOINTS_CONTROL, payload, qos=2)
            print(f"[运动] 已发送 waist 命令 ({len(joints_dict)} 个关节)")
        else:
            print(f"[运动] MQTT 连接失败, rc={rc}")
            done["error"] = True

    def on_message(client, userdata, msg):
        if msg.topic == TOPIC_COMMANDS_DONE:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                cmd = data.get("cmd")
                if cmd == "waist":
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
    parser = argparse.ArgumentParser(description="腰部运动到 JSON 记录的位姿 (模板)")
    parser.add_argument("--pose", type=str, default=POSE_FILE,
                        help=f"位姿 JSON 文件路径, 默认 {POSE_FILE}")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="等待运动完成超时(秒), 默认 15")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读取并打印, 不发送运动指令")
    args = parser.parse_args()

    pose_path = args.pose
    print(f"[模板] 位姿文件: {pose_path}")

    # 1. 读取位姿
    try:
        joints, ts = load_waist_joints(pose_path)
    except Exception as e:
        print(f"[模板] ✗ 读取位姿失败: {e}")
        return 1

    print(f"[模板] ✓ 已加载位姿 (采样时间: {ts})")
    print(f"[模板] 关节数: {len(joints)}")

    # 2. 校验完整性
    waist, missing = validate_joints(joints)
    print(f"\n[位姿] 腰部 → {[f'{p:.3f}' for p in waist]}")
    if missing:
        print(f"\n⚠ 缺失关节 (将补 0): {missing}")

    # 3. dry-run 退出
    if args.dry_run:
        print("\n[dry-run] 未发送运动指令")
        return 0

    # 4. 发送运动命令
    print(f"\n[运动] 正在发送命令并等待完成 (超时 {args.timeout}s)...")
    ok = send_waist_command(joints, timeout=args.timeout)
    if ok:
        print("[运动] ✓ 腰部已到达目标位姿")
        return 0
    else:
        print("[运动] ✗ 运动未完成")
        return 2


if __name__ == "__main__":
    sys.exit(main())
