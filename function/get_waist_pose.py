#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_waist_pose.py — 获取机器人腰部当前位姿并保存为 JSON

通过订阅 /humanoid/status/data 主题获取一次状态消息,
仅保留腰部相关位姿:
  - joints:        腰部 5 个关节角 {关节名: 位置}
  - timestamp:     采样时间戳

注: 底盘/手臂/头部位姿不记录

JSON 文件保存到 /data/wzd/position/ 下, 文件名格式:
  waist_YYYYMMDD_HHMMSS.json

用法:
  python get_waist_pose.py                    # 默认保存
  python get_waist_pose.py --name initial     # 指定文件名 (waist_initial.json)
  python get_waist_pose.py --timeout 5        # 设置订阅超时(秒)
  python get_waist_pose.py --show             # 在终端打印位姿
"""
import argparse
import json
import os
import time

import paho.mqtt.client as mqtt

# ===================== 配置 =====================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_STATUS_DATA = "/humanoid/status/data"

# 输出目录
POSITION_DIR = "/data/wzd/position"

# 腰部关节键 (与服务端 joints.py 一致)
WAIST_JOINT_KEYS = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
]


# ═══════════════════════════════════════════════════════════
#  位姿获取
# ═══════════════════════════════════════════════════════════

def fetch_pose(broker=MQTT_BROKER, port=MQTT_PORT, timeout=5.0):
    """订阅 /humanoid/status/data, 获取一次状态消息

    Returns
    -------
    dict or None
        状态消息字典; 超时返回 None
    """
    received = {"msg": None}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_STATUS_DATA, qos=0)
        else:
            print(f"[位姿] MQTT 连接失败, rc={rc}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            received["msg"] = payload
            client.disconnect()
        except Exception as e:
            print(f"[位姿] 解析消息失败: {e}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        print(f"[位姿] 连接 MQTT 失败: {e}")
        return None

    t_start = time.time()
    while received["msg"] is None and time.time() - t_start < timeout:
        client.loop(timeout=0.1)

    try:
        client.disconnect()
    except Exception:
        pass

    return received["msg"]


# ═══════════════════════════════════════════════════════════
#  显示与保存
# ═══════════════════════════════════════════════════════════

def _filter_waist_joints(joints):
    """从关节字典中筛选出腰部关节 (关节名含 'body')"""
    if not joints:
        return {}
    return {k: v for k, v in joints.items() if k in WAIST_JOINT_KEYS}


def print_pose(pose_msg):
    """在终端打印腰部位姿摘要"""
    if not pose_msg:
        print("[位姿] 无数据")
        return

    print(f"\n{'='*60}")
    print(f"采样时间: {pose_msg.get('timestamp', 'N/A')}")
    print(f"{'='*60}")

    joints = _filter_waist_joints(pose_msg.get("joints", {}))
    if joints:
        print(f"\n[腰部关节角] 共 {len(joints)} 个关节")
        for k in WAIST_JOINT_KEYS:
            if k in joints:
                print(f"  {k}: {joints[k]:.4f}")
    else:
        print("\n[腰部关节角] 无数据")

    print(f"\n{'='*60}")


def save_pose(pose_msg, out_dir=POSITION_DIR, name=None):
    """保存腰部位姿到 JSON 文件

    仅保留腰部相关字段 (joints 筛选为腰部关节, timestamp)

    Parameters
    ----------
    pose_msg : dict
        原始状态消息
    out_dir : str
        输出目录
    name : str or None
        文件名 (不含扩展名); None 则用时间戳

    Returns
    -------
    str
        保存的文件绝对路径
    """
    os.makedirs(out_dir, exist_ok=True)

    # 仅保留腰部相关字段
    waist_pose = {
        "timestamp": pose_msg.get("timestamp"),
        "joints": _filter_waist_joints(pose_msg.get("joints", {})),
    }

    if name:
        filename = f"waist_{name}.json"
    else:
        filename = f"waist_{time.strftime('%Y%m%d_%H%M%S')}.json"

    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(waist_pose, f, ensure_ascii=False, indent=2)

    return out_path


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="获取机器人腰部当前位姿并保存为 JSON")
    parser.add_argument("--name", type=str, default=None,
                        help="文件名 (不含扩展名), 默认用时间戳")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="订阅超时时间(秒), 默认 5")
    parser.add_argument("--show", action="store_true",
                        help="在终端打印位姿摘要")
    parser.add_argument("--out-dir", type=str, default=POSITION_DIR,
                        help=f"输出目录, 默认 {POSITION_DIR}")
    args = parser.parse_args()

    print(f"[位姿] 正在订阅 {TOPIC_STATUS_DATA} (超时 {args.timeout}s)...")
    pose_msg = fetch_pose(timeout=args.timeout)

    if pose_msg is None:
        print(f"[位姿] ✗ 在 {args.timeout}s 内未收到状态消息, 请确认服务已启动")
        return 1

    print(f"[位姿] ✓ 已获取位姿 (时间: {pose_msg.get('timestamp', 'N/A')})")

    if args.show:
        print_pose(pose_msg)

    out_path = save_pose(pose_msg, out_dir=args.out_dir, name=args.name)
    print(f"[位姿] 已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
