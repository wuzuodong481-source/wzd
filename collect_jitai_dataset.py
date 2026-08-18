#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_jitai_dataset.py — 腰部随机扰动数据采集

流程:
  1. 启动时获取当前腰部姿态作为初始姿态
  2. 循环 N 次:
     a. 在初始姿态基础上, 5 个腰部关节各自独立随机偏移 ±5° (角度→弧度)
     b. 发送腰部运动命令, 等待完成
     c. 稳定 1 秒
     d. 头部 RGB 拍照
     e. 以时间戳命名保存到 jitai/ 文件夹
     f. 回到初始姿态
  3. 采集 50 张后结束

用法:
  python collect_jitai_dataset.py                     # 默认 50 张
  python collect_jitai_dataset.py --num 30            # 采集 30 张
  python collect_jitai_dataset.py --deg 3             # 每次随机 ±3°
  python collect_jitai_dataset.py --stable 2.0        # 稳定 2 秒
  python collect_jitai_dataset.py --out /data/wzd/jitai   # 指定输出目录
"""
import argparse
import base64
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ===================== 配置 =====================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

TOPIC_STATUS_DATA    = "/humanoid/status/data"     # 获取当前位姿
TOPIC_JOINTS_CONTROL = "/humanoid/joints/control"  # 发送腰部运动命令
TOPIC_COMMANDS_DONE  = "/humanoid/commands/done"   # 接收运动完成通知
TOPIC_CAMERA_DATA    = "/humanoid/camera/data"     # 头部相机图像
TOPIC_CAMERA_CTRL    = "/humanoid/camera/control"  # 相机启停控制

# 腰部关节键 (与服务端一致)
WAIST_JOINT_KEYS = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
]

# 默认参数
DEFAULT_OUT_DIR    = "/data/wzd/jitai"
DEFAULT_NUM        = 50
DEFAULT_MAX_DEG    = 5.0          # 每个关节随机偏移最大角度 (°)
DEFAULT_STABLE_S   = 1.0          # 运动到位后稳定时间 (秒)
DEFAULT_MOVE_TO    = 8.0          # 腰部运动超时 (秒)
DEFAULT_CAM_TO     = 10.0         # 取图超时 (秒)


# ═══════════════════════════════════════════════════════════
#  1. 获取当前腰部位姿
# ═══════════════════════════════════════════════════════════

def fetch_current_waist(timeout=5.0):
    """订阅 /humanoid/status/data 获取一次状态消息, 返回腰部关节字典"""
    received = {"msg": None}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_STATUS_DATA, qos=0)
        else:
            print(f"[初始化] MQTT 连接失败, rc={rc}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            received["msg"] = payload
            client.disconnect()
        except Exception as e:
            print(f"[初始化] 解析消息失败: {e}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[初始化] 连接 MQTT 失败: {e}")
        return None

    t0 = time.time()
    while received["msg"] is None and time.time() - t0 < timeout:
        client.loop(timeout=0.1)
    try:
        client.disconnect()
    except Exception:
        pass

    if received["msg"] is None:
        return None

    joints_all = received["msg"].get("joints", {})
    waist = {k: float(joints_all[k]) for k in WAIST_JOINT_KEYS if k in joints_all}
    if len(waist) != len(WAIST_JOINT_KEYS):
        print(f"[初始化] ⚠ 关节数不匹配: 期望 {len(WAIST_JOINT_KEYS)}, 实际 {len(waist)}")
    return waist


# ═══════════════════════════════════════════════════════════
#  2. 腰部运动到指定位姿
# ═══════════════════════════════════════════════════════════

def move_waist_to(joints_dict, timeout=DEFAULT_MOVE_TO):
    """发送腰部运动命令, 等待 /humanoid/commands/done 通知"""
    payload = json.dumps({"command": "waist", "data": joints_dict})
    done = {"ok": False, "error": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_COMMANDS_DONE, qos=0)
            client.publish(TOPIC_JOINTS_CONTROL, payload, qos=2)
        else:
            done["error"] = True

    def on_message(client, userdata, msg):
        if msg.topic == TOPIC_COMMANDS_DONE:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                if data.get("cmd") == "waist":
                    done["ok"] = True
                    client.disconnect()
            except Exception:
                pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[运动] 连接 MQTT 失败: {e}")
        return False

    t0 = time.time()
    while not done["ok"] and not done["error"] and time.time() - t0 < timeout:
        client.loop(timeout=0.1)
    try:
        client.disconnect()
    except Exception:
        pass

    return done["ok"] and not done["error"]


# ═══════════════════════════════════════════════════════════
#  3. 头部 RGB 拍照
# ═══════════════════════════════════════════════════════════

def capture_head_color(timeout=DEFAULT_CAM_TO):
    """通过 MQTT 订阅 /humanoid/camera/data 获取一帧头部彩色图像"""
    received = {"img": None}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_CAMERA_DATA, qos=0)
            client.publish(TOPIC_CAMERA_CTRL, json.dumps({"command": "start"}), qos=0)
        else:
            print(f"[相机] MQTT 连接失败 rc={rc}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            b64 = payload.get("head_color")
            if b64:
                buf = base64.b64decode(b64)
                nparr = np.frombuffer(buf, dtype=np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    received["img"] = bgr
                    client.disconnect()
        except Exception as e:
            print(f"[相机] 解析失败: {e}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[相机] 连接 MQTT 失败: {e}")
        return None

    t0 = time.time()
    while received["img"] is None and time.time() - t0 < timeout:
        client.loop(timeout=0.2)

    # 停止相机流
    try:
        client.publish(TOPIC_CAMERA_CTRL, json.dumps({"command": "stop"}), qos=0)
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass

    return received["img"]


# ═══════════════════════════════════════════════════════════
#  4. 随机扰动生成
# ═══════════════════════════════════════════════════════════

def gen_random_pose(initial_pose, max_deg=DEFAULT_MAX_DEG):
    """在初始位姿基础上, 每个关节独立随机偏移 ±max_deg 度 (转弧度)

    Returns
    -------
    dict : 新的腰部关节字典
    list : 每个关节的偏移量(角度), 用于日志
    """
    max_rad = max_deg * (3.141592653589793 / 180.0)
    new_pose = {}
    deltas_deg = []
    for k in WAIST_JOINT_KEYS:
        if k in initial_pose:
            delta_rad = random.uniform(-max_rad, max_rad)
            new_pose[k] = initial_pose[k] + delta_rad
            deltas_deg.append(delta_rad * (180.0 / 3.141592653589793))
        else:
            new_pose[k] = 0.0
            deltas_deg.append(0.0)
    return new_pose, deltas_deg


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="腰部随机扰动数据采集")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"图像输出目录, 默认 {DEFAULT_OUT_DIR}")
    parser.add_argument("--num", type=int, default=DEFAULT_NUM,
                        help=f"采集数量, 默认 {DEFAULT_NUM}")
    parser.add_argument("--deg", type=float, default=DEFAULT_MAX_DEG,
                        help=f"每个关节随机偏移最大角度(°), 默认 {DEFAULT_MAX_DEG}")
    parser.add_argument("--stable", type=float, default=DEFAULT_STABLE_S,
                        help=f"运动到位后稳定时间(秒), 默认 {DEFAULT_STABLE_S}")
    parser.add_argument("--move-timeout", type=float, default=DEFAULT_MOVE_TO,
                        help=f"腰部运动超时(秒), 默认 {DEFAULT_MOVE_TO}")
    parser.add_argument("--cam-timeout", type=float, default=DEFAULT_CAM_TO,
                        help=f"取图超时(秒), 默认 {DEFAULT_CAM_TO}")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子, 指定后可复现")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("腰部随机扰动数据采集")
    print("=" * 60)
    print(f"  输出目录: {args.out}")
    print(f"  采集数量: {args.num} 张")
    print(f"  每个关节最大偏移: ±{args.deg}°")
    print(f"  稳定时间: {args.stable}s")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)

    # ── 步骤 1: 获取初始腰部位姿 ──
    print("\n[1/3] 获取当前腰部位姿作为初始姿态...")
    initial_pose = fetch_current_waist(timeout=5.0)
    if initial_pose is None or len(initial_pose) == 0:
        print("[初始化] ✗ 未获取到腰部位姿, 请确认 MQTT 服务已启动")
        return 1
    print(f"[初始化] ✓ 已获取初始位姿 ({len(initial_pose)} 个关节):")
    for k in WAIST_JOINT_KEYS:
        if k in initial_pose:
            print(f"        {k}: {initial_pose[k]:.4f} rad "
                  f"({initial_pose[k] * 57.2958:.2f}°)")

    # ── 步骤 2: 循环采集 ──
    print(f"\n[2/3] 开始采集 {args.num} 张图像...")
    success = 0
    fail = 0
    t_total = time.time()

    for i in range(1, args.num + 1):
        print(f"\n--- 第 {i}/{args.num} 张 ---")
        t0 = time.time()

        # 2.1 生成随机扰动位姿
        rand_pose, deltas = gen_random_pose(initial_pose, max_deg=args.deg)
        print(f"[扰动] 偏移(°): " + ", ".join(f"{d:+.2f}" for d in deltas))

        # 2.2 运动到随机位姿
        print("[运动] 移动到随机位姿...")
        ok = move_waist_to(rand_pose, timeout=args.move_timeout)
        if not ok:
            print("[运动] ✗ 运动超时, 跳过")
            fail += 1
            continue

        # 2.3 稳定等待
        if args.stable > 0:
            time.sleep(args.stable)

        # 2.4 拍照
        print("[相机] 拍照...")
        img = capture_head_color(timeout=args.cam_timeout)
        if img is None:
            print("[相机] ✗ 取图失败, 跳过")
            fail += 1
            # 回到初始位姿
            move_waist_to(initial_pose, timeout=args.move_timeout)
            continue

        # 2.5 保存
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() - int(time.time())) * 1000)
        fname = f"{ts}_{ms:03d}_{i:03d}.jpg"
        fpath = os.path.join(args.out, fname)
        cv2.imwrite(fpath, img)
        print(f"[保存] ✓ {fname}  ({img.shape[1]}x{img.shape[0]})")
        success += 1

        # 2.6 回到初始位姿
        print("[复位] 回到初始位姿...")
        ok = move_waist_to(initial_pose, timeout=args.move_timeout)
        if not ok:
            print("[复位] ⚠ 复位超时, 继续")

        dt = time.time() - t0
        print(f"[耗时] {dt:.1f}s")

    # ── 步骤 3: 总结 ──
    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"采集完成 (总耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  成功: {success} 张")
    print(f"  失败: {fail} 张")
    print(f"  输出: {args.out}")
    print(f"{'='*60}")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
