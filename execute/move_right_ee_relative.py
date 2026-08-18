#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""move_right_ee_relative.py — 通过 MQTT offset_move 命令控制末端相对移动

服务端已暴露 offset_move 命令 (单位: 毫米):
  {"command": "offset_move", "data": {"lx":0, "ly":0, "lz":0, "rx":0, "ry":0, "rz":30}}

坐标系: X+向前, Y+向左, Z+向上 (单位在命令中为毫米)

用法:
  python move_right_ee_relative.py                    # 右手抬升 30mm (默认)
  python move_right_ee_relative.py --rz 50            # 右手抬升 50mm
  python move_right_ee_relative.py --rz -20           # 右手下降 20mm
  python move_right_ee_relative.py --rx 10 --rz 30    # 右手向前10mm + 抬升30mm
  python move_right_ee_relative.py --lz 20            # 左手抬升 20mm
"""

import argparse
import json
import sys
import time

import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_CMD  = "/humanoid/commands/data"
TOPIC_DONE = "/humanoid/commands/done"


def send_offset_move(lx=0, ly=0, lz=0, rx=0, ry=0, rz=0, timeout=30.0):
    payload = json.dumps({
        "command": "offset_move",
        "data": {"lx": lx, "ly": ly, "lz": lz, "rx": rx, "ry": ry, "rz": rz}
    })
    done = {"ok": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_DONE, qos=0)
            client.publish(TOPIC_CMD, payload, qos=2)
            print(f"[发送] offset_move: L=({lx},{ly},{lz}) R=({rx},{ry},{rz}) mm")
        else:
            print(f"[错误] MQTT 连接失败 rc={rc}")
            done["ok"] = False
            client.disconnect()

    def on_message(client, userdata, msg):
        if msg.topic == TOPIC_DONE:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                cmd = data.get("cmd", "")
                if cmd == "offset_move":
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
        print(f"[错误] 连接 MQTT 失败: {e}")
        return False

    t0 = time.time()
    while not done["ok"] and time.time() - t0 < timeout:
        client.loop(timeout=0.1)

    try:
        client.disconnect()
    except Exception:
        pass

    if not done["ok"]:
        print(f"[警告] 等待完成超时 ({timeout}s)")
    return done["ok"]


def main():
    parser = argparse.ArgumentParser(description="末端相对移动 (通过MQTT offset_move, 单位: mm)")
    parser.add_argument("--lx", type=float, default=0, help="左手X偏移(mm), 正=向前")
    parser.add_argument("--ly", type=float, default=0, help="左手Y偏移(mm), 正=向左")
    parser.add_argument("--lz", type=float, default=0, help="左手Z偏移(mm), 正=向上")
    parser.add_argument("--rx", type=float, default=0, help="右手X偏移(mm), 正=向前")
    parser.add_argument("--ry", type=float, default=0, help="右手Y偏移(mm), 正=向左")
    parser.add_argument("--rz", type=float, default=30, help="右手Z偏移(mm), 正=向上, 默认30mm")
    parser.add_argument("--timeout", type=float, default=30.0, help="超时(秒), 默认30")
    args = parser.parse_args()

    total = abs(args.lx)+abs(args.ly)+abs(args.lz)+abs(args.rx)+abs(args.ry)+abs(args.rz)
    if total < 0.01:
        print("偏移量为零，无需移动")
        return 0

    print("=" * 55)
    print("末端相对移动 (MQTT offset_move, 单位: mm)")
    print(f"  左手: dx={args.lx:.1f} dy={args.ly:.1f} dz={args.lz:.1f}")
    print(f"  右手: dx={args.rx:.1f} dy={args.ry:.1f} dz={args.rz:.1f}")
    print("=" * 55)

    ok = send_offset_move(
        lx=args.lx, ly=args.ly, lz=args.lz,
        rx=args.rx, ry=args.ry, rz=args.rz,
        timeout=args.timeout,
    )
    if ok:
        print("[完成] ✓ 末端移动成功")
        return 0
    else:
        print("[错误] 末端移动未完成")
        return 1


if __name__ == "__main__":
    sys.exit(main())
