#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_pipeline.py — 抓取总控程序

流程:
  0. 初始化 - 张开夹爪 (防止初始为闭合状态)
  1. 待机姿态
  2. 预抓取姿态
  3. 抓取姿态
  4. 闭合夹爪
  5. 抬升姿态
  6. 回到待机姿态

运行:
  python pick_pipeline.py                    # 完整流程
  python pick_pipeline.py --dry-run          # 只打印不执行
  python pick_pipeline.py --skip-standby     # 跳过最后的待机复位
  python pick_pipeline.py --skip-init-open   # 跳过初始张开夹爪
  python pick_pipeline.py --pause SEC        # 每步之间停顿秒数 (默认 0.3)

环境:
  - MQTT 服务 (main.py) 已启动
  - 位姿 JSON 已录制: pose_standby/pre_pick/pick/lift.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import paho.mqtt.client as mqtt

# ═══════════════════════════════════════════════════════════
#  ★ 配置 ★
# ═══════════════════════════════════════════════════════════

WZD_DIR    = "/data/wzd"
EXEC_DIR   = os.path.join(WZD_DIR, "execute")
POSITION_DIR = os.path.join(WZD_DIR, "position")

# 各阶段对应的执行脚本
SCRIPT_STANDBY   = os.path.join(EXEC_DIR, "move_arms_to_standby.py")
SCRIPT_PRE_PICK  = os.path.join(EXEC_DIR, "move_arms_to_pre_pick.py")
SCRIPT_PICK      = os.path.join(EXEC_DIR, "move_arms_to_pick.py")
SCRIPT_LIFT      = os.path.join(EXEC_DIR, "move_arms_to_lift.py")

# 子脚本解释器 (含 paho-mqtt 的环境)
DEFAULT_PY = "/usr/bin/python3"

# MQTT 配置 (用于闭合夹爪)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_COMMANDS_DATA = "/humanoid/commands/data"

# ★ 末端工具参数 (参考 pick.py: 0.0=闭合, -0.7=张开) ★
RIGHT_GRIPPER_CLOSE_POS = 0.0
LEFT_GRIPPER_CLOSE_POS  = 0.0      # 左手同步闭合
RIGHT_GRIPPER_OPEN_POS  = -0.7     # 右手张开
LEFT_GRIPPER_OPEN_POS   = -0.7     # 左手张开


# ═══════════════════════════════════════════════════════════
#  运行子脚本
# ═══════════════════════════════════════════════════════════

def run_step(name, script, py_bin, dry_run=False, timeout=20.0):
    """调用 execute 文件夹下的脚本运动到位姿"""
    print(f"\n{'='*60}")
    print(f"[{name}] 开始")
    print(f"[{name}] 脚本: {script}")
    print(f"{'='*60}")
    if dry_run:
        print(f"[{name}] [DRY-RUN] 跳过")
        return True
    if not os.path.exists(script):
        print(f"[{name}] ✗ 脚本不存在: {script}")
        return False
    t0 = time.time()
    try:
        ret = subprocess.run([py_bin, "-u", script], check=False)
        dt = time.time() - t0
        ok = ret.returncode == 0
        print(f"[{name}] {'✓ 成功' if ok else '✗ 失败'} (耗时 {dt:.1f}s)")
        return ok
    except Exception as e:
        print(f"[{name}] ✗ 异常: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  末端工具控制
# ═══════════════════════════════════════════════════════════

def send_gripper(broker=MQTT_BROKER, port=MQTT_PORT,
                 left_pos=LEFT_GRIPPER_CLOSE_POS,
                 right_pos=RIGHT_GRIPPER_CLOSE_POS,
                 wait=1.0, timeout=5.0, tag="夹爪"):
    """发送夹爪控制命令 (grab)

    发送 {"command":"grab","data":{"left":L,"right":R}} 到 /humanoid/commands/data
    在 on_connect 回调中发布, 确保 MQTT 连接已建立;
    订阅 /humanoid/commands/done 等待执行完成, 超时后强制结束.

    left_pos / right_pos: 0.0=闭合, -0.7=张开 (参考 pick.py)
    wait: 收到完成通知后额外等待秒数, 让夹爪机械稳定
    timeout: 等待完成通知的超时秒数
    tag: 日志标签 (如 "夹爪闭合" / "夹爪张开")
    """
    payload = json.dumps({
        "command": "grab",
        "data": {
            "left":  left_pos,
            "right": right_pos,
        }
    })
    TOPIC_DONE = "/humanoid/commands/done"
    state = {"ok": False, "error": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(TOPIC_DONE, qos=0)
            client.publish(TOPIC_COMMANDS_DATA, payload, qos=2)
            print(f"[{tag}] 已发送 grab 命令 (left={left_pos}, right={right_pos})")
        else:
            print(f"[{tag}] MQTT 连接失败, rc={rc}")
            state["error"] = True

    def on_message(client, userdata, msg):
        if msg.topic == TOPIC_DONE:
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                if data.get("cmd") == "grab":
                    state["ok"] = True
                    client.disconnect()
            except Exception:
                pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        print(f"[{tag}] ✗ 连接 MQTT 失败: {e}")
        return False

    t_start = time.time()
    while not state["ok"] and not state["error"] and time.time() - t_start < timeout:
        client.loop(timeout=0.1)

    try:
        client.disconnect()
    except Exception:
        pass

    if state["error"]:
        print(f"[{tag}] ✗ 连接失败")
        return False
    if not state["ok"]:
        print(f"[{tag}] ⚠ 等待完成超时 ({timeout}s), 继续后续步骤")
    else:
        print(f"[{tag}] ✓ 收到完成通知")
        if wait > 0:
            time.sleep(wait)
    return True


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="抓取总控程序")
    parser.add_argument("--python-bin", default=DEFAULT_PY,
                        help=f"子脚本Python解释器 (默认: {DEFAULT_PY})")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="每步之间停顿秒数 (默认 0.1)")
    parser.add_argument("--gripper-wait", type=float, default=0.2,
                        help="夹爪动作收到完成通知后额外等待秒数 (默认 0.2)")
    parser.add_argument("--skip-standby", action="store_true",
                        help="跳过最后的待机复位")
    parser.add_argument("--skip-init-open", action="store_true",
                        help="跳过初始张开夹爪步骤")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印, 不执行")
    args = parser.parse_args()

    py = args.python_bin
    dry = args.dry_run
    pause = args.pause

    print("=" * 60)
    print("抓取总控流程")
    print("=" * 60)
    print(f"  流程: 张开夹爪 → 待机 → 预抓取 → 抓取 → 闭合夹爪 → 抬升 → 待机")
    print(f"  解释器: {py}")
    print(f"  每步停顿: {pause}s")
    print(f"  Dry-Run: {dry}")
    print("=" * 60)

    t_start = time.time()
    results = []

    def step(name, script):
        ok = run_step(name, script, py, dry_run=dry)
        results.append((name, ok))
        if not ok:
            print(f"\n⚠ {name} 失败, 终止流程")
            return False
        if pause > 0:
            time.sleep(pause)
        return True

    # ── 步骤 0: 初始化 - 张开夹爪 ──
    print(f"\n{'='*60}")
    print(f"[0.张开夹爪] 开始")
    print(f"{'='*60}")
    if dry:
        print("[0.张开夹爪] [DRY-RUN] 跳过")
        results.append(("0.张开夹爪", True))
    elif args.skip_init_open:
        print("[0.张开夹爪] 已跳过 (--skip-init-open)")
        results.append(("0.张开夹爪", True))
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_OPEN_POS,
                          right_pos=RIGHT_GRIPPER_OPEN_POS,
                          wait=args.gripper_wait, tag="0.张开夹爪")
        results.append(("0.张开夹爪", ok))
        if not ok:
            print("⚠ 张开夹爪失败, 终止流程")
            _summary(results, t_start)
            return 1
        print("[0.张开夹爪] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 1: 待机姿态 ──
    if not step("1.待机姿态", SCRIPT_STANDBY):
        _summary(results, t_start)
        return 1

    # ── 步骤 2: 预抓取 ──
    if not step("2.预抓取姿态", SCRIPT_PRE_PICK):
        _summary(results, t_start)
        return 1

    # ── 步骤 3: 抓取 ──
    if not step("3.抓取姿态", SCRIPT_PICK):
        _summary(results, t_start)
        return 1

    # ── 步骤 4: 闭合夹爪 ──
    print(f"\n{'='*60}")
    print(f"[4.闭合夹爪] 开始")
    print(f"{'='*60}")
    if dry:
        print("[4.闭合夹爪] [DRY-RUN] 跳过")
        results.append(("4.闭合夹爪", True))
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_CLOSE_POS,
                          right_pos=RIGHT_GRIPPER_CLOSE_POS,
                          wait=args.gripper_wait, tag="4.闭合夹爪")
        results.append(("4.闭合夹爪", ok))
        if not ok:
            print("⚠ 闭合夹爪失败, 终止流程")
            _summary(results, t_start)
            return 1
        print("[4.闭合夹爪] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 5: 抬升 ──
    if not step("5.抬升姿态", SCRIPT_LIFT):
        _summary(results, t_start)
        return 1

    # ── 步骤 6: 回到待机 ──
    if not args.skip_standby:
        if not step("6.回到待机姿态", SCRIPT_STANDBY):
            _summary(results, t_start)
            return 1

    _summary(results, t_start)
    return 0


def _summary(results, t_start):
    """打印总结"""
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"抓取流程总结 (总耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, ok in results:
        print(f"  {name}: {'✓ 成功' if ok else '✗ 失败'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
