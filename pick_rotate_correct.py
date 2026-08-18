#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pick_rotate_correct.py — 取货 + 旋转 + 角度纠偏流程

流程:
  Part A. 取货动作 (subprocess 调用 execute/ 脚本)
    0. 张开夹爪 (初始化, 防止初始为闭合状态)
    1. 待机姿态
    2. 预抓取姿态
    3. 抓取姿态
    4. 闭合夹爪
    5. 抬升姿态
    6. 回到待机姿态

  Part B. 底盘旋转
    7. 逆时针旋转 85° (cca.move_chassis, yaw_rad)

  Part C. 放货角度纠偏 (远程 GPU 推理)
    8. 使用 jitai.pt 模型远程推理进行角度纠偏 (cca.step_yaw_correct)
       - 启用 REMOTE_INFER, 通过 MQTT 请求外部 GPU 服务器
       - 远程失败时回退本地 CPU 推理 (默认)

运行环境:
  - MQTT 服务 (main.py) 已启动
  - 远程 GPU 推理服务 (yolo_infer_server.py) 已在电脑端启动:
      python3 yolo_infer_server.py --model /data/wzd/jitai.pt --device 0 --broker 192.168.168.18
  - 位姿 JSON 已录制: pose_standby/pre_pick/pick/lift.json
  - 模型文件: /data/wzd/jitai.pt

用法:
  python3 pick_rotate_correct.py                          # 完整流程
  python3 pick_rotate_correct.py --dry-run                # 只打印不执行
  python3 pick_rotate_correct.py --skip-pick              # 跳过取货部分
  python3 pick_rotate_correct.py --skip-rotate            # 跳过旋转
  python3 pick_rotate_correct.py --skip-correct           # 跳过角度纠偏
  python3 pick_rotate_correct.py --rotate-deg 85.0        # 旋转角度 (正值=逆时针)
  python3 pick_rotate_correct.py --no-fallback            # 远程失败时不回退本地
  python3 pick_rotate_correct.py --pause 0.1              # 取货每步停顿秒数
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

# ═══════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════
WZD_DIR      = "/data/wzd"
EXEC_DIR     = os.path.join(WZD_DIR, "execute")
POSITION_DIR = os.path.join(WZD_DIR, "position")

# 取货各阶段脚本
SCRIPT_STANDBY  = os.path.join(EXEC_DIR, "move_arms_to_standby.py")
SCRIPT_PRE_PICK = os.path.join(EXEC_DIR, "move_arms_to_pre_pick.py")
SCRIPT_PICK     = os.path.join(EXEC_DIR, "move_arms_to_pick.py")
SCRIPT_LIFT     = os.path.join(EXEC_DIR, "move_arms_to_lift.py")

# 子脚本解释器 (含 paho-mqtt 的环境)
DEFAULT_PY = "/usr/bin/python3"

# 放货角度纠偏模型
MODEL_PLACE = os.path.join(WZD_DIR, "jitai.pt")
# 放货纠偏输出目录
PLACE_OUTPUT_DIR = os.path.join(WZD_DIR, "correction_place")

# 默认旋转角度 (度, 正值=逆时针)
DEFAULT_ROTATE_DEG = 85.0

# ═══════════════════════════════════════════════════════════
#  MQTT 配置 (用于夹爪控制)
# ═══════════════════════════════════════════════════════════
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_COMMANDS_DATA = "/humanoid/commands/data"
TOPIC_DONE = "/humanoid/commands/done"

# ★ 末端工具参数 (参考 pick.py: 0.0=闭合, -0.7=张开) ★
RIGHT_GRIPPER_CLOSE_POS = 0.0
LEFT_GRIPPER_CLOSE_POS  = 0.0
RIGHT_GRIPPER_OPEN_POS  = -0.7
LEFT_GRIPPER_OPEN_POS   = -0.7

# ═══════════════════════════════════════════════════════════
#  导入底盘纠偏模块 (chassis_correct_all)
# ═══════════════════════════════════════════════════════════
sys.path.insert(0, WZD_DIR)
import chassis_correct_all as cca


# ═══════════════════════════════════════════════════════════
#  Part A: 取货动作 (subprocess 调用 execute/ 脚本)
# ═══════════════════════════════════════════════════════════

def run_step(name, script, py_bin, dry_run=False, timeout=20.0):
    """调用 execute 文件夹下的脚本运动到位姿

    Parameters
    ----------
    name : str       步骤名称 (用于日志)
    script : str     脚本绝对路径
    py_bin : str     Python 解释器路径
    dry_run : bool   True 则只打印不执行
    timeout : float  超时秒数 (仅参考, 实际由 subprocess 阻塞)
    """
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


def send_gripper(broker=MQTT_BROKER, port=MQTT_PORT,
                 left_pos=LEFT_GRIPPER_CLOSE_POS,
                 right_pos=RIGHT_GRIPPER_CLOSE_POS,
                 wait=0.2, timeout=5.0, tag="夹爪"):
    """发送夹爪控制命令 (grab)

    发送 {"command":"grab","data":{"left":L,"right":R}} 到 /humanoid/commands/data
    在 on_connect 回调中发布, 确保 MQTT 连接已建立;
    订阅 /humanoid/commands/done 等待执行完成, 超时后强制结束.

    Parameters
    ----------
    left_pos / right_pos : float   0.0=闭合, -0.7=张开 (参考 pick.py)
    wait : float                    收到完成通知后额外等待秒数
    timeout : float                 等待完成通知的超时秒数
    tag : str                       日志标签
    """
    payload = json.dumps({
        "command": "grab",
        "data": {"left": left_pos, "right": right_pos}
    })
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


def run_pick(py_bin, dry_run=False, pause=0.1, gripper_wait=0.2,
             skip_init_open=False, skip_standby=False):
    """执行取货动作 (Part A)

    流程: 张开夹爪 → 待机 → 预抓取 → 抓取 → 闭合夹爪 → 抬升 → 待机
    返回: (ok, results, t_elapsed)
    """
    print(f"\n{'#'*60}")
    print(f"# Part A: 取货动作")
    print(f"{'#'*60}")

    t_start = time.time()
    results = []

    def step(name, script):
        ok = run_step(name, script, py_bin, dry_run=dry_run)
        results.append((name, ok))
        if not ok:
            print(f"\n⚠ {name} 失败, 终止取货流程")
            return False
        if pause > 0:
            time.sleep(pause)
        return True

    # ── 步骤 0: 初始化 - 张开夹爪 ──
    print(f"\n{'='*60}")
    print(f"[0.张开夹爪] 开始")
    print(f"{'='*60}")
    if dry_run:
        print("[0.张开夹爪] [DRY-RUN] 跳过")
        results.append(("0.张开夹爪", True))
    elif skip_init_open:
        print("[0.张开夹爪] 已跳过 (--skip-init-open)")
        results.append(("0.张开夹爪", True))
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_OPEN_POS,
                          right_pos=RIGHT_GRIPPER_OPEN_POS,
                          wait=gripper_wait, tag="0.张开夹爪")
        results.append(("0.张开夹爪", ok))
        if not ok:
            print("⚠ 张开夹爪失败, 终止流程")
            return False, results, time.time() - t_start
        print("[0.张开夹爪] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 1-3, 5-6: 手臂运动 ──
    if not step("1.待机姿态", SCRIPT_STANDBY):
        return False, results, time.time() - t_start
    if not step("2.预抓取姿态", SCRIPT_PRE_PICK):
        return False, results, time.time() - t_start
    if not step("3.抓取姿态", SCRIPT_PICK):
        return False, results, time.time() - t_start

    # ── 步骤 4: 闭合夹爪 ──
    print(f"\n{'='*60}")
    print(f"[4.闭合夹爪] 开始")
    print(f"{'='*60}")
    if dry_run:
        print("[4.闭合夹爪] [DRY-RUN] 跳过")
        results.append(("4.闭合夹爪", True))
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_CLOSE_POS,
                          right_pos=RIGHT_GRIPPER_CLOSE_POS,
                          wait=gripper_wait, tag="4.闭合夹爪")
        results.append(("4.闭合夹爪", ok))
        if not ok:
            print("⚠ 闭合夹爪失败, 终止流程")
            return False, results, time.time() - t_start
        print("[4.闭合夹爪] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 5-6: 抬升 + 待机 ──
    if not step("5.抬升姿态", SCRIPT_LIFT):
        return False, results, time.time() - t_start
    if not skip_standby:
        if not step("6.回到待机姿态", SCRIPT_STANDBY):
            return False, results, time.time() - t_start

    return True, results, time.time() - t_start


# ═══════════════════════════════════════════════════════════
#  Part B: 底盘旋转
# ═══════════════════════════════════════════════════════════

def run_rotate(g2, deg, dry_run=False):
    """底盘旋转指定角度 (度, 正值=逆时针)

    Parameters
    ----------
    g2 : minth.G2    底盘控制对象 (None 时 dry_run)
    deg : float      旋转角度, 正值=逆时针, 负值=顺时针
    dry_run : bool   True 则只打印不执行
    """
    print(f"\n{'#'*60}")
    print(f"# Part B: 底盘旋转")
    print(f"{'#'*60}")
    print(f"[7.旋转] {deg:+.1f}° ({'逆时针' if deg > 0 else '顺时针'})")

    if dry_run:
        print(f"[7.旋转] [DRY-RUN] 跳过")
        return True

    yaw_rad = float(np.radians(deg))
    t0 = time.time()
    ok = cca.move_chassis(g2, yaw_rad=yaw_rad)
    dt = time.time() - t0
    if ok:
        time.sleep(cca.YAW_SETTLE_TIME)
        print(f"[7.旋转] ✓ 完成 (耗时 {dt:.1f}s, 等待 {cca.YAW_SETTLE_TIME}s 稳定)")
    else:
        print(f"[7.旋转] ✗ 失败 (耗时 {dt:.1f}s)")
    return ok


# ═══════════════════════════════════════════════════════════
#  Part C: 放货角度纠偏 (远程 GPU 推理)
# ═══════════════════════════════════════════════════════════

def run_place_correct(model, g2, dry_run=False):
    """放货角度纠偏 (使用 jitai.pt, 远程推理)

    Parameters
    ----------
    model : YOLO     本地模型 (远程失败时回退用, None 则不回退)
    g2 : minth.G2    底盘控制对象
    dry_run : bool   True 则只打印不执行
    """
    print(f"\n{'#'*60}")
    print(f"# Part C: 放货角度纠偏 (远程 GPU 推理)")
    print(f"{'#'*60}")
    print(f"[8.纠偏] 模型: {os.path.basename(MODEL_PLACE)}")
    print(f"[8.纠偏] 远程推理: 已启用 (失败回退本地: {cca.REMOTE_INFER_FALLBACK})")
    print(f"[8.纠偏] 目标: |angle| < {cca.YAW_THRESHOLD_DEG}°")
    print(f"[8.纠偏] 最大迭代: {cca.YAW_MAX_ITER}")

    if dry_run:
        print(f"[8.纠偏] [DRY-RUN] 跳过")
        return True

    r_yaw = cca.step_yaw_correct(model, g2, PLACE_OUTPUT_DIR, dry_run=False, reuse_img=None)
    converged = r_yaw.get("success") and r_yaw.get("converged", False)
    if converged:
        print(f"[8.纠偏] ✓ 收敛 (final_angle={r_yaw.get('final_angle', 0):.2f}°)")
    else:
        print(f"[8.纠偏] ⚠ 未完全收敛: {r_yaw.get('reason', '达到最大迭代')}")
    return converged


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="取货 + 旋转 + 角度纠偏流程")
    # 取货参数
    parser.add_argument("--python-bin", default=DEFAULT_PY,
                        help=f"子脚本(手臂)Python解释器 (默认: {DEFAULT_PY})")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="取货每步之间停顿秒数 (默认 0.1)")
    parser.add_argument("--gripper-wait", type=float, default=0.2,
                        help="夹爪动作收到完成通知后额外等待秒数 (默认 0.2)")
    parser.add_argument("--skip-init-open", action="store_true",
                        help="跳过初始张开夹爪步骤")
    parser.add_argument("--skip-standby", action="store_true",
                        help="跳过取货最后的待机复位")
    # 旋转参数
    parser.add_argument("--rotate-deg", type=float, default=DEFAULT_ROTATE_DEG,
                        help=f"旋转角度, 正值=逆时针 (默认: {DEFAULT_ROTATE_DEG})")
    # 纠偏参数
    parser.add_argument("--no-fallback", action="store_true",
                        help="远程推理失败时不回退到本地 CPU 推理")
    # 流程控制
    parser.add_argument("--skip-pick", action="store_true", help="跳过取货部分")
    parser.add_argument("--skip-rotate", action="store_true", help="跳过旋转")
    parser.add_argument("--skip-correct", action="store_true", help="跳过角度纠偏")
    parser.add_argument("--dry-run", action="store_true", help="只打印, 不执行")
    args = parser.parse_args()

    py = args.python_bin
    dry = args.dry_run

    # ── 启用远程推理 ──
    cca.REMOTE_INFER = True
    if args.no_fallback:
        cca.REMOTE_INFER_FALLBACK = False
    print(f"[配置] 远程 GPU 推理: 启用 (回退本地: {cca.REMOTE_INFER_FALLBACK})")

    # ── 流程概览 ──
    print("=" * 60)
    print("取货 + 旋转 + 角度纠偏流程")
    print("=" * 60)
    parts = []
    if not args.skip_pick:
        parts.append("取货(0-6)")
    if not args.skip_rotate:
        parts.append(f"旋转({args.rotate_deg:+.1f}°)")
    if not args.skip_correct:
        parts.append("角度纠偏(jitai.pt/远程)")
    print(f"  执行步骤: {' → '.join(parts)}")
    print(f"  解释器: {py}")
    print(f"  每步停顿: {args.pause}s")
    print(f"  Dry-Run: {dry}")
    print("=" * 60)

    t_total = time.time()
    all_results = []

    # ── Part A: 取货 ──
    if not args.skip_pick:
        ok, pick_results, pick_dt = run_pick(
            py, dry_run=dry, pause=args.pause,
            gripper_wait=args.gripper_wait,
            skip_init_open=args.skip_init_open,
            skip_standby=args.skip_standby,
        )
        all_results.extend(pick_results)
        if not ok:
            print(f"\n⚠ 取货流程失败, 终止")
            _summary(all_results, t_total)
            return 1
        print(f"\n[Part A] ✓ 取货完成 (耗时 {pick_dt:.1f}s)")
    else:
        print(f"\n[Part A] 已跳过 (--skip-pick)")

    # ── 连接机器人 (Part B/C 需要) ──
    g2 = None
    model = None
    if (not args.skip_rotate or not args.skip_correct) and not dry:
        print(f"\n[初始化] 连接机器人...")
        g2 = cca.setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")

        # 加载本地模型 (远程失败时回退用)
        if not args.skip_correct and cca.REMOTE_INFER_FALLBACK:
            print(f"[初始化] 加载本地回退模型: {MODEL_PLACE}")
            from ultralytics import YOLO
            model = YOLO(MODEL_PLACE)
            print(f"[初始化] ✓ 模型加载完成")

    try:
        # ── Part B: 旋转 ──
        if not args.skip_rotate:
            ok = run_rotate(g2, args.rotate_deg, dry_run=dry)
            all_results.append((f"7.旋转({args.rotate_deg:+.1f}°)", ok))
            if not ok:
                print(f"\n⚠ 旋转失败, 终止")
                _summary(all_results, t_total)
                return 1
        else:
            print(f"\n[Part B] 已跳过 (--skip-rotate)")

        # ── Part C: 角度纠偏 ──
        if not args.skip_correct:
            ok = run_place_correct(model, g2, dry_run=dry)
            all_results.append(("8.角度纠偏", ok))
            if not ok:
                print(f"\n⚠ 角度纠偏未收敛, 但流程继续")
        else:
            print(f"\n[Part C] 已跳过 (--skip-correct)")

    finally:
        if g2 is not None:
            g2.close()
            print(f"\n[清理] ✓ Minth 已断开")

    _summary(all_results, t_total)
    return 0


def _summary(results, t_start):
    """打印流程总结"""
    dt = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"流程总结 (总耗时 {dt:.1f}s)")
    print(f"{'='*60}")
    for name, ok in results:
        print(f"  {name}: {'✓ 成功' if ok else '✗ 失败'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
