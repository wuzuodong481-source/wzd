#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nav_pick_pipeline.py — 三步纠偏 + 取货 + 待机

流程:
  Part A. 取货三步纠偏 (best_new.pt, 远程 GPU 推理)
    1. 前后纠偏 (深度对齐 target_depth)
    2. 角度纠偏 (a/b 连线斜率归零)
    3. 左右纠偏 (a/b 中点对齐图像中心)
  Part B. 取货动作
    0. 张开夹爪 → 1.待机 → 2.预抓取 → 3.抓取 → 4.闭合夹爪 → 5.抬升 → 6.待机
  Part C. 停在待机姿态

★ 纠偏结果保存在 /data/wzd/1/ 文件夹
★ 任一纠偏异常立即停止机器人运动, 终止后续流程

用法:
  python3 nav_pick_pipeline.py                          # 完整流程
  python3 nav_pick_pipeline.py --dry-run                # 只打印不执行
  python3 nav_pick_pipeline.py --skip-correct           # 跳过三步纠偏
  python3 nav_pick_pipeline.py --skip-pick              # 跳过取货动作
  python3 nav_pick_pipeline.py --target-depth 750       # 前后纠偏目标深度 mm
  python3 nav_pick_pipeline.py --no-fallback            # 远程失败时不回退本地
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
MINTH_DIR    = "/data/wxf/wxf0721/runtime"

# 取货各阶段脚本
SCRIPT_STANDBY  = os.path.join(EXEC_DIR, "move_arms_to_standby.py")
SCRIPT_PRE_PICK = os.path.join(EXEC_DIR, "move_arms_to_pre_pick.py")
SCRIPT_PICK     = os.path.join(EXEC_DIR, "move_arms_to_pick.py")
SCRIPT_LIFT     = os.path.join(EXEC_DIR, "move_arms_to_lift.py")

# 子脚本解释器
DEFAULT_PY = "/usr/bin/python3"

# 模型文件
MODEL_PICK = os.path.join(WZD_DIR, "best_new.pt")

# 纠偏结果保存目录
DEFAULT_OUTPUT_DIR = "/data/wzd/1"

# 默认参数
DEFAULT_TARGET_DEPTH = 750      # 前后纠偏目标深度 mm

# ═══════════════════════════════════════════════════════════
#  MQTT 配置 (夹爪控制)
# ═══════════════════════════════════════════════════════════
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_COMMANDS_DATA = "/humanoid/commands/data"
TOPIC_DONE = "/humanoid/commands/done"

# 末端工具参数 (0.0=闭合, -0.7=张开)
RIGHT_GRIPPER_CLOSE_POS = 0.0
LEFT_GRIPPER_CLOSE_POS  = 0.0
RIGHT_GRIPPER_OPEN_POS  = -0.7
LEFT_GRIPPER_OPEN_POS   = -0.7

# ═══════════════════════════════════════════════════════════
#  导入底盘纠偏模块
# ═══════════════════════════════════════════════════════════
sys.path.insert(0, WZD_DIR)
import chassis_correct_all as cca


def setup_minth(broker=MQTT_BROKER, port=MQTT_PORT, timeout=60):
    """初始化 Minth G2 控制"""
    if MINTH_DIR not in sys.path:
        sys.path.insert(0, MINTH_DIR)
    import minth
    return minth.G2(broker=broker, port=port, timeout=timeout)


# ═══════════════════════════════════════════════════════════
#  Part A: 取货三步纠偏
# ═══════════════════════════════════════════════════════════

def run_pick_correct(model, g2, target_depth, output_dir, dry_run=False):
    """取货三步纠偏: 前后 → 角度 → 左右 (使用 best_new.pt, 远程推理)"""
    print(f"\n{'#'*60}")
    print(f"# Part A: 取货三步纠偏 (best_new.pt, 远程推理)")
    print(f"{'#'*60}")
    print(f"[纠偏] 模型: {os.path.basename(MODEL_PICK)}")
    print(f"[纠偏] 远程推理: 已启用 (失败回退本地: {cca.REMOTE_INFER_FALLBACK})")
    print(f"[纠偏] 目标深度: {target_depth}mm")
    print(f"[纠偏] 结果保存: {output_dir}")

    if dry_run:
        print(f"[纠偏] [DRY-RUN] 跳过")
        return True, {}

    results = {}
    reuse_img = None
    abort = False

    # 前后纠偏
    r_fb = cca.step_fb_correct(model, g2, target_depth, output_dir, dry_run=False)
    results["fb"] = r_fb
    if not r_fb.get("success"):
        print(f"[取货纠偏] ⚠⚠ 前后纠偏失败: {r_fb.get('reason')}")
        print(f"[取货纠偏] ⚠⚠ 终止后续纠偏, 停止机器人运动!")
        abort = True
    reuse_img = r_fb.get("color_img")

    # 角度纠偏
    if not abort:
        r_yaw = cca.step_yaw_correct(model, g2, output_dir, dry_run=False, reuse_img=reuse_img)
        results["yaw"] = r_yaw
        if not r_yaw.get("success"):
            print(f"[取货纠偏] ⚠⚠ 角度纠偏失败: {r_yaw.get('reason')}")
            print(f"[取货纠偏] ⚠⚠ 终止后续纠偏, 停止机器人运动!")
            abort = True
        reuse_img = r_yaw.get("color_img")

    # 左右纠偏
    if not abort:
        r_lr = cca.step_lr_correct(model, g2, output_dir, dry_run=False, reuse_img=reuse_img)
        results["lr"] = r_lr
        if not r_lr.get("success"):
            print(f"[取货纠偏] ⚠⚠ 左右纠偏失败: {r_lr.get('reason')}")
            print(f"[取货纠偏] ⚠⚠ 终止后续纠偏, 停止机器人运动!")
            abort = True

    # 汇总
    fb_ok  = r_fb.get("success") and r_fb.get("converged", False)
    yaw_ok = (not abort and "yaw" in results
              and r_yaw.get("success") and r_yaw.get("converged", False))
    lr_ok  = (not abort and "lr" in results
              and r_lr.get("success") and r_lr.get("converged", False))
    print(f"\n[取货纠偏] 前后={'✓' if fb_ok else '⚠'}  角度={'✓' if yaw_ok else '⚠'}  左右={'✓' if lr_ok else '⚠'}")

    no_error = not abort
    return no_error, results


# ═══════════════════════════════════════════════════════════
#  Part B: 取货动作
# ═══════════════════════════════════════════════════════════

def run_step(name, script, py_bin, dry_run=False):
    """调用 execute 文件夹下的脚本运动到位姿"""
    print(f"\n{'='*60}")
    print(f"[{name}] 开始")
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
    """发送夹爪控制命令"""
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
             skip_init_open=False):
    """执行取货动作

    流程: 张开夹爪 → 待机 → 预抓取 → 抓取 → 闭合夹爪 → 抬升 → 待机
    """
    print(f"\n{'#'*60}")
    print(f"# Part B: 取货动作")
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

    # 步骤 0: 张开夹爪
    print(f"\n{'='*60}")
    print(f"[0.张开夹爪] 开始")
    print(f"{'='*60}")
    if dry_run:
        print("[0.张开夹爪] [DRY-RUN] 跳过")
        results.append(("0.张开夹爪", True))
    elif skip_init_open:
        print("[0.张开夹爪] 已跳过")
        results.append(("0.张开夹爪", True))
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_OPEN_POS,
                          right_pos=RIGHT_GRIPPER_OPEN_POS,
                          wait=gripper_wait, tag="0.张开夹爪")
        results.append(("0.张开夹爪", ok))
        if not ok:
            return False, results, time.time() - t_start
        print("[0.张开夹爪] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # 步骤 1-3: 手臂运动
    if not step("1.待机姿态", SCRIPT_STANDBY):
        return False, results, time.time() - t_start
    if not step("2.预抓取姿态", SCRIPT_PRE_PICK):
        return False, results, time.time() - t_start
    if not step("3.抓取姿态", SCRIPT_PICK):
        return False, results, time.time() - t_start

    # 步骤 4: 闭合夹爪
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
            return False, results, time.time() - t_start
        print("[4.闭合夹爪] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # 步骤 5-6: 抬升 + 待机
    if not step("5.抬升姿态", SCRIPT_LIFT):
        return False, results, time.time() - t_start
    if not step("6.回到待机姿态", SCRIPT_STANDBY):
        return False, results, time.time() - t_start

    return True, results, time.time() - t_start


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="三步纠偏→取货→待机")
    # 取货动作参数
    parser.add_argument("--python-bin", default=DEFAULT_PY,
                        help=f"子脚本Python解释器 (默认: {DEFAULT_PY})")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="取货每步之间停顿秒数 (默认 0.1)")
    parser.add_argument("--gripper-wait", type=float, default=0.2,
                        help="夹爪动作额外等待秒数 (默认 0.2)")
    parser.add_argument("--skip-init-open", action="store_true",
                        help="跳过初始张开夹爪步骤")
    # 纠偏参数
    parser.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH,
                        help=f"前后纠偏目标深度mm (默认: {DEFAULT_TARGET_DEPTH})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"纠偏结果保存目录 (默认: {DEFAULT_OUTPUT_DIR})")
    # 远程推理
    parser.add_argument("--no-fallback", action="store_true",
                        help="远程推理失败时不回退到本地 CPU 推理")
    # 流程控制
    parser.add_argument("--skip-correct", action="store_true", help="跳过三步纠偏")
    parser.add_argument("--skip-pick", action="store_true", help="跳过取货动作")
    parser.add_argument("--dry-run", action="store_true", help="只打印, 不执行")
    args = parser.parse_args()

    py = args.python_bin
    dry = args.dry_run
    output_dir = args.output_dir

    # 启用远程推理
    cca.REMOTE_INFER = True
    if args.no_fallback:
        cca.REMOTE_INFER_FALLBACK = False
    print(f"[配置] 远程 GPU 推理: 启用 (回退本地: {cca.REMOTE_INFER_FALLBACK})")
    print(f"[配置] 纠偏结果保存目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 流程概览
    print("=" * 60)
    print("完整流程: 三步纠偏 → 取货 → 待机")
    print("=" * 60)
    parts = []
    if not args.skip_correct:
        parts.append("A.三步纠偏(best_new)")
    if not args.skip_pick:
        parts.append("B.取货动作(0-6)")
    parts.append("C.待机")
    print(f"  执行步骤: {' → '.join(parts)}")
    print(f"  取货目标深度: {args.target_depth}mm")
    print(f"  Dry-Run: {dry}")
    print("=" * 60)

    t_total = time.time()
    all_results = []

    # 连接机器人 + 加载模型
    g2 = None
    model_pick = None
    need_robot = (not args.skip_correct)
    if need_robot and not dry:
        print(f"\n[初始化] 连接机器人...")
        g2 = setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")

    if (not args.skip_correct) and not dry:
        if cca.REMOTE_INFER_FALLBACK:
            from ultralytics import YOLO
            print(f"[初始化] 加载取货回退模型: {MODEL_PICK}")
            model_pick = YOLO(MODEL_PICK)
            print(f"[初始化] ✓ 取货模型加载完成")

    try:
        # ── Part A: 三步纠偏 ──
        if not args.skip_correct:
            ok, _ = run_pick_correct(model_pick, g2, args.target_depth, output_dir, dry_run=dry)
            all_results.append(("A.三步纠偏", ok))
            if not ok:
                print(f"\n⚠⚠ 取货纠偏异常, 终止整个流程, 停止机器人运动!")
                _summary(all_results, t_total)
                return 1
        else:
            print(f"\n[Part A] 已跳过 (--skip-correct)")

        # ── Part B: 取货动作 ──
        if not args.skip_pick:
            ok, pick_results, pick_dt = run_pick(
                py, dry_run=dry, pause=args.pause,
                gripper_wait=args.gripper_wait,
                skip_init_open=args.skip_init_open,
            )
            all_results.extend(pick_results)
            if not ok:
                print(f"\n⚠ 取货流程失败, 终止")
                _summary(all_results, t_total)
                return 1
            print(f"\n[Part B] ✓ 取货完成 (耗时 {pick_dt:.1f}s)")
        else:
            print(f"\n[Part B] 已跳过 (--skip-pick)")

        # ── Part C: 待机姿态 (取货流程已包含待机, 此处确认) ──
        print(f"\n{'#'*60}")
        print(f"# Part C: 确认待机姿态")
        print(f"{'#'*60}")
        if not dry and not args.skip_pick:
            print("[待机] ✓ 已在待机姿态 (取货流程已完成)")
        else:
            print("[待机] (跳过)")

    finally:
        if g2 is not None:
            try:
                g2.close()
            except Exception:
                pass

    _summary(all_results, t_total)
    return 0


def _summary(all_results, t_start):
    """打印总结"""
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"流程总结 (耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, ok in all_results:
        print(f"  {name}: {'✓' if ok else '✗'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
