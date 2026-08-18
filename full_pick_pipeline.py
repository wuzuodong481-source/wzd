#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""full_pick_pipeline.py — 完整取货流程总控

流程 (4 个部分):
  Part A. 取货三步纠偏 (best_new.pt, 远程 GPU 推理)
    1. 前后纠偏 (深度对齐 target_depth)
    2. 角度纠偏 (a/b 连线斜率归零)
    3. 左右纠偏 (a/b 中点对齐图像中心)

  Part B. 取货动作 (subprocess 调用 execute/ 脚本)
    0. 张开夹爪 (初始化, 防止初始为闭合状态)
    1. 待机姿态
    2. 预抓取姿态
    3. 抓取姿态
    4. 闭合夹爪
    5. 抬升姿态
    6. 回到待机姿态

  Part C. 底盘旋转
    7. 逆时针旋转 85° (cca.move_chassis, yaw_rad)

  Part D. 放货角度纠偏 (jitai.pt, 远程 GPU 推理)
    8. 使用 jitai.pt 模型远程推理进行角度纠偏 (cca.step_yaw_correct)

★ 所有纠偏结果图统一保存在 /data/wzd/1/ 文件夹

运行环境:
  - MQTT 服务 (main.py) 已启动
  - 远程 GPU 推理服务 (yolo_infer_server.py) 已在电脑端启动:
      python3 yolo_infer_server.py --model /data/wzd/best_new.pt --device 0 --broker 192.168.168.18
      (放货阶段需要切换/同时加载 jitai.pt, 建议用两个服务进程或切换模型)
  - 位姿 JSON 已录制: pose_standby/pre_pick/pick/lift.json
  - 模型文件: /data/wzd/best_new.pt, /data/wzd/jitai.pt

用法:
  python3 full_pick_pipeline.py                            # 完整流程
  python3 full_pick_pipeline.py --dry-run                  # 只打印不执行
  python3 full_pick_pipeline.py --skip-correct-pick        # 跳过取货三步纠偏
  python3 full_pick_pipeline.py --skip-pick                # 跳过取货动作
  python3 full_pick_pipeline.py --skip-rotate              # 跳过旋转
  python3 full_pick_pipeline.py --skip-correct-place       # 跳过放货角度纠偏
  python3 full_pick_pipeline.py --rotate-deg 85.0          # 旋转角度 (正值=逆时针)
  python3 full_pick_pipeline.py --target-depth 750         # 取货前后纠偏目标深度 mm
  python3 full_pick_pipeline.py --no-fallback              # 远程失败时不回退本地
  python3 full_pick_pipeline.py --output-dir /data/wzd/1   # 纠偏结果保存目录
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

# 模型文件
MODEL_PICK  = os.path.join(WZD_DIR, "best_new.pt")   # 取货三步纠偏
MODEL_PLACE = os.path.join(WZD_DIR, "jitai.pt")      # 放货角度纠偏

# 纠偏结果统一保存目录
DEFAULT_OUTPUT_DIR = "/data/wzd/1"

# 默认参数
DEFAULT_ROTATE_DEG   = 85.0     # 旋转角度 (度, 正值=逆时针)
DEFAULT_TARGET_DEPTH = 750      # 取货前后纠偏目标深度 mm

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
#  Part B: 取货动作 (subprocess 调用 execute/ 脚本)
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
    """执行取货动作 (Part B)

    流程: 张开夹爪 → 待机 → 预抓取 → 抓取 → 闭合夹爪 → 抬升 → 待机
    返回: (ok, results, t_elapsed)
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

    # ── 步骤 1-3: 手臂运动 ──
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
#  Part A: 取货三步纠偏 (best_new.pt, 远程 GPU 推理)
# ═══════════════════════════════════════════════════════════

def run_pick_correct(model, g2, target_depth, output_dir, dry_run=False):
    """取货三步纠偏: 前后 → 角度 → 左右 (使用 best_new.pt, 远程推理)

    Parameters
    ----------
    model : YOLO     本地模型 (远程失败时回退用, None 则不回退)
    g2 : minth.G2    底盘控制对象
    target_depth : int   前后纠偏目标深度 (mm)
    output_dir : str     结果图保存目录
    dry_run : bool   True 则只打印不执行
    """
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
    abort = False  # 任一纠偏异常则终止后续步骤

    # 前后纠偏
    r_fb = cca.step_fb_correct(model, g2, target_depth, output_dir, dry_run=False)
    results["fb"] = r_fb
    if not r_fb.get("success"):
        print(f"[取货纠偏] ⚠⚠ 前后纠偏失败: {r_fb.get('reason')}")
        print(f"[取货纠偏] ⚠⚠ 终止后续纠偏, 停止机器人运动!")
        abort = True
    reuse_img = r_fb.get("color_img")

    # 角度纠偏 (前后异常则跳过)
    if not abort:
        r_yaw = cca.step_yaw_correct(model, g2, output_dir, dry_run=False, reuse_img=reuse_img)
        results["yaw"] = r_yaw
        if not r_yaw.get("success"):
            print(f"[取货纠偏] ⚠⚠ 角度纠偏失败: {r_yaw.get('reason')}")
            print(f"[取货纠偏] ⚠⚠ 终止后续纠偏, 停止机器人运动!")
            abort = True
        reuse_img = r_yaw.get("color_img")

    # 左右纠偏 (前序异常则跳过)
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
    # abort=True 表示有异常 (安全终止/检测失败), 返回 False 终止流程
    # 未收敛 (success=True, converged=False) 不算异常, 返回 True 继续流程
    no_error = not abort
    return no_error, results


# ═══════════════════════════════════════════════════════════
#  Part C: 底盘旋转
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
    print(f"# Part C: 底盘旋转")
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
#  Part D: 放货角度纠偏 (jitai.pt, 远程 GPU 推理)
# ═══════════════════════════════════════════════════════════

def run_place_correct(model, g2, output_dir, dry_run=False):
    """放货角度纠偏 (使用 jitai.pt, 远程推理)

    Parameters
    ----------
    model : YOLO     本地模型 (远程失败时回退用, None 则不回退)
    g2 : minth.G2    底盘控制对象
    output_dir : str     结果图保存目录
    dry_run : bool   True 则只打印不执行
    """
    print(f"\n{'#'*60}")
    print(f"# Part D: 放货角度纠偏 (jitai.pt, 远程推理)")
    print(f"{'#'*60}")
    print(f"[8.纠偏] 模型: {os.path.basename(MODEL_PLACE)}")
    print(f"[8.纠偏] 远程推理: 已启用 (失败回退本地: {cca.REMOTE_INFER_FALLBACK})")
    print(f"[8.纠偏] 目标: |angle| < {cca.YAW_THRESHOLD_DEG}°")
    print(f"[8.纠偏] 最大迭代: {cca.YAW_MAX_ITER}")
    print(f"[8.纠偏] 结果保存: {output_dir}")

    if dry_run:
        print(f"[8.纠偏] [DRY-RUN] 跳过")
        return True

    r_yaw = cca.step_yaw_correct(model, g2, output_dir, dry_run=False, reuse_img=None)
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
    parser = argparse.ArgumentParser(description="完整取货流程: 三步纠偏→取货→旋转→角度纠偏")
    # 取货动作参数
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
    # 取货纠偏参数
    parser.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH,
                        help=f"取货前后纠偏目标深度mm (默认: {DEFAULT_TARGET_DEPTH})")
    # 纠偏结果保存目录
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"纠偏结果保存目录 (默认: {DEFAULT_OUTPUT_DIR})")
    # 远程推理
    parser.add_argument("--no-fallback", action="store_true",
                        help="远程推理失败时不回退到本地 CPU 推理")
    # 流程控制
    parser.add_argument("--skip-correct-pick", action="store_true", help="跳过取货三步纠偏")
    parser.add_argument("--skip-pick", action="store_true", help="跳过取货动作")
    parser.add_argument("--skip-rotate", action="store_true", help="跳过旋转")
    parser.add_argument("--skip-correct-place", action="store_true", help="跳过放货角度纠偏")
    parser.add_argument("--dry-run", action="store_true", help="只打印, 不执行")
    args = parser.parse_args()

    py = args.python_bin
    dry = args.dry_run
    output_dir = args.output_dir

    # ── 启用远程推理 ──
    cca.REMOTE_INFER = True
    if args.no_fallback:
        cca.REMOTE_INFER_FALLBACK = False
    print(f"[配置] 远程 GPU 推理: 启用 (回退本地: {cca.REMOTE_INFER_FALLBACK})")
    print(f"[配置] 纠偏结果保存目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ── 流程概览 ──
    print("=" * 60)
    print("完整取货流程: 三步纠偏 → 取货 → 旋转 → 角度纠偏")
    print("=" * 60)
    parts = []
    if not args.skip_correct_pick:
        parts.append("A.取货三步纠偏(best_new)")
    if not args.skip_pick:
        parts.append("B.取货动作(0-6)")
    if not args.skip_rotate:
        parts.append(f"C.旋转({args.rotate_deg:+.1f}°)")
    if not args.skip_correct_place:
        parts.append("D.放货角度纠偏(jitai)")
    print(f"  执行步骤: {' → '.join(parts)}")
    print(f"  解释器: {py}")
    print(f"  每步停顿: {args.pause}s")
    print(f"  取货目标深度: {args.target_depth}mm")
    print(f"  Dry-Run: {dry}")
    print("=" * 60)

    t_total = time.time()
    all_results = []

    # ── 连接机器人 + 加载模型 (任一纠偏部分需要) ──
    g2 = None
    model_pick = None
    model_place = None
    need_robot = (not args.skip_correct_pick) or (not args.skip_rotate) or (not args.skip_correct_place)
    if need_robot and not dry:
        print(f"\n[初始化] 连接机器人...")
        g2 = cca.setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")

        # 加载推理模型
        if cca.USE_RHINO_INFER:
            if not args.skip_correct_pick:
                print(f"[初始化] 辉羲 RPU 芯片推理: {cca.RHINO_REF_MODEL}")
                model_pick = cca.RhinoInfer(cca.RHINO_REF_MODEL)
                print(f"[初始化] ✓ 辉羲 RPU 取货模型就绪")
            if not args.skip_correct_place:
                print(f"[初始化] 辉羲 RPU 芯片推理: {cca.RHINO_REF_MODEL}")
                model_place = cca.RhinoInfer(cca.RHINO_REF_MODEL)
                print(f"[初始化] ✓ 辉羲 RPU 放货模型就绪")
        elif cca.REMOTE_INFER_FALLBACK:
            from ultralytics import YOLO
            if not args.skip_correct_pick:
                print(f"[初始化] 加载取货回退模型: {MODEL_PICK}")
                model_pick = YOLO(MODEL_PICK)
                print(f"[初始化] ✓ 取货模型加载完成")
            if not args.skip_correct_place:
                print(f"[初始化] 加载放货回退模型: {MODEL_PLACE}")
                model_place = YOLO(MODEL_PLACE)
                print(f"[初始化] ✓ 放货模型加载完成")
        else:
            print(f"[初始化] 远程推理无回退模式, 跳过本地模型加载")

    try:
        # ── Part A: 取货三步纠偏 ──
        if not args.skip_correct_pick:
            ok, _ = run_pick_correct(model_pick, g2, args.target_depth, output_dir, dry_run=dry)
            all_results.append(("A.取货三步纠偏", ok))
            if not ok:
                print(f"\n⚠⚠ 取货纠偏异常, 终止整个流程, 停止机器人运动!")
                _summary(all_results, t_total)
                return 1
        else:
            print(f"\n[Part A] 已跳过 (--skip-correct-pick)")

        # ── Part B: 取货动作 ──
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
            print(f"\n[Part B] ✓ 取货完成 (耗时 {pick_dt:.1f}s)")
        else:
            print(f"\n[Part B] 已跳过 (--skip-pick)")

        # ── Part C: 旋转 ──
        if not args.skip_rotate:
            ok = run_rotate(g2, args.rotate_deg, dry_run=dry)
            all_results.append((f"C.旋转({args.rotate_deg:+.1f}°)", ok))
            if not ok:
                print(f"\n⚠ 旋转失败, 终止")
                _summary(all_results, t_total)
                return 1
        else:
            print(f"\n[Part C] 已跳过 (--skip-rotate)")

        # ── Part D: 放货角度纠偏 ──
        if not args.skip_correct_place:
            ok = run_place_correct(model_place, g2, output_dir, dry_run=dry)
            all_results.append(("D.放货角度纠偏", ok))
            if not ok:
                print(f"\n⚠ 角度纠偏未收敛, 但流程继续")
        else:
            print(f"\n[Part D] 已跳过 (--skip-correct-place)")

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
