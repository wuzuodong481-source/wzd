#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nav_correct_pipeline.py — 地图导航 + 取货纠偏 + 取货动作 + 转向 + 放货纠偏 + 后退

流程 (全自动不间断):
  Part 0. 初始化: 双臂回待机姿态 + 张开夹爪
  Part A. 地图导航 0 → 1 → 2 号点
  Part B. 取货三步纠偏 (best_new.ref, 辉羲 RPU 推理)
          前后 → 角度 → 左右
  Part B2. 取货动作 (subprocess 调用 execute/ 脚本)
          待机 → 预抓取 → 抓取 → 夹爪关闭 → 抬升 → 待机
  Part C. 底盘逆时针旋转 90°
  Part D. 放货三步纠偏 (jitai_new.ref, 辉羲 RPU 推理)
          角度 → 前后 → 左右
  Part E. 底盘后退 3m

★ 全流程不间断运行, 无用户确认
★ 任一异常立即停止机器人运动, 终止后续流程
★ 默认执行 Part 0/A/B/B2; Part C/D/E 需用 --skip-* 控制是否跳过

用法:
  python3 nav_correct_pipeline.py                    # 完整流程
  python3 nav_correct_pipeline.py --dry-run          # 只打印不执行
  python3 nav_correct_pipeline.py --skip-nav         # 跳过地图导航
  python3 nav_correct_pipeline.py --skip-pick        # 跳过取货纠偏
  python3 nav_correct_pipeline.py --skip-pick-action # 跳过取货动作
  python3 nav_correct_pipeline.py --skip-init        # 跳过初始化 (双臂+夹爪)
  python3 nav_correct_pipeline.py --skip-rotate      # 跳过旋转90°
  python3 nav_correct_pipeline.py --skip-place       # 跳过放货纠偏
  python3 nav_correct_pipeline.py --skip-backoff     # 跳过后退3m
  python3 nav_correct_pipeline.py --target-depth 750      # 取货前后纠偏目标深度 mm
  python3 nav_correct_pipeline.py --place-target-depth 800 # 放货前后纠偏目标深度 mm

验证取货流程 (跳过放货后续):
  python3 nav_correct_pipeline.py --skip-rotate --skip-place --skip-backoff
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import paho.mqtt.client as mqtt

# ═══════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════
WZD_DIR   = "/data/wzd"
MINTH_DIR = "/data/wxf/wxf0721/runtime"

# 导入纠偏总控模块
sys.path.insert(0, WZD_DIR)
import chassis_correct_all as cca          # 取货纠偏 (best_new.ref)
import jitai_correct_all   as jca          # 放货纠偏 (jitai_new.ref)
from rhino_infer import RhinoInfer

# 导入 minth 机器人控制库
sys.path.insert(0, MINTH_DIR)
from minth import G2


# ═══════════════════════════════════════════════════════════
#  导航点配置
# ═══════════════════════════════════════════════════════════
NAV_POINTS   = [0, 1, 2]                    # 导航路径: 0 → 1 → 2
ROTATE_RAD   = math.pi / 2                  # 逆时针旋转 90° = π/2
PICK_OUTPUT  = "/data/wzd/correction_all"   # 取货纠偏结果输出目录
PLACE_OUTPUT = "/data/wzd/jitai_correct_all"  # 放货纠偏结果输出目录


# ═══════════════════════════════════════════════════════════
#  双臂/夹爪控制配置
# ═══════════════════════════════════════════════════════════
EXEC_DIR = os.path.join(WZD_DIR, "execute")
SCRIPT_STANDBY  = os.path.join(EXEC_DIR, "move_arms_to_standby.py")
SCRIPT_PRE_PICK = os.path.join(EXEC_DIR, "move_arms_to_pre_pick.py")
SCRIPT_PICK     = os.path.join(EXEC_DIR, "move_arms_to_pick.py")
SCRIPT_LIFT     = os.path.join(EXEC_DIR, "move_arms_to_lift.py")
SCRIPT_PRE_PLACE = os.path.join(EXEC_DIR, "move_arms_to_pre_place.py")
SCRIPT_PLACE     = os.path.join(EXEC_DIR, "move_arms_to_place.py")
SCRIPT_POSE     = os.path.join(EXEC_DIR, "move_arms_to_pose.py")  # 通用位姿脚本

# 双臂初始化姿态 (对称水平伸出)
POSE_INITIAL_FILE = os.path.join(WZD_DIR, "position", "pose_initial.json")

# 子脚本解释器
PY_BIN = sys.executable

# MQTT 主题 (夹爪控制)
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883
TOPIC_COMMANDS_DATA = "/humanoid/commands/data"
TOPIC_DONE = "/humanoid/commands/done"

# 夹爪位置 (参考 pick.py: 0.0=闭合, -0.7=张开)
LEFT_GRIPPER_OPEN_POS  = -0.7
RIGHT_GRIPPER_OPEN_POS = -0.7
LEFT_GRIPPER_CLOSE_POS  = 0.0
RIGHT_GRIPPER_CLOSE_POS = 0.0


# ═══════════════════════════════════════════════════════════
#  Part 0: 初始化 (双臂回初始姿态 + 张开夹爪)
# ═══════════════════════════════════════════════════════════
def send_gripper(left_pos, right_pos, wait=0.2, timeout=5.0, tag="夹爪"):
    """发送夹爪控制命令 (grab)

    发送 {"command":"grab","data":{"left":L,"right":R}} 到 /humanoid/commands/data
    订阅 /humanoid/commands/done 等待执行完成

    Args:
        left_pos / right_pos: 0.0=闭合, -0.7=张开
        wait: 收到完成通知后额外等待秒数
        timeout: 等待完成通知的超时秒数
        tag: 日志标签
    Returns:
        bool: True=成功
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
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
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


def run_arm_script(name, script, pose_file=None, dry_run=False, timeout=20.0):
    """调用 execute/ 下的脚本运动到位姿

    Args:
        name: 步骤名称 (用于日志)
        script: 脚本绝对路径
        pose_file: 可选, 通过 --pose 参数传给脚本
        dry_run: 只打印不执行
        timeout: 超时秒数 (仅参考)
    Returns:
        bool: True=成功
    """
    print(f"\n{'='*60}")
    print(f"[{name}] 开始")
    print(f"[{name}] 脚本: {script}" + (f" --pose {pose_file}" if pose_file else ""))
    print(f"{'='*60}")
    if dry_run:
        print(f"[{name}] [DRY-RUN] 跳过")
        return True
    if not os.path.exists(script):
        print(f"[{name}] ✗ 脚本不存在: {script}")
        return False

    cmd = [PY_BIN, "-u", script]
    if pose_file:
        cmd += ["--pose", pose_file]

    t0 = time.time()
    try:
        ret = subprocess.run(cmd, check=False)
        dt = time.time() - t0
        ok = ret.returncode == 0
        print(f"[{name}] {'✓ 成功' if ok else '✗ 失败'} (耗时 {dt:.1f}s)")
        return ok
    except Exception as e:
        print(f"[{name}] ✗ 异常: {e}")
        return False


def run_init(dry_run=False):
    """Part 0: 初始化 (双臂回初始姿态 + 张开夹爪)

    流程:
        1. 双臂运动到 pose_initial.json (对称水平伸出)
        2. 张开双侧夹爪

    Returns:
        bool: True=成功
    """
    print(f"\n{'#'*60}")
    print(f"# Part 0: 初始化 (双臂回初始姿态 + 张开夹爪)")
    print(f"{'#'*60}")

    # 1. 双臂回初始姿态
    if not run_arm_script("0.1 双臂回初始姿态", SCRIPT_POSE,
                          pose_file=POSE_INITIAL_FILE, dry_run=dry_run):
        print(f"\n⚠⚠ 双臂回初始姿态失败, 终止流程")
        return False

    # 2. 张开夹爪
    print(f"\n{'='*60}")
    print(f"[0.2 张开夹爪] 开始")
    print(f"{'='*60}")
    if dry_run:
        print(f"[0.2 张开夹爪] [DRY-RUN] 跳过")
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_OPEN_POS,
                          right_pos=RIGHT_GRIPPER_OPEN_POS,
                          tag="0.2 张开夹爪")
        if not ok:
            print(f"\n⚠⚠ 张开夹爪失败, 终止流程")
            return False
        print(f"[0.2 张开夹爪] ✓ 完成")

    print(f"\n[初始化] ✓ 全部完成")
    return True


# ═══════════════════════════════════════════════════════════
#  Part B2: 取货动作 (待机 → 预抓取 → 抓取 → 夹爪关闭 → 抬升 → 待机)
# ═══════════════════════════════════════════════════════════
def run_pick_action(dry_run=False, pause=0.3, gripper_wait=0.2):
    """Part B2: 取货动作

    流程:
        1. 待机姿态 (pose_standby.json)
        2. 预抓取姿态 (pose_pre_pick.json)
        3. 抓取姿态 (pose_pick.json)
        4. 夹爪关闭
        5. 抬升姿态 (pose_lift.json)
        6. 回到待机姿态 (pose_standby.json)

    Args:
        dry_run: 只打印不执行
        pause: 步骤间停顿秒数
        gripper_wait: 夹爪动作完成后额外等待秒数
    Returns:
        bool: True=成功
    """
    print(f"\n{'#'*60}")
    print(f"# Part B2: 取货动作")
    print(f"  顺序: 待机 → 预抓取 → 抓取 → 夹爪关闭 → 抬升 → 待机")
    print(f"{'#'*60}")

    t_start = time.time()

    # ── 步骤 1: 待机姿态 ──
    if not run_arm_script("1.待机姿态", SCRIPT_STANDBY, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 2: 预抓取姿态 ──
    if not run_arm_script("2.预抓取姿态", SCRIPT_PRE_PICK, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 3: 抓取姿态 ──
    if not run_arm_script("3.抓取姿态", SCRIPT_PICK, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 4: 夹爪关闭 ──
    print(f"\n{'='*60}")
    print(f"[4.夹爪关闭] 开始")
    print(f"{'='*60}")
    if dry_run:
        print(f"[4.夹爪关闭] [DRY-RUN] 跳过")
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_CLOSE_POS,
                          right_pos=RIGHT_GRIPPER_CLOSE_POS,
                          wait=gripper_wait, tag="4.夹爪关闭")
        if not ok:
            print(f"\n⚠⚠ 夹爪关闭失败, 终止流程")
            return False
        print(f"[4.夹爪关闭] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 5: 抬升姿态 ──
    if not run_arm_script("5.抬升姿态", SCRIPT_LIFT, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 6: 回到待机姿态 ──
    if not run_arm_script("6.回到待机姿态", SCRIPT_STANDBY, dry_run=dry_run):
        return False

    dt = time.time() - t_start
    print(f"\n[取货动作] ✓ 全部完成 (耗时 {dt:.1f}s)")
    return True


# ═══════════════════════════════════════════════════════════
#  Part E2: 放货动作
# ═══════════════════════════════════════════════════════════
def run_place_action(dry_run=False, pause=0.3, gripper_wait=0.2):
    """Part E2: 放货动作

    流程:
        1. 待机姿态 (pose_standby.json)
        2. 预放货姿态 (pose_pre_place.json)
        3. 放货姿态 (pose_place.json)
        4. 夹爪打开
        5. 回到待机姿态 (pose_standby.json)

    Args:
        dry_run: 只打印不执行
        pause: 步骤间停顿秒数
        gripper_wait: 夹爪动作完成后额外等待秒数
    Returns:
        bool: True=成功
    """
    print(f"\n{'#'*60}")
    print(f"# Part E2: 放货动作")
    print(f"  顺序: 待机 → 预放货 → 放货 → 夹爪打开 → 待机")
    print(f"{'#'*60}")

    t_start = time.time()

    # ── 步骤 1: 待机姿态 ──
    if not run_arm_script("1.待机姿态", SCRIPT_STANDBY, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 2: 预放货姿态 ──
    if not run_arm_script("2.预放货姿态", SCRIPT_PRE_PLACE, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 3: 放货姿态 ──
    if not run_arm_script("3.放货姿态", SCRIPT_PLACE, dry_run=dry_run):
        return False
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 4: 夹爪打开 ──
    print(f"\n{'='*60}")
    print(f"[4.夹爪打开] 开始")
    print(f"{'='*60}")
    if dry_run:
        print(f"[4.夹爪打开] [DRY-RUN] 跳过")
    else:
        ok = send_gripper(left_pos=LEFT_GRIPPER_OPEN_POS,
                          right_pos=RIGHT_GRIPPER_OPEN_POS,
                          wait=gripper_wait, tag="4.夹爪打开")
        if not ok:
            print(f"\n⚠⚠ 夹爪打开失败, 终止流程")
            return False
        print(f"[4.夹爪打开] ✓ 完成")
    if pause > 0:
        time.sleep(pause)

    # ── 步骤 5: 回到待机姿态 ──
    if not run_arm_script("5.回到待机姿态", SCRIPT_STANDBY, dry_run=dry_run):
        return False

    dt = time.time() - t_start
    print(f"\n[放货动作] ✓ 全部完成 (耗时 {dt:.1f}s)")
    return True


# ═══════════════════════════════════════════════════════════
#  Part A: 地图导航
# ═══════════════════════════════════════════════════════════
def run_nav(g2, points, dry_run=False):
    """按顺序导航到地图点
    Args:
        g2: minth.G2 实例
        points: 导航点列表, 如 [0, 1, 2]
        dry_run: 只打印不执行
    Returns:
        bool: True=全部到达, False=任一点失败
    """
    print(f"\n{'='*60}")
    print(f"Part A: 地图导航")
    print(f"  路径: {' → '.join(str(p) for p in points)}")
    print(f"{'='*60}")

    if dry_run:
        for p in points:
            print(f"[导航] [DRY-RUN] g2.GO({p})")
        print(f"[导航] [DRY-RUN] ✓ 完成")
        return True

    for i, pt in enumerate(points):
        print(f"\n[导航] 第 {i+1}/{len(points)} 段: g2.GO({pt})")
        t0 = time.time()
        try:
            ok = g2.GO(pt)
        except Exception as e:
            print(f"[导航] ✗ g2.GO({pt}) 异常: {e}")
            return False
        dt = time.time() - t0
        if ok:
            print(f"[导航] ✓ 到达 {pt} 号点 (耗时 {dt:.1f}s)")
        else:
            print(f"[导航] ✗ 到达 {pt} 号点失败 (耗时 {dt:.1f}s)")
            print(f"[导航] ⚠⚠ 终止后续导航和流程")
            return False
        # 点间小停顿
        if i < len(points) - 1:
            time.sleep(1.0)

    print(f"\n[导航] ✓ 全部导航完成")
    return True


# ═══════════════════════════════════════════════════════════
#  Part B: 取货三步纠偏 (best_new.ref)
# ═══════════════════════════════════════════════════════════
def run_pick_correct(g2, model_pick, target_depth, dry_run=False):
    """取货三步纠偏: 前后 → 角度 → 左右
    Args:
        g2: minth.G2 实例
        model_pick: best_new.ref RhinoInfer 实例
        target_depth: 前后纠偏目标深度 mm
        dry_run: 只打印不执行
    Returns:
        (bool, dict): (是否无异常, 结果字典)
    """
    print(f"\n{'='*60}")
    print(f"Part B: 取货三步纠偏 (best_new.ref, 辉羲 RPU 推理)")
    print(f"  顺序: 前后 → 角度 → 左右")
    print(f"  目标深度: {target_depth}mm")
    print(f"  结果保存: {PICK_OUTPUT}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[取货纠偏] [DRY-RUN] 跳过")
        return True, {}

    results = {}
    reuse_img = None
    abort = False

    # 前后纠偏
    r_fb = cca.step_fb_correct(model_pick, g2, target_depth, PICK_OUTPUT, dry_run=False)
    results["fb"] = r_fb
    if not r_fb.get("success"):
        print(f"[取货纠偏] ⚠⚠ 前后纠偏失败: {r_fb.get('reason')}")
        abort = True
    reuse_img = r_fb.get("color_img")

    # 角度纠偏
    if not abort:
        r_yaw = cca.step_yaw_correct(model_pick, g2, PICK_OUTPUT,
                                     dry_run=False, reuse_img=reuse_img)
        results["yaw"] = r_yaw
        if not r_yaw.get("success"):
            print(f"[取货纠偏] ⚠⚠ 角度纠偏失败: {r_yaw.get('reason')}")
            abort = True
        reuse_img = r_yaw.get("color_img")

    # 左右纠偏
    if not abort:
        r_lr = cca.step_lr_correct(model_pick, g2, PICK_OUTPUT,
                                   dry_run=False, reuse_img=reuse_img)
        results["lr"] = r_lr
        if not r_lr.get("success"):
            print(f"[取货纠偏] ⚠⚠ 左右纠偏失败: {r_lr.get('reason')}")
            abort = True

    # 汇总
    fb_ok  = r_fb.get("success") and r_fb.get("converged", False)
    yaw_ok = (not abort and "yaw" in results
              and r_yaw.get("success") and r_yaw.get("converged", False))
    lr_ok  = (not abort and "lr" in results
              and r_lr.get("success") and r_lr.get("converged", False))
    print(f"\n[取货纠偏] 前后={'✓' if fb_ok else '⚠'}  "
          f"角度={'✓' if yaw_ok else '⚠'}  "
          f"左右={'✓' if lr_ok else '⚠'}")

    return (not abort), results


# ═══════════════════════════════════════════════════════════
#  Part C: 底盘旋转 90°
# ═══════════════════════════════════════════════════════════
def run_rotate(g2, rotate_rad=ROTATE_RAD, dry_run=False):
    """底盘逆时针旋转 90°
    Args:
        g2: minth.G2 实例
        rotate_rad: 旋转角度 (弧度), 正=逆时针
        dry_run: 只打印不执行
    Returns:
        bool: True=成功
    """
    deg = math.degrees(rotate_rad)
    print(f"\n{'='*60}")
    print(f"Part C: 底盘旋转")
    print(f"  角度: {deg:+.1f}° ({'逆时针' if rotate_rad > 0 else '顺时针'})")
    print(f"{'='*60}")

    if dry_run:
        print(f"[旋转] [DRY-RUN] g2.REL(yaw_rad={rotate_rad:.4f})")
        return True

    t0 = time.time()
    try:
        ok = g2.REL({"x": 0.0, "y": 0.0, "yaw_rad": rotate_rad})
    except Exception as e:
        print(f"[旋转] ✗ 异常: {e}")
        return False
    dt = time.time() - t0
    if ok:
        print(f"[旋转] ✓ 完成 (耗时 {dt:.1f}s)")
    else:
        print(f"[旋转] ✗ 失败 (耗时 {dt:.1f}s)")
    return ok


# ═══════════════════════════════════════════════════════════
#  Part D: 放货三步纠偏 (jitai_new.ref)
# ═══════════════════════════════════════════════════════════
def run_place_correct(g2, model_place, target_depth, dry_run=False):
    """放货三步纠偏 (角度 → 前后 → 左右)
    Args:
        g2: minth.G2 实例
        model_place: jitai_new.ref RhinoInfer 实例
        target_depth: 前后纠偏目标深度 mm
        dry_run: 只打印不执行
    Returns:
        (bool, dict): (是否无异常, 结果字典)
    """
    print(f"\n{'='*60}")
    print(f"Part D: 放货三步纠偏 (jitai_new.ref, 辉羲 RPU 推理)")
    print(f"  顺序: 角度 → 前后 → 左右")
    print(f"  前后目标深度: {target_depth}mm")
    print(f"  结果保存: {PLACE_OUTPUT}")
    print(f"{'='*60}")

    os.makedirs(PLACE_OUTPUT, exist_ok=True)

    if dry_run:
        print(f"[放货纠偏] [DRY-RUN] 跳过")
        return True, {}

    results = {}

    # 步骤 1/3: 角度纠偏 (必须拍照)
    r_yaw = jca.step_yaw_correct(model_place, g2, PLACE_OUTPUT,
                                 dry_run=False, reuse_img=None)
    results["yaw"] = r_yaw
    if not r_yaw.get("success"):
        print(f"[放货纠偏] ⚠⚠ 角度纠偏失败: {r_yaw.get('reason')}")
        return False, results
    yaw_ok = r_yaw.get("success") and r_yaw.get("converged", False)
    # 角度纠偏后位置已变, 不复用图像
    last_img = None

    # 步骤 2/3: 前后纠偏 (不复用图像, 重新拍照)
    r_fb = jca.step_fb_correct(model_place, g2, target_depth, PLACE_OUTPUT,
                               dry_run=False)
    results["fb"] = r_fb
    if not r_fb.get("success"):
        print(f"[放货纠偏] ⚠⚠ 前后纠偏失败: {r_fb.get('reason')}")
        return False, results
    fb_ok = r_fb.get("success") and r_fb.get("converged", False)
    last_img = r_fb.get("last_color_path")

    # 步骤 3/3: 左右纠偏 (复用前一步彩色图)
    r_lr = jca.step_lr_correct(model_place, g2, PLACE_OUTPUT,
                               dry_run=False, reuse_img=last_img)
    results["lr"] = r_lr
    if not r_lr.get("success"):
        print(f"[放货纠偏] ⚠⚠ 左右纠偏失败: {r_lr.get('reason')}")
        return False, results
    lr_ok = r_lr.get("success") and r_lr.get("converged", False)

    print(f"\n[放货纠偏] 角度={'✓' if yaw_ok else '⚠'}  "
          f"前后={'✓' if fb_ok else '⚠'}  "
          f"左右={'✓' if lr_ok else '⚠'}")
    return True, results


# ═══════════════════════════════════════════════════════════
#  Part E: 底盘后退 (放货完成后)
# ═══════════════════════════════════════════════════════════
BACKOFF_DISTANCE_M = 3.0   # 后退距离 (米)


def run_backoff(g2, distance_m, dry_run=False):
    """底盘后退指定距离
    Args:
        g2: minth.G2 实例
        distance_m: 后退距离 (米, 正数)
        dry_run: 只打印不执行
    Returns:
        bool: True=成功
    """
    print(f"\n{'='*60}")
    print(f"Part E: 底盘后退")
    print(f"  距离: {distance_m:.2f}m")
    print(f"{'='*60}")

    if dry_run:
        print(f"[后退] [DRY-RUN] g2.REL(x={-distance_m:.3f})")
        return True

    t0 = time.time()
    try:
        ok = g2.REL({"x": -distance_m, "y": 0.0, "yaw_rad": 0.0})
    except Exception as e:
        print(f"[后退] ✗ 异常: {e}")
        return False
    dt = time.time() - t0
    if ok:
        print(f"[后退] ✓ 完成 (耗时 {dt:.1f}s)")
    else:
        print(f"[后退] ✗ 失败 (耗时 {dt:.1f}s)")
    return ok


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="地图导航 + 取货纠偏 + 转向 + 放货纠偏 (辉羲 RPU 推理)"
    )
    p.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p.add_argument("--skip-init", action="store_true", help="跳过初始化 (双臂回初始+张开夹爪)")
    p.add_argument("--skip-nav", action="store_true", help="跳过地图导航")
    p.add_argument("--skip-pick", action="store_true", help="跳过取货纠偏")
    p.add_argument("--skip-pick-action", action="store_true", help="跳过取货动作 (双臂+夹爪)")
    p.add_argument("--skip-rotate", action="store_true", help="跳过旋转 90°")
    p.add_argument("--skip-place", action="store_true", help="跳过放货纠偏")
    p.add_argument("--skip-place-action", action="store_true", help="跳过放货动作 (双臂+夹爪)")
    p.add_argument("--skip-backoff", action="store_true", help="跳过底盘后退 3m")
    p.add_argument("--target-depth", type=int, default=750,
                   help="取货前后纠偏目标深度 mm (默认 750)")
    p.add_argument("--place-target-depth", type=int, default=800,
                   help="放货前后纠偏目标深度 mm (默认 800)")
    return p.parse_args()


def main():
    args = parse_args()
    t_total_start = time.time()

    print(f"{'='*60}")
    print(f"地图导航 + 取货纠偏 + 转向 + 放货纠偏")
    print(f"  导航路径: {' → '.join(str(p) for p in NAV_POINTS)}")
    print(f"  取货模型: best_new.ref (辉羲 RPU)")
    print(f"  放货模型: jitai_new.ref (辉羲 RPU)")
    print(f"  旋转角度: {math.degrees(ROTATE_RAD):+.1f}° (逆时针)")
    print(f"  取货目标深度: {args.target_depth}mm")
    print(f"  放货目标深度: {args.place_target_depth}mm")
    print(f"  后退距离: {BACKOFF_DISTANCE_M}m")
    print(f"  Dry Run: {args.dry_run}")
    print(f"{'='*60}")

    # ─────────────────────────────────────────────────
    # 1. 加载模型 (辉羲 RPU 推理)
    # ─────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n[初始化] 加载辉羲 RPU 模型...")
        print(f"[初始化] 取货模型: {cca.RHINO_REF_MODEL}")
        model_pick = RhinoInfer(cca.RHINO_REF_MODEL)
        print(f"[初始化] ✓ 取货模型就绪")

        print(f"[初始化] 放货模型: {jca.RHINO_REF_MODEL}")
        model_place = RhinoInfer(jca.RHINO_REF_MODEL)
        print(f"[初始化] ✓ 放货模型就绪")
    else:
        model_pick = None
        model_place = None

    # ─────────────────────────────────────────────────
    # 2. 连接机器人
    # ─────────────────────────────────────────────────
    if args.dry_run:
        print(f"[初始化] [DRY-RUN] 跳过机器人连接")
        g2 = None
    else:
        print(f"\n[初始化] 连接机器人...")
        # 复用 cca 的 setup_minth (timeout=120)
        g2 = cca.setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")

    # ─────────────────────────────────────────────────
    # 3. Part 0: 初始化 (双臂回初始姿态 + 张开夹爪)
    # ─────────────────────────────────────────────────
    if not args.skip_init:
        init_ok = run_init(dry_run=args.dry_run)
        if not init_ok:
            print(f"\n⚠⚠ 初始化失败, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part 0: 初始化")

    # ─────────────────────────────────────────────────
    # 4. Part A: 地图导航
    # ─────────────────────────────────────────────────
    if not args.skip_nav:
        nav_ok = run_nav(g2, NAV_POINTS, dry_run=args.dry_run)
        if not nav_ok:
            print(f"\n⚠⚠ 地图导航失败, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part A: 地图导航")

    # ─────────────────────────────────────────────────
    # 5. Part B: 取货三步纠偏
    # ─────────────────────────────────────────────────
    if not args.skip_pick:
        pick_ok, pick_results = run_pick_correct(
            g2, model_pick, args.target_depth, dry_run=args.dry_run
        )
        if not pick_ok:
            print(f"\n⚠⚠ 取货纠偏异常, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part B: 取货纠偏")

    # ─────────────────────────────────────────────────
    # 6. Part B2: 取货动作 (待机 → 预抓取 → 抓取 → 夹爪关闭 → 抬升 → 待机)
    # ─────────────────────────────────────────────────
    if not args.skip_pick_action:
        pick_action_ok = run_pick_action(dry_run=args.dry_run)
        if not pick_action_ok:
            print(f"\n⚠⚠ 取货动作失败, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part B2: 取货动作")

    # ─────────────────────────────────────────────────
    # 7. Part C: 底盘旋转 90°
    # ─────────────────────────────────────────────────
    if not args.skip_rotate:
        rot_ok = run_rotate(g2, ROTATE_RAD, dry_run=args.dry_run)
        if not rot_ok:
            print(f"\n⚠⚠ 底盘旋转失败, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part C: 底盘旋转")

    # ─────────────────────────────────────────────────
    # 6. Part D: 放货三步纠偏 (角度 → 前后 → 左右)
    # ─────────────────────────────────────────────────
    if not args.skip_place:
        place_ok, place_results = run_place_correct(
            g2, model_place, args.place_target_depth, dry_run=args.dry_run
        )
        if not place_ok:
            print(f"\n⚠⚠ 放货纠偏异常, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part D: 放货纠偏")

    # ─────────────────────────────────────────────────
    # 7. Part E2: 放货动作 (待机 → 预放货 → 放货 → 夹爪打开 → 待机)
    # ─────────────────────────────────────────────────
    if not args.skip_place_action:
        place_action_ok = run_place_action(dry_run=args.dry_run)
        if not place_action_ok:
            print(f"\n⚠⚠ 放货动作失败, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part E2: 放货动作")

    # ─────────────────────────────────────────────────
    # 8. Part E: 底盘后退 3m (放货完成后)
    # ─────────────────────────────────────────────────
    if not args.skip_backoff:
        back_ok = run_backoff(g2, BACKOFF_DISTANCE_M, dry_run=args.dry_run)
        if not back_ok:
            print(f"\n⚠⚠ 底盘后退失败, 终止流程")
            if g2 is not None:
                g2.close()
            return 1
    else:
        print(f"\n[跳过] Part E: 底盘后退")

    # ─────────────────────────────────────────────────
    # 8. 结束
    # ─────────────────────────────────────────────────
    if g2 is not None:
        g2.close()

    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"全流程完成 ✓  总耗时 {t_total:.1f}s")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
