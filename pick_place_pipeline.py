#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pick_place_pipeline.py — 取放货流程总控

流程:
  1. 双臂回到初始化姿态 (pose_initial.json)
  2. 导航到取货点 (g2.GO)
  3. 取货纠偏: 前后→角度→左右 (best_new.pt)
  4. 取货姿态 (pose_pick.json)
  5. 粗转85° 逆时针 (move_chassis)
  6. 放货角度纠偏 (jitai.pt, ±0.5°)
  7. 放货姿态 (pose_place.json)

运行环境: 需使用含 ultralytics + paho-mqtt 的 Python 解释器
  python pick_place_pipeline.py                    # 完整流程
  python pick_place_pipeline.py --dry-run          # 只打印不执行
  python pick_place_pipeline.py --skip-nav         # 跳过导航
  python pick_place_pipeline.py --skip-pick-correct# 跳过取货纠偏
  python pick_place_pipeline.py --skip-place-correct # 跳过放货纠偏
  python pick_place_pipeline.py --python-bin PY    # 指定子脚本(手臂)解释器

★ 使用前需准备:
  1. 录制取货姿态: python function/get_current_pose.py --name pick
  2. 录制放货姿态: python function/get_current_pose.py --name place
  3. 修改下方 PICK_POINT 为实际取货点编号
  4. 确认 ROTATE_DEG 旋转方向 (正值=逆时针, 负值=顺时针)
  5. 确认 MQTT 服务 (main.py) 已启动
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

# ═══════════════════════════════════════════════════════════
#  ★★★ 路径配置 (使用前务必修改) ★★★
# ═══════════════════════════════════════════════════════════

WZD_DIR = "/data/wzd"
EXEC_DIR = os.path.join(WZD_DIR, "execute")
POSE_DIR = os.path.join(WZD_DIR, "position")

# 位姿 JSON 文件
POSE_INITIAL = os.path.join(POSE_DIR, "pose_initial.json")   # 初始化姿态
POSE_PICK    = os.path.join(POSE_DIR, "pose_pick.json")      # ← 取货姿态 (需录制)
POSE_PLACE   = os.path.join(POSE_DIR, "pose_place.json")     # ← 放货姿态 (需录制)

# YOLO 模型
MODEL_PICK  = os.path.join(WZD_DIR, "best_new.pt")   # 取货纠偏 (a/b 检测)
MODEL_PLACE = os.path.join(WZD_DIR, "jitai.pt")      # 放货纠偏 (a/b 检测)

# 底盘参数
PICK_POINT = 1                # ← 取货点编号 (地图点位)
ROTATE_DEG = 85.0             # ← 粗转角度 (正值=逆时针, 负值=顺时针)
TARGET_DEPTH = 750            # 取货纠偏目标深度 (mm)

# 输出目录
PICK_OUTPUT_DIR  = os.path.join(WZD_DIR, "correction_all")    # 取货纠偏结果
PLACE_OUTPUT_DIR = os.path.join(WZD_DIR, "correction_place")  # 放货纠偏结果

# 子脚本解释器 (用于调用 move_arms_to_pose.py)
DEFAULT_PY = "/data/wxf/wxf/yolo/yolo-env/bin/python"
MOVE_ARMS  = os.path.join(EXEC_DIR, "move_arms_to_pose.py")

# ═══════════════════════════════════════════════════════════
#  导入纠偏模块 (chassis_correct_all)
# ═══════════════════════════════════════════════════════════

sys.path.insert(0, WZD_DIR)
import chassis_correct_all as cca
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════
#  步骤封装
# ═══════════════════════════════════════════════════════════

def run_step(name, cmd, dry_run=False):
    """执行子脚本步骤 (subprocess)"""
    print(f"\n{'='*60}")
    print(f"[{name}] 开始")
    print(f"[{name}] 命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    if dry_run:
        print(f"[{name}] [DRY-RUN] 跳过")
        return {"ok": True, "returncode": 0}
    t0 = time.time()
    try:
        ret = subprocess.run(cmd, check=False)
        dt = time.time() - t0
        ok = ret.returncode == 0
        print(f"[{name}] {'✓ 成功' if ok else '✗ 失败'} (耗时 {dt:.1f}s)")
        return {"ok": ok, "returncode": ret.returncode}
    except Exception as e:
        print(f"[{name}] ✗ 异常: {e}")
        return {"ok": False, "returncode": -1}


def step_move_arms(name, pose_file, py_bin, dry_run=False, timeout=30.0):
    """双臂运动到 JSON 位姿 (subprocess 调用 move_arms_to_pose.py)"""
    if not os.path.exists(pose_file):
        print(f"[{name}] ✗ 位姿文件不存在: {pose_file}")
        return {"ok": False, "returncode": 2}
    cmd = [py_bin, "-u", MOVE_ARMS, "--pose", pose_file, "--timeout", str(timeout)]
    return run_step(name, cmd, dry_run=dry_run)


def step_nav(g2, point, dry_run=False):
    """导航到地图点位"""
    print(f"\n{'='*60}")
    print(f"[导航] 前往 {point} 号点")
    print(f"{'='*60}")
    if dry_run:
        print(f"[导航] [DRY-RUN] 跳过")
        return {"ok": True}
    t0 = time.time()
    ok = g2.GO(point)
    dt = time.time() - t0
    print(f"[导航] {'✓ 到达' if ok else '✗ 失败'} (耗时 {dt:.1f}s)")
    return {"ok": ok}


def step_rotate(g2, deg, dry_run=False):
    """底盘旋转指定角度 (度, 正值=逆时针)"""
    print(f"\n{'='*60}")
    print(f"[旋转] {deg:+.1f}° ({'逆时针' if deg > 0 else '顺时针'})")
    print(f"{'='*60}")
    if dry_run:
        print(f"[旋转] [DRY-RUN] 跳过")
        return {"ok": True}
    yaw_rad = np.radians(deg)
    ok = cca.move_chassis(g2, yaw_rad=yaw_rad)
    if ok:
        time.sleep(cca.YAW_SETTLE_TIME)
        print(f"[旋转] ✓ 完成 (等待 {cca.YAW_SETTLE_TIME}s 稳定)")
    else:
        print(f"[旋转] ✗ 失败")
    return {"ok": ok}


def step_pick_correct(model, g2, target_depth, dry_run=False):
    """取货纠偏: 前后 → 角度 → 左右 (使用 best_new.pt)"""
    print(f"\n{'='*60}")
    print(f"[取货纠偏] 前后→角度→左右 (模型: {os.path.basename(MODEL_PICK)})")
    print(f"{'='*60}")

    results = {}
    reuse_img = None

    # 前后纠偏
    r_fb = cca.step_fb_correct(model, g2, target_depth, PICK_OUTPUT_DIR, dry_run)
    results["fb"] = r_fb
    if not r_fb["success"]:
        print(f"[取货纠偏] ⚠ 前后纠偏失败: {r_fb.get('reason')}")
    reuse_img = r_fb.get("color_img")

    # 角度纠偏
    r_yaw = cca.step_yaw_correct(model, g2, PICK_OUTPUT_DIR, dry_run, reuse_img=reuse_img)
    results["yaw"] = r_yaw
    if not r_yaw["success"]:
        print(f"[取货纠偏] ⚠ 角度纠偏失败: {r_yaw.get('reason')}")
    reuse_img = r_yaw.get("color_img")

    # 左右纠偏
    r_lr = cca.step_lr_correct(model, g2, PICK_OUTPUT_DIR, dry_run, reuse_img=reuse_img)
    results["lr"] = r_lr
    if not r_lr["success"]:
        print(f"[取货纠偏] ⚠ 左右纠偏失败: {r_lr.get('reason')}")

    # 汇总
    fb_ok = r_fb.get("success") and r_fb.get("converged", False)
    yaw_ok = r_yaw.get("success") and r_yaw.get("converged", False)
    lr_ok = r_lr.get("success") and r_lr.get("converged", False)
    print(f"\n[取货纠偏] 前后={'✓' if fb_ok else '⚠'}  角度={'✓' if yaw_ok else '⚠'}  左右={'✓' if lr_ok else '⚠'}")
    return {"ok": fb_ok and yaw_ok and lr_ok, "details": results}


def step_place_correct(model, g2, dry_run=False):
    """放货角度纠偏 (仅角度, 使用 jitai.pt, ±0.5°)"""
    print(f"\n{'='*60}")
    print(f"[放货纠偏] 角度 (模型: {os.path.basename(MODEL_PLACE)}, 目标: ±{cca.YAW_THRESHOLD_DEG}°)")
    print(f"{'='*60}")

    r_yaw = cca.step_yaw_correct(model, g2, PLACE_OUTPUT_DIR, dry_run, reuse_img=None)
    converged = r_yaw.get("success") and r_yaw.get("converged", False)
    if converged:
        print(f"[放货纠偏] ✓ 收敛 (final_angle={r_yaw.get('final_angle', '?'):.2f}°)")
    else:
        print(f"[放货纠偏] ⚠ 未完全收敛: {r_yaw.get('reason', '达到最大迭代')}")
    return {"ok": converged, "details": r_yaw}


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="取放货流程总控")
    parser.add_argument("--python-bin", default=DEFAULT_PY,
                        help=f"子脚本(手臂)Python解释器 (默认: {DEFAULT_PY})")
    parser.add_argument("--target-depth", type=int, default=TARGET_DEPTH,
                        help=f"取货纠偏目标深度mm (默认: {TARGET_DEPTH})")
    parser.add_argument("--pick-point", type=int, default=PICK_POINT,
                        help=f"取货点编号 (默认: {PICK_POINT})")
    parser.add_argument("--rotate-deg", type=float, default=ROTATE_DEG,
                        help=f"粗转角度, 正值=逆时针 (默认: {ROTATE_DEG})")
    parser.add_argument("--skip-nav", action="store_true", help="跳过导航")
    parser.add_argument("--skip-pick-correct", action="store_true", help="跳过取货纠偏")
    parser.add_argument("--skip-place-correct", action="store_true", help="跳过放货纠偏")
    parser.add_argument("--skip-init-pose", action="store_true", help="跳过初始化姿态")
    parser.add_argument("--stop-on-fail", action="store_true", help="任一步失败即终止")
    parser.add_argument("--dry-run", action="store_true", help="只打印, 不执行")
    args = parser.parse_args()

    py = args.python_bin
    dry = args.dry_run

    print("=" * 60)
    print("取放货流程总控")
    print("=" * 60)
    print(f"  取货点: {args.pick_point} 号")
    print(f"  粗转: {args.rotate_deg:+.1f}°")
    print(f"  目标深度: {args.target_depth}mm")
    print(f"  取货模型: {MODEL_PICK}")
    print(f"  放货模型: {MODEL_PLACE}")
    print(f"  初始姿态: {POSE_INITIAL}")
    print(f"  取货姿态: {POSE_PICK}")
    print(f"  放货姿态: {POSE_PLACE}")
    print(f"  Dry-Run: {dry}")
    print("=" * 60)

    t_start = time.time()
    results = []

    def _record(name, r):
        results.append((name, r))
        return r["ok"]

    def _should_stop(r):
        return (not r["ok"]) and (args.stop_on_fail or r.get("returncode") in (2, -1))

    # ── 加载模型 ──
    model_pick = None
    model_place = None
    if not dry:
        print(f"\n[初始化] 加载取货模型: {MODEL_PICK}")
        model_pick = YOLO(MODEL_PICK)
        print(f"[初始化] 加载放货模型: {MODEL_PLACE}")
        model_place = YOLO(MODEL_PLACE)
        print(f"[初始化] ✓ 模型加载完成")
    else:
        print(f"\n[初始化] [DRY-RUN] 跳过模型加载")

    # ── 连接机器人 ──
    g2 = None
    if not dry:
        print(f"[初始化] 连接机器人...")
        g2 = cca.setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")
    else:
        print(f"[初始化] [DRY-RUN] 跳过机器人连接")

    os.makedirs(PICK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLACE_OUTPUT_DIR, exist_ok=True)

    try:
        # ── 步骤 1: 初始化姿态 ──
        if not args.skip_init_pose:
            r = step_move_arms("步骤1-初始化姿态", POSE_INITIAL, py, dry_run=dry)
            ok = _record("1.初始化姿态", r)
            if _should_stop(r):
                _summary(results, t_start)
                return 1

        # ── 步骤 2: 导航到取货点 ──
        if not args.skip_nav:
            r = step_nav(g2, args.pick_point, dry_run=dry)
            ok = _record("2.导航到取货点", r)
            if _should_stop(r):
                _summary(results, t_start)
                return 1

        # ── 步骤 3: 取货纠偏 ──
        if not args.skip_pick_correct:
            r = step_pick_correct(model_pick, g2, args.target_depth, dry_run=dry)
            ok = _record("3.取货纠偏", r)
            if _should_stop(r):
                _summary(results, t_start)
                return 1

        # ── 步骤 4: 取货姿态 ──
        r = step_move_arms("步骤4-取货姿态", POSE_PICK, py, dry_run=dry)
        ok = _record("4.取货姿态", r)
        if _should_stop(r):
            _summary(results, t_start)
            return 1

        # ── 步骤 5: 粗转 ──
        r = step_rotate(g2, args.rotate_deg, dry_run=dry)
        ok = _record("5.粗转", r)
        if _should_stop(r):
            _summary(results, t_start)
            return 1

        # ── 步骤 6: 放货角度纠偏 ──
        if not args.skip_place_correct:
            r = step_place_correct(model_place, g2, dry_run=dry)
            ok = _record("6.放货纠偏", r)
            if _should_stop(r):
                _summary(results, t_start)
                return 1

        # ── 步骤 7: 放货姿态 ──
        r = step_move_arms("步骤7-放货姿态", POSE_PLACE, py, dry_run=dry)
        ok = _record("7.放货姿态", r)
        if _should_stop(r):
            _summary(results, t_start)
            return 1

    finally:
        if g2 is not None:
            g2.close()
            print(f"\n[清理] Minth 连接已关闭")

    _summary(results, t_start)
    return 0


def _summary(results, t_start):
    """打印总结"""
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"取放货流程总结 (总耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, r in results:
        status = "✓ 成功" if r["ok"] else "✗ 失败"
        print(f"  {name}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
