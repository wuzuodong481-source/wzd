#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""full_pipeline.py — 机器人流程总控

流程:
  1. 双臂回到初始化姿态 (pose_initial.json)
  2. 底盘按 0→1→2→3→4→0 顺序导航
  3. 执行底盘综合纠偏 (前后 → 角度 → 左右)
  4. 双臂调整到 134903.json 记录的姿态
  5. 双臂调整到 142909.json 记录的姿态 (结束姿态)

依赖程序 (按调用顺序):
  - /data/wzd/execute/move_arms_to_pose.py   (双臂位姿移动)
  - /data/wzd/chassis_run_012340.py            (底盘顺序导航)
  - /data/wzd/chassis_correct_all.py           (底盘综合纠偏)

用法:
  python full_pipeline.py                        # 完整流程
  python full_pipeline.py --skip-nav             # 跳过导航
  python full_pipeline.py --skip-correct         # 跳过纠偏
  python full_pipeline.py --dry-run              # 只打印, 不实际执行
  python full_pipeline.py --python-bin /path/py  # 自定义Python解释器
"""
import argparse
import os
import subprocess
import sys
import time

# ===================== 路径配置 =====================
WZD_DIR = "/data/wzd"
EXEC_DIR = os.path.join(WZD_DIR, "execute")
POSE_DIR = os.path.join(WZD_DIR, "position")

POSE_INITIAL = os.path.join(POSE_DIR, "pose_initial.json")
POSE_134903  = os.path.join(POSE_DIR, "pose_20260810_134903.json")
POSE_142909  = os.path.join(POSE_DIR, "pose_20260810_161035.json")

MOVE_ARMS    = os.path.join(EXEC_DIR, "move_arms_to_pose.py")
CHASSIS_RUN  = os.path.join(WZD_DIR, "chassis_run_012340.py")
CHASSIS_CORR = os.path.join(WZD_DIR, "chassis_correct_all.py")
INFER_SERVER = os.path.join(WZD_DIR, "yolo_infer_server.py")

# 默认 Python 解释器 (含 ultralytics + paho-mqtt)
DEFAULT_PY = "/data/wxf/wxf/yolo/yolo-env/bin/python"


# ═══════════════════════════════════════════════════════════
#  步骤封装
# ═══════════════════════════════════════════════════════════

def run_step(name, cmd, dry_run=False):
    """执行一个子步骤

    Parameters
    ----------
    name : str  步骤名称
    cmd : list  命令列表 (不含 python 解释器, 调用方提供)
    dry_run : bool  True 则只打印不执行

    Returns
    -------
    dict  {"ok": bool, "returncode": int}
        ok=True: 正常完成 (退出码 0)
        ok=False, returncode=2: 异常崩溃
        ok=False, returncode=其他: 正常退出但失败 (如未收敛)
    """
    print(f"\n{'='*60}")
    print(f"[{name}] 开始执行")
    print(f"[{name}] 命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[{name}] [DRY-RUN] 跳过执行")
        return {"ok": True, "returncode": 0}

    t0 = time.time()
    try:
        ret = subprocess.run(cmd, check=False)
        dt = time.time() - t0
        if ret.returncode == 0:
            print(f"\n[{name}] ✓ 成功 (耗时 {dt:.1f}s)")
            return {"ok": True, "returncode": 0}
        else:
            tag = "异常崩溃" if ret.returncode == 2 else "失败"
            print(f"\n[{name}] ✗ {tag} (返回码 {ret.returncode}, 耗时 {dt:.1f}s)")
            return {"ok": False, "returncode": ret.returncode}
    except Exception as e:
        print(f"\n[{name}] ✗ 异常: {e}")
        return {"ok": False, "returncode": -1}


def step_move_arms(name, pose_file, py_bin, dry_run=False, timeout=30.0):
    """双臂运动到指定 JSON 位姿"""
    if not os.path.exists(pose_file):
        print(f"[{name}] ✗ 位姿文件不存在: {pose_file}")
        return {"ok": False, "returncode": 2}  # 视为崩溃
    cmd = [py_bin, "-u", MOVE_ARMS, "--pose", pose_file, "--timeout", str(timeout)]
    return run_step(name, cmd, dry_run=dry_run)


def step_chassis_nav(py_bin, points, pause, dry_run=False, skip_fail=False):
    """底盘顺序导航"""
    cmd = [py_bin, "-u", CHASSIS_RUN,
           "--points", *map(str, points),
           "--pause", str(pause)]
    if skip_fail:
        cmd.append("--skip-fail")
    return run_step("底盘导航", cmd, dry_run=dry_run)


def step_chassis_correct(py_bin, dry_run=False, remote_infer=False, infer_server_proc=None):
    """底盘综合纠偏 (前后→角度→左右)

    remote_infer=True 时自动启动推理服务器 (GPU), 纠偏完成后停止
    """
    cmd = [py_bin, "-u", CHASSIS_CORR]
    if remote_infer:
        cmd.append("--remote-infer")

    # 自动启动推理服务器 (如果未启动)
    if remote_infer and infer_server_proc is None and not dry_run:
        print(f"\n[推理服务器] 启动 GPU 推理服务...")
        server_cmd = [py_bin, "-u", INFER_SERVER, "--device", "0"]
        infer_server_proc = subprocess.Popen(
            server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # 等待服务器就绪 (检测 MQTT 订阅成功)
        t0 = time.time()
        ready = False
        while time.time() - t0 < 30:
            line = infer_server_proc.stdout.readline()
            if not line:
                if infer_server_proc.poll() is not None:
                    print(f"[推理服务器] ✗ 启动失败")
                    return {"ok": False, "returncode": 2}  # 视为崩溃
                continue
            txt = line.decode("utf-8", errors="ignore").rstrip()
            print(f"[推理服务器] {txt}")
            if "订阅" in txt or "等待请求" in txt:
                ready = True
                break
        if not ready:
            print(f"[推理服务器] ✗ 启动超时")
            infer_server_proc.terminate()
            return {"ok": False, "returncode": 2}  # 视为崩溃
        print(f"[推理服务器] ✓ 已就绪")

    result = run_step("底盘纠偏", cmd, dry_run=dry_run)

    # 停止推理服务器
    if remote_infer and infer_server_proc is not None:
        print(f"\n[推理服务器] 停止 GPU 推理服务...")
        infer_server_proc.terminate()
        try:
            infer_server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            infer_server_proc.kill()
        print(f"[推理服务器] ✓ 已停止")

    return result


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="机器人流程总控")
    parser.add_argument("--python-bin", default=DEFAULT_PY,
                        help=f"Python 解释器 (默认: {DEFAULT_PY})")
    parser.add_argument("--points", type=int, nargs="+", default=[0,1,2,3,4,0],
                        help="导航点位顺序 (默认: 0 1 2 3 4 0)")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="每个导航点停留时间(秒), 默认 2.0")
    parser.add_argument("--skip-nav", action="store_true", help="跳过底盘导航")
    parser.add_argument("--skip-correct", action="store_true", help="跳过底盘纠偏")
    parser.add_argument("--skip-init-pose", action="store_true", help="跳过回到初始化姿态")
    parser.add_argument("--skip-final-poses", action="store_true",
                        help="跳过结束前的双臂姿态调整")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="任一步失败即终止 (默认: 继续)")
    parser.add_argument("--remote-infer", action="store_true",
                        help="启用 GPU 远程推理 (自动启动/停止 yolo_infer_server.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印流程, 不实际执行")
    args = parser.parse_args()

    py = args.python_bin
    dry = args.dry_run

    print("=" * 60)
    print("机器人流程总控")
    print("=" * 60)
    print(f"  Python: {py}")
    print(f"  导航点位: {' → '.join(map(str, args.points))}")
    print(f"  每点停留: {args.pause}s")
    print(f"  初始化姿态: {POSE_INITIAL}")
    print(f"  中间姿态: {POSE_134903}")
    print(f"  结束姿态: {POSE_142909}")
    print(f"  GPU推理: {'启用' if args.remote_infer else '本地CPU'}")
    print(f"  Dry-Run: {dry}")
    print("=" * 60)

    t_start = time.time()
    results = []

    def _is_crash(r):
        """判断是否为异常崩溃 (returncode == 2 或 -1)"""
        return (not r["ok"]) and r["returncode"] in (2, -1)

    # ── 步骤 1: 双臂回到初始化姿态 ──
    if not args.skip_init_pose:
        r = step_move_arms("步骤1-回到初始化姿态", POSE_INITIAL, py, dry_run=dry)
        results.append(("1.初始化姿态", r))
        if not r["ok"]:
            if _is_crash(r) or args.stop_on_fail:
                tag = "异常崩溃" if _is_crash(r) else "失败"
                print(f"\n! 步骤1{tag}, 终止流程")
                _summary(results, t_start)
                return 1

    # ── 步骤 2: 底盘顺序导航 ──
    if not args.skip_nav:
        r = step_chassis_nav(py, args.points, args.pause, dry_run=dry)
        results.append(("2.底盘导航", r))
        if not r["ok"]:
            if _is_crash(r) or args.stop_on_fail:
                tag = "异常崩溃" if _is_crash(r) else "失败"
                print(f"\n! 步骤2{tag}, 终止流程")
                _summary(results, t_start)
                return 1

    # ── 步骤 3: 底盘综合纠偏 ──
    if not args.skip_correct:
        r = step_chassis_correct(py, dry_run=dry, remote_infer=args.remote_infer)
        results.append(("3.底盘纠偏", r))
        if not r["ok"]:
            if _is_crash(r):
                # 崩溃: 必须终止 (后续双臂运动有风险)
                print("\n! 步骤3异常崩溃, 终止流程 (崩溃不可恢复)")
                _summary(results, t_start)
                return 1
            elif args.stop_on_fail:
                # 未收敛 + --stop-on-fail: 终止
                print("\n! 步骤3未收敛, 终止流程 (--stop-on-fail)")
                _summary(results, t_start)
                return 1
            else:
                # 未收敛, 默认继续 (双臂运动风险低)
                print("\n⚠ 步骤3纠偏未完全收敛, 继续后续双臂姿态调整")

    # ── 步骤 4: 双臂调整到 134903 姿态 ──
    if not args.skip_final_poses:
        r = step_move_arms("步骤4-调整到134903姿态", POSE_134903, py, dry_run=dry)
        results.append(("4.134903姿态", r))
        if not r["ok"]:
            if _is_crash(r) or args.stop_on_fail:
                tag = "异常崩溃" if _is_crash(r) else "失败"
                print(f"\n! 步骤4{tag}, 终止流程")
                _summary(results, t_start)
                return 1

        # ── 步骤 5: 双臂调整到 142909 姿态 ──
        r = step_move_arms("步骤5-调整到142909姿态", POSE_142909, py, dry_run=dry)
        results.append(("5.142909姿态", r))
        if not r["ok"]:
            if _is_crash(r) or args.stop_on_fail:
                tag = "异常崩溃" if _is_crash(r) else "失败"
                print(f"\n! 步骤5{tag}, 终止流程")
                _summary(results, t_start)
                return 1

    _summary(results, t_start)
    return 0


def _summary(results, t_start):
    """打印总结"""
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"流程总结 (总耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, r in results:
        if r["ok"]:
            status = "✓ 成功"
        elif r["returncode"] == 2:
            status = "✗ 异常崩溃"
        else:
            status = "⚠ 未收敛"
        print(f"  {name}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
