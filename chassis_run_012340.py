#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
底盘地图点位顺序运行程序

按 0 → 1 → 2 → 3 → 4 → 0 的顺序导航到当前使用中地图的指定点位。
每个点位之间停留 2 秒，便于观察和后续动作衔接。

导航接口: G2.GO(num)
    - 通过 MQTT 向 /humanoid/commands/data 发送 {"command":"go","data":num}
    - 等待 /humanoid/commands/done 回复，确认到达
    - 超时 120 秒返回失败

用法:
  python chassis_run_012340.py                  # 默认顺序 0-1-2-3-4-0
  python chassis_run_012340.py --points 0 1 2 0 # 自定义点位顺序
  python chassis_run_012340.py --broker localhost --port 1883
  python chassis_run_012340.py --pause 3        # 每个点停留 3 秒
  python chassis_run_012340.py --skip-fail      # 某点失败也继续下一步
"""
import argparse
import sys
import time

# ===================== 路径配置 (可后续修改) =====================
# Minth 控制库路径
MINTH_DIR = "/data/wxf/wxf0721/runtime"
# MQTT 配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# 默认点位顺序
DEFAULT_POINTS = [0, 1, 2, 3, 4, 0]

# 每个点到达后停留时间 (秒)
DEFAULT_PAUSE = 2.0

# 导航超时 (秒) — GO 命令路径规划+移动耗时较长
NAV_TIMEOUT = 120


# ═══════════════════════════════════════════════════════════
#  机器人控制
# ═══════════════════════════════════════════════════════════

def setup_minth(broker, port, timeout):
    """加载 minth 模块并初始化 G2 控制"""
    if MINTH_DIR not in sys.path:
        sys.path.insert(0, MINTH_DIR)
    import minth
    g2 = minth.G2(broker=broker, port=port, timeout=timeout)
    return g2


# ═══════════════════════════════════════════════════════════
#  顺序导航
# ═══════════════════════════════════════════════════════════

def run_points(g2, points, pause, skip_fail):
    """按顺序导航到每个点位

    Parameters
    ----------
    g2 : minth.G2
        已连接的 G2 控制实例
    points : list[int]
        点位编号顺序
    pause : float
        每个点到达后停留时间 (秒)
    skip_fail : bool
        True: 某点失败也继续; False: 任一点失败即终止

    Returns
    -------
    dict
        {"total": N, "success": M, "failed_points": [...]}
    """
    total = len(points)
    success = 0
    failed = []

    print(f"\n{'='*60}")
    print(f"开始顺序导航: {' → '.join(str(p) for p in points)}")
    print(f"共 {total} 个点位, 每点停留 {pause:.1f}s")
    print(f"{'='*60}")

    t_start = time.time()

    for i, pt in enumerate(points, 1):
        print(f"\n[{i}/{total}] 导航到 {pt} 号点 ...")
        t0 = time.time()
        ok = g2.GO(pt)
        dt = time.time() - t0

        if ok:
            success += 1
            print(f"✓ 到达 {pt} 号点 (耗时 {dt:.1f}s)")
            if pause > 0:
                time.sleep(pause)
        else:
            failed.append(pt)
            print(f"✗ 导航到 {pt} 号点失败 (超时 {dt:.1f}s)")
            if not skip_fail:
                print(f"! 终止后续导航 (使用 --skip-fail 可继续)")
                break

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"导航完成: 成功 {success}/{total}, 失败 {len(failed)} 个 {failed if failed else ''}")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"{'='*60}")

    return {"total": total, "success": success, "failed_points": failed}


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    global MQTT_BROKER, MQTT_PORT
    parser = argparse.ArgumentParser(description="底盘地图点位顺序运行")
    parser.add_argument("--points", type=int, nargs="+", default=DEFAULT_POINTS,
                        help=f"点位编号顺序 (默认: {' '.join(map(str, DEFAULT_POINTS))})")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE,
                        help=f"每个点到达后停留时间(秒), 默认 {DEFAULT_PAUSE}")
    parser.add_argument("--skip-fail", action="store_true",
                        help="某点导航失败也继续下一步 (默认遇错即停)")
    parser.add_argument("--broker", default=MQTT_BROKER,
                        help=f"MQTT broker (默认: {MQTT_BROKER})")
    parser.add_argument("--port", type=int, default=MQTT_PORT,
                        help=f"MQTT 端口 (默认: {MQTT_PORT})")
    parser.add_argument("--timeout", type=int, default=NAV_TIMEOUT,
                        help=f"导航超时(秒), 默认 {NAV_TIMEOUT}")
    args = parser.parse_args()

    MQTT_BROKER = args.broker
    MQTT_PORT = args.port

    # 连接机器人
    print(f"[连接] MQTT {MQTT_BROKER}:{MQTT_PORT} (超时 {args.timeout}s) ...")
    g2 = setup_minth(MQTT_BROKER, MQTT_PORT, args.timeout)
    print(f"[连接] ✓ Minth G2 已就绪")

    try:
        run_points(g2, args.points, args.pause, args.skip_fail)
    except KeyboardInterrupt:
        print(f"\n[中断] 用户取消")
    finally:
        g2.close()
        print(f"[连接] 已关闭")


if __name__ == "__main__":
    main()
