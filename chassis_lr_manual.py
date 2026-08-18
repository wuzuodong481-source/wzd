#!/usr/bin/env python3
"""
底盘左右横移手动控制程序

支持两种模式:
  1. 交互模式: 启动后输入距离, 实时控制底盘左右横移
  2. 命令行模式: 通过参数直接指定移动距离

坐标系:
  正值 = 左移 (y+)
  负值 = 右移 (y-)
  单位: 米 (支持小数, 如 0.1 = 10cm)

用法:
  # 交互模式 (推荐)
  python chassis_lr_manual.py

  # 命令行模式
  python chassis_lr_manual.py 0.1        # 左移 0.1m
  python chassis_lr_manual.py -0.1       # 右移 0.1m
  python chassis_lr_manual.py 0.2        # 左移 0.2m

  # 自定义参数
  python chassis_lr_manual.py --broker localhost --port 1883
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


# ═══════════════════════════════════════════════════════════
#  机器人控制
# ═══════════════════════════════════════════════════════════

def setup_minth():
    """加载 minth 模块并初始化 G2 控制"""
    if MINTH_DIR not in sys.path:
        sys.path.insert(0, MINTH_DIR)
    import minth
    g2 = minth.G2(broker=MQTT_BROKER, port=MQTT_PORT, timeout=60)
    return g2


def move_chassis_relative(g2, dx_m=0.0, dy_m=0.0, yaw_rad=0.0):
    """底盘相对运动 (dx 前, dy 左, yaw 逆时针)"""
    return g2._send_and_wait("go_rel", {"x": dx_m, "y": dy_m, "yaw_rad": yaw_rad})


# ═══════════════════════════════════════════════════════════
#  交互模式
# ═══════════════════════════════════════════════════════════

def interactive_loop(g2):
    """交互式横移控制"""
    print(f"\n{'='*60}")
    print(f"底盘横移交互控制")
    print(f"{'='*60}")
    print(f"命令说明:")
    print(f"  输入数字       → 左移该距离(米), 正=左, 负=右")
    print(f"  示例:")
    print(f"    0.1   → 左移 10cm")
    print(f"    -0.1  → 右移 10cm")
    print(f"    0.05  → 左移 5cm")
    print(f"    -0.2  → 右移 20cm")
    print(f"")
    print(f"  q / quit / exit → 退出")
    print(f"  h / help        → 显示帮助")
    print(f"  0               → 原地不动 (测试连接)")
    print(f"{'='*60}")

    total_moved = 0.0  # 累计横移量 (正=净左移)

    while True:
        try:
            user_input = input(f"\n[{total_moved*100:+.1f}cm 累计] 输入横移距离(米)> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n退出")
            break

        if not user_input:
            continue

        # 命令处理
        cmd = user_input.lower()
        if cmd in ("q", "quit", "exit"):
            print(f"退出")
            break
        if cmd in ("h", "help", "?"):
            print(f"输入数字(米): 正=左移, 负=右移. 如 0.1=左10cm, -0.1=右10cm")
            continue

        # 解析距离
        try:
            distance_m = float(user_input)
        except ValueError:
            print(f"✗ 无效输入: {user_input} (请输入数字)")
            continue

        # 安全限制
        if abs(distance_m) > 0.5:
            confirm = input(f"⚠ 距离较大 ({distance_m}m), 确认执行? [y/N] ").strip().lower()
            if confirm != "y":
                print(f"已取消")
                continue

        # 执行移动
        direction = "左" if distance_m > 0 else ("右" if distance_m < 0 else "原地")
        print(f"→ {direction}移 {abs(distance_m)*100:.1f} cm ...")
        t0 = time.time()
        ok = move_chassis_relative(g2, dy_m=distance_m)
        dt = time.time() - t0

        if ok:
            total_moved += distance_m
            print(f"✓ 完成 (耗时 {dt:.1f}s)  累计: {total_moved*100:+.1f} cm")
        else:
            print(f"✗ 移动失败 (耗时 {dt:.1f}s)")

    print(f"\n{'='*60}")
    print(f"会话结束. 累计横移: {total_moved*100:+.1f} cm")
    print(f"{'='*60}")
    return total_moved


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    global MQTT_BROKER, MQTT_PORT
    parser = argparse.ArgumentParser(description="底盘左右横移手动控制")
    parser.add_argument("distance", type=float, nargs="?", default=None,
                        help="横移距离(米), 正=左 负=右. 不指定则进入交互模式")
    parser.add_argument("--broker", default=MQTT_BROKER, help=f"MQTT broker (默认: {MQTT_BROKER})")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help=f"MQTT 端口 (默认: {MQTT_PORT})")
    args = parser.parse_args()

    MQTT_BROKER = args.broker
    MQTT_PORT = args.port

    # 连接机器人
    print(f"[连接] MQTT {MQTT_BROKER}:{MQTT_PORT} ...")
    g2 = setup_minth()
    print(f"[连接] ✓ Minth 已就绪")

    try:
        if args.distance is not None:
            # 命令行模式: 单次移动
            distance_m = args.distance
            direction = "左" if distance_m > 0 else ("右" if distance_m < 0 else "原地")
            print(f"\n→ {direction}移 {abs(distance_m)*100:.1f} cm ...")
            t0 = time.time()
            ok = move_chassis_relative(g2, dy_m=distance_m)
            dt = time.time() - t0
            if ok:
                print(f"✓ 完成 (耗时 {dt:.1f}s)")
            else:
                print(f"✗ 移动失败 (耗时 {dt:.1f}s)")
                sys.exit(1)
        else:
            # 交互模式
            interactive_loop(g2)
    finally:
        g2.close()


if __name__ == "__main__":
    main()
